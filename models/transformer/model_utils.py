# ============================================================
# Wrapper class (unchanged public API, now DDP-safe save/load)
# ============================================================

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TrajectoryTransformer", "TrajectoryModel"]

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE with dynamic extension: if seq_len > current tables, we rebuild cos/sin.
    head_dim must be even.
    """
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even.")
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = 0  # will be set by _build_tables
        # initialize empty buffers; we’ll build them on first use
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self._build_tables(max_seq_len, device=torch.device("cpu"))

    def _build_tables(self, seq_len: int, device, dtype=torch.float32):
        inv = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv)  # [S, d/2]
        cos = torch.cos(freqs)  # [S, d/2]
        sin = torch.sin(freqs)  # [S, d/2]
        # store in float32; cast later to match q/k dtype
        self.cos = cos
        self.sin = sin
        self.max_seq_len = seq_len

    def _maybe_extend(self, seq_len: int, device, dtype):
        if seq_len > self.max_seq_len or self.cos.device != device:
            # rebuild on the right device; keep tables in float32 for stability
            self._build_tables(seq_len, device=device)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

    def apply(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # ensure we have tables large enough and on the same device
        dev = q.device
        self._maybe_extend(seq_len, device=dev, dtype=q.dtype)

        # [S, d/2] -> repeat even/odd to [S, d], then broadcast to [1,1,S,d]
        cos = self.cos[:seq_len].repeat_interleave(2, dim=-1)[None, None, :, :]
        sin = self.sin[:seq_len].repeat_interleave(2, dim=-1)[None, None, :, :]
        # cast to q/k dtype for correctness
        cos = cos.to(dtype=q.dtype, device=dev)
        sin = sin.to(dtype=q.dtype, device=dev)

        q_rope = (q * cos) + (self._rotate_half(q) * sin)
        k_rope = (k * cos) + (self._rotate_half(k) * sin)
        return q_rope, k_rope

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float, expansion: float = 4.0):
        super().__init__()
        inner = int((2.0 / 3.0) * expansion * d_model)
        self.up = nn.Linear(d_model, 2 * inner, bias=True)
        self.down = nn.Linear(inner, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.up(x).chunk(2, -1)
        x = F.silu(gate) * val
        return self.down(self.drop(x))


class MultiHeadGroupQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_query_heads: int, n_kv_heads: int,
                 dropout: float, rope: RotaryPositionalEmbedding):
        super().__init__()
        if d_model % n_query_heads != 0:
            raise ValueError("d_model must be divisible by n_query_heads")
        if n_query_heads % n_kv_heads != 0:
            raise ValueError("n_query_heads must be a multiple of n_kv_heads")

        self.n_q = n_query_heads
        self.n_kv = n_kv_heads
        self.group_size = self.n_q // self.n_kv
        self.head_dim = d_model // self.n_q
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, self.n_q  * self.head_dim, bias=True)
        self.k_proj = nn.Linear(d_model, self.n_kv * self.head_dim, bias=True)
        self.v_proj = nn.Linear(d_model, self.n_kv * self.head_dim, bias=True)

        self.out = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_q,  self.head_dim).transpose(1, 2)  # [B,Hq,S,d]
        k = self.k_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)  # [B,Hkv,S,d]
        v = self.v_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)  # [B,Hkv,S,d]

        q, k = self.rope.apply(q, k, seq_len=S)

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)  # [B,Hq,S,d]
            v = v.repeat_interleave(self.group_size, dim=1)  # [B,Hq,S,d]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,Hq,S,S]
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)                              # [B,Hq,S,d]
        out = out.transpose(1, 2).contiguous().view(B, S, D)     # [B,S,D]
        out = self.out(out)
        out = self.proj_drop(out)
        return out

class PreNormEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_query_heads: int,
                 n_kv_heads: int, dropout: float, rope: RotaryPositionalEmbedding):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadGroupQueryAttention(d_model, n_query_heads, n_kv_heads, dropout, rope)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = SwiGLUFeedForward(d_model, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        x = x + self.drop1(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

class PreNormEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int,
                 n_query_heads: int, n_kv_heads: int,
                 dropout: float, max_seq_len: int):
        super().__init__()
        rope = RotaryPositionalEmbedding(head_dim=d_model // n_query_heads,
                                         max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([
            PreNormEncoderLayer(d_model, n_query_heads, n_kv_heads, dropout, rope)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.final_norm(x)

# ... your GQA code (MultiHeadGroupQueryAttention, PreNormEncoder*, TrajectoryTransformer) ...

# ============================================================
# Wrapper class (DDP-safe; now passes kv_heads through)
# ============================================================
class TrajectoryTransformer(nn.Module):
    def __init__(self,
                 feature_size: int,
                 num_classes: int,
                 d_model: int = 128,
                 nhead: int = 8,                 # query heads
                 num_layers: int = 4,
                 window_size: int = 100,
                 dropout: float = 0.1,
                 kv_heads: Optional[int] = None  # <-- NEW
                 ):
        super().__init__()
        self.nhead = nhead
        self.kv_heads = kv_heads if kv_heads is not None else max(1, nhead // 2)

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if nhead % self.kv_heads != 0:
            raise ValueError("nhead must be a multiple of kv_heads")

        self.embedding = nn.Linear(feature_size, d_model)
        self.encoder = PreNormEncoder(
            num_layers=num_layers,
            d_model=d_model,
            n_query_heads=nhead,
            n_kv_heads=self.kv_heads,
            dropout=dropout,
            max_seq_len=window_size,
        )
        self.attention_weights_layer = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, position_ids=None, src_key_padding_mask=None):
        x = self.embedding(x)
        out = self.encoder(x, src_key_padding_mask)
        attn_scores = self.attention_weights_layer(out).squeeze(-1)
        if src_key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(src_key_padding_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        return self.classifier(self.dropout(pooled))



def _unwrap(module: nn.Module) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP
    return module.module if isinstance(module, DDP) else module

class TrajectoryModel:
    def __init__(self, feature_columns, label_encoder, device=None, use_amp=False):
        self.feature_columns = feature_columns
        self.label_encoder = label_encoder
        self.num_classes = len(label_encoder.classes_)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model: nn.Module = None
        self.criterion = None
        self.optimizer = None
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    def prepare_model(
        self,
        window_size: int = 100,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        kv_heads: Optional[int] = None,
    ):
        core = TrajectoryTransformer(
            feature_size=len(self.feature_columns),
            num_classes=self.num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            window_size=window_size,
            dropout=dropout,
            kv_heads=kv_heads,
        ).to(self.device)

        # init weights before wrapping
        for name, p in core.named_parameters():
            if p.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0)

        # DDP (if launched with torchrun / WORLD_SIZE>1)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            core = torch.nn.parallel.DistributedDataParallel(
                core, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )

        self.model = core
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(_unwrap(self.model).parameters(), lr=lr, weight_decay=weight_decay)

    def train_one_epoch(self, train_loader, gradient_clip: float = 1.0):
        self.model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for sequences, masks, labels in train_loader:
            sequences = sequences.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True) if masks is not None else None
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(sequences, None, src_key_padding_mask=masks)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(_unwrap(self.model).parameters(), gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(sequences, None, src_key_padding_mask=masks)
                loss = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(_unwrap(self.model).parameters(), gradient_clip)
                self.optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            total_samples += bs
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()

        avg_loss = running_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for sequences, masks, labels in data_loader:
            sequences = sequences.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True) if masks is not None else None
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(sequences, None, src_key_padding_mask=masks)
                loss = self.criterion(outputs, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return avg_loss, accuracy

    @torch.no_grad()
    def predict(self, data_loader):
        self.model.eval()
        all_labels, all_preds = [], []
        for sequences, masks, labels in data_loader:
            sequences = sequences.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True) if masks is not None else None
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(sequences, None, src_key_padding_mask=masks)
                _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        return all_labels, all_preds

    def save_model(self, path: str):
        torch.save(_unwrap(self.model).state_dict(), path)

    def load_model(self, path: str, strict: bool = True):
        state = torch.load(path, map_location=self.device)
        core = _unwrap(self.model)
        has_prefix = any(k.startswith("module.") for k in state.keys())
        target_wrapped = hasattr(self.model, "module")

        if has_prefix and not target_wrapped:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        elif (not has_prefix) and target_wrapped:
            state = {f"module.{k}": v for k, v in state.items()}

        core.load_state_dict(state, strict=strict)
        self.model.to(self.device)