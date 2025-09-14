#!/usr/bin/env python3
"""
Script to restructure the models directory for GitHub push.
Keeps: source code, logs, configs, notebooks
Removes: model checkpoints (.pth), joblib files, __pycache__, large binaries
"""

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_clean_structure():
    """Create a clean models directory structure"""
    
    source_dir = Path("/data/A-SpeedTransformer/models")
    target_dir = Path("/data/A-SpeedTransformer/models_clean")
    
    # Create target directory
    target_dir.mkdir(exist_ok=True)
    
    # Files/directories to exclude
    exclude_patterns = {
        '*.pth',           # PyTorch model files
        '*.joblib',        # Scikit-learn model files  
        '*.pkl',           # Pickle files
        '__pycache__',     # Python cache
        '*.pyc',           # Compiled Python
        'best_model.pth',  # Specific model files
        'model.pth',
        'checkpoint.pth'
    }
    
    # Files/directories to include
    include_patterns = {
        '*.py',            # Python source code
        '*.log',           # Training logs
        '*.sh',            # Shell scripts
        '*.ipynb',         # Jupyter notebooks
        '*.md',            # Markdown files
        '*.txt',           # Text files
        '*.json',          # Config files
        '*.yaml',          # Config files
        '*.yml',           # Config files
        '*.png',           # Plots and figures
        '*.jpg',           # Images
        '*.jpeg',          # Images
        'README*',         # README files
        'requirements*',   # Requirements files
    }
    
    def should_exclude(path: Path) -> bool:
        """Check if a path should be excluded"""
        for pattern in exclude_patterns:
            if path.match(pattern):
                return True
        return False
    
    def should_include(path: Path) -> bool:
        """Check if a path should be included"""
        # Always include directories (we'll filter contents)
        if path.is_dir():
            return True
            
        for pattern in include_patterns:
            if path.match(pattern):
                return True
        return False
    
    def copy_structure(src: Path, dst: Path):
        """Recursively copy directory structure with filtering"""
        
        for item in src.iterdir():
            src_path = src / item.name
            dst_path = dst / item.name
            
            # Skip excluded items
            if should_exclude(src_path):
                logger.info(f"Excluding: {src_path}")
                continue
            
            if src_path.is_dir():
                # Create directory and recurse
                dst_path.mkdir(exist_ok=True)
                copy_structure(src_path, dst_path)
                
                # Remove empty directories
                try:
                    if not any(dst_path.iterdir()):
                        dst_path.rmdir()
                        logger.info(f"Removed empty directory: {dst_path}")
                except OSError:
                    pass
                    
            elif should_include(src_path):
                # Copy file
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied: {src_path} -> {dst_path}")
    
    logger.info("Starting restructure...")
    copy_structure(source_dir, target_dir)
    
    # Create a summary structure file
    summary_file = target_dir / "STRUCTURE.md"
    with open(summary_file, 'w') as f:
        f.write("# Models Directory Structure\n\n")
        f.write("This directory contains the clean version of the models for GitHub.\n\n")
        f.write("## Included:\n")
        f.write("- Source code (.py files)\n")
        f.write("- Training logs (.log files)\n")
        f.write("- Shell scripts (.sh files)\n")
        f.write("- Jupyter notebooks (.ipynb files)\n")
        f.write("- Configuration files (.json, .yaml)\n")
        f.write("- Documentation (.md, .txt files)\n")
        f.write("- Plots and figures (.png, .jpg)\n\n")
        f.write("## Excluded:\n")
        f.write("- Model checkpoints (.pth files)\n")
        f.write("- Joblib files (.joblib files)\n")
        f.write("- Python cache (__pycache__)\n")
        f.write("- Pickle files (.pkl files)\n\n")
        f.write("## Directory Structure:\n\n")
        
        # Generate directory tree
        def write_tree(path: Path, f, level=0):
            items = sorted([p for p in path.iterdir() if p.is_dir() or should_include(p)])
            for item in items:
                indent = "  " * level
                if item.is_dir():
                    f.write(f"{indent}- {item.name}/\n")
                    write_tree(item, f, level + 1)
                else:
                    f.write(f"{indent}- {item.name}\n")
        
        write_tree(target_dir, f)
    
    logger.info(f"Restructuring complete! Clean version created at: {target_dir}")
    
    # Print size comparison
    def get_size(path: Path) -> int:
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total
    
    original_size = get_size(source_dir)
    clean_size = get_size(target_dir)
    
    logger.info(f"Original size: {original_size / (1024**2):.1f} MB")
    logger.info(f"Clean size: {clean_size / (1024**2):.1f} MB")
    logger.info(f"Space saved: {(original_size - clean_size) / (1024**2):.1f} MB ({((original_size - clean_size) / original_size * 100):.1f}%)")

if __name__ == "__main__":
    create_clean_structure()
