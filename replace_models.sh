#!/bin/bash

# Script to replace the original models directory with the clean version
# WARNING: This will permanently delete the original models directory

echo "⚠️  WARNING: This will replace the original models directory with the clean version."
echo "The original directory (3.2GB) will be permanently deleted."
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    echo "Backing up original directory to models_backup..."
    mv models models_backup
    
    echo "Moving clean directory to models..."
    mv models_clean models
    
    echo "✅ Successfully replaced models directory!"
    echo "Original directory backed up as models_backup"
    echo "New models directory is 6.7MB (99.8% size reduction)"
else
    echo "Operation cancelled."
fi
