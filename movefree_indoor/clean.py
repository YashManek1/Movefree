"""
Label Cleaner & Fixer
Scans all labels and removes classes that are out of bounds (>= 19).
"""

import os
from pathlib import Path
from tqdm import tqdm

def fix_labels(dataset_path="datasets/movefree_combined"):
    labels_dir = Path(dataset_path) / "labels"
    
    # Max valid class ID (0 to 18)
    MAX_CLASS_ID = 18 
    
    print(f"🧹 Scanning labels in {labels_dir}...")
    
    files = list(labels_dir.rglob("*.txt"))
    
    fixed_count = 0
    removed_count = 0
    
    for file_path in tqdm(files):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            changed = False
            
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                class_id = int(parts[0])
                
                # Check if class is valid
                if class_id <= MAX_CLASS_ID:
                    new_lines.append(line)
                else:
                    # OPTIONAL: Remap specific common errors if we knew them
                    # But safest is to just drop unknown high-id classes
                    changed = True
                    removed_count += 1
            
            # Overwrite file if changes needed
            if changed:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"✅ Done! Fixed {fixed_count} files.")
    print(f"🗑️ Removed {removed_count} invalid object labels.")

if __name__ == "__main__":
    fix_labels()
