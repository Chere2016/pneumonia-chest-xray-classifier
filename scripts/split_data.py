import os
import shutil
import random
from collections import defaultdict

def get_patient_id(filename, cls):
    if cls == 'PNEUMONIA':
        return filename.split('_')[0]
    else:
        parts = filename.split('-')
        if len(parts) > 1:
            return "-".join(parts[:-1])
        else:
            return filename.split('.')[0]

def main():
    data_dir = '/home/falcon/student1/mscs/medical_classifier/data'
    old_splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    # 1. Map patient IDs to their absolute file paths and class
    # patient_map[patient_id] = [(class, abs_path), ...]
    patient_map = defaultdict(list)
    
    for split in old_splits:
        for cls in classes:
            cls_dir = os.path.join(data_dir, split, cls)
            if os.path.exists(cls_dir):
                for img in os.listdir(cls_dir):
                    if img.endswith(('.jpeg', '.jpg', '.png')):
                        pid = get_patient_id(img, cls)
                        abs_path = os.path.join(cls_dir, img)
                        patient_map[pid].append((cls, abs_path, img))
                        
    patient_ids = list(patient_map.keys())
    print(f"Total unique patients found: {len(patient_ids)}")
    
    # 2. Shuffle and split patient IDs (80 / 10 / 10)
    random.seed(42)
    random.shuffle(patient_ids)
    
    n_patients = len(patient_ids)
    train_end = int(n_patients * 0.8)
    val_end = train_end + int(n_patients * 0.1)
    
    train_pids = set(patient_ids[:train_end])
    val_pids = set(patient_ids[train_end:val_end])
    test_pids = set(patient_ids[val_end:])
    
    print(f"Allocating Patients -> Train: {len(train_pids)}, Val: {len(val_pids)}, Test: {len(test_pids)}")
    
    # 3. Create new directory structure
    new_data_dir = os.path.join(data_dir, 'new_split')
    new_splits = ['train', 'val', 'test']
    for split in new_splits:
        for cls in classes:
            os.makedirs(os.path.join(new_data_dir, split, cls), exist_ok=True)
            
    # 4. Move images to new structure
    def get_target_split(pid):
        if pid in train_pids: return 'train'
        elif pid in val_pids: return 'val'
        else: return 'test'
        
    moved_count = 0
    for pid, files in patient_map.items():
        target_split = get_target_split(pid)
        for cls, old_path, img_name in files:
            new_path = os.path.join(new_data_dir, target_split, cls, img_name)
            # Use move instead of copy for speed, but move cross-device can be slow.
            # In the same filesystem, os.rename or shutil.move is instant.
            shutil.move(old_path, new_path)
            moved_count += 1
            
    print(f"Successfully moved {moved_count} images to new leak-proof split.")
    
    # 5. Cleanup old directories and swap in the new ones
    for split in old_splits:
        shutil.rmtree(os.path.join(data_dir, split))
        
    for split in new_splits:
        shutil.move(os.path.join(new_data_dir, split), os.path.join(data_dir, split))
        
    os.rmdir(new_data_dir)
    print("Cleanup complete. The `data` directory now contains the leak-proof split.")

if __name__ == '__main__':
    main()
