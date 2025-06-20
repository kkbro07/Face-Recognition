import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

source_dir = 'dataset'
train_dir = 'dataset_train'
test_dir = 'dataset_test'

for path in [train_dir, test_dir]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

people = os.listdir(source_dir)

for person in tqdm(people, desc="Splitting dataset"):
    person_path = os.path.join(source_dir, person)
    files = os.listdir(person_path)
    train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)

    for split, split_files in [('train', train_files), ('test', test_files)]:
        dest_dir = train_dir if split == 'train' else test_dir
        os.makedirs(os.path.join(dest_dir, person), exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(person_path, f), os.path.join(dest_dir, person, f))

print("[INFO] Step 1 complete: Dataset split into 70-30.")
