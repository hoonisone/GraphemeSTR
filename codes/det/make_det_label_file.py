from pathlib import Path
import random




# data_dir 구조
"""
- data_dir
    - dataset1
        - images
        - label.txt
    - dataset2
        - images
        - label.txt 
"""

# Input config: 어떤 데이터셋들을 사용할 것인지
data_dir = Path("/home/datasets") # 모든 데이터 셋들이 담긴 폴더
label_dir = Path("/home/dataset_labels")

label_file_list = [
    data_dir/"AIHUB/korean_image/label.txt"
]

# Output: 어떤 비율로 어떻게 나눌지
random.seed(100)
orders = [
    ["ai_hub_det", [80, 10, 10]],
    ["ai_hub_det", [20, 10, 70]],
    ["ai_hub_det", [16,  4, 80]],
    ["ai_hub_det", [8, 2, 90]],
]

# label_file을 모두 읽어 드려 하나로 모으기
print("1. Load all label files")
label_list = []
for label_file in label_file_list:
    labels = open(label_file).readlines()
    labels = [label.strip("\n").split("\t") for label in labels]
    labels = [f"{str((label_file.absolute().parent/path).relative_to(data_dir))}\t{label}" for path, label in labels]
    label_list += labels
    random.shuffle(label_list)

# 전체 레이블을 나누어 레이블 파일로 저장
print("2. split and save label files")
for name, [train, val, test] in orders:
    total = sum([train, val, test])
    train_n = int(train/total*len(label_list))
    val_n = int(val/total*len(label_list))
    test_n = int(test/total*len(label_list))
    
    train_labels = label_list[:train_n]
    val_labels = label_list[train_n:-test_n]
    test_labels = label_list[-test_n:]
    infer_list = [label.split("\t")[0] for label in test_labels]
    

    name = f"{name}_{train:02}_{val:02}_{test:2}"
    (label_dir/name).mkdir(parents=True, exist_ok=True)
    print(label_dir/name)
    open(label_dir/name/"train_label.txt", "w").write("\n".join(train_labels))
    open(label_dir/name/"val_label.txt", "w").write("\n".join(val_labels))
    open(label_dir/name/"test_label.txt", "w").write("\n".join(test_labels))
    open(label_dir/name/"infer_list.txt", "w").write("\n".join(infer_list))