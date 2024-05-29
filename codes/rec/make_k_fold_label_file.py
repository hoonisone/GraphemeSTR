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
label_root = Path("/home/dataset_labels")

orders = [
#     [dataset_labels_dir, k in k_fold]
    [label_root/"ai_hub_det_08_02_90", 5]
]
# Output: 어떤 비율로 어떻게 나눌지
# random.seed(100) 필요 없을 듯

# label_file을 모두 읽어 드려 하나로 모으기


for label_dir, k in orders:
    print(f"""Generate k_fold dataset_labels for "{label_dir}" """)
    print("1. Load all label files")
    train = [line.strip("\n") for line in open(label_dir/"train_label.txt").readlines()]
    val = [line.strip("\n") for line in open(label_dir/"val_label.txt").readlines()]
    # test = [line.strip("\n") for line in open(label_dir/"test_label.txt").readlines()]
    # infer = [line.strip("\n") for line in open(label_dir/"infer_list.txt").readlines()]
    labels = train + val

    # 전체 레이블을 나누어 레이블 파일로 저장
    print("2. split labels in k segments")
    segments = []
    s_size = int(len(labels)/k) # segment_size
    for i in range(k-1):
        segments.append(labels[s_size*i:s_size*(i+1)])
    segments.append(labels[s_size*(k-1):])
    
    print("3. save")
    for i in range(k):
        train = sum([[] if i==j else segment for j, segment in enumerate(segments)], [])
        val = segments[i]
        new_label_dir = Path(str(label_dir)+f"_k_fold_{k}_{i+1}")
        new_label_dir.mkdir(parents=True, exist_ok=True)
        open(new_label_dir/"train_label.txt", "w").write("\n".join(train))
        open(new_label_dir/"val_label.txt", "w").write("\n".join(val))