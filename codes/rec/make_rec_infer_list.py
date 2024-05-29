from pathlib import Path
import shutil

data_dir = Path("/home/rec/dataset2")
test_label_file = Path("/home/rec/dataset2/test_label.txt")
sample_num = -1
save_path = Path("/home/rec/dataset2/infer_list.txt")

with open(test_label_file, "r") as f:
    lines = f.readlines()
    lines = [line.rstrip("\n") for line in lines]
    
    
paths = [str(data_dir/line.split("\t")[0]) for line in lines][:sample_num]


with open(save_path, "w") as f:
    f.write("\n".join(paths))
