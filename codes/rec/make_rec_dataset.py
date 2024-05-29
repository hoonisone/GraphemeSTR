from pathlib import Path
from PIL import Image
import json
import os
import shutil
import tqdm


def main(det_data_dir, det_label_file, rec_data_dir, rec_image_dir, rec_label_file):
    
    print(f"(1/4) 기존 데이터 체크")
    if rec_data_dir.exists() | rec_image_dir.exists() | rec_label_file.exists():
        while True:
            answer = input(f"""아래 경로에 이미 데이터가 있습니다. \
                            {rec_data_dir} \
                            {rec_image_dir} \
                            {rec_label_file} \
                           삭제 하고 진행하시겠습니까? (y/n):""")
            if answer in ["y", "yes"]:       
                # 기존 내용을 삭제하고 계속 수행     
                if rec_data_dir.exists():
                    shutil.rmtree(rec_data_dir)
                if rec_image_dir.exists():
                     shutil.rmtree(rec_image_dir)
                if rec_label_file.exists():
                    rec_label_file.unlink()
                break
            elif answer == ["n", "no"]:
                return # 바로 종료
            else:
                print("Please 'y' or 'n'")
                continue
            
    rec_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"(2/4) detection label file 로드 ('{det_label_file}')")
    with open(det_label_file, "r") as f:
        lines = [line.strip("\n").split("\t") for line in f.readlines()]

    print(f"(3/4) Image cropping & Save at ('{rec_image_dir}')")
    img_num = 1
    pass_num = 0
    xxx_num = 0
    for line in tqdm.tqdm(lines[:]):
        img = Image.open(det_data_dir/line[0])
        labels = json.loads(line[1])
        for label in labels:
            try:
                text = label["transcription"]
                points = label["points"]
                if text == "xxx":
                    xxx_num += 1
                    continue
                else:
                    bbox = points[0] + points[2]
                    cropped_img = img.crop(bbox)
                    save_path = rec_image_dir/f"word_{img_num}.png"
                    cropped_img.save(save_path)
                    relative_path = str(save_path.relative_to(rec_data_dir))
                    label = f'{relative_path}\t{text}'
                    with open(rec_label_file, "a") as f:
                        f.write(label+"\n")
            except:
                pass_num += 1
                print(f"error and pass ({pass_num}): {text}, {bbox}")
            
            img_num += 1
    print(f"total:{img_num}, xxx: {xxx_num}, pass:{pass_num}")
                
    print(f"(4/4) 정리")
    # 마지막 개행 하나 지우기
    lines = open(rec_label_file, "r").read()
    with open(rec_label_file, "w") as f:
        f.write(lines[:-1])


def f(mode):
    return {
        f"det_data_dir":Path(f"/home/det/dataset"),
        f"det_label_file":Path(f"/home/det/dataset/{mode}_label.txt"),
        f"rec_data_dir":Path(f"/home/rec/dataset"),
        f"rec_image_dir":Path(f"/home/rec/dataset/{mode}"),
        f"rec_label_file":Path(f"/home/rec/dataset/{mode}_label.txt"),
    }
# task_list = [f("train"), f("val"), f("test")]
task_list = [f("test")]

for i, task in enumerate(task_list):
    main(task["det_data_dir"], task["det_label_file"], task["rec_data_dir"], task["rec_image_dir"], task["rec_label_file"])