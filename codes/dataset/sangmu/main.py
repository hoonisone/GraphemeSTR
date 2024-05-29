from pathlib import Path

path = "E:\workspace\paddleocr\origin_datasets\sangmu_real_image_det"
target_path = "E:\workspace\paddleocr\origin_datasets\sangmu_real_image_det2"


print("모든 파일 이름 로드")##########################################
files = Path(path).glob("*")
files = [str(file) for file in files]

print("파일 이름 유효성 체크")##########################################
for file in files:
    assert file[:-4]+".jpg" in files
    assert file[:-4]+".txt" in files


print("이름만 추출 (확장자 제거 및 중복 제거)")##########################################
names = [file[:-4] for file in files]
names = list(set(names))

print("이름 변경 (심플하게)")##########################################
print(names)
# rename
# for i, name in enumerate(names):
#     i += 1
#     for ext in [".jpg", ".txt"]:
#         path = Path(name+ext)
#         print(path.exists())
#         target_path = path.parent/f"{i}{ext}"
#         path.rename(target_path)

print("이름 변경 (심플하게)")##########################################
img = Image.open(det_data_dir/line[0])
cropped_img = img.crop(bbox)
# 모든 파일 이름 로드
# Path(target_path).mkdir(parents=True, exist_ok=True)
