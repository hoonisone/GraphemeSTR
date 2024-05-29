from PaddleOCR import paddleocr
import json
from pathlib import Path


def save_result(predict, image_path, save_dir):
    image_file_name = Path(image_path).stem
    file_path = save_dir/f"{image_file_name}.json"

    with open(file_path, "w") as f:
        f.write(json.dumps(predict))
    
def main(image_dir, prediction_dir):
    image_paths, predicts = paddleocr.main(str(image_dir))
    # PADDLEOCR에서 제공하는 코드이다. 설정된 디렉터리 안에 모든 이미지에 대해 추론을 한다.
    # 결과값은 반환되도록 수정함

    for predict, image_path in zip(predicts, image_paths):
        save_result(predict, image_path, save_dir = prediction_dir)

if __name__=="__main__":
    root_dir = Path("test1")
    image_dir = root_dir/"images"
    prediction_dir = root_dir/"predicted"
    main(image_dir, prediction_dir)

