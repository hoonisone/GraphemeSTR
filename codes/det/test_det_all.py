import os
from rec.config import *


# 테스트 될 모델 설정 정보들
model_and_configs=[
    ml_PP_OCRv3_det_pretrained,
    
    en_PP_OCRv3_det_pretrained,
    
    ch_PP_OCRv3_det_pretrained,
    ch_ppocr_mobile_v2_det_pretrained,
    ch_ppocr_server_v2_det_pretrained,
    
    MobileNetV3_large_x0_5_pretrained,
    MobileNetV3_large_x0_5_tuned
]

# 모델에 관계없이 동일한 설정
data_dir="/home/det/dataset"
label_file_list=["/home/det/dataset/test_label.txt"]
test_save_path="/home/det/resource/test"

for model_and_config in model_and_configs:
    model_name = model_and_config["model_name"]
    config = model_and_config["config"]
    
    command = f"""
    python /home/code/PaddleOCR/tools/eval.py \
        -c {config} \
        -o  Global.pretrained_model={model_name} \
            Global.test_save_path={test_save_path} \
            Eval.dataset.data_dir={data_dir} \
            Eval.dataset.label_file_list={label_file_list}
    """

    os.system(command)