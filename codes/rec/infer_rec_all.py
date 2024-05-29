import os
from rec_config import *

model_and_configs=[
    # make_PP_OCR_config("korean_PP-OCRv3_rec", "finetune", "pretrained"),
    make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_80_10_10", "finetune", "pretrained", config=None)
    # make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_80_10_10", "finetune", "pretrained", config=None)
]

# 모델에 관계없이 동일한 설정

for model_and_config in model_and_configs:
    model_name = model_and_config["model_name"]
    config = model_and_config["config"]
    infer_save_dir = model_and_config["infer_save_dir"]
    
    command = f"""
    python /home/code/PaddleOCR/tools/infer_rec.py \
    -c {model_and_config["config"]} \
    -o Global.pretrained_model={model_and_config["model_name"]}\
        Global.save_inference_dir={model_and_config["infer_save_dir"]} \
        Global.save_res_path={model_and_config["infer_save_dir"]}/predicts.txt \
        Global.use_visualdl=true \
        Infer.data_dir={model_and_config["infer_data_dir"]} \
        Infer.infer_file_list={model_and_config["infer_list_file_list"]}
    """

    os.system(command)