import os
from det_config import *


model_and_configs=[
    ml_PP_OCRv3_det_pretrained,
    ml_PP_OCRv3_det_tuned,
    en_PP_OCRv3_det_pretrained,
    ch_PP_OCRv3_det_pretrained,
    
    MobileNetV3_large_x0_5_pretrained,
    MobileNetV3_large_x0_5_tuned
]

# 모델에 관계없이 동일한 설정
infer_img="/home/det/dataset/infer_samples"
data_dir="/home/det/dataset"
infer_file_list = ["/home/det/dataset/infer_list.txt"]
for model_and_config in model_and_configs:
    model_name = model_and_config["model_name"]
    config = model_and_config["config"]
    infer_save_dir = model_and_config["infer_save_dir"]
    
    command = f"""
    model_name={model_name}
    config={config}
    python /home/code/PaddleOCR/tools/infer_det.py \
    -c {config} \
    -o Global.infer_img={infer_img} \
        Global.pretrained_model={model_name}\
        Global.save_inference_dir={infer_save_dir} \
        Global.save_res_path={infer_save_dir}/predicts.txt \
        Global.use_visualdl=true \
        Infer.data_dir={data_dir} \
        Infer.infer_file_list={infer_file_list}
    """
    

        # Global.save_model_dir=/home/resource/inference/${model_name} \
    os.system(command)