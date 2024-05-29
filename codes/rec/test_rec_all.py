import os
from config import *




model_and_configs=[
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_1", "finetune", "latest", config=None)],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_2", "finetune", "latest", config=None)],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_3", "finetune", "latest", config=None)],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_4", "finetune", "latest", config=None)],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_5", "finetune", "latest", config=None)],
]

with open("test_rec_all.sh", "w") as f:
    for model_and_config in model_and_configs:
        command = f""" python /home/code/PaddleOCR/tools/eval.py \
            -c {model_and_config["config"]} \
            -o Global.pretrained_model={model_and_config["pretrained_model"]} \
                Global.test_save_path={model_and_config["test_save_path"]} \
                Global.character_dict_path={model_and_config["character_dict_path"]} \
                Eval.dataset.data_dir={model_and_config["test_data_dir"]} \
                Eval.dataset.label_file_list={model_and_config["test_label_file_list"]}
        """

        f.write(command)