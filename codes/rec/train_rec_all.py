import os
from config import *

model_and_configs=[
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_1", "finetune", "latest", config=None), 30],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_2", "finetune", "latest", config=None), 30],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_3", "finetune", "latest", config=None), 30],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_4", "finetune", "latest", config=None), 30],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_5", "finetune", "latest", config=None), 30],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_1", "finetune", "latest", config=None), 50],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_2", "finetune", "latest", config=None), 50],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_3", "finetune", "latest", config=None), 50],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_4", "finetune", "latest", config=None), 50],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_5", "finetune", "latest", config=None), 50],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_1", "finetune", "latest", config=None), 70],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_2", "finetune", "latest", config=None), 70],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_3", "finetune", "latest", config=None), 70],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_4", "finetune", "latest", config=None), 70],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_5", "finetune", "latest", config=None), 70],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_1", "finetune", "latest", config=None), 90],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_2", "finetune", "latest", config=None), 90],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_3", "finetune", "latest", config=None), 90],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_4", "finetune", "latest", config=None), 90],
    [make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_5", "finetune", "latest", config=None), 90],
]

with open("train_rec_all.sh", "w") as f:
    for model_and_config, epoch in model_and_configs:
        command = f"""python /home/code/PaddleOCR/tools/train.py \
        -c {model_and_config["config"]} \
        -o Global.epoch_num={epoch} \
            Global.pretrained_model={model_and_config["pretrained_model"]} \
            Global.print_batch_step={1} \
            Global.save_epoch_step=1 \
            Global.eval_batch_step=[0,100] \
            Global.checkpoints={model_and_config["checkpoints"]} \
            Global.save_model_dir={model_and_config["save_model_dir"]} \
            Train.dataset.data_dir={model_and_config["train_data_dir"]} \
            Train.dataset.label_file_list={model_and_config["train_label_file_list"]} \
            Eval.dataset.data_dir={model_and_config["val_data_dir"]} \
            Eval.dataset.label_file_list={model_and_config["val_label_file_list"]}\n
        """
        f.write(command)