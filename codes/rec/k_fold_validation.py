
from config import *

from pathlib import Path
def k_fold_validation(shel_path, orders, save_path, start_epoch, end_epoch, task):
    with open(shel_path, "w") as f:    
        for epoch in range(start_epoch, end_epoch+1):
            # 모든 k_fold에 대해 epoch 번째 가중치 파일이 없으면 종료
            if any([not (Path(order["save_model_dir"])/f"iter_epoch_{epoch}.pdparams").exists() for order in orders]):
                break
            for order in orders:
                model = str(Path(order["save_model_dir"])/f"iter_epoch_{epoch}") 
                order["checkpoints"] = model
                order["pretrained_model"] = model
                order["test_save_path"] = save_path
                print(model)
                
                
                
                command = f""" python /home/code/PaddleOCR/tools/eval.py\
                    -c {order["config"]}\
                    -o Global.pretrained_model={order["pretrained_model"]}\
                        Global.test_save_path={order["test_save_path"]}\
                        Global.character_dict_path={order["character_dict_path"]}\
                        Eval.dataset.data_dir={order[f"{task}_data_dir"]}\
                        Eval.dataset.label_file_list={order[f"{task}_label_file_list"]}\n"""
                f.write(command)
orders = [
    # make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_1", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_2", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_3", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_4", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("rec", "korean_PP-OCRv3_rec", "rec_08_02_90_k_fold_5_5", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("det", "ml_PP-OCRv3_det", "ai_hub_det_08_02_90_k_fold_5_1", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("det", "ml_PP-OCRv3_det", "ai_hub_det_08_02_90_k_fold_5_2", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("det", "ml_PP-OCRv3_det", "ai_hub_det_08_02_90_k_fold_5_3", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("det", "ml_PP-OCRv3_det", "ai_hub_det_08_02_90_k_fold_5_4", "finetune", "pretrained", config=None),
    # make_PP_OCR_config("det", "ml_PP-OCRv3_det", "ai_hub_det_08_02_90_k_fold_5_5", "finetune", "pretrained", config=None),
    make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_1", "finetune", "pretrained", config=None),
    make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_2", "finetune", "pretrained", config=None),
    make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_3", "finetune", "pretrained", config=None),
    make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_4", "finetune", "pretrained", config=None),
    make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_5", "finetune", "pretrained", config=None),
    
]

for i, epoch in enumerate(range(1, 51, 3)):
    start_epoch = epoch
    end_epoch = epoch+3
    shel_path=f"/home/code/det/k_fold_validation_{i+1}.sh"
    save_path=f"/home/output/det_k_fold_results.csv"
    for task in ["train", "val"]
    k_fold_validation(shel_path, orders, save_path, start_epoch, end_epoch, task)