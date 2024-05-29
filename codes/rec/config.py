from pathlib import Path
from pretrained_models import *
DEFAILT_CONFIG = {
    "ml_PP-OCRv3_det":"/home/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml",
    "korean_PP-OCRv3_rec": "/home/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml",
    "MobileNetV3_large_x0_5":"/home/configs/det/det_mv3_db.yml"
}

DEFAILT_LANGUAGE_DICT = {
    "ml_PP-OCRv3_det":None,
    "korean_PP-OCRv3_rec": "/home/code/PaddleOCR/ppocr/utils/dict/korean_dict.txt",
    "MobileNetV3_large_x0_5":None
}


def make_PP_OCR_config(task, model_name, labelset, mode, version, config=None):
    assert task in ["det", "rec"]
    assert mode in ["scratch", "finetune"]
    assert (version in ["latest", "best_accuracy"]) or (version[:11] == "iter_epoch_" and 0 < int(version[11:])), f"{version} is invalid"

    
    labelset_name = "_".join(labelset.path.parts[3:])
    output_path = f"/home/output/{task}___{model_name}___{labelset_name}___{'default_config'}___{mode}"

    config = DEFAILT_CONFIG[model_name] if config == None else config
    
    model = f"{output_path}/trained_model/{version}"
    if version in ["latest", "best_accuracy"]:
        if not Path(model+".pdparams").exists():
            if mode == "scratch":
                model = ""
            elif mode == "finetune":
                model = PretrainedModelDB().get_model(model_name)["pretrained"]
    
    if version == "pretrained":
        if mode == "scratch":
            model = ""
        elif mode == "finetune":
            model = PretrainedModelDB().get_model(model_name)["pretrained"]
  
    data_dir=f"/home/datasets"
    train_label_file_list=[str(labelset.train)] 
    val_label_file_list=[str(labelset.eval)]
    test_label_file_list=[str(labelset.test)]
    infer_list_file_list=[str(labelset.infer)]
    

    
    save_model_dir = f"{output_path}/trained_model"
    infer_save_dir = f"{output_path}/infer_result/{version}"
    
    pretrained_model= model    
    checkpoints = model
    
    character_dict_path = DEFAILT_LANGUAGE_DICT[model_name]

    return {
        "model_name":model,
        "config":config,
        "save_model_dir":save_model_dir,
        "infer_save_dir":infer_save_dir,
        "pretrained_model":pretrained_model,
        "checkpoints":checkpoints,
        "train_data_dir":data_dir,
        "train_label_file_list":train_label_file_list,
        "val_data_dir":data_dir,
        "val_label_file_list":val_label_file_list,
        "test_data_dir":data_dir,
        "test_label_file_list":test_label_file_list,
        "infer_data_dir":data_dir,
        "infer_list_file_list":infer_list_file_list,
        "test_save_path":f"/home/output/{task}_results.csv",
        "character_dict_path":character_dict_path,
    }

if __name__=="__main__":
    make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_5", "finetune", "latest", config=None)
    