def make_PP_OCR_config(model_name, mode):
    return {
        "model_name":"/home/det/model/{model_name}/{mode}/model/best_accuracy",
        "config":"/home/code/PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml",
        "infer_save_dir":"/home/det/model/{model_name}/{mode}/infer"
    }
    
ml_PP_OCRv3_det_slim_pretrained = make_PP_OCR_config("ml_PP-OCRv3_det_slim", "pretrained")
ml_PP_OCRv3_det_pretrained = make_PP_OCR_config("ml_PP-OCRv3_det", "pretrained")
ml_PP_OCRv3_det_tuned = make_PP_OCR_config("ml_PP-OCRv3_det", "tuned")
ml_PP_OCRv3_det_inference = make_PP_OCR_config("ml_PP-OCRv3_det", "inference")
en_PP_OCRv3_det_pretrained = make_PP_OCR_config("en_PP-OCRv3_det", "pretrained")
ch_PP_OCRv3_det_pretrained = make_PP_OCR_config("ch_PP-OCRv3_det", "pretrained")
ch_ppocr_mobile_v2_det_pretrained = make_PP_OCR_config("ch_ppocr_mobile_v2_det", "pretrained")
ch_ppocr_server_v2_det_pretrained = make_PP_OCR_config("ch_ppocr_server_v2_det", "pretrained")
MobileNetV3_large_x0_5_pretrained = make_PP_OCR_config("MobileNetV3_large_x0_5", "pretrained")
MobileNetV3_large_x0_5_tuned = make_PP_OCR_config("MobileNetV3_large_x0_5", "tuned")