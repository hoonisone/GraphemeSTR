########## Test Model ##########
# paddle ml_PP-OCRv3_det_slim (pretrained)
# model_name=/home/resource/model/pretrained/ml_PP-OCRv3_det_slim/best_accuracy
# config=/home/code/PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml

# paddle ml_PP-OCRv3_det (pretrained)
model_name=/home/resource/model/pretrained/ml_PP-OCRv3_det/best_accuracy
config=/home/code/PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml

# MobileNet(pretrained)
# model_name=/home/resource/model/pretrained/MobileNetV3_large_x0_5/best_accuracy
# config=/home/code/PaddleOCR/configs/det/det_mv3_db.yml

# MobileNet(tuned) 13epoch
# model_name=/home/resource/model/tuned/MobileNetV3_large_x0_5/best_accuracy
# config=/home/code/PaddleOCR/configs/det/det_mv3_db.yml

# paddle ml_PP-OCRv3_det (inference)
# model_name=/home/resource/model/infer/ml_PP-OCRv3_det/inference
# config=/home/code/PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml


########## Test Dataset ##########
data_dir=/home/dataset
label_file_list=("/home/dataset/test_label2.txt")
test_save_path=/home/resource/test
########## Command ##########
python /home/code/PaddleOCR/tools/eval.py \
    -c ${config} \
    -o  Global.pretrained_model=${model_name} \
        Global.test_save_path=${test_save_path} \
        Eval.dataset.data_dir=${data_dir} \
        Eval.dataset.label_file_list=${label_file_list}
echo ${model_name}