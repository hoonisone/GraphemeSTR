# model_name="pretrained_model/ml_PP-OCRv3_det/best_accuracy"
# config="det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml" # 모델에 맞는 config 파일 경로 (/home/PaddleOCR/configs 기준 상대경로)

name="MobileNetV3_large_x0_5_pretrained"
config="det/det_mv3_db.yml"
python ../PaddleOCR/tools/train.py \
    -c ../PaddleOCR/configs/det/det_mv3_db.yml \
    -o Global.pretrained_model=/home/resource/model/tuned/${name}/latest \
        Global.save_model_dir=/home/resource/model/tuned/${name} \
        Train.dataset.data_dir=/home/dataset \
        Train.dataset.label_file_list=["/home/dataset/train_label.txt"] \
        Eval.dataset.data_dir=/home/dataset \
        Eval.dataset.label_file_list=["/home/dataset/val_label.txt"] \