python ../PaddleOCR/tools/eval.py \
    -c ../PaddleOCR/configs/det/det_mv3_db.yml \
    -o Eval.dataset.label_file_list=/home/dataset/eval_label.txt




name="MobileNetV3_large_x0_5_pretrained"
config="det/det_mv3_db.yml"

python ../PaddleOCR/tools/train.py \
    -c ../PaddleOCR/configs/det/det_mv3_db.yml \
    -o Global.pretrained_model=/home/resource/model/pretrained/${name} \
        Global.save_model_dir=/home/resource/model/tuned/${name} \
        Train.dataset.data_dir=/home/dataset \
        Train.dataset.label_file_list=["/home/dataset/train_label.txt"] \
        Eval.dataset.data_dir=/home/dataset \
        Eval.dataset.label_file_list=["/home/dataset/val_label.txt"] \