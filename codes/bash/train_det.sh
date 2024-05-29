########## Dataset ##########
train_data_dir=/home/dataset
train_lebel_file_list=["/home/dataset/train_label.txt"]
eval_data_dir=/home/dataset
eval_lebel_file_list=["/home/dataset/val_label.txt"]
# train_batch_size_per_card=128
# train_num_workers=8
# eval_batch_size_per_card=1
# eval_num_workers=2

########## Model & Config
# MobileNetV3_large_x0_5 (pretrained)
# config=/home/code/PaddleOCR/configs/det/det_mv3_db.yml
# pretrained_model=/home/resource/model/pretrained/MobileNetV3_large_x0_5/best_accuracy
# checkpoints=None
# save_model_dir=/home/resource/model/tuned/MobileNetV3_large_x0_5

# MobileNetV3_large_x0_5 (tuned)
# config=/home/code/PaddleOCR/configs/det/det_mv3_db.yml
# pretrained_model=/home/resource/model/tuned/MobileNetV3_large_x0_5/latest
# checkpoints=/home/resource/model/tuned/MobileNetV3_large_x0_5/latest
# save_model_dir=/home/resource/model/tuned/MobileNetV3_large_x0_5

# ml_PP-OCRv3_det (pretrained)
config=/home/code/PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml
pretrained_model=/home/resource/model/pretrained/ml_PP-OCRv3_det/best_accuracy
checkpoints=/home/resource/model/pretrained/ml_PP-OCRv3_det/best_accuracy
save_model_dir=/home/resource/model/tuned/ml_PP-OCRv3_det
# train_batch_size_per_card=64



python /home/code/PaddleOCR/tools/train.py \
    -c ${config} \
    -o Global.pretrained_model=${pretrained_model} \
        Global.print_batch_step=1 \
        Global.eval_batch_step={[0, 2000]} \
        Global.checkpoints=${checkpoints} \
        Global.save_model_dir=${save_model_dir} \
        Train.dataset.data_dir=${train_data_dir} \
        Train.dataset.label_file_list=${train_lebel_file_list} \
        Eval.dataset.data_dir=${eval_data_dir} \
        Eval.dataset.label_file_list=${eval_lebel_file_list} \
        # Train.loader.batch_size_per_card=${train_batch_size_per_card} \
        # Train.loader.num_workers=${train_num_workers} \
        # Eval.loader.batch_size_per_card=${train_batch_size_per_card} \
        # Eval.loader.num_workers=${train_num_workers} \