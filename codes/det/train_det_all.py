from rec.config import *
from outputs import OutputDB
from labelsets import LabelsetDB
from pretrained_models import PretrainedModelDB
########## Dataset ##########
orders=[    

]
for epoch in range(50, 101, 10):
    # orders.append([make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_1", "finetune", "latest", config=None), epoch])
    # orders.append([make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_2", "finetune", "latest", config=None), epoch])
    orders.append([make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_3", "finetune", "latest", config=None), epoch])
    orders.append([make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_4", "finetune", "latest", config=None), epoch])
    orders.append([make_PP_OCR_config("det", "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90_k_fold_5_5", "finetune", "latest", config=None), epoch])

with open("train_det_all.sh", "w") as f:
    for order, epoch in orders:
        command = f""" python /home/code/PaddleOCR/tools/train.py \
        -c {order["config"]} \
        -o Global.epoch_num={epoch} \
            Global.pretrained_model={order["pretrained_model"]} \
            Global.print_batch_step={1} \
            Global.save_epoch_step=1 \
            Global.eval_batch_step=[0,2000] \
            Global.checkpoints={order["checkpoints"]} \
            Global.save_model_dir={order["save_model_dir"]} \
            Train.dataset.data_dir={order["train_data_dir"]} \
            Train.dataset.label_file_list={order["train_label_file_list"]} \
            Eval.dataset.data_dir={order["val_data_dir"]} \
            Eval.dataset.label_file_list={order["val_label_file_list"]}\n"""
        
        # Train.loader.batch_size_per_card={train_batch_size_per_card} \
        # Train.loader.num_workers=${train_num_workers} \
        # Eval.loader.batch_size_per_card={train_batch_size_per_card} \
        # Eval.loader.num_workers={train_num_workers} \


        f.write(command)
        

        command = f""" python /home/code/PaddleOCR/tools/train.py \
        -c /home/outputs/output1/config.yml \
            