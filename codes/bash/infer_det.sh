

infer_img="/home/dataset/samples"

model_name_list=("pretrained/MobileNetV3_large_x0_5_pretrained"
"pretrained/ml_PP-OCRv3_det/best_accuracy"
"pretrained/ml_PP-OCRv3_det_slim/best_accuracy"
"pretrained/en_PP-OCRv3_det/best_accuracy"
)
config_list=("det/det_mv3_db.yml"
"det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml"
"det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml"
"det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml"
)

for i in 0 1 2 3
do
    model_name=${model_name_list[i]}
    config=${config_list[i]}
    python ../PaddleOCR/tools/infer_det.py \
    -c ../PaddleOCR/configs/${config} \
    -o Global.infer_img=${infer_img} \
        Global.pretrained_model=/home/resource/model/${model_name} \
        Global.save_model_dir=/home/resource/inference/${model_name} \
        Global.save_inference_dir=/home/resource/inference/${model_name} \
        Global.save_res_path=/home/resource//inference/${model_name}/predicts.txt \
        Global.use_visualdl=true
done