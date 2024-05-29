# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml
python PaddleOCR/tools/train.py --log_dir=./debug/ --gpus '0' -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  PaddleOCR/tools/train.py -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  PaddleOCR/tools/train.py -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml





#############Train
# det
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python PaddleOCR/tools/train.py -c PaddleOCR/configs/det/det_mv3_db.yml

# rec
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python PaddleOCR/tools/train.py -c PaddleOCR/configs/rec/multi_language/rec_multi_language_lite_train.yml




#############Evaluation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python PaddleOCR/tools/eval.py -c PaddleOCR/configs/det/det_mv3_db.yml



#############Test
python PaddleOCR/tools/infer_det.py -c PaddleOCR/configs/det/det_mv3_db.yml -o Global.infer_img={이미지 경로 또는 폴더 디렉터리}
