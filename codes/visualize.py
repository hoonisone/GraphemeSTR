import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import json
import bbox_visualizer as bbv
from PIL import ImageFont, ImageDraw, Image
from PaddleOCR.ppocr.utils.utility import check_and_read, get_image_file_list
import glob
from pathlib import Path
import cv2
# fontpath = "fonts/gulim.ttc"
fontpath = "usr/share/fonts/nanum/NanumGothic.ttf"
fontpath= "C:\Windows\Fonts\malgun.ttf"
# fontpath = "usr/share/fonts/truetype/dejavu/DejaBuSansMono.ttf"
# ImageFont.truetype(fontpath, 20)

def change_box_representation(left_top, right_top, right_bottom, left_bottom):
    res = [left_top[0], left_top[1], right_top[0], left_bottom[1]]
    return [int(x) for x in res]

def load_data(root_dir, image_name):
    
    image_path = f"{root_dir}/images/{image_name}"
    result_path = f"{root_dir}/predicts/{image_name.split('.')[0]}.json"

    img = np.array(pilimg.open(image_path))


    with open(result_path, "r") as f:
        result = json.loads(f.readline())[0]

    boxes = [change_box_representation(*x[0]) for x in result]
    labels = [x[1][0] for x in result]
    return [img, boxes, labels]

def load_image(image_path):
    return np.array(pilimg.open(image_path))

def get_prediction_path(predicted_dir, image_path):
    return Path(predicted_dir)/(f"{Path(image_path).stem}"+".json")

def load_prediction(prediction_path):

    with open(prediction_path, "r") as f:
        result = json.loads(f.readline())[0]

    boxes = [change_box_representation(*x[0]) for x in result]
    poligons = [x[0] for x in result]
    labels = [x[1][0] for x in result]
    return [boxes, poligons, labels]


def draw_multiple_labels(img, boxes, labels, font_size = None):
    font_size = font_size if font_size else max(2, max(img.shape[0], img.shape[1])/30)
    

    img_pil = Image.fromarray(img)
    # fontpath = "fonts/gulim.ttc"
    font = ImageFont.truetype(fontpath, font_size)
    draw = ImageDraw.Draw(img_pil)
    
    for box, label in zip(boxes, labels):
        box[1] -= font_size
        draw.text(box, label, font=font, fill=(255,0,0))
    return img_pil


def draw_multiple_boxes(img, boxes):
    # boxes = [[x, y, x+w, x+h]for x, y, w, h in boxes]
    print(img.shape[0])
    return bbv.draw_multiple_rectangles(img, boxes, thickness=max(2, int(img.shape[0]/200)), bbox_color=(0,255,0))

def get_image_file_name_list(image_dir):
    image_path_list = get_image_file_list(image_dir)
    image_file_name_list = [x.split("\\")[-1] for x in image_path_list]
    return image_file_name_list
def draw_poligons(img, poligons):
    for poligon in poligons:
        poligon = np.array(poligon,dtype=np.int32)
        img = cv2.polylines(img,[poligon],True,(0,255,0),4)
    return img

def main(image_dir, predicted_dir, visualized_dir):
    # change all path paremeters into Path object
    image_dir = Path(image_dir)
    predicted_dir = Path(predicted_dir)
    visualized_dir = Path(visualized_dir)

    # for each all images, load + visualize + save
    image_path_list = list(Path(image_dir).glob("*"))

    for image_path in image_path_list:
        try:

            # load image and prediction
            img = load_image(image_path)
            prediction_path = get_prediction_path(predicted_dir, image_path)
            boxes, poligons, labels = load_prediction(prediction_path)
            
            # visualize
            # img = draw_multiple_boxes(img, boxes)
            # pts1 = np.array([[783.0, 40.0], [1177.0, 51.0], [1174.0, 200.0], [780.0, 189.0]],dtype=np.int32)
            img = draw_poligons(img, poligons)
            # for poligon in poligons:
            #     print(poligon)
            #     poligon = np.array(poligon,dtype=np.int32)
            #     img = cv2.polylines(img,[poligon],True,(255,255,255),4)

            img = draw_multiple_labels(img, boxes, labels)
            plt.imshow(img)

            # save
            plt.savefig(visualized_dir/(Path(image_path).stem+".png"), dpi=1200)
        
        except Exception as e:
            print(f"error: {e}")
            continue

if __name__=="__main__":
    root_dir = Path("/home/resource/간판여행2")
    image_dir = root_dir/"images"
    predicted_dir = root_dir/"predicted"
    visualized_dir = root_dir/"visualized"
    main(image_dir, predicted_dir, visualized_dir)



    