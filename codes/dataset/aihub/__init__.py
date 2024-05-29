
from pathlib import Path
from dataset import *
import json
from pathlib import Path
from PIL import Image
import os
import numpy as np
from config_loader import YamlConfigLoader
import project
with open(Path(os.path.realpath(__file__)).parent/"preprocess/organize.json", encoding='UTF8') as f:
    ALL_PATH_PAIR = json.load(f)
    

DEFALT_CONFIG_PATH = Path(os.path.realpath(__file__)).parent/"config.yml"


class DirChecklist:
    # @staticmethod
    # def get_checklist(path):
    #     with open(path) as f:
    #         return yaml.load(f, Loader=yaml.FullLoader)    
    
    @staticmethod
    def get_checked_dir_list(checklist):
        return DirChecklist._get_checked_dir_list(checklist, "")  
    
    @staticmethod
    def _get_checked_dir_list(checklist, path):
        # checklist = dict(checklist)
        path = Path(path)
        if isinstance(checklist, dict):
            if checklist["all"] == True:
                return [path]
            elif checklist["all"] == False:
                return []
            else:
                del checklist["all"]
                return sum([DirChecklist._get_checked_dir_list(v, path/k) for k, v in checklist.items()], [])

        if checklist == True:
            return [path]
        else:
            return []

class HangulRealImageDataset(Dataset_Loader):
    """
        AI HUB에서 제공하는 '야외 실제 촬영 한글 이미지'의 데이터를 관리하는 클래스
        데이터 정리, 로드 등 기능 제공 
        데이터를 직접 다운해야 함
        
    """
    

    def get_x(self, index):
        return self.load_x(self.get_x_path(index))
    

    def get_y(self, index):
        return self.load_y(self.get_y_path(index))
    
    def __len__(self):
        return len(self.sample_list)
    
    def __init__(self, args=None):
        args = args if args is not None else YamlConfigLoader.load_config(DEFALT_CONFIG_PATH)
        self.args = args
        
        checklist = args.checklist
        
        self.dataset_dir = f"{project.PROJECT_ROOT}/{args.dataset_dir}"

        self.checked_dir_list = self.get_checked_dir_list(checklist)
        
        print("*******Checked Dir List*******")
        for dir in self.checked_dir_list:
            print(dir)
        print("******************************")
        self.sample_list = self.get_all_sample_list()
        super().__init__()

    def get_checked_dir_list(self, checklist_dir_list):
        checked_dir_list = DirChecklist.get_checked_dir_list(checklist_dir_list)
        return [Path(self.dataset_dir)/x for x in checked_dir_list]
    
    def get_x_path(self, index):
        return self.sample_list[index][0]
    
    def get_y_path(self, index):
        return self.sample_list[index][1]
    
    def load_x(self, path):
        return Image.open(path)
    
    def load_y(self, path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
        
    def get_all_sample_list(self):
        img_ext = self.args.x_ext
        label_ext = self.args.y_ext
        sample_list = []
                
        for checked_dir in self.checked_dir_list:
            for img_path in sorted(Path(checked_dir).rglob(f"*.{img_ext}")):
                label_path = img_path.parent/f"{img_path.stem}.{label_ext}"
                label_path = Path(str(label_path).replace("source", "label"))
                sample_list.append([img_path, label_path])
        return sample_list


    
class HangulRealImage_BoxDetectionDataset(Dataset_Converter):
    # @override
    def get_x(self, i):
        return self.dataset_loader.get_x(i)
    
    # @override
    def get_y(self, i):
        y = self.dataset_loader.get_y(i)

        label = y
        result = []
        for annotation in label["annotations"]:
            try:
                
                x, y, w, h = annotation["bbox"]
                upper_left = [x, y]
                upper_right = [x+w, y]
                bottom_right = [x+w, y+h]
                bottom_left = [x, y+h]
            except:
                continue
            result.append({"bbox":[upper_left, upper_right, bottom_right, bottom_left], "label":annotation["text"]})
        return result

    def get_x_path(self, index):
        return self.dataset_loader.get_x_path(index)
    
    def get_y_path(self, index):
        return self.dataset_loader.get_y_path(index)

def show_data_example_function():
    import sys
    sys.path.append("/home/code")


    DEFALT_CONFIG_PATH = "/home/code/dataset/aihub/hangul_real_image_dataset/config.yml"
    config = YamlConfigLoader.load_config(DEFALT_CONFIG_PATH)  
    dataset = HangulRealImageDataset(config)
    box_det_dataset = dataset.get_box_detection_dataset()

    sample_idx = 10
    x, y = box_det_dataset[sample_idx]
    box_det_dataset.show_xy(x, y) # 레이블링된 이미지 출력
    box_det_dataset.show_y(y) # 레이블 출력
