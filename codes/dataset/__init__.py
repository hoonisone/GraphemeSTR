# from abc import *
# from PIL import Image
# import bbox_visualizer as bbv

# class OpenDataset(metaclass=ABCMeta):
#     def __init__(self, args):
#         self.args = args
#         self.sample_list = self.get_all_sample_list()

#     @abstractmethod
#     def load_x(self, path):
#         pass
    
#     @abstractmethod
#     def load_y(self, path):
#         pass                
    
#     def get_x_path(self, index):
#         return self.sample_list[index][0]
    
#     def get_y_path(self, index):
#         return self.sample_list[index][1]
    
#     def get_x(self, index):
#         return self.load_x(self.get_x_path(index))
    
#     def get_y(self, index):
#         return self.load_y(self.get_y_path(index))
    
#     def __len__(self):
#         return len(self.sample_list)
    
#     def __getitem__(self, index):
#         return [self.get_x(index), self.get_y(index)]
    
#     def get_sample_path(self, index):
#         return self.sample_list[index]
    
#     @abstractmethod
#     def get_all_sample_list(self):
#         pass

#     def get_box_detection_dataset():
#         return None
    
# class OpenDatasetDecorator:
#     def __init__(self, dataset):
#         self.__dataset = dataset
        
#     def __len__(self):
#         return len(self.__dataset)

#     def convert_x(self, x):
#         return x
    
#     def convert_y(self, y):
#         return y
    
#     def get_x(self, index):
#         x = self.__dataset.get_x(index)
#         return self.convert_x(x)
    
#     def get_y(self, index):
#         y = self.__dataset.get_y(index)
#         return self.convert_y(y)
    
#     def __getitem__(self, index):
#         return [self.get_x(index), self.get_y(index)]
    
#     def get_x_path(self, index):
#         return self.__dataset.get_x_path(index)
    
#     def get_y_path(self, index):
#         return self.__dataset.get_y_path(index)
    
#     @staticmethod
#     def xywh_to_polygon(x, y, w, h):
#         return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]

#     @staticmethod
#     def centered_xywh_to_polygon(cx, cy, w, h):
#         x = cx-w/2
#         y = cy-h/2
#         return Datase.xywh_to_polygon(x, y, w, h)

#     @staticmethod
#     def polygon_to_xywh(polygon):
#         x = [point[0] for point in polygon]
#         y = [point[1] for point in polygon]
#         min_x, max_x = min(x), max(x)
#         min_y, max_y = min(y), max(y)
#         x, y, w, h = min_x, min_y, max_x-min_x, max_y-min_y
#         return [x, y, w, h]

#     @staticmethod
#     def polygon_to_centered_xywh(polygon):
#         x = [point[0] for point in polygon]
#         y = [point[1] for point in polygon]
#         min_x, max_x = min(x), max(x)
#         min_y, max_y = min(y), max(y)
#         x, y, w, h = min_x, min_y, max_x-min_x, max_y-min_y
#         return [x+w/2, y+h/2, w, h]
        
            
# class BoxDetectionDataset(OpenDatasetDecorator):
#     @abstractmethod
#     def convert_x(self, x):
#         # numpy image [3, w, h]
#         pass
    
#     @abstractmethod
#     def convert_y(self, x):
#         # return: label list
#         # label: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  
#         # order: (top-left, top-right, bottom-right, bottom-left)
#         pass


            
#     def show_x(self, x):
#         Image.fromarray(x).show()
    
#     def show_y(self, y):
#         for v in y:
#             label = v["label"]
#             bbox = v["bbox"]
#             print(f"{label:-10s}: {bbox}")
    
#     def show_xy(self, x, y):
#         labels = [v["label"] for v in y]
#         poligon_boxes = [v["bbox"] for v in y]

#         xywh_bboxes = self.poligon_to_xywh(poligon_boxes)
#         min_max_xy_bboxes = [[x, y, x+w, y+h]for x, y, w, h in xywh_bboxes]
#         x = bbv.draw_multiple_rectangles(x, min_max_xy_bboxes, thickness=2, bbox_color=(0,0,0))
#         self.show_x(x)
        
from abc import *
from PIL import Image
import json
from pathlib import Path

class Dataset_Loader(metaclass=ABCMeta):
    def __init__(self):
        pass
        
    @abstractmethod    
    def get_x(self, index):
        pass
    
    @abstractmethod
    def get_y(self, index):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

    @staticmethod
    def xywh_to_polygon(x, y, w, h):
        return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]

    @staticmethod
    def centered_xywh_to_polygon(cx, cy, w, h):
        x = cx-w/2
        y = cy-h/2
        return Dataset_Loader.xywh_to_polygon(x, y, w, h)
    
    @staticmethod
    def polygon_to_xywh(polygon):
        x = [point[0] for point in polygon]
        y = [point[1] for point in polygon]
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        x, y, w, h = min_x, min_y, max_x-min_x, max_y-min_y
        return [x, y, w, h]

    @staticmethod
    def polygon_to_centered_xywh(polygon):
        x = [point[0] for point in polygon]
        y = [point[1] for point in polygon]
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        x, y, w, h = min_x, min_y, max_x-min_x, max_y-min_y
        return [x+w/2, y+h/2, w, h]
    
    @staticmethod
    def load_image(path):
        return Image.open(path)
    
class PPOCR_STD_Dataset_Loader(Dataset_Loader):
    def __init__(self, path, label_file_name):
        label_path = Path(path)/label_file_name
        with open(label_path) as f:
            lines = [line.rstrip("\n") for line in f.readlines()]
            self.x_path_list = [Path(path)/line.split("\t")[0] for line in lines]
            self.y_list = [json.loads(line.split("\t")[1]) for line in lines]        

    def get_x(self, index):
        return {"image":self.load_image(self.x_path_list[index])}
        
    def get_y(self, index):
        y = self.y_list[index]
        return [{"text":v["transcription"], "polygon":v["points"]} for v in y]
    
    def __len__(self):
        return len(self.y_list)
    
class Dataset_Converter(metaclass=ABCMeta):
    def __init__(self, dataset_loader):
        self.dataset_loader = dataset_loader
        
    def __len__(self):
        return len(self.dataset_loader)
    @abstractmethod
    def get_x(self, index):
        pass

    @abstractmethod
    def get_y(self, index):
        pass