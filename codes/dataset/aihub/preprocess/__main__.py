import sys
sys.path.append("/home/code")

from pathlib import Path
import os
import json
import argparse
from config_loader import YamlConfigLoader

with open(Path(os.path.realpath(__file__)).parent/"organize.json") as f:
    ALL_PATH_PAIR = json.load(f)
    
class HangulRealImageDatasetOganizer:
    """
        AIHUB 야외 실제 촬영 한글 이미지 데이터 셋에 대해
        폴더를 정리해줌
    """
    # def __init__(self, origin_root):
        # self.root = origin_root
        # assert self.check_all_dir_exist(), f"The root '{origin_root}' has to contain all ai-hub data."
        # assert self.check_all_pair_exist(), f"The renewal imfor is not correct"
    
    @staticmethod
    def unzip(root):
        ""
        # extensions = OpenDataset.check_zip_extensions(self.args.Unzip.extensions)
        origin_root = Path(root) # zip 파일들이 있는 데이터 셋의 root
        target_root = Path(root) # zip 파일들의 압축을 풀어 경로를 유지한 체 저장하기 시작할 대상 root
        
        for origin_path in sorted(origin_root.rglob(f"*.zip")):
            
            target_dir = target_root/origin_path.relative_to(origin_root).parent/origin_path.stem    # 파일을 저장할 디렉터리
            
            if target_dir.exists():
                print(f"Note: The target file '{target_dir}' already exist. It's not done to unzip this file. For doint this, delete the target file")
                continue
            
            target_dir.mkdir(parents=True, exist_ok=True)                                               # 유효성 체크
            
            print(f"unzip -o -O cp949 '{str(origin_path)}' -d '{str(target_dir)}'")
            os.system(f"unzip -o -O cp949 '{str(origin_path)}' -d '{str(target_dir)}'")         # 압축 해제 후 저장

    @staticmethod
    def clean_zip(root):
        for target_path in sorted(Path(root).rglob(f"*.zip")):
            target_path.unlink()
            
    @staticmethod
    def get_path_pair_where_origin_exist(root):
        # pair중 origin이 root 내에 존재하는 pair만 추출하여 반환
        existing_pair = []
        for origin, target in ALL_PATH_PAIR:
            origin = Path(root)/origin
            if (origin).exists():
                existing_pair.append((origin, target))
        return existing_pair
    
    @staticmethod
    def newly_organize(root):
        # 기존의 길고 복잡한 경로를 깔끔하게 정리해줌
        # 대응되는 경로를 미리 정해져 있음
        path_pairs = HangulRealImageDatasetOganizer.get_path_pair_where_origin_exist(root) # origin이 존재하는 pair에 대해서만
                
        for origin, target in path_pairs:
            origin = Path(root)/origin
            target = Path(root)/target
            target.parent.mkdir(parents=True, exist_ok=True)
            origin.rename(target)
            print(f"'{origin}' -> '{target}'")
    
    @staticmethod
    def get_existing_target_dir(root):
        # 전체 target_dir 중 실제로 존재하는 것 만 찾아서 반환
        existing_target_dir = []
        for origin, target in ALL_PATH_PAIR:
            target = Path(root)/origin
            if (target).exists():
                existing_target_dir.append(target)
        return existing_target_dir # 실제로 변경 가능한 수정 사항만 담음
    
    @staticmethod
    def check_valid(root):
        # 변경된 디렉터리들이 유효한지 체크
        existing_target_list = HangulRealImageDatasetOganizer.get_existing_target_dir(root)
        for target in existing_target_list:
            if ("source" in target) and (target.replace("source", "label") not in existing_target_list):
                label_dir = target.replace("source", "label") 
                print(f"The image dir '{target}' needs label dir '{label_dir}'")
                assert False, "Not Invalid Dataset"

    @staticmethod
    def clean_origin_dir(root):
        dir_list = ["030.야외 실제 촬영 한글 이미지", "야외 실제 촬영 한글 이미지"]
        
        for dir in dir_list:
            dir = Path(dir)
            if dir.exists():
                dir.unlink()
                


def main(config):
    # dataset_dir = config["Global"]["dataset_dir"]
    dataset_dir = config.dataset_dir
    n = 5
    print(f"(1/{n}) root 내에 모든 zip 파일 압축 해제")#######################################################
    HangulRealImageDatasetOganizer.unzip(dataset_dir) # root 내에 모든 zip 파일 압축 해제
    
    print(f"(2/{n}) 남아있는 모든 zip 파일 제거")#######################################################
    HangulRealImageDatasetOganizer.clean_zip(dataset_dir) # 남아있는 모든 zip 파일 제거
    
    print(f"(3/{n}) 새롭게 디렉터리 구조화 (경로 정리)")#######################################################
    HangulRealImageDatasetOganizer.newly_organize(dataset_dir) # 새롭게 디렉터리 구조화 (경로 정리)
    
    print(f"(4/{n}) 이미지와 레이블이 잘 대응되어 존재하는지 점검")#######################################################
    HangulRealImageDatasetOganizer.check_valid(dataset_dir) # 이미지와 레이블이 잘 대응되어 존재하는지 점검
    
    print(f"(5/{n}) 불필요한 기존 폴더 제거 (빈 폴더 제거)")#######################################################
    HangulRealImageDatasetOganizer.clean_origin_dir(dataset_dir) # 불필요한 기존 폴더 제거 (빈 폴더 제거)



DEFALT_CONFIG_PATH = "/home/code/dataset/aihub/hangul_real_image_dataset/config.yml"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=DEFALT_CONFIG_PATH)
    return  parser.parse_args()

def load_config(path):
    return YamlConfigLoader.load_config(path)  

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)    
    main(config)
    
