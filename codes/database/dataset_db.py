from pathlib import Path
import project
from .db import DB
import pandas as pd
import yaml
class DatasetDB(DB):
    # 데이터 셋을 관리하는 DB
    DIR = "./datasets"
    CONFIG_NAME = "dataset_config.yml"
    def __init__(self):

        super().__init__(DatasetDB.DIR, DatasetDB.CONFIG_NAME)    
    
    def make(self, id):
        assert id not in self.get_all_id(), f"the dataset '{id}' already exists!"
        
        (Path(project.PROJECT_ROOT)/self.dir/id).mkdir(parents=True, exist_ok=True)
        
        config = {
            "virtual":{
                "flag": False,
                "origin_dataset": ""                 
            },
            "task":[],
            "labelfiles":[],
            "options": []
        }
        self.update_config(id, config)
    
    def make_virtual(self, id, origin_dataset):
        assert id not in self.get_all_id(), f"the dataset '{id}' already exists!"
        assert origin_dataset in self.get_all_id(), f"the origin dataset '{origin_dataset}' does not exist!"
        
        (Path(project.PROJECT_ROOT)/self.dir/id).mkdir(parents=True, exist_ok=True)
        
        config = {
            "virtual":{
                "flag": True,
                "origin_dataset": origin_dataset                 
            },
            "task":[],
            "labelfiles":[],
            "options": []
        }
        self.update_config(id, config)
    
    def get_all_labels(self, id, relative_to=None):
        assert id in self.get_all_id(), f"the dataset '{id}' does not exist!"
        
        # 모든 레이블 파일을 로드하여 리스트로 합쳐 반환
        config = self.get_config(id, relative_to="absolute")
        labels = sum([self.load_text_file(file) for file in config["labelfiles"]], [])
        
        if relative_to:
            labels = [label.split("\t") for label in labels]
            labels = [f"{self.relative_to(id, path, relative_to=relative_to)}\t{label}" for path, label in labels]
       
        return labels
    
    def get_size(self, id):
        return len(self.get_all_labels(id))
    
    def get_config(self, id, relative_to=None):
        config = super().get_config(id)
        
        if relative_to:
            config["labelfiles"] = [self.relative_to(id, file, relative_to=relative_to) for file in config["labelfiles"]]
        return config
    
    def get_df(self):
        ides = self.get_all_id()
        df = pd.DataFrame()
        for id in ides:
            config = self.get_config(id)
            new_df = pd.DataFrame([{"id": id, "task": config["task"], "size": self.get_size(id)}])
            df = pd.concat([df, new_df])
        return df
        
    @staticmethod
    def load_text_file(path):
        with open(path) as f: return [line.rstrip('\n') for line in f.readlines()]

if __name__ == "__main__":
    mdb = DatasetDB()
    print(mdb.get_all_id())
    id = mdb.get_all_id()[0]
    print(id)
    print(mdb.get_config(id))