from .db import DB
import project
from pathlib import Path
import os
import subprocess

class ModelDB(DB):    
    DIR = "./models"
    ROOT = f"{project.PROJECT_ROOT}/{DIR}"
    CONFIG_NAME = "model_config.yml"
    
    def __init__(self):
        super().__init__(self.ROOT, self.CONFIG_NAME)
    
    def make_inference_model(self, id):
        root = str(Path(self.ROOT)/id).replace("\\", "/")
        config = root+"/pretrained_model/config.yml"
        checkpoint = root+"/pretrained_model/pretrained"
        pretrained = root+"/pretrained_model/pretrained"
        save_inferencd = root+"/inference_model"
        command = f"""python code/PaddleOCR/tools/export_model.py \
        -c {config} \
        -o Global.pretrained_model={pretrained} \
        Global.save_inference_dir={save_inferencd} \
        Global.checkpoints={checkpoint}"""
        
        config = self.get_config(id, abs_path=False)
        config["inference_model_dir"] = "{ROOT}/{ID}/inference_model"
        config["inference_model"] = "{ROOT}/{ID}/inference_model/inference.pdmodel"
        config["inference_model_weight"] = "{ROOT}/{ID}/inference_model/inference.pdiparams"
        self.update_config(id, config)
        
        with open(Path(self.root)/"make_inference_model.sh", "a") as f:
            f.write(command+"\n")
    
    def make_config(self, id):
        config = {
            "name":id,
            "train_config":"{ROOT}/{ID}/pretrained_model/config.yml",
            "pretrained_model_dir":"{ROOT}/{ID}/pretrained_model",
            "pretrained_model_weight":"{ROOT}/{ID}/pretrained_model/pretrained.pdparams",
            
            
            "inference_model_dir":None,
            "inference_model":None,
            "inference_model_weight":None,
        }
        return config
    
    def make(self, id):
        assert id not in self.get_all_id(), f"The id '{id}' already exists."
        (Path(self.ROOT)/id/"pretrained").mkdir(parents=True, exist_ok=True)
        (Path(self.ROOT)/id/"inference").mkdir(parents=True, exist_ok=True)
        config = self.make_config(id)
        self.update_config(id, config)
        
    def get_config(self, id, relative_to=None):
        config = super().get_config(id)
        
        path_keys = ["inference_model", "inference_model_dir", "inference_model_weight",
                     "pretrained_model_dir", "pretrained_model_weight", "train_config"]

        if relative_to:
            for k in path_keys:
                config[k] = self.relative_to(id, config[k], relative_to=relative_to)
        return config

        
    def check_config(self, id):
        config = self.get_config(id)

        for v in config.values():
            print(Path(v), Path(v).exists())
        
    def get_inference_model(self, id):    
        config = self.get_config(id)
        return f"{self.ROOT}/{config['inference_model']}"
    
    def get_inference_model_dir(self, id):    
        config = self.get_config(id)
        return f"{self.ROOT}/{config['inference_model_dir']}"
    
if __name__ == "__main__":
    mdb = ModelDB()
    print(mdb.get_all_id())
    print()
    id = mdb.get_all_id()[0]
    print(mdb.get_config(id))