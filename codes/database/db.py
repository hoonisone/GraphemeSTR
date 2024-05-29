import project

import yaml
from pathlib import Path
class DB:
    # 지정된 디렉터리 안에 모든 데이터를 저장하며
    # 마치 DB 처럼 쿼리를 제공하는 클래스
    # 모든 메타 데이터는 config.yaml 파일로 관리 
    PROJECT_ROOT = project.PROJECT_ROOT
    
    def __init__(self, dir, config_name):
        self.dir = dir
        self.config_name = config_name

        
    def get_all_id(self):
        path_list = (Path(project.PROJECT_ROOT)/self.dir).glob("*")
        element_list = []
        for path in path_list:
            if path.is_dir() and self.is_target(path):
                element_list.append(path.name)
        return element_list
        
    def is_target(self, path):
        return len(list(path.glob(self.config_name))) == 1
        
    def get_path(self, id):
        return str(Path(self.PROJECT_ROOT)/self.dir/id/self.config_name).replace("\\", "/")
    
    def get_config(self, id):
        
        path = self.get_path(id)
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if config == None:
            config = {}
        # config["id"] = id
        # if abs_path:
        #     config = self.replace(config, "{DIR}", self.dir)
        #     config = self.replace(config, "{ID}", id)
        return config
    
    def update_config(self, id, config):
        path = self.get_path(id)
        with open(path, "w") as f:
            yaml.dump(config, f)

    def replace(self, value, target , replace):
        # config 안에 모든 값들에 대해 target을 replace로 대체해줌
        if isinstance(value, str):
            return value.replace(target, replace)
        if isinstance(value, dict):
            return {k:self.replace(v, target, replace) for k, v in value.items()}
        if isinstance(value, list):
            return [self.replace(x, target, replace) for x in value]
        return value

    def relative_to(self, id, path, relative_to):    
        assert relative_to in ["dir", "project", "absolute"], f"relative_to should be 'dir', 'project', or 'absolute' but {relative_to} is given"
        if relative_to == "dir":
            return str(Path(id)/path).replace("\\", "/")
        elif relative_to == "project":
            return str(Path(self.DIR)/id/path).replace("\\", "/")
        elif relative_to == "absolute":
            return str(Path(self.PROJECT_ROOT)/self.DIR/id/path).replace("\\", "/")
        else:
            return path
