import yaml
import dotdict

class ConfigLoader:
    @staticmethod
    def check_all(args):
        if "SUPPORTED" not in args:
            return
        ConfigLoader.check(args, args.SUPPORTED)
    
    @staticmethod
    def check(value, supported_values):
        if supported_values  == None:
            return
        
        if isinstance(value, dotdict.dotdict):
            for key in value.keys():
                ConfigLoader.check(value.get(key), supported_values.get(key))
            return
        
        if isinstance(value, list):
            for v in value:
                assert v in supported_values, f"The value {v} is not supported. Check config"            
        else:
            assert value in supported_values, f"The value {value} is not supported. Check config"
            
class YamlConfigLoader(ConfigLoader):
    @staticmethod
    def load_config(config_path):
        with open(config_path, encoding='utf-8') as f:
            config = dotdict.dotdict(yaml.load(f,  Loader=yaml.FullLoader))
            YamlConfigLoader.check_all(config)
            return config    