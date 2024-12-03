import os
from pathlib import Path
import yaml

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        config_path = os.environ.get("EXO_CONFIG", "~/.exo/config.yml")
        config_path = os.path.expanduser(config_path)
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
            
    def get_model_path(self, model_id: str) -> Optional[str]:
        return self.config.get("model_paths", {}).get(model_id) 