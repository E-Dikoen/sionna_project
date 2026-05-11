"""
配置管理模块
支持 YAML/JSON 格式加载参数
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """配置管理类"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        初始化配置
        
        Args:
            config_dict: 配置字典
        """
        self._config = config_dict or {}
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str):
        """从 JSON 文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔的多级键"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value):
        """设置配置值"""
        self._config[key] = value
    
    def to_dict(self):
        """转换为字典"""
        return self._config.copy()
    
    def update(self, other_dict: Dict[str, Any]):
        """更新配置"""
        self._config.update(other_dict)
    
    def save_yaml(self, path: str):
        """保存为 YAML 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def save_json(self, path: str):
        """保存为 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2)


# 默认配置
DEFAULT_CONFIG = {
    'system': {
        'name': 'sionna_simulation',
        'version': '0.1.0'
    },
    'simulation': {
        'batch_size': 100,
        'num_bits': 10000,
        'ebno_dbs': [0, 2, 4, 6, 8, 10]
    },
    'modulation': {
        'type': 'qam',
        'num_bits_per_symbol': 4
    },
    'channel': {
        'type': 'awgn',
        'coderate': 1.0
    },
    'output': {
        'save_results': True,
        'plot': True,
        'result_dir': './results'
    }
}


def get_default_config():
    """获取默认配置"""
    return Config(DEFAULT_CONFIG)
