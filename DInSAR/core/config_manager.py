#!/usr/bin/env python3
"""
配置管理模块
功能：管理系统配置和参数
"""

import os
import yaml
import json
from typing import Any, Dict, Optional


class ConfigManager:
    """配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config: Dict[str, Any] = {}
        self.config_file = config_file
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            self._set_default_config()
    
    def _set_default_config(self):
        """设置默认配置"""
        self.config = {
            'system': {
                'name': 'GMTSAR-Python',
                'version': '1.0.0',
                'log_level': 'INFO'
            },
            'paths': {
                'data_dir': './data',
                'output_dir': './output',
                'temp_dir': './temp',
                'log_dir': './logs'
            },
            'processing': {
                'num_threads': 4,
                'gpu_enabled': False,
                'chunk_size': 1024
            },
            'preprocessing': {
                'l1_processor': 'dj1',
                'orbit_type': 'precise',
                'radiometric_correction': True
            },
            'registration': {
                'method': 'cross_correlation',
                'window_size': 64,
                'search_window': 64,
                'max_iterations': 100
            },
            'interferometry': {
                'filter_enabled': True,
                'filter_window': 5,
                'coherence_threshold': 0.3
            },
            'unwrapping': {
                'algorithm': 'snaphu',
                'budget': 1000
            },
            'geocoding': {
                'projection': 'UTM',
                'datum': 'WGS84'
            }
        }
    
    def load_config(self, config_file: str):
        """加载配置文件
        
        Args:
            config_file: 配置文件路径
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        ext = os.path.splitext(config_file)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
        elif ext == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
        
        if loaded_config:
            self._merge_config(loaded_config)
    
    def _merge_config(self, loaded_config: Dict[str, Any]):
        """合并加载的配置
        
        Args:
            loaded_config: 加载的配置
        """
        self._deep_update(self.config, loaded_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典
        
        Args:
            base_dict: 基础字典
            update_dict: 更新字典
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, config_file: Optional[str] = None):
        """保存配置文件
        
        Args:
            config_file: 配置文件路径
        """
        file_path = config_file or self.config_file
        
        if not file_path:
            raise ValueError("未指定配置文件路径")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        elif ext == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键（支持点分隔的路径，如 'system.name'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值
        
        Args:
            key: 配置键（支持点分隔的路径，如 'system.name'）
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置节
        
        Args:
            section: 配置节名称
            
        Returns:
            配置节字典
        """
        return self.config.get(section, {})
    
    def update_section(self, section: str, values: Dict[str, Any]):
        """更新配置节
        
        Args:
            section: 配置节名称
            values: 要更新的值
        """
        if section not in self.config:
            self.config[section] = {}
        
        self._deep_update(self.config[section], values)


def create_default_config(output_file: str):
    """创建默认配置文件
    
    Args:
        output_file: 输出配置文件路径
    """
    manager = ConfigManager()
    manager.save_config(output_file)
    print(f"默认配置文件已创建: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='配置管理工具')
    parser.add_argument('--create', type=str, help='创建默认配置文件')
    args = parser.parse_args()
    
    if args.create:
        create_default_config(args.create)
