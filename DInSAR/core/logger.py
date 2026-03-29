#!/usr/bin/env python3
"""
日志系统模块
功能：记录系统运行状态和错误
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class Logger:
    """日志管理类"""
    
    _instance: Optional['Logger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not Logger._initialized:
            self._setup_logging()
            Logger._initialized = True
    
    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger('GMTSAR')
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def setup_file_logging(self, log_dir: str = './logs', 
                           log_level: str = 'DEBUG',
                           max_bytes: int = 10 * 1024 * 1024,
                           backup_count: int = 5):
        """设置文件日志
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
        """
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'gmtsar.log')
        
        level = getattr(logging, log_level.upper(), logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        error_log_file = os.path.join(log_dir, 'gmtsar_error.log')
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str):
        """记录调试信息
        
        Args:
            message: 日志消息
        """
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录一般信息
        
        Args:
            message: 日志消息
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告信息
        
        Args:
            message: 日志消息
        """
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """记录错误信息
        
        Args:
            message: 日志消息
            exc_info: 是否包含异常信息
        """
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str):
        """记录严重错误信息
        
        Args:
            message: 日志消息
        """
        self.logger.critical(message)
    
    def log(self, level: int, message: str):
        """记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
        """
        self.logger.log(level, message)
    
    def set_level(self, level: str):
        """设置日志级别
        
        Args:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level_value = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(level_value)
        
        for handler in self.logger.handlers:
            handler.setLevel(level_value)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    if name:
        return logging.getLogger(f'GMTSAR.{name}')
    return logging.getLogger('GMTSAR')


def setup_logging(log_dir: str = './logs', 
                  log_level: str = 'INFO',
                  console: bool = True):
    """设置日志系统
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        console: 是否输出到控制台
    """
    logger = Logger()
    
    logger.setup_file_logging(log_dir=log_dir, log_level=log_level)
    
    if not console:
        for handler in logger.logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                logger.logger.removeHandler(handler)
    
    return logger


if __name__ == '__main__':
    logger = Logger()
    
    logger.info("测试日志系统")
    logger.debug("调试信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    
    print("日志系统测试完成")
