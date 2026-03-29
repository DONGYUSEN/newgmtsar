#!/usr/bin/env python3
"""
并行处理模块
功能：提供并行处理能力，支持多线程、多进程和GPU加速
"""

import os
import numpy as np
from typing import Callable, List, Any, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import multiprocessing as mp


class ParallelProcessor:
    """并行处理类"""
    
    def __init__(self, num_workers: Optional[int] = None, 
                 use_gpu: bool = False,
                 chunk_size: int = 1024):
        """初始化并行处理器
        
        Args:
            num_workers: 工作线程/进程数量
            use_gpu: 是否使用GPU
            chunk_size: 数据块大小
        """
        self.num_workers = num_workers or cpu_count()
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
    
    def map_threads(self, func: Callable, items: List[Any], 
                    max_workers: Optional[int] = None) -> List[Any]:
        """使用线程池映射函数
        
        Args:
            func: 要执行的函数
            items: 输入项列表
            max_workers: 最大工作线程数
            
        Returns:
            结果列表
        """
        if max_workers is None:
            max_workers = self.num_workers
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in items}
            
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        return results
    
    def map_processes(self, func: Callable, items: List[Any],
                     max_workers: Optional[int] = None) -> List[Any]:
        """使用进程池映射函数
        
        Args:
            func: 要执行的函数
            items: 输入项列表
            max_workers: 最大工作进程数
            
        Returns:
            结果列表
        """
        if max_workers is None:
            max_workers = self.num_workers
        
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in items}
            
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        return results
    
    def chunk_data(self, data: np.ndarray) -> List[np.ndarray]:
        """将数据分块
        
        Args:
            data: 输入数据数组
            
        Returns:
            数据块列表
        """
        total_size = data.shape[0]
        chunks = []
        
        for i in range(0, total_size, self.chunk_size):
            chunk = data[i:min(i + self.chunk_size, total_size)]
            chunks.append(chunk)
        
        return chunks
    
    def parallel_apply(self, func: Callable, data: np.ndarray,
                      axis: int = 0, use_threads: bool = True) -> np.ndarray:
        """并行应用函数到数据
        
        Args:
            func: 要应用的函数
            data: 输入数据数组
            axis: 应用轴
            use_threads: 是否使用线程（否则使用进程）
            
        Returns:
            结果数组
        """
        if axis != 0:
            data = np.moveaxis(data, axis, 0)
        
        chunks = self.chunk_data(data)
        
        if use_threads:
            results = self.map_threads(func, chunks)
        else:
            results = self.map_processes(func, chunks)
        
        return np.concatenate(results, axis=0)
    
    def batch_process(self, items: List[Any], process_func: Callable,
                     batch_size: int = 10,
                     use_threads: bool = True) -> List[Any]:
        """批处理项目
        
        Args:
            items: 项目列表
            process_func: 处理函数
            batch_size: 批大小
            use_threads: 是否使用线程
            
        Returns:
            处理结果列表
        """
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        def process_batch(batch):
            return [process_func(item) for item in batch]
        
        if use_threads:
            batch_results = self.map_threads(process_batch, batches)
        else:
            batch_results = self.map_processes(process_batch, batches)
        
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results


class GPUProcessor:
    """GPU处理类"""
    
    def __init__(self, device_id: int = 0):
        """初始化GPU处理器
        
        Args:
            device_id: GPU设备ID
        """
        self.device_id = device_id
        self.available = False
        self._check_gpu()
    
    def _check_gpu(self):
        """检查GPU是否可用"""
        try:
            import cupy as cp
            self.cp = cp
            self.available = True
        except ImportError:
            try:
                import torch
                self.torch = torch
                self.available = torch.cuda.is_available()
            except ImportError:
                self.available = False
    
    def to_device(self, data: np.ndarray) -> Any:
        """将数据移动到GPU
        
        Args:
            data: NumPy数组
            
        Returns:
            GPU上的数据
        """
        if not self.available:
            return data
        
        if hasattr(self, 'cp'):
            return self.cp.asarray(data)
        elif hasattr(self, 'torch'):
            return self.torch.from_numpy(data).cuda()
        
        return data
    
    def to_host(self, data: Any) -> np.ndarray:
        """将数据从GPU移动到主机
        
        Args:
            data: GPU上的数据
            
        Returns:
            NumPy数组
        """
        if not self.available:
            return data
        
        if hasattr(self, 'cp'):
            return self.cp.asnumpy(data)
        elif hasattr(self, 'torch'):
            return data.cpu().numpy()
        
        return data
    
    def apply_func(self, func: Callable, data: np.ndarray) -> np.ndarray:
        """在GPU上应用函数
        
        Args:
            func: 要应用的函数
            data: 输入数据
            
        Returns:
            结果数组
        """
        if not self.available:
            return func(data)
        
        device_data = self.to_device(data)
        result = func(device_data)
        
        return self.to_host(result)


def parallel_map(func: Callable, items: List[Any],
                 num_workers: Optional[int] = None,
                 use_threads: bool = True) -> List[Any]:
    """并行映射函数（便捷函数）
    
    Args:
        func: 要执行的函数
        items: 输入项列表
        num_workers: 工作线程/进程数量
        use_threads: 是否使用线程
        
    Returns:
        结果列表
    """
    processor = ParallelProcessor(num_workers=num_workers)
    
    if use_threads:
        return processor.map_threads(func, items)
    else:
        return processor.map_processes(func, items)


def parallel_apply(func: Callable, data: np.ndarray,
                  axis: int = 0,
                  num_workers: Optional[int] = None) -> np.ndarray:
    """并行应用函数（便捷函数）
    
    Args:
        func: 要应用的函数
        data: 输入数据数组
        axis: 应用轴
        num_workers: 工作进程数量
        
    Returns:
        结果数组
    """
    processor = ParallelProcessor(num_workers=num_workers)
    return processor.parallel_apply(func, data, axis)


if __name__ == '__main__':
    def test_func(x):
        return x * 2
    
    items = list(range(10))
    
    processor = ParallelProcessor(num_workers=4)
    
    results = processor.map_threads(test_func, items)
    print(f"线程结果: {results}")
    
    results = processor.map_processes(test_func, items)
    print(f"进程结果: {results}")
    
    print("并行处理测试完成")
