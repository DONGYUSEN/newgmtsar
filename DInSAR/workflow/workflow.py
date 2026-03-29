#!/usr/bin/env python3
"""
工作流管理模块
功能：管理处理流程，调度任务执行
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from core.parallel_processor import ParallelProcessor


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class Task:
    """任务类"""
    name: str
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    depends_on: List[str] = field(default_factory=list)
    description: str = ""
    
    def duration(self) -> Optional[float]:
        """获取任务执行时长（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'result': str(self.result) if self.result else None,
            'error': str(self.error) if self.error else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration(),
            'depends_on': self.depends_on
        }


class Workflow:
    """工作流类"""
    
    def __init__(self, name: str, description: str = ""):
        """初始化工作流
        
        Args:
            name: 工作流名称
            description: 工作流描述
        """
        self.name = name
        self.description = description
        self.tasks: Dict[str, Task] = {}
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status: TaskStatus = TaskStatus.PENDING
    
    def add_task(self, task: Task):
        """添加任务
        
        Args:
            task: 任务对象
        """
        self.tasks[task.name] = task
    
    def add_tasks(self, tasks: List[Task]):
        """批量添加任务
        
        Args:
            tasks: 任务列表
        """
        for task in tasks:
            self.add_task(task)
    
    def get_task(self, name: str) -> Optional[Task]:
        """获取任务
        
        Args:
            name: 任务名称
            
        Returns:
            任务对象
        """
        return self.tasks.get(name)
    
    def get_ready_tasks(self) -> List[Task]:
        """获取就绪任务（所有依赖都已完成）
        
        Returns:
            就绪任务列表
        """
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                deps_completed = True
                for dep_name in task.depends_on:
                    dep_task = self.tasks.get(dep_name)
                    if dep_task and dep_task.status != TaskStatus.COMPLETED:
                        deps_completed = False
                        break
                if deps_completed:
                    ready.append(task)
        return ready
    
    def execute(self, max_concurrent: int = 1) -> Dict[str, Task]:
        """执行工作流
        
        Args:
            max_concurrent: 最大并发任务数
            
        Returns:
            任务结果字典
        """
        self.started_at = datetime.now()
        self.status = TaskStatus.RUNNING
        
        processor = ParallelProcessor(num_workers=max_concurrent)
        
        while True:
            ready_tasks = self.get_ready_tasks()
            
            if not ready_tasks:
                break
            
            for task in ready_tasks:
                try:
                    task.status = TaskStatus.RUNNING
                    task.start_time = datetime.now()
                    
                    task.result = task.func(*task.args, **task.kwargs)
                    
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = e
                finally:
                    task.end_time = datetime.now()
            
            if all(t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] for t in self.tasks.values()):
                break
        
        self.completed_at = datetime.now()
        
        if all(t.status == TaskStatus.COMPLETED for t in self.tasks.values()):
            self.status = TaskStatus.COMPLETED
        elif any(t.status == TaskStatus.FAILED for t in self.tasks.values()):
            self.status = TaskStatus.FAILED
        
        return self.tasks
    
    def get_status(self) -> Dict:
        """获取工作流状态
        
        Returns:
            状态字典
        """
        return {
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'tasks': {name: task.to_dict() for name, task in self.tasks.items()}
        }
    
    def save(self, filepath: str):
        """保存工作流定义
        
        Args:
            filepath: 保存路径
        """
        workflow_def = {
            'name': self.name,
            'description': self.description,
            'tasks': [
                {
                    'name': task.name,
                    'description': task.description,
                    'depends_on': task.depends_on
                }
                for task in self.tasks.values()
            ]
        }
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(workflow_def, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(workflow_def, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Workflow':
        """加载工作流定义
        
        Args:
            filepath: 工作流定义文件路径
            
        Returns:
            工作流对象
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            if ext in ['.yaml', '.yml']:
                workflow_def = yaml.safe_load(f)
            else:
                workflow_def = json.load(f)
        
        workflow = cls(workflow_def['name'], workflow_def.get('description', ''))
        
        return workflow


class WorkflowTemplate:
    """工作流模板类"""
    
    TEMPLATES = {
        'dinsar_basic': {
            'name': 'DInSAR基本处理流程',
            'description': '基本的DInSAR处理流程，从L1数据到地理编码结果',
            'steps': [
                {'name': 'preprocessing', 'description': '数据预处理'},
                {'name': 'registration', 'description': '图像配准'},
                {'name': 'interferogram', 'description': '干涉图生成'},
                {'name': 'filtering', 'description': '干涉图滤波'},
                {'name': 'unwrapping', 'description': '相位解缠'},
                {'name': 'geocoding', 'description': '地理编码'}
            ]
        },
        'ps_basic': {
            'name': 'PS基线处理流程',
            'description': '永久散射体(PS)处理流程',
            'steps': [
                {'name': 'preprocessing', 'description': '数据预处理'},
                {'name': 'ps_selection', 'description': 'PS点选取'},
                {'name': 'ps_interferogram', 'description': 'PS干涉图生成'},
                {'name': 'ps_network', 'description': 'PS网络构建'},
                {'name': 'ps_timeseries', 'description': 'PS时间序列分析'}
            ]
        }
    }
    
    @classmethod
    def get_template(cls, template_name: str) -> Optional[Dict]:
        """获取模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板字典
        """
        return cls.TEMPLATES.get(template_name)
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """列出所有模板
        
        Returns:
            模板名称列表
        """
        return list(cls.TEMPLATES.keys())


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, workspace: str = './workspace'):
        """初始化工作流管理器
        
        Args:
            workspace: 工作目录
        """
        self.workspace = workspace
        self.workflows: Dict[str, Workflow] = {}
        os.makedirs(workspace, exist_ok=True)
    
    def create_workflow(self, name: str, description: str = "") -> Workflow:
        """创建工作流
        
        Args:
            name: 工作流名称
            description: 工作流描述
            
        Returns:
            工作流对象
        """
        workflow = Workflow(name, description)
        self.workflows[name] = workflow
        return workflow
    
    def get_workflow(self, name: str) -> Optional[Workflow]:
        """获取工作流
        
        Args:
            name: 工作流名称
            
        Returns:
            工作流对象
        """
        return self.workflows.get(name)
    
    def execute_workflow(self, name: str, max_concurrent: int = 1) -> Dict[str, Task]:
        """执行工作流
        
        Args:
            name: 工作流名称
            max_concurrent: 最大并发任务数
            
        Returns:
            任务结果字典
        """
        workflow = self.workflows.get(name)
        if not workflow:
            raise ValueError(f"工作流不存在: {name}")
        
        return workflow.execute(max_concurrent)
    
    def save_workflow(self, name: str, filepath: Optional[str] = None):
        """保存工作流
        
        Args:
            name: 工作流名称
            filepath: 保存路径
        """
        workflow = self.workflows.get(name)
        if not workflow:
            raise ValueError(f"工作流不存在: {name}")
        
        if filepath is None:
            filepath = os.path.join(self.workspace, f"{name}.yaml")
        
        workflow.save(filepath)
    
    def load_workflow(self, filepath: str) -> Workflow:
        """加载工作流
        
        Args:
            filepath: 工作流定义文件路径
            
        Returns:
            工作流对象
        """
        workflow = Workflow.load(filepath)
        self.workflows[workflow.name] = workflow
        return workflow
    
    def list_workflows(self) -> List[str]:
        """列出所有工作流
        
        Returns:
            工作流名称列表
        """
        return list(self.workflows.keys())


if __name__ == '__main__':
    def test_task(x):
        import time
        time.sleep(0.1)
        return x * 2
    
    workflow = Workflow('test_workflow', '测试工作流')
    
    task1 = Task('task1', test_task, args=(1,), description='任务1')
    task2 = Task('task2', test_task, args=(2,), description='任务2', depends_on=['task1'])
    task3 = Task('task3', test_task, args=(3,), description='任务3', depends_on=['task1'])
    
    workflow.add_tasks([task1, task2, task3])
    
    print("执行前:")
    for name, task in workflow.tasks.items():
        print(f"  {name}: {task.status.value}")
    
    workflow.execute(max_concurrent=2)
    
    print("\n执行后:")
    for name, task in workflow.tasks.items():
        print(f"  {name}: {task.status.value}, 结果: {task.result}")
    
    print("\n工作流状态:")
    print(workflow.get_status())
