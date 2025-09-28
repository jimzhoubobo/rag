#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目常量定义模块
统一管理项目中使用的所有常量
"""

import os


class ProjectConstants:
    """项目常量类"""
    
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 向量库存储目录
    CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
    
    # 数据目录
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    
    # 处理后数据目录
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
    
    # 模型目录
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    
    # 日志目录
    LOGS_DIR = "/apps/logs/tcm-rag-qa"
    
    # 向量库构建批大小
    VECTOR_STORE_BATCH_SIZE = 100
    
    @classmethod
    def get_chroma_db_path(cls):
        """
        获取向量库的绝对路径
        
        Returns:
            str: 向量库的绝对路径
        """
        return cls.CHROMA_DB_DIR
    
    @classmethod
    def get_data_path(cls):
        """
        获取数据目录的绝对路径
        
        Returns:
            str: 数据目录的绝对路径
        """
        return cls.DATA_DIR
    
    @classmethod
    def get_processed_data_path(cls):
        """
        获取处理后数据目录的绝对路径
        
        Returns:
            str: 处理后数据目录的绝对路径
        """
        return cls.PROCESSED_DATA_DIR
    
    @classmethod
    def get_models_path(cls):
        """
        获取模型目录的绝对路径
        
        Returns:
            str: 模型目录的绝对路径
        """
        return cls.MODELS_DIR


# 为了方便使用，也可以定义模块级常量
PROJECT_ROOT = ProjectConstants.PROJECT_ROOT
CHROMA_DB_DIR = ProjectConstants.CHROMA_DB_DIR
DATA_DIR = ProjectConstants.DATA_DIR
PROCESSED_DATA_DIR = ProjectConstants.PROCESSED_DATA_DIR
MODELS_DIR = ProjectConstants.MODELS_DIR
VECTOR_STORE_BATCH_SIZE = ProjectConstants.VECTOR_STORE_BATCH_SIZE