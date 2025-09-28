#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chroma向量库版本管理模块
实现双缓冲版本管理，保留5个历史版本，并在切换前验证功能
"""

import os
import shutil
from datetime import datetime
from log.logger import logger
from etl.vector_builder import build_vector_store, load_vector_store
from constant.constants import CHROMA_DB_DIR


class VectorVersionManager:
    """Chroma向量库版本管理器"""
    
    def __init__(self, base_directory: str = CHROMA_DB_DIR):
        """
        初始化版本管理器
        
        Args:
            base_directory (str): 向量库基础目录
        """
        self.base_directory = base_directory
        self.active_version_file = os.path.join(base_directory, "active_version.txt")
        self.version_prefix = "chroma_v"
        
    def _get_current_version(self) -> str:
        """
        获取当前活动版本
        
        Returns:
            str: 当前版本号，如果不存在则返回None
        """
        if not os.path.exists(self.active_version_file):
            return None
            
        with open(self.active_version_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _set_current_version(self, version: str):
        """
        设置当前活动版本
        
        Args:
            version (str): 版本号
        """
        with open(self.active_version_file, 'w', encoding='utf-8') as f:
            f.write(version)
    
    def _get_all_versions(self) -> list:
        """
        获取所有版本目录
        
        Returns:
            list: 版本目录列表，按创建时间排序（新到旧）
        """
        version_dirs = []
        for item in os.listdir(self.base_directory):
            if item.startswith(self.version_prefix):
                version_path = os.path.join(self.base_directory, item)
                if os.path.isdir(version_path):
                    version_dirs.append((item, os.path.getctime(version_path)))
        
        # 按创建时间排序，新的在前
        version_dirs.sort(key=lambda x: x[1], reverse=True)
        return [version[0] for version in version_dirs]
    
    def _get_next_version(self) -> str:
        """
        获取下一个版本号
        
        Returns:
            str: 下一个版本号
        """
        versions = self._get_all_versions()
        if not versions:
            return f"{self.version_prefix}001"
        
        # 获取最新版本号并加1
        latest_version = versions[0]
        version_num = int(latest_version.replace(self.version_prefix, ""))
        next_version_num = version_num + 1
        return f"{self.version_prefix}{next_version_num:03d}"
    
    def _cleanup_old_versions(self):
        """
        清理多余的旧版本，只保留最新的5个版本
        """
        versions = self._get_all_versions()
        if len(versions) <= 5:
            return
        
        # 删除多余的旧版本
        for old_version in versions[5:]:
            old_version_path = os.path.join(self.base_directory, old_version)
            try:
                shutil.rmtree(old_version_path)
                logger.info(f"已删除旧版本: {old_version}")
            except Exception as e:
                logger.error(f"删除旧版本 {old_version} 失败: {e}")
    
    def _validate_version(self, version_path: str) -> bool:
        """
        验证版本功能是否正常
        
        Args:
            version_path (str): 版本路径
            
        Returns:
            bool: 验证是否通过
        """
        try:
            vector_store = load_vector_store(version_path)
            # 简单测试向量库是否可以正常加载和查询
            # 这里可以添加更复杂的验证逻辑
            vector_store.similarity_search("测试", k=1)
            logger.info(f"版本 {version_path} 功能验证通过")
            return True
        except Exception as e:
            logger.error(f"版本 {version_path} 功能验证失败: {e}")
            return False
    
    def create_new_version(self, documents, batch_size: int = 50) -> str:
        """
        创建新版本向量库
        
        Args:
            documents: 文档列表
            batch_size (int): 批处理大小
            
        Returns:
            str: 新版本号
        """
        next_version = self._get_next_version()
        version_path = os.path.join(self.base_directory, next_version)
        
        logger.info(f"开始创建新版本: {next_version}")
        
        try:
            # 构建新版本向量库
            build_vector_store(documents, version_path, batch_size)
            logger.info(f"新版本 {next_version} 创建完成")
            return next_version
        except Exception as e:
            logger.error(f"创建新版本 {next_version} 失败: {e}")
            # 清理失败的版本目录
            if os.path.exists(version_path):
                shutil.rmtree(version_path)
            raise
    
    def switch_to_version(self, version: str) -> bool:
        """
        切换到指定版本
        
        Args:
            version (str): 目标版本号
            
        Returns:
            bool: 切换是否成功
        """
        version_path = os.path.join(self.base_directory, version)
        
        # 验证版本是否存在
        if not os.path.exists(version_path):
            logger.error(f"版本 {version} 不存在")
            return False
        
        # 验证版本功能是否正常
        if not self._validate_version(version_path):
            logger.error(f"版本 {version} 功能验证失败，取消切换")
            return False
        
        # 执行切换
        try:
            self._set_current_version(version)
            logger.info(f"成功切换到版本: {version}")
            return True
        except Exception as e:
            logger.error(f"切换到版本 {version} 失败: {e}")
            return False
    
    def switch_to_new_version(self, documents, batch_size: int = 50) -> bool:
        """
        创建新版本并切换到新版本
        
        Args:
            documents: 文档列表
            batch_size (int): 批处理大小
            
        Returns:
            bool: 是否成功创建并切换到新版本
        """
        try:
            # 创建新版本
            new_version = self.create_new_version(documents, batch_size)
            
            # 验证新版本功能
            version_path = os.path.join(self.base_directory, new_version)
            if not self._validate_version(version_path):
                logger.error(f"新版本 {new_version} 功能验证失败，取消切换")
                # 清理验证失败的版本
                shutil.rmtree(version_path)
                return False
            
            # 切换到新版本
            if self.switch_to_version(new_version):
                # 清理旧版本
                self._cleanup_old_versions()
                return True
            else:
                # 切换失败，清理新版本
                shutil.rmtree(version_path)
                return False
                
        except Exception as e:
            logger.error(f"创建并切换到新版本失败: {e}")
            return False
    
    def get_active_version_path(self) -> str:
        """
        获取当前活动版本的路径
        
        Returns:
            str: 当前活动版本路径
        """
        current_version = self._get_current_version()
        if not current_version:
            return None
        return os.path.join(self.base_directory, current_version)
    
    def list_versions(self) -> list:
        """
        列出所有版本信息
        
        Returns:
            list: 版本信息列表
        """
        versions = self._get_all_versions()
        current_version = self._get_current_version()
        
        version_info = []
        for version in versions:
            version_path = os.path.join(self.base_directory, version)
            create_time = datetime.fromtimestamp(os.path.getctime(version_path))
            is_active = version == current_version
            
            version_info.append({
                "version": version,
                "create_time": create_time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_active": is_active,
                "path": version_path
            })
        
        return version_info


# 全局版本管理器实例
vector_version_manager = VectorVersionManager()