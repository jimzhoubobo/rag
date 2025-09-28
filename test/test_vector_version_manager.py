#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量库版本管理器单元测试
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from etl.vector_version_manager import VectorVersionManager


class TestVectorVersionManager(unittest.TestCase):
    """VectorVersionManager测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp()
        self.version_manager = VectorVersionManager(self.test_dir)
        
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.version_manager.base_directory, self.test_dir)
        self.assertEqual(self.version_manager.active_version_file, 
                         os.path.join(self.test_dir, "active_version.txt"))
        self.assertEqual(self.version_manager.version_prefix, "chroma_v")
        
    def test_get_next_version(self):
        """测试获取下一个版本号"""
        # 第一个版本
        version1 = self.version_manager._get_next_version()
        self.assertEqual(version1, "chroma_v001")
        
        # 创建一个版本目录
        version1_path = os.path.join(self.test_dir, version1)
        os.makedirs(version1_path)
        
        # 下一个版本
        version2 = self.version_manager._get_next_version()
        self.assertEqual(version2, "chroma_v002")
        
    def test_get_all_versions(self):
        """测试获取所有版本"""
        # 没有版本时
        versions = self.version_manager._get_all_versions()
        self.assertEqual(versions, [])
        
        # 创建几个版本目录
        version_dirs = ["chroma_v001", "chroma_v002", "chroma_v003"]
        for version_dir in version_dirs:
            os.makedirs(os.path.join(self.test_dir, version_dir))
            
        # 获取所有版本
        versions = self.version_manager._get_all_versions()
        self.assertEqual(len(versions), 3)
        self.assertIn("chroma_v001", versions)
        self.assertIn("chroma_v002", versions)
        self.assertIn("chroma_v003", versions)
        
    def test_get_current_version(self):
        """测试获取当前版本"""
        # 没有活动版本文件时
        current_version = self.version_manager._get_current_version()
        self.assertIsNone(current_version)
        
        # 设置当前版本
        test_version = "chroma_v001"
        self.version_manager._set_current_version(test_version)
        
        # 获取当前版本
        current_version = self.version_manager._get_current_version()
        self.assertEqual(current_version, test_version)
        
    def test_get_active_version_path(self):
        """测试获取活动版本路径"""
        # 没有活动版本时
        active_path = self.version_manager.get_active_version_path()
        self.assertIsNone(active_path)
        
        # 设置当前版本并创建目录
        test_version = "chroma_v001"
        self.version_manager._set_current_version(test_version)
        version_path = os.path.join(self.test_dir, test_version)
        os.makedirs(version_path)
        
        # 获取活动版本路径
        active_path = self.version_manager.get_active_version_path()
        self.assertEqual(active_path, version_path)
        
    @patch('etl.vector_version_manager.build_vector_store')
    @patch('etl.vector_version_manager.load_vector_store')
    def test_create_new_version(self, mock_load_vector_store, mock_build_vector_store):
        """测试创建新版本"""
        # 模拟build_vector_store函数
        mock_build_vector_store.return_value = Mock()
        
        documents = ["文档1", "文档2"]
        new_version = self.version_manager.create_new_version(documents)
        
        # 验证返回的版本号
        self.assertEqual(new_version, "chroma_v001")
        
        # 验证build_vector_store被调用
        mock_build_vector_store.assert_called_once()
        
    @patch('etl.vector_version_manager.load_vector_store')
    def test_validate_version(self, mock_load_vector_store):
        """测试验证版本"""
        # 模拟load_vector_store和相似性搜索
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = ["结果"]
        mock_load_vector_store.return_value = mock_vector_store
        
        version_path = os.path.join(self.test_dir, "chroma_v001")
        os.makedirs(version_path)
        
        # 验证版本
        result = self.version_manager._validate_version(version_path)
        self.assertTrue(result)
        
        # 验证load_vector_store被调用
        mock_load_vector_store.assert_called_once_with(version_path)
        
    @patch('etl.vector_version_manager.load_vector_store')
    def test_validate_version_failure(self, mock_load_vector_store):
        """测试验证版本失败"""
        # 模拟load_vector_store抛出异常
        mock_load_vector_store.side_effect = Exception("加载失败")
        
        version_path = os.path.join(self.test_dir, "chroma_v001")
        os.makedirs(version_path)
        
        # 验证版本
        result = self.version_manager._validate_version(version_path)
        self.assertFalse(result)
        
    def test_cleanup_old_versions(self):
        """测试清理旧版本"""
        # 创建6个版本目录
        version_dirs = ["chroma_v001", "chroma_v002", "chroma_v003", 
                       "chroma_v004", "chroma_v005", "chroma_v006"]
        for version_dir in version_dirs:
            os.makedirs(os.path.join(self.test_dir, version_dir))
            
        # 清理旧版本
        self.version_manager._cleanup_old_versions()
        
        # 检查剩余版本数量
        remaining_versions = self.version_manager._get_all_versions()
        self.assertEqual(len(remaining_versions), 5)
        
    def test_list_versions(self):
        """测试列出版本信息"""
        # 创建几个版本目录
        version_dirs = ["chroma_v001", "chroma_v002"]
        for version_dir in version_dirs:
            os.makedirs(os.path.join(self.test_dir, version_dir))
            
        # 设置当前版本
        self.version_manager._set_current_version("chroma_v002")
        
        # 获取版本信息列表
        version_info = self.version_manager.list_versions()
        
        # 验证返回的信息
        self.assertEqual(len(version_info), 2)
        self.assertEqual(version_info[0]["version"], "chroma_v002")
        self.assertTrue(version_info[0]["is_active"])
        self.assertEqual(version_info[1]["version"], "chroma_v001")
        self.assertFalse(version_info[1]["is_active"])


if __name__ == '__main__':
    unittest.main()