#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档处理器测试用例
"""

import os
import sys
import tempfile
import shutil
import unittest

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from etl.document_processor import load_and_process_documents, extract_content_and_store


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp()
        self.processed_data_dir = os.path.join(self.test_dir, "processed_data")
        
        # 创建测试文本文件
        self.txt_file = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文档。\n用于测试文档处理功能。\n包含多行内容。")
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)
    
    def test_load_and_process_documents_with_txt(self):
        """测试加载和处理TXT文档"""
        # 执行文档加载和处理
        documents = load_and_process_documents(self.test_dir)
        
        # 验证结果
        self.assertGreater(len(documents), 0, "应该至少加载一个文档")
        self.assertIn("测试文档", documents[0].page_content, "文档内容应该包含测试文本")
    
    def test_extract_content_and_store(self):
        """测试内容提取和存储功能"""
        # 先加载文档
        documents = load_and_process_documents(self.test_dir)
        
        # 提取并存储内容
        extracted_content = extract_content_and_store(documents, self.processed_data_dir)
        
        # 验证结果
        self.assertEqual(len(extracted_content), len(documents), "提取的内容数量应该与文档数量一致")
        
        # 验证存储文件是否存在
        stored_files = os.listdir(self.processed_data_dir)
        self.assertGreater(len(stored_files), 0, "应该创建存储文件")
    
    def test_load_and_process_documents_empty_directory(self):
        """测试加载空目录"""
        # 创建空目录
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        
        # 执行文档加载和处理
        documents = load_and_process_documents(empty_dir)
        
        # 验证结果
        self.assertEqual(len(documents), 0, "空目录应该返回空文档列表")
    
    def test_extract_content_and_store_without_storage(self):
        """测试不指定存储路径的内容提取"""
        # 先加载文档
        documents = load_and_process_documents(self.test_dir)
        
        # 提取内容但不存储
        extracted_content = extract_content_and_store(documents)
        
        # 验证结果
        self.assertEqual(len(extracted_content), len(documents), "提取的内容数量应该与文档数量一致")


if __name__ == "__main__":
    unittest.main()