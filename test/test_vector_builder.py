#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量库构建器测试用例
"""

import os
import sys
import tempfile
import shutil
import unittest

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from etl.vector_builder import init_embedding, build_vector_store, load_vector_store, update_vector_store
from langchain.docstore.document import Document


class TestVectorBuilder(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp()
        self.vector_store_dir = os.path.join(self.test_dir, "vector_store")
        self.test_documents = [
            Document(page_content="这是测试文档1的内容。", metadata={"source": "test1"}),
            Document(page_content="这是测试文档2的内容。", metadata={"source": "test2"})
        ]
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)
    
    def test_init_embedding(self):
        """测试初始化嵌入模型"""
        # 初始化嵌入模型
        embedding = init_embedding()
        
        # 验证结果
        self.assertIsNotNone(embedding, "嵌入模型应该初始化成功")
        
        # 测试嵌入功能
        test_texts = ["测试文本1", "测试文本2"]
        embeddings = embedding.embed_documents(test_texts)
        self.assertEqual(len(embeddings), 2, "应该为每个文本生成嵌入向量")
        # 检查嵌入向量维度(根据使用的模型可能不同)
        self.assertGreater(len(embeddings[0]), 0, "嵌入向量维度应该大于0")
    
    def test_build_vector_store(self):
        """测试构建向量库"""
        # 构建向量库
        vector_store = build_vector_store(self.test_documents, self.vector_store_dir)
        
        # 验证结果
        self.assertIsNotNone(vector_store, "向量库应该构建成功")
        self.assertTrue(os.path.exists(self.vector_store_dir), "向量库目录应该存在")
        self.assertTrue(os.listdir(self.vector_store_dir), "向量库目录不应该为空")
    
    def test_load_vector_store(self):
        """测试加载向量库"""
        # 先构建向量库
        build_vector_store(self.test_documents, self.vector_store_dir)
        
        # 加载向量库
        vector_store = load_vector_store(self.vector_store_dir)
        
        # 验证结果
        self.assertIsNotNone(vector_store, "向量库应该加载成功")
        
        # 测试相似性搜索
        query = "测试文档"
        docs = vector_store.similarity_search(query, k=1)
        self.assertGreater(len(docs), 0, "应该找到相似的文档")
    
    def test_update_vector_store(self):
        """测试更新向量库"""
        # 先构建向量库
        vector_store = build_vector_store(self.test_documents, self.vector_store_dir)
        initial_count = vector_store._collection.count()
        
        # 创建新文档用于更新
        new_documents = [
            Document(page_content="这是新增的测试文档3内容。", metadata={"source": "test3"})
        ]
        
        # 更新向量库
        updated_vector_store = update_vector_store(new_documents, self.vector_store_dir)
        updated_count = updated_vector_store._collection.count()
        
        # 验证结果
        self.assertIsNotNone(updated_vector_store, "向量库应该更新成功")
        self.assertEqual(updated_count, initial_count + 1, "向量库中的文档数量应该增加1")


if __name__ == "__main__":
    unittest.main()