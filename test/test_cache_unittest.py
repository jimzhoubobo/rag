#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缓存功能单元测试
使用unittest框架测试缓存装饰器的功能
"""

import unittest
import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from rag.rag_core import load_vector_store_with_cache
from constant.constants import ProjectConstants


class TestCacheFunctionality(unittest.TestCase):
    """缓存功能测试类"""

    def test_cache_hit(self):
        """测试缓存命中功能"""
        # 第一次调用，应该实际加载向量库
        vector_store1 = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
        
        # 第二次调用，应该使用缓存
        vector_store2 = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
        
        # 验证两次调用返回相同对象
        self.assertIs(vector_store1, vector_store2, "缓存未生效，两次调用返回了不同对象")

    def test_cache_clear(self):
        """测试缓存清除功能"""
        # 确保有缓存数据
        vector_store_before_clear = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())

        # 清除缓存
        load_vector_store_with_cache.clear_cache()

        # 再次调用，应该重新加载向量库
        vector_store_after_clear = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())

        # 验证缓存确实被清除了
        self.assertIsNot(vector_store_before_clear, vector_store_after_clear, "缓存清除失败，两次调用返回了相同对象")

    def test_clear_cache_method_exists(self):
        """测试clear_cache方法是否存在"""
        # 检查函数是否有clear_cache属性
        self.assertTrue(hasattr(load_vector_store_with_cache, 'clear_cache'), "函数缺少clear_cache方法")
        
        # 检查clear_cache是否可调用
        self.assertTrue(callable(load_vector_store_with_cache.clear_cache), "clear_cache不是可调用的方法")


if __name__ == "__main__":
    # 运行测试
    unittest.main()