#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缓存功能测试脚本
用于测试缓存装饰器的功能，包括缓存命中和缓存清除
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from rag.rag_core import load_vector_store_with_cache
from constant.constants import ProjectConstants


def test_cache_functionality():
    """测试缓存功能"""
    print("=== 测试缓存功能 ===")
    
    # 第一次调用，应该实际加载向量库
    print("1. 首次调用函数:")
    vector_store1 = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
    print(f"   首次调用完成，返回对象ID: {id(vector_store1)}")

    # 等待一小段时间
    time.sleep(1)

    # 第二次调用，应该使用缓存
    print("2. 第二次调用(应该使用缓存):")
    vector_store2 = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
    print(f"   第二次调用完成，返回对象ID: {id(vector_store2)}")
    print(f"   两次调用返回相同对象: {vector_store1 is vector_store2}")

    # 验证缓存确实生效
    assert vector_store1 is vector_store2, "缓存未生效，两次调用返回了不同对象"
    print("   ✓ 缓存功能正常工作")


def test_cache_clear():
    """测试缓存清除功能"""
    print("\n=== 测试缓存清除功能 ===")
    
    # 确保有缓存数据
    vector_store_before_clear = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
    print(f"1. 清除缓存前调用，返回对象ID: {id(vector_store_before_clear)}")

    # 清除缓存
    print("2. 清除缓存:")
    load_vector_store_with_cache.clear_cache()
    print("   缓存已清除")

    # 再次调用，应该重新加载向量库
    print("3. 清除缓存后再次调用(应该重新加载):")
    vector_store_after_clear = load_vector_store_with_cache([], ProjectConstants.get_chroma_db_path())
    print(f"   清除缓存后调用完成，返回对象ID: {id(vector_store_after_clear)}")
    print(f"   清除缓存前后返回不同对象: {vector_store_before_clear is not vector_store_after_clear}")

    # 验证缓存确实被清除了
    assert vector_store_before_clear is not vector_store_after_clear, "缓存清除失败，两次调用返回了相同对象"
    print("   ✓ 缓存清除功能正常工作")


def main():
    """主测试函数"""
    print("缓存功能测试开始")
    print("=" * 50)
    
    try:
        test_cache_functionality()
        test_cache_clear()
        
        print("\n" + "=" * 50)
        print("所有缓存功能测试通过!")
        return True
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)