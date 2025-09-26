#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试DeepSeek API调用
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# 加载环境变量
load_dotenv()

def test_deepseek_api():
    """测试DeepSeek API调用"""
    print("=== 测试DeepSeek API调用 ===")
    
    try:
        from langchain_openai import ChatOpenAI
        
        # 获取API密钥
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("错误: 未设置DEEPSEEK_API_KEY环境变量")
            return
            
        # 创建LLM实例
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=1000,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        print("LLM实例创建成功")
        
        # 测试简单调用
        print("测试简单调用...")
        response = llm.invoke("你好，请简单介绍一下 yourself")
        print(f"响应: {response.content}")
        
        print("DeepSeek API测试完成")
        
    except Exception as e:
        print(f"DeepSeek API测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deepseek_api()