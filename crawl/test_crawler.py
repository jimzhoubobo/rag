#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
爬虫测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baidu_baike_crawler import BaiduBaikeCrawler
import random
import time

def main():
    """测试主函数"""
    # 创建爬虫实例
    crawler = BaiduBaikeCrawler()
    
    # 测试爬取单个词条
    test_keywords = ["中医", "针灸", "中药"]
    
    print("开始测试爬虫...")
    for keyword in test_keywords:
        print(f"\n=== 测试爬取词条: {keyword} ===")
        content_data = crawler.get_baike_content(keyword)
        
        if content_data:
            print(f"标题: {content_data['title']}")
            print(f"内容长度: {len(content_data['content'])} 字符")
            print(f"部分内容预览: {content_data['content'][:100]}...")
            
            # 保存内容
            crawler.save_content(content_data, format="json")
            crawler.save_content(content_data, format="txt")
            print("内容已保存")
        else:
            print("未能获取到内容")
        
        # 随机延时
        time.sleep(random.uniform(1, 3))


if __name__ == "__main__":
    main()