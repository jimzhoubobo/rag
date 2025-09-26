#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
定时ETL任务
每30分钟执行一次文档加载和处理任务
"""

import schedule
import time
import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from etl.document_processor import load_and_process_documents, extract_content_and_store


def etl_job():
    """
    ETL任务函数
    """
    print(f"[{datetime.now()}] 开始执行ETL任务...")
    
    try:
        # 定义数据路径
        data_path = "./data"
        processed_data_path = "./processed_data"
        
        # 确保数据目录存在
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)
        
        # 加载和处理文档
        documents = load_and_process_documents(data_path)
        
        # 提取内容并存储
        extracted_content = extract_content_and_store(documents, processed_data_path)
        
        print(f"[{datetime.now()}] ETL任务执行完成，共处理 {len(extracted_content)} 个文档片段")
        
    except Exception as e:
        print(f"[{datetime.now()}] ETL任务执行出错: {e}")


def run_scheduler():
    """
    运行定时调度器
    """
    # 每30分钟执行一次
    schedule.every(30).minutes.do(etl_job)
    
    # 每天凌晨5点执行一次
    schedule.every().day.at("05:00").do(etl_job)
    
    # 立即执行一次
    etl_job()
    
    print(f"[{datetime.now()}] 定时任务已启动:")
    print("  - 每30分钟执行一次")
    print("  - 每天凌晨5点执行一次")
    
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    run_scheduler()