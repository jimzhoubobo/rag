#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试定时任务调度配置
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

def test_schedule_config():
    """测试定时任务调度配置"""
    print("=== 测试定时任务调度配置 ===")
    
    try:
        import schedule
        print("Schedule库导入成功")
        
        # 测试调度配置
        # 每30分钟执行一次
        job1 = schedule.every(30).minutes.do(lambda: print("每30分钟任务"))
        print("每30分钟调度配置成功")
        
        # 每天凌晨5点执行一次
        job2 = schedule.every().day.at("05:00").do(lambda: print("每天凌晨5点任务"))
        print("每天凌晨5点调度配置成功")
        
        # 显示所有已调度的任务
        print(f"已调度任务数量: {len(schedule.jobs)}")
        for i, job in enumerate(schedule.jobs):
            print(f"  任务 {i+1}: {job}")
            
        # 清理测试任务
        schedule.cancel_job(job1)
        schedule.cancel_job(job2)
        
        print("定时任务调度配置测试通过")
        
    except Exception as e:
        print(f"定时任务调度配置测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_schedule_config()