#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缓存装饰器模块
提供缓存功能，支持过期时间和自动更新
"""

import time
import functools
from log.logger import logger


def ttl_cache(expire_time=600):
    """
    TTL缓存装饰器
    
    Args:
        expire_time (int): 缓存过期时间（秒），默认10分钟(600秒)
        
    Returns:
        function: 装饰器函数
    """
    def decorator(func):
        # 缓存存储
        cache_data = {
            'value': None,
            'timestamp': 0,
            'lock': False  # 防止并发重复加载
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # 检查缓存是否有效
            if (cache_data['value'] is not None and 
                (current_time - cache_data['timestamp']) < expire_time and 
                not cache_data['lock']):
                logger.debug(f"使用 {func.__name__} 的缓存结果")
                return cache_data['value']
            
            # 防止并发重复加载
            if cache_data['lock']:
                # 等待其他线程加载完成
                while cache_data['lock']:
                    time.sleep(0.1)
                # 再次检查缓存
                if (cache_data['value'] is not None and 
                    (time.time() - cache_data['timestamp']) < expire_time):
                    return cache_data['value']
            
            # 加锁并加载新数据
            cache_data['lock'] = True
            try:
                logger.info(f"缓存过期或不存在，调用 {func.__name__} 重新加载数据")
                result = func(*args, **kwargs)
                cache_data['value'] = result
                cache_data['timestamp'] = current_time
                logger.info(f"{func.__name__} 数据已缓存，将在 {expire_time} 秒后过期")
                return result
            except Exception as e:
                logger.error(f"执行 {func.__name__} 时出错: {e}")
                # 如果有缓存数据，即使过期也返回
                if cache_data['value'] is not None:
                    logger.info(f"返回过期的缓存数据以保证服务可用性")
                    return cache_data['value']
                # 没有缓存数据且执行出错，抛出异常
                raise
            finally:
                # 解锁
                cache_data['lock'] = False
        
        # 添加清除缓存的方法
        def clear_cache():
            cache_data['value'] = None
            cache_data['timestamp'] = 0
            logger.info(f"{func.__name__} 的缓存已被清除")
        
        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator