#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
百度百科经络相关条目爬虫
用于爬取经络相关的百科内容，为RAG系统准备数据
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
import random
import logging

from constant.constants import ProjectConstants

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaiduBaikeCrawler:
    """百度百科爬虫类"""
    
    def __init__(self, output_dir=ProjectConstants.get_data_path()):
        """
        初始化爬虫
        
        Args:
            output_dir (str): 数据保存目录
        """
        self.output_dir = output_dir
        self.session = requests.Session()
        
        # 更丰富的User-Agent列表，随机选择
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59',
        ]
        
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def get_baike_content(self, keyword):
        """
        获取百度百科条目内容
        
        Args:
            keyword (str): 百科词条关键词
            
        Returns:
            dict: 包含标题、内容等信息的字典
        """
        try:
            # 随机选择User-Agent
            self.session.headers.update({
                'User-Agent': random.choice(self.user_agents)
            })
            
            # 构造搜索URL
            search_url = f"https://baike.baidu.com/item/{keyword}"
            logger.info(f"正在爬取百度百科词条: {keyword}")
            
            # 添加随机延时
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(search_url, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                logger.error(f"请求失败，状态码: {response.status_code}")
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 检查是否被反爬虫拦截
            if "百度安全验证" in response.text or "验证码" in response.text:
                logger.warning("遇到百度安全验证，稍后重试")
                time.sleep(random.uniform(5, 10))
                return None
                
            # 提取标题
            title_elem = soup.find('h1')
            title = title_elem.text.strip() if title_elem else keyword
            
            # 提取主要内容
            content = ""
            
            # 查找百科内容的主要区域
            main_content = soup.find('div', class_='lemma-summary')
            if main_content:
                content += main_content.get_text(strip=True)
            
            # 查找详细内容区域
            detail_content = soup.find('div', class_='para')
            if detail_content:
                content += "\n" + detail_content.get_text(strip=True)
            
            # 如果没找到标准结构，尝试其他方式提取
            if not content:
                # 查找所有段落内容
                paragraphs = soup.find_all('div', class_='para')
                for p in paragraphs[:20]:  # 限制段落数量
                    content += p.get_text(strip=True) + "\n"
            
            # 清理内容
            content = self._clean_content(content)
            
            if not content:
                logger.warning(f"未找到有效的百科内容: {keyword}")
                return None
                
            # 构造结果
            result = {
                "title": title,
                "keyword": keyword,
                "url": search_url,
                "content": content,
                "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "百度百科"
            }
            
            logger.info(f"成功爬取词条: {title}")
            return result
            
        except Exception as e:
            logger.error(f"爬取词条 '{keyword}' 时出错: {e}")
            return None
    
    def _clean_content(self, content):
        """
        清理内容，去除无关信息
        
        Args:
            content (str): 原始内容
            
        Returns:
            str: 清理后的内容
        """
        if not content:
            return ""
            
        # 去除多余的空白字符
        content = re.sub(r'\s+', ' ', content)
        
        # 去除编辑相关链接和按钮文本
        content = re.sub(r'编辑.*?锁定', '', content)
        content = re.sub(r'讨论.*?分享', '', content)
        
        # 去除特殊符号和多余内容
        content = re.sub(r'\[.*?\]', '', content)  # 去除引用标记
        content = re.sub(r'词条图册.*', '', content)  # 去除图册相关
        content = re.sub(r'参考资料.*', '', content)  # 去除参考资料
        content = re.sub(r'百度百科.*?内容开放平台', '', content, flags=re.DOTALL)  # 去除平台信息
        
        return content.strip()
    
    def save_content(self, content_data, format="json"):
        """
        保存内容到文件
        
        Args:
            content_data (dict): 内容数据
            format (str): 保存格式，支持json和txt
        """
        if not content_data:
            return
            
        title = content_data.get('title', 'unknown')
        # 清理文件名中的特殊字符
        safe_title = re.sub(r'[^\w\u4e00-\u9fff\-_]', '_', title)[:50]
        
        if format == "json":
            filename = f"{safe_title}.json"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content_data, f, ensure_ascii=False, indent=2)
        elif format == "txt":
            filename = f"{safe_title}.txt"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"标题: {content_data.get('title', '')}\n")
                f.write(f"来源: {content_data.get('source', '')}\n")
                f.write(f"URL: {content_data.get('url', '')}\n")
                f.write(f"爬取时间: {content_data.get('crawl_time', '')}\n")
                f.write("-" * 50 + "\n")
                f.write(content_data.get('content', ''))
        
        logger.info(f"内容已保存到: {filepath}")
    
    def crawl_tcm_keywords(self, keywords=None):
        """
        爬取经络相关关键词
        
        Args:
            keywords (list): 关键词列表，如果为None则使用默认列表
        """
        if keywords is None:
            # 默认的经络相关关键词
            keywords = [
                "经络", "中药", "针灸", "推拿", "拔罐",
                "刮痧", "艾灸", "经络", "穴位", "气血",
                "阴阳", "五行", "脏腑", "脉象", "辨证论治",
                "四诊", "望闻问切", "中药方剂", "中草药", "中药材"
            ]
        
        logger.info(f"开始爬取 {len(keywords)} 个经络相关词条")
        
        success_count = 0
        for i, keyword in enumerate(keywords, 1):
            logger.info(f"进度: {i}/{len(keywords)} - 爬取词条: {keyword}")
            
            # 获取内容
            content_data = self.get_baike_content(keyword)
            
            if content_data:
                # 保存为JSON格式
                self.save_content(content_data, format="json")
                # 保存为TXT格式
                self.save_content(content_data, format="txt")
                success_count += 1
            else:
                logger.warning(f"未能获取词条内容: {keyword}")
            
            # 添加延时，避免请求过于频繁
            time.sleep(random.uniform(2, 5))
        
        logger.info(f"爬取完成，成功获取 {success_count}/{len(keywords)} 个词条")


def main():
    """主函数"""
    # 创建爬虫实例
    crawler = BaiduBaikeCrawler(output_dir=ProjectConstants.get_data_path())
    
    # 爬取经络相关词条
    crawler.crawl_tcm_keywords()


if __name__ == "__main__":
    main()