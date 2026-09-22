"""
网络工具模块 (Web Tools)

【核心职责与定位】：
本模块为 Agent 提供与外部互联网交互的底层能力支撑，包含两项核心功能：
1. 网络搜索 (search_web)：通过 Google 搜索 API (Serper) 检索关键词，并自动深入抓取返回的所有相关网页正文；
2. 网页抓取与解析 (fetch_webpage)：通过 HTTP 请求下载目标网页 HTML，剔除无用标签（脚本、样式、导航、页脚），
   清洗转化为格式规整的纯文本。

【免 Key 运行机制】：
为了让没有申请 Serper API Key 的读者也能零成本复现第二章的上下文压缩实验，
本模块内置了全自动的“模拟数据 (Mock Data) 降级引擎”。在无 API Key 或网络请求报错时，
自动提供内置的真实 Wikipedia 长文本，保障 Agent 调研与压缩逻辑畅通运行。
"""

import json
import html
import re
import logging
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urlparse, urljoin
import time
from config import Config

# 配置模块日志输出格式
logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class WebTools:
    """
    网络搜索与网页正文抓取工具箱
    
    封装了网络请求、HTML 解析清洗、html2text Markdown 转换、
    内存防重复抓取缓存以及离线 Mock 模拟降级等功能。
    """
    
    def __init__(self):
        """
        初始化网络工具箱
        
        初始化并配置：
        1. Serper API 搜索密钥（读取自环境变量配置）；
        2. html2text 转换器参数（保留超链接、忽略图片装饰、禁止自动换行断词等）；
        3. 页面内存缓存字典 page_cache（防止同一次任务中重复下载解析同一 URL）。
        """
        self.serper_api_key = Config.SERPER_API_KEY
        
        # 配置 HTML 到纯文本/Markdown 的转换器
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False        # 保留超链接，便于 Agent 追踪引用来源
        self.html_converter.ignore_images = True       # 忽略图片，节省无关 Token
        self.html_converter.ignore_emphasis = False     # 保留加粗/斜体等强调文本
        self.html_converter.body_width = 0             # 设为 0 表示不限制单行宽度（禁止自动强制换行断字）
        self.html_converter.single_line_break = True   # 单换行模式，避免生成过多空行
        
        # 页面缓存字典（key: url, value: 抓取解析后的结果字典）
        self.page_cache = {}
    
    def search_web(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        根据检索词搜索互联网并抓取返回的网页全文
        
        【执行流水线】：
        1. Key 校验：若未配置 SERPER_API_KEY，自动降级调用 _get_mock_search_results；
        2. 调用 Serper API (POST /search)：获取 Google 检索到的 organic 自然排名结果；
        3. 深入网页正文（Deep Crawl）：
           遍历每条搜索结果中的链接 (link)，调用 self.fetch_webpage(url) 下载并解析网页全文；
        4. 组装复合数据：将标题、URL、搜索短摘要 (snippet) 以及抓取到的网页长文本 (content) 整合打包返回；
        5. 异常保护：网络异常、HTTP 错误或超时时，统一安全降级为 Mock 数据。
        
        参数:
            query: 搜索关键词（如 "OpenAI co-founders current affiliations"）
            num_results: 返回的搜索结果数量（默认 5 篇网页）
            
        返回:
            Dict[str, Any]: 包含 query、num_results、results 列表及时间戳的完整检索数据字典
        """
        try:
            # ---------------------------------------------------------
            # 步骤 1: 检查是否存在 Serper API Key，未配置则降级为 Mock 模式
            # ---------------------------------------------------------
            if not self.serper_api_key:
                logger.warning("No Serper API key, using mock results")
                return self._get_mock_search_results(query)
            
            logger.info(f"Searching web for: {query}")
            
            # ---------------------------------------------------------
            # 步骤 2: 发起 Serper Google 搜索 API 请求
            # ---------------------------------------------------------
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'q': query,
                'num': num_results
            }
            
            response = requests.post(
                f"{Config.SERPER_BASE_URL}/search",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            # 若 API 响应状态码异常，降级为 Mock 数据
            if response.status_code != 200:
                logger.error(f"Serper API error: {response.status_code}")
                return self._get_mock_search_results(query)
            
            data = response.json()
            
            # ---------------------------------------------------------
            # 步骤 3: 遍历搜索结果，逐一抓取并解析网页正文
            # ---------------------------------------------------------
            results = []
            organic_results = data.get('organic', [])[:num_results]
            
            for result in organic_results:
                url = result.get('link', '')
                if url:
                    # 调用 fetch_webpage 抓取清洗该网页全文
                    page_content = self.fetch_webpage(url)
                    
                    results.append({
                        'title': result.get('title', ''),
                        'url': url,
                        'snippet': result.get('snippet', ''),
                        'content': page_content.get('content', ''),
                        'content_length': len(page_content.get('content') or ''),
                        'fetch_success': page_content.get('success', False)
                    })
                    
                    # 请求间隔微延时 (0.5秒)，避免对目标站点造成过大爬虫压力
                    time.sleep(0.5)
            
            return {
                'query': query,
                'num_results': len(results),
                'results': results,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}")
            # 任意环节网络异常时，兜底返回内置的 Mock 搜索结果
            return self._get_mock_search_results(query)
    
    def fetch_webpage(self, url: str) -> Dict[str, Any]:
        """
        抓取指定 URL 网页正文并转换为清洗后的纯文本
        
        【清洗流水线与去噪设计】：
        1. 内存缓存检查：若该 URL 已经抓取过，直接从 self.page_cache 命中返回，避免重复耗时；
        2. 网络请求伪装：携带主流浏览器的 User-Agent 请求头，降低被目标站点反爬拦截的概率；
        3. DOM 树强力去噪 (DOM Decompose)：
           使用 BeautifulSoup 解析后，直接剔除五类噪音标签：
           - <script>: 避免将 JavaScript 代码塞进上下文；
           - <style>: 避免将 CSS 样式表代码塞进上下文；
           - <nav>: 避免将网站顶部的菜单栏链接当作文章内容；
           - <header>: 避免包含通用横幅标语；
           - <footer>: 避免将底部的版权声明、备案号等法律冗余信息记入；
        4. 文本排版转化：通过 html2text 将剩余的核心 DOM 结构转换为清晰的 Markdown/纯文本；
        5. 长度熔断截断：若清洗后的正文超过 Config.MAX_WEBPAGE_LENGTH，主动截断并打上 [Content truncated...] 标记；
        6. 错误缓存保护：即使某次请求因 404 或超时失败，也将其记录在缓存中，避免后续重复无意义重试。
        
        参数:
            url: 目标网页的完整 URL 地址
            
        返回:
            Dict[str, Any]: 包含 url、title、content、content_length、success、error 等字段的字典
        """
        try:
            # ---------------------------------------------------------
            # 步骤 1: 内存缓存命中检测
            # ---------------------------------------------------------
            if url in self.page_cache:
                logger.info(f"Using cached content for: {url}")
                return self.page_cache[url]
            
            logger.info(f"Fetching webpage: {url}")
            
            # ---------------------------------------------------------
            # 步骤 2: 发起真实 HTTP GET 请求（模拟浏览器标头）
            # ---------------------------------------------------------
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # ---------------------------------------------------------
            # 步骤 3: 使用 BeautifulSoup 解析 HTML 并彻底剔除噪音元素
            # ---------------------------------------------------------
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 剔除无用的脚本、样式表、导航栏、页眉与页脚
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # ---------------------------------------------------------
            # 步骤 4: 转换 HTML 为纯文本并过滤空白行
            # ---------------------------------------------------------
            text_content = self.html_converter.handle(str(soup))
            
            lines = text_content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # 过滤纯空行以及可能残存的导航标记
                if line and not line.startswith('#'):
                    cleaned_lines.append(line)
            
            cleaned_text = '\n'.join(cleaned_lines)
            
            # ---------------------------------------------------------
            # 步骤 5: 单页字符数上限截断保护
            # ---------------------------------------------------------
            if len(cleaned_text) > Config.MAX_WEBPAGE_LENGTH:
                cleaned_text = cleaned_text[:Config.MAX_WEBPAGE_LENGTH] + "\n\n[Content truncated...]"
            
            # 提取网页标题 Title
            title = 'No title'
            if soup.title:
                raw_title = soup.title.get_text()
                # 还原 HTML 转义字符并剥离尖括号标签
                cleaned_title = html.unescape(re.sub(r'<[^>]+>', '', raw_title)).strip()
                if cleaned_title:
                    title = cleaned_title

            result = {
                'url': url,
                'title': title,
                'content': cleaned_text,
                'content_length': len(cleaned_text),
                'success': True,
                'timestamp': time.time()
            }
            
            # 存入运行时缓存
            self.page_cache[url] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching webpage {url}: {str(e)}")
            
            error_result = {
                'url': url,
                'title': 'Error',
                'content': f"Failed to fetch webpage: {str(e)}",
                'content_length': 0,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
            
            # 即使抓取失败也缓存该错误结果，避免同一 URL 反复超时重试拖垮主流程
            self.page_cache[url] = error_result
            
            return error_result
    
    def _get_mock_search_results(self, query: str) -> Dict[str, Any]:
        """
        获取离线模拟检索数据 (Mock Search Engine)
        
        【设计目的与原理】：
        当用户未在 .env 中配置 SERPER_API_KEY，或外部网络搜索因防火墙/配额耗尽报错时，
        本函数充当内置的“离线微型维基百科库”。
        
        【内置数据集】：
        真实收录了维基百科及权威科技媒体关于 OpenAI 联合创始人的长篇事实档案：
        1. "openai": OpenAI 成立背景、全体 11 位早期创始人清单及其截至 2024 年的现状；
        2. "sam altman": Sam Altman 出任 CEO、2023 年 11 月短暂风波后重返及天使投资背景；
        3. "elon musk": 马斯克 2018 年退出 OpenAI 董事会及 2023 年创立 xAI 的完整记录；
        4. "ilya sutskever": 前首席科学家 Ilya Sutskever 2024 年 5 月离职创立 SSI (Safe Superintelligence) 的记录。
        
        【匹配机制】：
        将检索词 query 转为小写，进行关键词子串匹配。若命中则返回对应的模拟网页与正文；
        若均未命中则返回通用的 Mock 数据。
        
        参数:
            query: 搜索词
            
        返回:
            Dict[str, Any]: 结构与真实搜索 API 完全一致的结果字典 (带 'mock': True 标记)
        """
        # 内置预置的高仿真百科与新闻正文数据
        mock_data = {
            "openai": [
                {
                    'title': 'OpenAI - Wikipedia',
                    'url': 'https://en.wikipedia.org/wiki/OpenAI',
                    'snippet': 'OpenAI was founded in 2015 by Sam Altman, Elon Musk, Ilya Sutskever, Greg Brockman, Wojciech Zaremba, and John Schulman...',
                    'content': '''OpenAI was founded in December 2015 by Sam Altman, Elon Musk, Ilya Sutskever, Greg Brockman, Wojciech Zaremba, and John Schulman.
                    
The organization was founded with the goal of advancing digital intelligence in a way that benefits humanity. 

Current Status of Co-founders (as of 2024):
- Sam Altman: CEO of OpenAI (returned after brief departure in November 2023)
- Elon Musk: Left OpenAI board in 2018, founded xAI in 2023
- Ilya Sutskever: Former Chief Scientist, left OpenAI in May 2024, co-founded Safe Superintelligence Inc.
- Greg Brockman: President and Chairman of OpenAI
- Wojciech Zaremba: Head of Language and Code Generation at OpenAI
- John Schulman: Co-founder, left OpenAI in August 2024 to join Anthropic

Additional early members:
- Andrej Karpathy: Former Director of AI at Tesla, briefly returned to OpenAI, now independent
- Dario Amodei: Left to co-found Anthropic in 2021
- Daniela Amodei: Left to co-found Anthropic in 2021'''
                }
            ],
            "sam altman": [
                {
                    'title': 'Sam Altman - CEO of OpenAI',
                    'url': 'https://example.com/sam-altman',
                    'snippet': 'Sam Altman is the CEO of OpenAI...',
                    'content': 'Sam Altman is currently the CEO of OpenAI. He briefly left the company in November 2023 but returned after employee protests. He is also known for his work at Y Combinator and various investments in startups.'
                }
            ],
            "elon musk": [
                {
                    'title': 'Elon Musk launches xAI',
                    'url': 'https://example.com/elon-musk-ai',
                    'snippet': 'Elon Musk founded xAI in 2023...',
                    'content': 'Elon Musk, who co-founded OpenAI in 2015, left the board in 2018 citing conflicts of interest with Tesla\'s AI development. In 2023, he founded xAI, a new AI company focused on understanding the universe. He is also CEO of Tesla, SpaceX, and owner of X (formerly Twitter).'
                }
            ],
            "ilya sutskever": [
                {
                    'title': 'Ilya Sutskever launches Safe Superintelligence',
                    'url': 'https://example.com/ilya-sutskever',
                    'snippet': 'Ilya Sutskever left OpenAI to start SSI...',
                    'content': 'Ilya Sutskever, former Chief Scientist at OpenAI, left the company in May 2024 after nearly a decade. He co-founded Safe Superintelligence Inc. (SSI) with Daniel Gross and Daniel Levy, focusing on building safe AGI.'
                }
            ]
        }
        
        # 将输入查询转换为小写后进行前缀/子串匹配
        query_lower = query.lower()
        for key in mock_data:
            if key in query_lower:
                results = []
                for item in mock_data[key]:
                    results.append({
                        'title': item['title'],
                        'url': item['url'],
                        'snippet': item['snippet'],
                        'content': item['content'],
                        'content_length': len(item['content']),
                        'fetch_success': True
                    })
                
                return {
                    'query': query,
                    'num_results': len(results),
                    'results': results,
                    'timestamp': time.time(),
                    'mock': True
                }
        
        # 若未命中任何预设关键词，返回通用兜底模拟数据
        return {
            'query': query,
            'num_results': 1,
            'results': [{
                'title': 'Mock Search Result',
                'url': 'https://example.com',
                'snippet': 'This is a mock search result for testing',
                'content': 'Mock content for testing when no API key is available.',
                'content_length': 50,
                'fetch_success': True
            }],
            'timestamp': time.time(),
            'mock': True
        }
    
    def clear_cache(self):
        """
        清空网页内存缓存
        
        【调用场景】：
        在运行策略对比实验（ExperimentRunner）时，每当切换到一个全新的压缩策略，
        必须调用本方法重置缓存，确保各策略在公平无残留的历史环境下独立运行评测。
        """
        self.page_cache.clear()
        logger.info("Page cache cleared")
