"""
上下文压缩策略模块 (Context Compression Strategies)

【模块核心职责】：
本模块实现了针对大语言模型智能体 (LLM Agent) 外部工具调用结果的多种上下文压缩算法与策略。
在多步调研或长程推理任务中，外部工具（如网页搜索、网页抓取）返回的内容往往包含数万甚至数十万字符的冗长网页文本。
若直接将原始文本填入对话历史，会迅速耗尽上下文窗口（Context Window）或产生昂贵的 Token 费用。
本模块提供了从“完全不压缩”到“意图感知”、“带引用压缩”和“滑动窗口回溯压缩”等一系列对比策略，
用于评估不同压缩机制对 Token 开销、信息保留度与 Agent 任务完成率的影响。
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
import tiktoken
from config import Config


def _reasoning_safe_temperature(model, requested=1.0):
    """
    推理模型（Reasoning Models）温度参数安全适配函数
    
    【背景说明】：
    部分新型深度推理模型（如 Kimi K3、OpenAI o1/GPT-5、部分思考模型）具有内置的思维链 (Chain of Thought)。
    这类模型通常固定仅支持 temperature=1.0（若传入 0.0 或 0.3 会抛出 HTTP 400 校验异常）。
    本函数在检测到此类推理模型时强制返回 1.0，而在常规非推理模型（如 Gemini、DeepSeek、Doubao 等）
    上则保留用户指定的温度值（如用于摘要任务的低温 0.3，以保证确定性和事实准确度）。
    """
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested


def _reasoning_safe_max_tokens(model, requested, reasoning_budget=2048):
    """
    推理模型输出 Token 上限安全适配函数
    
    【背景说明】：
    具备内部思考过程（reasoning_content）的模型在输出最终可见答案前，会先在内部消耗一部分 max_tokens 预算
    来记录其思考链。如果给摘要任务分配的 max_tokens 太小（例如仅设 300~500 tokens），模型的内部思考就会把
    预算完全耗尽，导致最终返回给用户的正文被截断甚至变为空文本（None/Empty）。
    因此，对于此类模型，额外补充 reasoning_budget（默认 2048 tokens）的思考缓冲区，确保摘要正文有足够的生成空间。
    """
    m = str(model or "").lower().replace("/", "-")
    if "kimi-k3" in m or "gpt-5" in m:
        return requested + reasoning_budget
    return requested

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """
    上下文压缩策略枚举类
    
    定义了实验中可对比的 6 种典型上下文管理方案：
    1. NO_COMPRESSION: 原始保留策略（基准组，不进行任何压缩，原样拼接网页全文）；
    2. NON_CONTEXT_AWARE_INDIVIDUAL: 无上下文感知单页摘要（对抓取的每个网页分别独立调用一次 LLM 生成摘要后拼接）；
    3. NON_CONTEXT_AWARE_COMBINED: 无上下文感知合并摘要（将所有网页先拼成一篇长文本，仅调用 1 次 LLM 生成综合摘要）；
    4. CONTEXT_AWARE: 意图上下文感知摘要（将当前检索 Query 及多轮任务意图作为上下文传给 LLM，针对性提取事实）；
    5. CONTEXT_AWARE_CITATIONS: 意图感知带来源引用（在意图感知的基础上，保留 [1], [2] 来源编号并附带 URL 引用列表）；
    6. WINDOWED_CONTEXT: 滑动窗口延迟压缩（平时不压缩，保留 100% 原始全文，直到 Prompt 触及 80% 警戒线时触发回溯压缩）。
    """
    NO_COMPRESSION = "no_compression"
    NON_CONTEXT_AWARE_INDIVIDUAL = "non_context_aware_individual_summary"  # 逐页独立摘要后拼接
    NON_CONTEXT_AWARE_COMBINED = "non_context_aware_combined_summary"     # 整体合并后单次摘要
    CONTEXT_AWARE = "context_aware_summary"                               # 结合当前 Query 的意图感知摘要
    CONTEXT_AWARE_CITATIONS = "context_aware_with_citations"             # 意图感知且保留来源引用链接
    WINDOWED_CONTEXT = "windowed_context"                                 # 滑动窗口阈值延迟压缩


@dataclass
class CompressedContent:
    """
    压缩结果数据容器 (Data Class)
    
    记录每次压缩操作前后的体积变化、压缩产物文本以及元数据，便于量化评估压缩比与溯源信息。
    
    属性:
        original_length: 压缩前原始文本的字符数
        compressed_length: 压缩后生成的精炼文本字符数
        content: 压缩后的最终正文字符串（将被写入 Agent 对话上下文）
        citations: 抽取的引用源信息列表（如 [{'id': '[1]', 'title': '...', 'url': '...'}]）
        strategy: 产生该压缩结果所采用的策略枚举
        timestamp: 压缩完成时的 ISO 格式时间戳
    """
    original_length: int
    compressed_length: int
    content: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    strategy: CompressionStrategy = CompressionStrategy.NO_COMPRESSION
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ContextCompressor:
    """
    上下文压缩器核心引擎
    
    负责调度和执行各种上下文压缩策略，封装了 LLM Client 调用、流式输出、Token 估算及容错回退机制。
    """
    
    def __init__(self, strategy: CompressionStrategy, api_key: str, enable_streaming: bool = True):
        """
        初始化上下文压缩器
        
        参数:
            strategy: 选定的压缩策略 (CompressionStrategy 枚举)
            api_key: 大模型 API Key
            enable_streaming: 是否开启流式打印（为 True 时可以在终端实时看到生成摘要的打字效果）
        """
        self.strategy = strategy
        self.enable_streaming = enable_streaming
        
        # 统一解析模型配置：通过 Config.resolve_llm() 解析 API Key、Base URL 及具体模型名称
        resolved_key, resolved_base_url, resolved_model = Config.resolve_llm()
        
        # 创建标准 OpenAI SDK 客户端（显式设置超时时间为 60 秒，避免经本地网络代理访问时出现超时中断）
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
            timeout=60.0
        )
        self.model = resolved_model
        
        # 初始化 Token 分词计数器（默认使用 GPT-4 的 cl100k_base 编码表）
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Context compressor initialized with strategy: {strategy.value}, streaming: {enable_streaming}")
    
    def count_tokens(self, text: str) -> int:
        """
        精确计算给定文本的 Token 数量
        
        优先使用 tiktoken 进行分词计算；若分词库不可用或出现异常，采用经典的快速经验公式（1 token ≈ 4 字符）兜底。
        
        参数:
            text: 待计算的文本字符串
            
        返回:
            int: 估算/计算出的 Token 总数
        """
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # 容错降级：按平均每个 Token 约等于 4 个字符进行换算
            return len(text) // 4
    
    def compress_search_results(
        self, 
        search_results: Dict[str, Any],
        query: str,
        current_context: Optional[str] = None
    ) -> CompressedContent:
        """
        压缩工具检索结果的总路由器方法 (Strategy Router)
        
        根据当前实例绑定的压缩策略，将原始搜索结果路由分发到对应的私有实现方法中进行处理。
        
        参数:
            search_results: 来自 WebTools 的原始搜索结果字典（包含 organic 网页标题、URL 及全文）
            query: 触发本次搜索的原始查询关键词（如 'OpenAI co-founders'）
            current_context: 当前对话历史或近期工具调用意图摘要（供意图感知策略参考）
            
        返回:
            CompressedContent: 封装了压缩后文本、字符数对比及策略类型的结构体对象
        """
        # 1. 策略：不压缩
        if self.strategy == CompressionStrategy.NO_COMPRESSION:
            return self._no_compression(search_results)
            
        # 2. 策略：无上下文感知的单页独立摘要
        elif self.strategy == CompressionStrategy.NON_CONTEXT_AWARE_INDIVIDUAL:
            return self._non_context_aware_individual_summary(search_results)
            
        # 3. 策略：无上下文感知的合并后整体摘要
        elif self.strategy == CompressionStrategy.NON_CONTEXT_AWARE_COMBINED:
            return self._non_context_aware_combined_summary(search_results)
            
        # 4. 策略：结合 Query 与近期意图的上下文感知摘要
        elif self.strategy == CompressionStrategy.CONTEXT_AWARE:
            return self._context_aware_summary(search_results, query, current_context)
            
        # 5. 策略：意图感知且显式保留 [1], [2] 来源引用与链接
        elif self.strategy == CompressionStrategy.CONTEXT_AWARE_CITATIONS:
            return self._context_aware_with_citations(search_results, query, current_context)
            
        # 6. 策略：滑动窗口延迟压缩（搜索阶段暂不压缩，保留原始全文，等到接近 80% 窗口时由 agent.py 触发）
        elif self.strategy == CompressionStrategy.WINDOWED_CONTEXT:
            return self._no_compression(search_results)
            
        else:
            raise ValueError(f"Unknown compression strategy: {self.strategy}")
    
    def compress_for_history(
        self,
        content: str,
        tool_name: str,
        query: str,
        preserve_citations: bool = True
    ) -> CompressedContent:
        """
        对已存入历史消息的单条工具执行结果进行针对性回溯压缩
        
        【适用场景】：
        专门用于滑动窗口策略 (WINDOWED_CONTEXT)。当 Agent 多轮运行后检测到 Prompt Token 占用超过 80% 警戒线时，
        agent.py 会遍历历史中的所有旧 tool 消息，调用本方法将原本冗长的工具返回结果压缩为精简要点。
        
        【处理逻辑】：
        1. 截取前 10,000 字符送入大模型，避免单次历史压缩请求本身超出上下文限制；
        2. 携带原始工具名和触发该工具的 query（意图导向），要求大模型提取与该 query 紧密相关的核心事实、人名、任职与日期；
        3. 根据 preserve_citations 参数决定是否要求附带 [Source: URL] 来源标记；
        4. 支持流式打印压缩进度，若 API 请求异常则自动降级为字符硬截断（前 2000 字符）。
        
        参数:
            content: 历史 tool 消息中存放的原始网页/搜索文本
            tool_name: 产生该内容的工具名称 (如 'search_web')
            query: 当时触发该工具调用的查询词 (用于指导大模型保留哪些相关事实)
            preserve_citations: 是否保留事实来源与引用 URL
            
        返回:
            CompressedContent: 压缩后的精炼内容对象
        """
        original_length = len(content)
        
        try:
            # 构建精炼历史的提示词
            prompt = f"""Compress the following {tool_name} results into a concise summary that preserves key information.
Focus on information relevant to: {query}

Original content:
{content[:10000]}

Requirements:
1. Keep all important facts, names, dates, and affiliations
2. Remove redundant information
3. Maintain clarity and coherence
{"4. Include [Source: URL] citations for important facts" if preserve_citations else ""}
5. Maximum length: {Config.SUMMARY_MAX_TOKENS} tokens

Provide a focused summary:"""

            # 记录 Prompt 长度与 Token 数量
            prompt_tokens = self.count_tokens(prompt)
            logger.info(f"Simple summary request - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} chars")

            # 分支 A：流式生成并实时输出到终端
            if self.enable_streaming:
                print(f"\n📝 Creating simple summary...\n", flush=True)
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=_reasoning_safe_temperature(self.model, 0.3),
                    max_tokens=_reasoning_safe_max_tokens(self.model, Config.SUMMARY_MAX_TOKENS),
                    stream=True
                )
                
                summary_parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        # 注意：此处变量名使用 delta_text 而非 content，避免遮盖入参 content，
                        # 确保即使流中途失败，后方的降级截断逻辑仍能正确访问原始输入 content
                        delta_text = chunk.choices[0].delta.content
                        print(delta_text, end="", flush=True)
                        summary_parts.append(delta_text)
                print("\n")
                compressed = "".join(summary_parts)
            # 分支 B：非流式同步调用
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=_reasoning_safe_temperature(self.model, 0.3),
                    max_tokens=_reasoning_safe_max_tokens(self.model, Config.SUMMARY_MAX_TOKENS)
                )
                compressed = response.choices[0].message.content or ""
            
            return CompressedContent(
                original_length=original_length,
                compressed_length=len(compressed),
                content=compressed,
                strategy=CompressionStrategy.WINDOWED_CONTEXT
            )
            
        except Exception as e:
            logger.error(f"Error compressing for history: {str(e)}")
            # 容错降级方案：大模型调用失败时直接硬截断前 2000 字符并附带截断提示，防止阻塞主流程
            truncated = content[:2000] + "\n\n[Content truncated for history...]"
            return CompressedContent(
                original_length=original_length,
                compressed_length=len(truncated),
                content=truncated,
                strategy=CompressionStrategy.WINDOWED_CONTEXT
            )
    
    def _no_compression(self, search_results: Dict[str, Any]) -> CompressedContent:
        """
        策略 1：不压缩策略 (No Compression) - 基准对比组
        
        【实现机制】：
        完全不进行任何过滤、截断或大模型摘要。
        遍历搜索出来的所有网页，原封不动地将标题 (Title)、网址 (URL)、摘要 (Snippet)
        以及爬取的全文 (Full Content) 按照模板拼接成格式化字符串直接返回。
        
        【特征与意义】：
        作为整个上下文压缩实验的基准参照组（Baseline）。
        优点是保持 100% 信息保真度，不产生压缩阶段额外的 LLM 调用开销；
        缺点是单次搜索结果往往消耗几千到上万 Token，在长程复杂任务中极易发生上下文溢出（Context Overflow）。
        """
        all_content = []
        total_length = 0
        
        for result in search_results.get('results', []):
            content = f"""
===== Search Result =====
Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Snippet: {result.get('snippet', 'N/A')}

Full Content:
{result.get('content', 'No content available')}
========================
"""
            all_content.append(content)
            total_length += len(result.get('content') or '')
        
        full_content = "\n\n".join(all_content)
        
        return CompressedContent(
            original_length=total_length,
            compressed_length=len(full_content),
            content=full_content,
            strategy=CompressionStrategy.NO_COMPRESSION
        )
    
    def _non_context_aware_individual_summary(self, search_results: Dict[str, Any]) -> CompressedContent:
        """
        策略 2A：无上下文感知的单页独立摘要 (Individual Summary)
        
        【设计思想与执行流程】：
        1. 遍历本次搜索返回的每一个网页结果；
        2. 对每一个网页（截取前 5,000 字符），分别独立发起一次大模型摘要请求：
           - 提示词只要求对该网页进行 2~3 段的通用内容归纳；
           - 此时大模型并不知晓 Agent 当前正在调查什么具体问题（即“无上下文感知”）；
           - 生成的单页摘要限制在 300 Token 以内；
        3. 将所有网页生成的独立摘要按 Source、URL、Summary 格式用换行拼接在一起。
        
        【优劣势分析】：
        - 优点：每篇网页独立处理，不会相互混淆干扰，且便于并行化；
        - 缺点：
          a. API 调用频次高：若一次返回 5 个网页，就需要额外消耗 5 次独立的大模型 HTTP 调用，产生较高的网络延迟与调用费用；
          b. 缺乏针对性：由于模型不知道具体的检索目标，摘要中可能保留了大量无关的主题背景，而漏掉了 Agent 真正想找的细节。
        """
        summaries = []
        total_original = 0
        
        for result in search_results.get('results', []):
            if not result.get('content'):
                continue
                
            original_content = result.get('content', '')
            total_original += len(original_content)
            
            try:
                # 构建针对单个页面的通用摘要提示词（截取前 5000 字符）
                prompt = f"""Summarize the following webpage content in 2-3 paragraphs:

Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}

Content:
{original_content[:5000]}

Provide a concise summary:"""

                # 记录单次摘要请求的 Prompt 消耗
                prompt_tokens = self.count_tokens(prompt)
                logger.info(f"Non-context-aware summary - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} chars")

                # 支持流式打印单篇摘要生成过程
                if self.enable_streaming:
                    print(f"\n📝 Summarizing: {result.get('title', 'N/A')[:50]}...", end=" ", flush=True)
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=_reasoning_safe_temperature(self.model, 0.3),
                        max_tokens=_reasoning_safe_max_tokens(self.model, 300),
                        stream=True
                    )
                    
                    summary_parts = []
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            print(content, end="", flush=True)
                            summary_parts.append(content)
                    print()
                    summary = "".join(summary_parts)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=_reasoning_safe_temperature(self.model, 0.3),
                        max_tokens=_reasoning_safe_max_tokens(self.model, 300)
                    )
                    summary = response.choices[0].message.content or ""
                
                # 记录该页摘要
                summaries.append(f"""
Source: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Summary: {summary}
""")
                
            except Exception as e:
                logger.error(f"Error summarizing page: {str(e)}")
                # 单页摘要失败时降级：使用搜索引擎原本返回的短摘要 snippet 兜底
                summaries.append(f"""
Source: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Summary: {result.get('snippet', 'No summary available')}
""")
        
        compressed_content = "\n".join(summaries)
        
        return CompressedContent(
            original_length=total_original,
            compressed_length=len(compressed_content),
            content=compressed_content,
            strategy=CompressionStrategy.NON_CONTEXT_AWARE_INDIVIDUAL
        )
    
    def _non_context_aware_combined_summary(self, search_results: Dict[str, Any]) -> CompressedContent:
        """
        策略 2B：无上下文感知的合并后整体摘要 (Combined Summary)
        
        【设计思想与执行流程】：
        1. 针对策略 2A 中“调用次数过多、网络开销大”的痛点进行优化；
        2. 先将本次搜索返回的所有网页正文（每页截取前 5,000 字符）以模板形式拼接合并为单一的长文本；
        3. **仅发起 1 次大模型调用**，让大模型阅读所有网页内容并生成一份统一的综合摘要；
        4. 限制生成摘要最大 Token 上限为 Config.SUMMARY_MAX_TOKENS。
        
        【对比结论】：
        - 相比策略 2A，API 调用次数从 N 次骤降为 1 次，整体耗时显著降低；
        - 但由于仍然不具备上下文意图感知，大模型仍是在做泛化的全篇总结，无法聚焦于特定细分问题。
        """
        # 1. 汇总所有网页正文
        all_content = []
        total_original = 0
        max_chars_per_page = 5000  # 限制每个页面最大截取字符数，防止单页过大导致发送给摘要模型的请求超限
        
        for result in search_results.get('results', []):
            if result.get('content'):
                original_content = result.get('content', '')
                total_original += len(original_content)
                
                limited_content = original_content[:max_chars_per_page]
                
                all_content.append(f"""
===== Page: {result.get('title', 'N/A')} =====
URL: {result.get('url', 'N/A')}
Content: {limited_content}
""")
        
        # 若没有获取到任何网页内容，返回空结果
        if not all_content:
            return CompressedContent(
                original_length=0,
                compressed_length=0,
                content="No content available",
                strategy=CompressionStrategy.NON_CONTEXT_AWARE_COMBINED
            )
        
        combined_content = "\n\n".join(all_content)
        
        try:
            # 2. 构建合并摘要提示词
            prompt = f"""Summarize the following combined webpage content comprehensively:

{combined_content}

Requirements:
1. Create a comprehensive summary covering all pages
2. Include key information from each source
3. Maintain factual accuracy
4. Maximum length: {Config.SUMMARY_MAX_TOKENS} tokens

Provide a comprehensive summary:"""

            prompt_tokens = self.count_tokens(prompt)
            logger.info(f"Non-context-aware combined summary - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} chars")

            # 3. 仅发起单次大模型调用生成综合摘要
            if self.enable_streaming:
                print(f"\n📄 Creating combined summary for all {len(search_results.get('results', []))} pages...\n", flush=True)
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates comprehensive summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=_reasoning_safe_temperature(self.model, 0.3),
                    max_tokens=_reasoning_safe_max_tokens(self.model, Config.SUMMARY_MAX_TOKENS),
                    stream=True
                )
                
                summary_parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        summary_parts.append(content)
                    print("\n")
                summary = "".join(summary_parts)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates comprehensive summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=_reasoning_safe_temperature(self.model, 0.3),
                    max_tokens=_reasoning_safe_max_tokens(self.model, Config.SUMMARY_MAX_TOKENS)
                )
                summary = response.choices[0].message.content or ""
            
            return CompressedContent(
                original_length=total_original,
                compressed_length=len(summary),
                content=summary,
                strategy=CompressionStrategy.NON_CONTEXT_AWARE_COMBINED
            )
            
        except Exception as e:
            logger.error(f"Error creating combined summary: {str(e)}")
            # 容错降级：直接将所有搜索结果自带的短 snippet 拼接返回
            fallback = "\n\n".join([
                f"{r.get('title', 'N/A')}: {r.get('snippet', 'No summary available')}"
                for r in search_results.get('results', [])
            ])
            return CompressedContent(
                original_length=total_original,
                compressed_length=len(fallback),
                content=fallback,
                strategy=CompressionStrategy.NON_CONTEXT_AWARE_COMBINED
            )
    
    def _context_aware_summary(
        self, 
        search_results: Dict[str, Any],
        query: str,
        current_context: Optional[str] = None
    ) -> CompressedContent:
        """
        策略 3：意图上下文感知摘要 (Context-Aware Summarization)
        
        【核心设计思想与突破】：
        传统摘要（如策略 2A、2B）最大的弊端在于“盲目总结”——大模型不知道用户/Agent 当前在找什么，
        往往花很多字数去总结网站的历史背景、免责声明或无关介绍，反而漏掉了最核心的事实。
        
        【本策略的具体实现】：
        1. 在 Prompt 中显式注入当前搜索的关键词 query（如 'Ilya Sutskever current affiliation'）；
        2. 若存在多轮上下文 current_context（如最近 3 次的搜索意图），一并作为背景输入；
        3. 强化 Prompt 约束规则：
           - 严格只保留与该 query 紧密相关的核心信息；
           - 优先提取最新/当前的状态、具体人名、日期、组织任职机构；
           - 最大输出长度限制在 Config.SUMMARY_MAX_TOKENS 以内。
        
        【效果】：
        在大幅降低 Token 占用（通常减少 80%~90%）的同时，最大化保留了解答目标问题所需的高价值信息。
        """
        # 1. 汇总所有网页正文，每页截取前 5000 字符防止请求模型过载
        all_content = []
        total_original = 0
        max_chars_per_page = 5000
        
        for result in search_results.get('results', []):
            if result.get('content'):
                original_content = result.get('content', '')
                total_original += len(original_content)
                
                limited_content = original_content[:max_chars_per_page]
                
                all_content.append(f"""
Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Content: {limited_content}
""")
        
        combined_content = "\n\n".join(all_content)
        
        try:
            # 2. 注入检索词 Query 与前序上下文，构建意图定向摘要提示词
            prompt = f"""Given the search query: "{query}"
{f"Current context: {current_context[:1000]}" if current_context else ""}

Analyze the following search results and provide a focused summary that directly addresses the query.
Focus on extracting information most relevant to answering: {query}

Search Results:
{combined_content}

Requirements:
1. Focus only on information relevant to the query
2. Prioritize current/recent information
3. Include specific names, dates, and affiliations
4. Maximum length: {Config.SUMMARY_MAX_TOKENS} tokens

Provide a query-focused summary:"""

            prompt_tokens = self.count_tokens(prompt)
            logger.info(f"Context-aware summary - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} chars")

            # 3. 发起大模型推理调用
            if self.enable_streaming:
                print(f"\n🎯 Creating context-aware summary for query: '{query[:50]}...'\n", flush=True)
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates focused, context-aware summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=_reasoning_safe_temperature(self.model, 0.3),
                    max_tokens=_reasoning_safe_max_tokens(self.model, Config.SUMMARY_MAX_TOKENS),
                    stream=True
                )
                
                summary_parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        summary_parts.append(content)
                    print("\n")
                summary = "".join(summary_parts)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates focused, context-aware summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=_reasoning_safe_temperature(self.model, 0.3),
                    max_tokens=_reasoning_safe_max_tokens(self.model, Config.SUMMARY_MAX_TOKENS)
                )
                summary = response.choices[0].message.content or ""
            
            return CompressedContent(
                original_length=total_original,
                compressed_length=len(summary),
                content=summary,
                strategy=CompressionStrategy.CONTEXT_AWARE
            )
            
        except Exception as e:
            logger.error(f"Error creating context-aware summary: {str(e)}")
            # 容错降级方案：拼接搜索引擎 snippet
            fallback = "\n\n".join([r.get('snippet', '') for r in search_results.get('results', [])])
            return CompressedContent(
                original_length=total_original,
                compressed_length=len(fallback),
                content=fallback,
                strategy=CompressionStrategy.CONTEXT_AWARE
            )
    
    def _context_aware_with_citations(
        self,
        search_results: Dict[str, Any],
        query: str,
        current_context: Optional[str] = None
    ) -> CompressedContent:
        """
        策略 4：意图感知且带来源引用 (Context-Aware with Citations)
        
        【解决的核心痛点——压缩后的“信息失真与无法溯源”】：
        大模型在高度压缩信息时，容易将不同网页的事实张冠李戴，或者丢失信息出处，
        导致 Agent 在输出最终报告时无法向用户证明结论的真实来源。
        
        【本策略的具体实现】：
        1. 编号标记 (Source Tagging)：
           给抓取到的每一个网页分配独立的数字角标，例如 [1], [2], [3]，并在正文拼接时显式加上编号；
        2. 行内引用强约束 (Inline Citations)：
           Prompt 要求大模型：在提炼事实时，必须在对应陈述后标注引用角标（如 "成立了 SSI [2]"）；
        3. 尾部追加来源清单 (Source Reference List)：
           在生成的摘要正文末尾，自动拼接结构化的来源清单（包含 [序号] 网页标题 - 原始 URL）；
        4. 元数据回传：
           将完整的来源列表记录在 CompressedContent 的 citations 字段中，供后续评测或系统做可信度校验。
        """
        sources = []
        all_content = []
        total_original = 0
        max_chars_per_page = 5000
        
        # 1. 为每个来源分配 [1], [2] 标号并建立清单
        for i, result in enumerate(search_results.get('results', [])):
            if result.get('content'):
                source_id = f"[{i+1}]"
                original_content = result.get('content', '')
                total_original += len(original_content)
                
                limited_content = original_content[:max_chars_per_page]
                
                sources.append({
                    'id': source_id,
                    'title': result.get('title', 'N/A'),
                    'url': result.get('url', 'N/A')
                })
                
                all_content.append(f"""
{source_id} Title: {result.get('title', 'N/A')}
Content: {limited_content}
""")
        
        combined_content = "\n\n".join(all_content)
        
        try:
            # 2. 构建要求带角标行内引用的提示词
            prompt = f"""Given the search query: "{query}"
{f"Current context: {current_context[:1000]}" if current_context else ""}

Analyze the following search results and provide a focused summary with citations.

Search Results (with source IDs):
{combined_content}

Requirements:
1. Focus on information relevant to: {query}
2. Include inline citations using [1], [2], etc. for each fact
3. Prioritize current/recent information
4. Include specific names, dates, and affiliations with citations
5. Maximum length: {Config.SUMMARY_MAX_TOKENS} tokens

Provide a query-focused summary with citations:"""

            prompt_tokens = self.count_tokens(prompt)
            logger.info(f"Citation-based summary - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} chars")

            # 3. 发起调用生成带角标的精炼摘要
            if self.enable_streaming:
                print(f"\n📚 Creating summary with citations for: '{query[:50]}...'\n", flush=True)
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates focused summaries with proper citations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=_reasoning_safe_temperature(self.model, 0.3),
                    max_tokens=_reasoning_safe_max_tokens(self.model, Config.SUMMARY_MAX_TOKENS),
                    stream=True
                )
                
                summary_parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        summary_parts.append(content)
                    print("\n")
                summary = "".join(summary_parts)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates focused summaries with proper citations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=_reasoning_safe_temperature(self.model, 0.3),
                    max_tokens=_reasoning_safe_max_tokens(self.model, Config.SUMMARY_MAX_TOKENS)
                )
                summary = response.choices[0].message.content or ""
            
            # 4. 在正文末尾结构化追加所有引用的 URL 来源列表
            source_list = "\n\nSources:\n"
            for source in sources:
                source_list += f"{source['id']} {source['title']} - {source['url']}\n"
            
            final_content = summary + source_list
            
            return CompressedContent(
                original_length=total_original,
                compressed_length=len(final_content),
                content=final_content,
                citations=sources,
                strategy=CompressionStrategy.CONTEXT_AWARE_CITATIONS
            )
            
        except Exception as e:
            logger.error(f"Error creating summary with citations: {str(e)}")
            # 降级方案：带编号的 snippet 拼接
            fallback = "\n\n".join([
                f"[{i+1}] {r.get('title', '')}: {r.get('snippet', '')}"
                for i, r in enumerate(search_results.get('results', []))
            ])
            return CompressedContent(
                original_length=total_original,
                compressed_length=len(fallback),
                content=fallback,
                citations=sources,
                strategy=CompressionStrategy.CONTEXT_AWARE_CITATIONS
            )
    
    def estimate_tokens(self, text: str) -> int:
        """
        粗略估算给定文本的 Token 消耗数
        
        使用通用的快速经验法则（1 Token 约等于 4 个字符）进行估算，
        适用于无需精准计费的轻量级监控或快速长度排查。
        
        参数:
            text: 输入文本
            
        返回:
            int: 估算的 Token 数
        """
        return len(text) // 4
