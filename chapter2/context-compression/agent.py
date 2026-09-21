"""
Context Compression Research Agent with Streaming Support
"""

import json
import logging
import time
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from config import Config
from web_tools import WebTools
from compression_strategies import (
    CompressionStrategy,
    ContextCompressor,
    CompressedContent
)


def _reasoning_safe_temperature(model, requested=1.0):
    """Reasoning models (Kimi K3, GPT-5, ...) only accept temperature=1.
    Return 1 for those; otherwise the requested value so non-reasoning
    providers (Doubao, DeepSeek, older Moonshot) are unchanged."""
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested

# Configure logging
logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """
    单次工具调用记录 (Tool Call)
    记录 Agent 在某一步执行了哪个工具、传入了什么参数、原始输出结果以及压缩后的结果。
    """
    tool_name: str                                          # 调用的工具名（如 'search_web' 或 'fetch_webpage'）
    arguments: Dict[str, Any]                              # 模型生成的调用参数（如 {'query': '...'}）
    result: Optional[Any] = None                           # 工具执行后返回的原始内容
    compressed_result: Optional[CompressedContent] = None   # 经过压缩策略处理后的精简内容
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[str] = None                               # 服务端返回的 tool_call id，用于消息配对


@dataclass
class AgentTrajectory:
    """
    Agent 执行轨迹追踪器 (Trajectory Tracker)
    全程记录 Agent 在长任务中的资源使用情况，包括 Token 计数、耗时、压缩策略和是否溢出。
    """
    tool_calls: List[ToolCall] = field(default_factory=list) # 历史上所有的工具调用列表
    total_tokens_used: int = 0                              # 累积消耗的 Token 总量
    prompt_tokens_used: int = 0                             # 累积消耗的输入 Prompt Token
    completion_tokens_used: int = 0                         # 累积消耗的输出 Completion Token
    last_prompt_tokens: int = 0                             # 最近一次 API 调用的 Prompt Token（代表当前瞬时上下文大小）
    context_overflows: int = 0                              # 触发上下文溢出（超出硬性窗口上限）的次数
    compression_strategy: CompressionStrategy = CompressionStrategy.NO_COMPRESSION # 当前使用的压缩策略
    start_time: float = field(default_factory=time.time)    # 任务启动时间戳
    end_time: Optional[float] = None                        # 任务完成时间戳


class ResearchAgent:
    """
    具备上下文压缩与长流程研究能力的 ReAct 智能体
    能够自主调用搜索工具检索信息，并在多轮交互中根据选定的策略动态压缩对话历史，
    保证在完成复杂长任务的同时不突破大模型的上下文预算。
    """
    
    def __init__(
        self, 
        api_key: str,
        compression_strategy: CompressionStrategy = CompressionStrategy.NO_COMPRESSION,
        verbose: bool = False,
        enable_streaming: bool = True
    ):
        """
        Initialize the research agent
        
        Args:
            api_key: API key for Moonshot/Kimi
            compression_strategy: Strategy for context compression
            verbose: Enable verbose logging
            enable_streaming: Enable streaming responses
        """
        # Moonshot 官方 key 存在则直连；否则回退 OpenRouter（见 Config.resolve_llm）。
        resolved_key, resolved_base_url, resolved_model = Config.resolve_llm()
        # 设置超时时间为 60 秒，避免走本地网络代理时因 SSL 握手耗时导致 Request timed out
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
            timeout=60.0
        )
        self.model = resolved_model
        self.compression_strategy = compression_strategy
        self.verbose = verbose
        self.enable_streaming = enable_streaming
        
        # Initialize tools
        self.web_tools = WebTools()
        self.compressor = ContextCompressor(compression_strategy, api_key, enable_streaming)
        
        # Initialize trajectory
        self.trajectory = AgentTrajectory(compression_strategy=compression_strategy)
        
        # Initialize conversation history
        self.conversation_history = []
        self._init_system_prompt()
        
        logger.info(f"Agent initialized with compression strategy: {compression_strategy.value}")
    
    def _init_system_prompt(self):
        """
        初始化系统提示词 (System Prompt)
        明确 Agent 的角色设定、核心研究任务目标、工作规范与可用工具约束。
        """
        # 动态获取当前系统日期，确保模型在搜索事实时能够感知“今天”的时间边界
        from datetime import datetime
        today = datetime.now()
        date_string = today.strftime("%A, %B %d, %Y")
        
        self.conversation_history = [
            {
                "role": "system",
                "content": f"""You are a research assistant tasked with finding information about OpenAI co-founders.

Your task is to:
1. First, search for and identify ALL OpenAI co-founders
2. Then, search for EACH co-founder individually to find their CURRENT affiliations
3. Compile a comprehensive report with current status for each co-founder

Important instructions:
- Be thorough and systematic - search for each person individually
- Focus on CURRENT affiliations, not historical roles
- Include company names, positions, and any recent changes
- If someone left a position, note where they went
- When you have gathered all information, provide a FINAL ANSWER with a complete list

Available tools:
- search_web: Search the web for information
- fetch_webpage: Fetch specific webpage content

Start by searching for the complete list of OpenAI co-founders.

TODAY'S DATE: {date_string}"""
            }
        ]
    
    def _get_tools_description(self) -> List[Dict[str, Any]]:
        """
        获取暴露给大模型的工具描述列表（遵循 OpenAI 标准 Function Calling 规范）
        声明了两个工具：
        1. search_web: 根据关键词搜索互联网并抓取网页内容；
        2. fetch_webpage: 抓取指定 URL 的纯文本网页正文。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information. Returns multiple search results with content from each webpage.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_webpage",
                    "description": "Fetch and extract text content from a specific webpage URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the webpage to fetch"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[Any, Optional[CompressedContent]]:
        """
        执行模型指定的外部工具，并按选定的压缩策略对执行结果进行可选压缩
        
        参数:
            tool_name: 要执行的工具名称 (如 'search_web' 或 'fetch_webpage')
            arguments: 传入工具的参数字典 (例如 {'query': '...', 'num_results': 5})
            
        返回:
            Tuple[Any, Optional[CompressedContent]]:
                - 第一个元素: 工具执行的原始结果数据 (Python 对象/字典/列表)
                - 第二个元素: 若进行了压缩，返回 CompressedContent 对象；若未压缩或不支持压缩则为 None
        """
        # 参数容错保护：若模型传入非 dict 类型参数（如 None 或字符串），统一重置为空字典
        if not isinstance(arguments, dict):
            arguments = {}

        # -------------------------------------------------------------
        # 工具分支 1: search_web 网络搜索
        # -------------------------------------------------------------
        if tool_name == "search_web":
            # 校验必需参数 query
            if "query" not in arguments:
                return {"error": "Missing required argument 'query' for search_web"}, None
            try:
                # 调用底层 WebTools 执行真实/模拟搜索与网页抓取
                result = self.web_tools.search_web(**arguments)
            except Exception as e:
                logger.error(f"Failed to execute search_web: {e}")
                return {"error": f"Failed to execute search_web: {e}"}, None
            
            # 应用当前选定的压缩策略 (如 LLM 总结、选择性保留或原始保留)
            query = arguments.get('query', '')
            # 提取近期工具调用轨迹摘要，作为压缩上下文（便于针对性压缩）
            current_context = self._get_current_context_summary()
            compressed = self.compressor.compress_search_results(
                result, 
                query, 
                current_context
            )
            
            return result, compressed
            
        # -------------------------------------------------------------
        # 工具分支 2: fetch_webpage 抓取指定网页全文
        # -------------------------------------------------------------
        elif tool_name == "fetch_webpage":
            # 校验必需参数 url
            if "url" not in arguments:
                return {"error": "Missing required argument 'url' for fetch_webpage"}, None
            try:
                # 调用底层 WebTools 下载并用 BeautifulSoup 解析出网页纯文本
                result = self.web_tools.fetch_webpage(**arguments)
            except Exception as e:
                logger.error(f"Failed to execute fetch_webpage: {e}")
                return {"error": f"Failed to execute fetch_webpage: {e}"}, None
            
            # 对于单独抓取的页面，通常是 Agent 针对某条信息做跟进阅读，
            # 这里默认不即时压缩，保留原文本供 Agent 精细提取
            return result, None
        else:
            # 未知工具名称错误处理
            return {"error": f"Unknown tool: {tool_name}"}, None
    
    def _get_current_context_summary(self) -> str:
        """
        获取当前执行轨迹的上下文摘要（用于辅助上下文敏感的压缩算法）
        
        工作机制：
        提取最近最多 3 次历史工具调用的搜索 query，拼接成形如：
        'Previous search: A | Previous search: B' 的短文本。
        压缩模块（如 LLM 摘要器）可据此获悉 Agent 此刻的探索意图，从而在压缩搜索结果时
        重点保留与意图最相关的高价值信息。
        """
        if not self.trajectory.tool_calls:
            return ""
        
        # 提取最近最多 3 次工具调用
        recent_calls = self.trajectory.tool_calls[-3:]
        context_parts = []
        
        for call in recent_calls:
            context_parts.append(f"Previous search: {call.arguments.get('query', 'N/A')}")
        
        return " | ".join(context_parts)
    
    def _handle_windowed_compression(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对消息历史执行基于滑动窗口阈值（Windowed Context）的延迟动态压缩策略
        
        【设计思想与触发逻辑】：
        普通的即时压缩（如 LLM/Selective）会在工具执行完毕后立即对结果做压缩。
        而“滑动窗口压缩”采取延迟压缩策略：
        1. 在对话初期/上下文充裕时，保留 100% 原始工具结果，让大模型阅读最完整的信息；
        2. 仅当最后一次请求的 Prompt Token 消耗超过预设上下文窗口的 80% 警戒线时，才触发回溯压缩；
        3. 遍历历史消息中的所有 tool 角色消息，利用大模型或启发式规则将其压缩为核心要点并打上标记；
        4. 压缩后的消息带有 [COMPRESSED] 标记，防止后续轮次重复压缩。
        
        参数:
            messages: 当前对话历史消息列表 (包含 system, user, assistant, tool 等角色)
            
        返回:
            List[Dict[str, Any]]: 处理后的消息列表（超阈值时历史 tool 消息被压缩并打标）
        """
        # 非滑动窗口策略直接原样返回
        if self.compression_strategy != CompressionStrategy.WINDOWED_CONTEXT:
            return messages
        
        # -------------------------------------------------------------
        # 1. 阈值检查：80% 上下文窗口警戒线
        # 注意：此处必须使用上一轮模型实际消耗的 Prompt Token 数 (last_prompt_tokens)，
        # 而不能使用累加器 (prompt_tokens_used)。因为多轮对话中前序历史会反复作为 Prompt 发送，
        # 累加器是 O(N^2) 增长的，若用累加器会在窗口远未占满时就误触发压缩。
        # -------------------------------------------------------------
        context_threshold = Config.CONTEXT_WINDOW_SIZE * 0.8

        if self.trajectory.last_prompt_tokens <= context_threshold:
            logger.debug(f"Windowed compression: 上下文使用量未达阈值 ({self.trajectory.last_prompt_tokens:,}/{context_threshold:.0f} tokens)")
            return messages  # 尚未达到警戒线，保留完整未压缩历史

        logger.info(f"⚠️ 上下文占用超过 80% 警戒线 ({self.trajectory.last_prompt_tokens:,}/{Config.CONTEXT_WINDOW_SIZE} tokens) - 开始回溯压缩历史工具消息")
        
        # 压缩标记：已压缩消息的前缀标识，防止下一轮重复压缩同一条消息
        COMPRESSION_MARKER = "[COMPRESSED]"
        
        # -------------------------------------------------------------
        # 2. 统计需要压缩的 tool 消息数量
        # -------------------------------------------------------------
        tool_messages_to_compress = []
        already_compressed_count = 0
        
        for i, msg in enumerate(messages):
            if msg.get('role') == 'tool':
                original_content = msg.get('content', '')
                if original_content.startswith(COMPRESSION_MARKER):
                    already_compressed_count += 1
                else:
                    tool_messages_to_compress.append((i, msg))
        
        total_tool_messages = already_compressed_count + len(tool_messages_to_compress)
        
        # 若所有工具消息均已被压缩，则无需任何操作
        if not tool_messages_to_compress:
            logger.debug(f"Windowed compression: 所有 {total_tool_messages} 条工具消息此前已全部完成压缩")
            return messages
        
        logger.info(f"📊 正在压缩 {len(tool_messages_to_compress)} 条未压缩工具消息 (共 {total_tool_messages} 条)")
        
        # -------------------------------------------------------------
        # 3. 逐条对未压缩的历史 tool 消息执行上下文感知压缩
        # -------------------------------------------------------------
        compressed_messages = []
        compressed_in_this_pass = 0
        
        for i, msg in enumerate(messages):
            if msg.get('role') == 'tool':
                original_content = msg.get('content', '')
                
                # 检查是否已包含压缩标记
                if original_content.startswith(COMPRESSION_MARKER):
                    # 已压缩过，直接保留
                    compressed_messages.append(msg)
                else:
                    # 发现未压缩的工具返回结果，执行压缩
                    compressed_in_this_pass += 1
                    
                    # 尝试从 trajectory 中回溯该 tool 消息对应的 search query，以提供针对性压缩线索
                    tool_call_id = msg.get('tool_call_id')
                    query = "Information search"  # 默认兜底 query
                    
                    for call in self.trajectory.tool_calls:
                        if call.id is not None and call.id == tool_call_id:
                            query = call.arguments.get('query', query)
                            break
                    
                    logger.debug(f"正在压缩第 {compressed_in_this_pass}/{len(tool_messages_to_compress)} 条工具消息 (索引 {i}, 检索词: {query[:50]}...)")
                    # 调用压缩器：针对历史记录压缩，且显式保留来源引用 (preserve_citations=True)
                    compressed = self.compressor.compress_for_history(
                        original_content,
                        'search_web',
                        query,
                        preserve_citations=True
                    )
                    logger.debug(f"压缩效果: {compressed.original_length:,} 字符 → {compressed.compressed_length:,} 字符")
                    
                    # 构造带有统一标记和压缩前后体积说明的新内容
                    compressed_content = (
                        f"{COMPRESSION_MARKER} "
                        f"[Original: {compressed.original_length:,} chars → Compressed: {compressed.compressed_length:,} chars]\n"
                        f"{compressed.content}"
                    )
                    
                    # 组装替换后的 tool 消息
                    compressed_messages.append({
                        **msg,
                        'content': compressed_content
                    })
            else:
                # 非 tool 消息（system, user, assistant）不参与此压缩，完整保留
                compressed_messages.append(msg)
        
        logger.info(f"✅ 本轮成功压缩了 {compressed_in_this_pass} 条历史工具消息")
        
        return compressed_messages
    
    def _stream_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        以流式 (Streaming) 方式向大模型发起请求，实时输出思考/回复并组装 Tool Calls
        
        【流式处理的关键难点与设计】：
        1. 实时文本打印：将 delta.content 逐字打印到控制台，实现打字机效果。
        2. 工具调用增量拼接 (Streaming Tool Calls)：
           在大模型流式输出函数调用时，tool_calls 的 arguments 是作为一段段短字符串分批吐出的
           （例如 '{"qu' -> 'ery": "' -> 'OpenAI"' -> '}'）。
           本方法维护 current_tool_calls 列表，根据 chunk 中的 index 持续将 arguments 片段拼接到对应的工具调用中。
        3. Gemini 兼容性支持 (thought_signature)：
           Google Gemini 2.0/Flash 系列模型在使用 OpenAI 兼容接口时，其工具调用会携带思考签名 (extra_content)。
           在多轮对话把 assistant 消息传回给模型时，若丢失该签名会报错 400。因此此处专门提取并保留 extra_content。
        4. Token 使用量采集：
           启用 stream_options={"include_usage": True}，在流的末尾捕获精确的 Token 消耗数据，
           并更新 AgentTrajectory（包括当前轮次的 prompt_tokens，用于滑动窗口阈值判定）。
        
        参数:
            messages: 当前发送给模型的对话历史消息
            
        返回:
            Dict[str, Any]: 完整组装好的标准 assistant 消息字典 {"role": "assistant", "content": ..., "tool_calls": [...]}
        """
        try:
            # 发起流式 API 调用
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._get_tools_description(),
                tool_choice="auto",
                temperature=_reasoning_safe_temperature(self.model, Config.MODEL_TEMPERATURE),
                max_tokens=Config.MODEL_MAX_TOKENS,
                stream=True,
                stream_options={"include_usage": True}  # 要求在流结束前返回 Token 消耗数据
            )
            
            collected_chunks = []
            collected_messages = []
            current_tool_calls = []
            usage_data = None
            
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # 迭代处理流式返回的数据块 (Chunks)
            for chunk in stream:
                collected_chunks.append(chunk)
                
                # 捕获 Token 使用量（通常在最后一个 chunk 或独立的 usage chunk 中）
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage_data = chunk.usage
                
                # 处理文本或工具调用增量
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    # 1. 文本内容流式输出
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        print(content, end="", flush=True)
                        collected_messages.append(content)
                    
                    # 2. 工具调用增量拼装
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            idx = tool_call_delta.index if tool_call_delta.index is not None else (len(current_tool_calls) - 1 if current_tool_calls else 0)
                            if idx < 0:
                                idx = 0
                            # 动态扩容当前工具调用槽位
                            while len(current_tool_calls) <= idx:
                                current_tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            # 填充工具 call id
                            if tool_call_delta.id:
                                current_tool_calls[idx]["id"] = tool_call_delta.id
                            # 填充工具名称与累加参数字符串
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    current_tool_calls[idx]["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    current_tool_calls[idx]["function"]["arguments"] += tool_call_delta.function.arguments
                            # 保留 Google Gemini 等模型需要的 thought_signature 等 extra_content 元数据
                            if hasattr(tool_call_delta, 'extra_content') and tool_call_delta.extra_content:
                                current_tool_calls[idx]["extra_content"] = tool_call_delta.extra_content
            
            print("\n", flush=True)
            
            # 记录并统计实际消耗的 Token
            if usage_data:
                prompt_tokens = usage_data.prompt_tokens if hasattr(usage_data, 'prompt_tokens') else 0
                completion_tokens = usage_data.completion_tokens if hasattr(usage_data, 'completion_tokens') else 0
                total_tokens = usage_data.total_tokens if hasattr(usage_data, 'total_tokens') else 0
                
                logger.info(f"🔢 API Token Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

                # 更新轨迹指标（注意：last_prompt_tokens 专供滑动窗口 80% 阈值检查使用）
                self.trajectory.last_prompt_tokens = prompt_tokens
                self.trajectory.prompt_tokens_used += prompt_tokens
                self.trajectory.completion_tokens_used += completion_tokens
                self.trajectory.total_tokens_used += total_tokens
            
            # 组装最终完整的 assistant 响应字典
            complete_message = {
                "role": "assistant",
                "content": "".join(collected_messages) if collected_messages else None
            }
            
            if current_tool_calls:
                complete_message["tool_calls"] = current_tool_calls
            
            return complete_message
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise
    
    def _non_streaming_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        以非流式 (同步阻塞) 方式向大模型发起请求
        
        适用于调试、简单场景或不需要打字机流式输出的环境。
        同样负责：
        1. 提取 response.usage 更新 AgentTrajectory；
        2. 转换 message 为字典，并完整保留 tool_calls 及其 extra_content (Gemini 思考签名)；
        3. 控制台打印模型回复。
        
        参数:
            messages: 当前对话历史消息
            
        返回:
            Dict[str, Any]: 标准格式的 assistant 消息字典
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self._get_tools_description(),
            tool_choice="auto",
            temperature=_reasoning_safe_temperature(self.model, Config.MODEL_TEMPERATURE),
            max_tokens=Config.MODEL_MAX_TOKENS,
            stream=False
        )
        
        message = response.choices[0].message
        
        # 提取并记录 Token 消耗
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            logger.info(f"🔢 API Token Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

            # 更新轨迹统计指标
            self.trajectory.last_prompt_tokens = prompt_tokens
            self.trajectory.prompt_tokens_used += prompt_tokens
            self.trajectory.completion_tokens_used += completion_tokens
            self.trajectory.total_tokens_used += total_tokens
        
        # 转换为标准的字典格式
        message_dict = {
            "role": "assistant",
            "content": message.content
        }
        
        # 提取工具调用列表，并保留 Google Gemini 必需的 extra_content (thought_signature)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            message_dict["tool_calls"] = []
            for tc in message.tool_calls:
                tc_item = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                if hasattr(tc, 'extra_content') and tc.extra_content:
                    tc_item["extra_content"] = tc.extra_content
                message_dict["tool_calls"].append(tc_item)
        
        # 在控制台展示模型回复
        if message.content:
            print(f"\n🤖 Assistant: {message.content}\n")
        
        return message_dict
    
    def execute_research(self, max_iterations: int = 15) -> Dict[str, Any]:
        """
        执行完整的 Agent 调研任务（核心 ReAct 智能体循环）
        
        【核心运行流程与状态机】：
        1. 初始化提示词：将用户初始任务（调研 OpenAI 联合创始人的最新现状）加入消息队列；
        2. 启动 ReAct (Reasoning + Acting) 循环，最多迭代 max_iterations 轮：
           a. 上下文滑动窗口压缩：若策略为 WINDOWED_CONTEXT，检测是否超过 80% 阈值，若超标则触发回溯压缩；
           b. 上下文溢出熔断检测：在 NO_COMPRESSION 策略下，若 Prompt Token 超过 80% 警戒线，
              直接报告 Context Window Exceeded 错误，用于向读者演示“不压缩策略下的上下文崩溃极限”；
           c. 大模型推理：根据 enable_streaming 选择流式或非流式发起调用；
           d. 分支判定：
              - 若模型返回 tool_calls：进入 Action 阶段，解析参数、调用 search_web/fetch_webpage，
                根据当前策略决定使用压缩后文本还是完整文本，构造 role="tool" 消息写回上下文；
              - 若模型未返回 tool_calls 且带有文本：说明模型完成调研并得出最终结论，结束循环；
        3. 统计并返回耗时、轨迹指标、成功状态与最终答案。
        
        参数:
            max_iterations: 最大允许的交互轮数（防止无限调用死循环，默认 15）
            
        返回:
            Dict[str, Any]: 包含 final_answer、trajectory、iterations、success、execution_time 的结果字典
        """
        # 1. 注入初始用户任务提示
        self.conversation_history.append({
            "role": "user",
            "content": "Please research and find the current affiliations of all OpenAI co-founders."
        })
        
        messages = self.conversation_history.copy()
        iteration = 0
        final_answer = None
        
        print("\n" + "="*60)
        print(f"Starting research with {self.compression_strategy.value} strategy")
        print("="*60)
        
        # 2. 进入 ReAct 核心交互主循环
        while iteration < max_iterations:
            iteration += 1
            print(f"\n📍 Iteration {iteration}/{max_iterations}")
            
            try:
                # ---------------------------------------------------------
                # 阶段 2.1: 滑动窗口延迟压缩检查
                # 若策略为 WINDOWED_CONTEXT，且前一轮 prompt tokens > 80% 阈值，
                # 将对话历史中所有未压缩的 tool 消息进行回溯压缩
                # ---------------------------------------------------------
                if self.compression_strategy == CompressionStrategy.WINDOWED_CONTEXT:
                    messages = self._handle_windowed_compression(messages)
                
                # 打印累加的 Token 消耗情况（方便监控实验开销）
                print(f"📊 Cumulative Token Usage - Prompt: {self.trajectory.prompt_tokens_used:,}, Completion: {self.trajectory.completion_tokens_used:,}, Total: {self.trajectory.total_tokens_used:,}")
                
                # ---------------------------------------------------------
                # 阶段 2.2: 上下文溢出监测 (Context Overflow Check)
                # 本实验模拟 128k 窗口。若上一次请求的 Prompt 大小超过 128k * 80% = 102,400 tokens：
                # - 如果策略是 NO_COMPRESSION，则主动抛出上下文溢出错误，证明不压缩会导致 Agent 崩溃
                # ---------------------------------------------------------
                if self.trajectory.total_tokens_used > 0:  # 仅在至少执行过一次请求后检查
                    if self.trajectory.last_prompt_tokens > Config.CONTEXT_WINDOW_SIZE * 0.8:
                        logger.warning(f"Approaching context limit: {self.trajectory.last_prompt_tokens:,} prompt tokens in last request")
                        self.trajectory.context_overflows += 1

                        if self.compression_strategy == CompressionStrategy.NO_COMPRESSION:
                            print("\n⚠️ Context overflow detected! This demonstrates the limitation of no compression.")
                            return {
                                "error": f"Context window exceeded - {self.trajectory.last_prompt_tokens:,} tokens in last request (limit: {Config.CONTEXT_WINDOW_SIZE})",
                                "trajectory": self.trajectory,
                                "iterations": iteration
                            }
                
                # ---------------------------------------------------------
                # 阶段 2.3: 大模型思考决策 (Reasoning)
                # ---------------------------------------------------------
                if self.enable_streaming:
                    message = self._stream_response(messages)
                else:
                    message = self._non_streaming_response(messages)
                
                # ---------------------------------------------------------
                # 阶段 2.4: 执行模型决策动作 (Acting / Tool Execution)
                # ---------------------------------------------------------
                if message.get('tool_calls'):
                    # 将模型的 assistant (含 tool_calls) 消息先推入上下文
                    messages.append(message)

                    if message.get('content'):
                        print(f"\n🤖 Assistant: {message['content']}")
                    
                    # 遍历并执行本次请求中模型生成的所有工具调用
                    for tool_call in message['tool_calls']:
                        function_name = tool_call['function']['name']
                        raw_args = tool_call['function'].get('arguments') or "{}"
                        
                        # 健壮的 JSON 反序列化：兼容 dict、bytes、str 或模型格式异常的各种情况
                        try:
                            if isinstance(raw_args, dict):
                                function_args = raw_args
                            elif isinstance(raw_args, (bytes, bytearray)):
                                function_args = json.loads(raw_args.decode("utf-8"))
                            elif isinstance(raw_args, str):
                                function_args = json.loads(raw_args)
                            else:
                                function_args = json.loads(str(raw_args))

                            if not isinstance(function_args, dict):
                                logger.warning(
                                    "Tool argument JSON is not an object, proceeding with empty object: %r",
                                    raw_args,
                                )
                                function_args = {}
                        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                            # 模型若偶尔吐出非法 JSON，容错处理为空参数，保证主流程不中断崩溃
                            function_args = {}
                            logger.warning(
                                "Tool argument is not valid JSON, proceeding with empty object: %r",
                                raw_args,
                            )
                        
                        print(f"\n🔧 Executing: {function_name}")
                        print(f"   Args: {function_args}")
                        
                        # 调用工具并根据即时压缩策略处理结果
                        result, compressed = self._execute_tool(function_name, function_args)
                        
                        # 完整记录该次调用的详细信息至轨迹 Trajectory 中
                        tool_call_record = ToolCall(
                            tool_name=function_name,
                            arguments=function_args,
                            result=result,
                            compressed_result=compressed,
                            id=tool_call['id']
                        )
                        self.trajectory.tool_calls.append(tool_call_record)
                        
                        # 决定最终加入对话历史 (messages) 的文本内容：
                        # - 若有压缩结果且策略不是 NO_COMPRESSION，采用压缩后的精炼文本
                        # - 否则采用完整的原始 JSON 数据（无压缩或滑动窗口初期的策略）
                        if compressed and self.compression_strategy != CompressionStrategy.NO_COMPRESSION:
                            tool_content = compressed.content
                            print(f"   ✂️ Compressed: {compressed.original_length:,} → {compressed.compressed_length:,} chars")
                        else:
                            if function_name == "search_web":
                                tool_content = json.dumps(result, indent=2)
                            else:
                                tool_content = json.dumps(result)
                        
                        # 将工具执行结果作为 role="tool" 消息存入对话历史，供下一轮模型阅读
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call['id'],
                            "content": tool_content
                        }
                        messages.append(tool_msg)
                        
                        print(f"   📄 Result size: {len(tool_content):,} characters")
                
                # ---------------------------------------------------------
                # 阶段 2.5: 调研任务收敛 (Final Answer)
                # 模型没有再发出任何 tool_calls，而是给出了文本回答，标志着调研结束
                # ---------------------------------------------------------
                elif message.get('content'):
                    messages.append(message)
                    final_answer = message['content']
                    logger.info("Final answer found")
                    break
                    
            except Exception as e:
                logger.error(f"Error during research: {str(e)}")
                return {
                    "error": str(e),
                    "trajectory": self.trajectory,
                    "iterations": iteration
                }
        
        # 3. 统计结束时间与耗时
        self.trajectory.end_time = time.time()
        
        return {
            "final_answer": final_answer,
            "trajectory": self.trajectory,
            "iterations": iteration,
            "success": final_answer is not None,
            "execution_time": self.trajectory.end_time - self.trajectory.start_time
        }
    
    def reset(self):
        """
        重置 Agent 的内部运行时状态
        
        清空并重建轨迹 Trajectory、重新初始化系统提示词、清空 WebTools 的网页搜索缓存，
        使得 Agent 可以立即以干净的状态投入下一次独立的策略对比实验。
        """
        self.trajectory = AgentTrajectory(compression_strategy=self.compression_strategy)
        self._init_system_prompt()
        self.web_tools.clear_cache()
        logger.info("Agent state reset")
