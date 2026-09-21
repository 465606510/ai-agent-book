#!/usr/bin/env python3
"""
Context Compression Strategies Comparison Experiment
"""

import os
import sys
import json
import time

# python .\experiment.py -s context_aware
# Windows 控制台默认代码页通常是 GBK (cp936)，打印 Emoji 会报 UnicodeEncodeError
# 强制 stdout 和 stderr 采用 UTF-8 编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict
from colorama import init, Fore, Style
from tqdm import tqdm

from config import Config
from agent import ResearchAgent
from compression_strategies import CompressionStrategy

# 初始化 colorama，使 Windows 终端能正常输出带颜色的文本（autoreset=True 表示每次 print 自动重置颜色）
init(autoreset=True)


# ==============================================================================
# 策略映射字典 (STRATEGY_CHOICES)
# 作用：将命令行传参的简短别名（例如 -s context_aware）映射为内部枚举 CompressionStrategy
# 顺序与《深入理解 AI Agent》第 2 章“实验 2-10 ★★★：上下文压缩策略对比”完全对齐
# ==============================================================================
STRATEGY_CHOICES = {
    # 1. 无压缩基线：原始网页全量塞入消息历史，用于观察不加控制时上下文迅速溢出（Overflow）的现象
    "no_compression": CompressionStrategy.NO_COMPRESSION,
    # 2. 单页独立摘要：每抓取一个网页就让模型总结一次，再拼接起来。简单但丢失跨网页的关联关系
    "individual": CompressionStrategy.NON_CONTEXT_AWARE_INDIVIDUAL,
    # 3. 全局合并摘要：将本次工具调用涉及的全部网页合并后一次性总结。结构连贯，但单页归因较弱
    "combined": CompressionStrategy.NON_CONTEXT_AWARE_COMBINED,
    # 4. 上下文/目标感知摘要（推荐）：带着当前研究的目标问题去提炼网页信息，只保留直接相关的关键事实
    "context_aware": CompressionStrategy.CONTEXT_AWARE,
    # 5. 带引用的目标感知摘要：在策略 4 的基础上，强制保留网页 URL 和出处引用，方便后续追问与核查
    "citations": CompressionStrategy.CONTEXT_AWARE_CITATIONS,
    # 6. 滑动窗口：仅保留最近几轮工具调用的输出，最早的历史直接丢弃。简单但容易丢失早期关键记忆
    "windowed": CompressionStrategy.WINDOWED_CONTEXT,
}

# 所有支持的策略列表（用于默认跑全量测试）
ALL_STRATEGIES = list(STRATEGY_CHOICES.values())


class ExperimentRunner:
    """
    实验运行与评估控制器 (Experiment Runner)
    负责统一调度不同压缩策略的执行、收集 Agent 的运行轨迹与资源消耗、输出横向对比报告并持久化到 JSON 文件。
    """
    
    def __init__(self, api_key: str, results_file: Optional[str] = None,
                 enable_streaming: bool = False):
        """
        初始化实验运行器

        参数:
            api_key: 模型调用 API Key
            results_file: 结果 JSON 文件的指定存储路径（若不指定，默认保存在 results/experiment_<时间戳>.json）
            enable_streaming: 是否在终端实时流式打印大模型与压缩过程的文字输出（默认 False 以保持对比表格干净整洁）
        """
        self.api_key = api_key
        self.results = []
        self.enable_streaming = enable_streaming

        # 创建必要的输出目录（如 results/ 和 cache/）
        Config.create_directories()

        # 确定评测结果的保存路径
        if results_file:
            self.results_file = results_file
            parent = os.path.dirname(self.results_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_file = os.path.join(Config.RESULTS_DIR, f"experiment_{timestamp}.json")

    def run_single_strategy(self, strategy: CompressionStrategy, verbose: bool = False) -> Dict[str, Any]:
        """
        运行单个压缩策略的独立实验

        参数:
            strategy: 待评测的压缩策略枚举（例如 CompressionStrategy.CONTEXT_AWARE）
            verbose: 是否开启详细调试日志输出
            
        返回:
            包含该策略完整运行指标（metrics）与最终研究答案（final_answer）的字典
        """
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}Testing Strategy: {Fore.YELLOW}{strategy.value}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # 1. 为当前策略单独实例化一个 ResearchAgent 智能体
        agent = ResearchAgent(
            api_key=self.api_key,
            compression_strategy=strategy,
            verbose=verbose,
            enable_streaming=self.enable_streaming  # 默认关闭流式输出，保持终端表格整洁
        )
        
        start_time = time.time()
        
        try:
            # 2. 启动智能体的自主研究主循环（最大迭代轮数由 Config.MAX_ITERATIONS 限制）
            result = agent.execute_research(max_iterations=Config.MAX_ITERATIONS)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 3. 收集并分析执行轨迹（trajectory）中的核心指标
            trajectory = result.get('trajectory')
            
            # 整理指标卡片：成功与否、迭代轮数、工具调用数、溢出次数、总耗时、Token 用量
            metrics = {
                'strategy': strategy.value,
                'success': result.get('success', False),
                'iterations': result.get('iterations', 0),
                'tool_calls': len(trajectory.tool_calls) if trajectory else 0,
                'context_overflows': trajectory.context_overflows if trajectory else 0,
                'execution_time': execution_time,
                'total_tokens': trajectory.total_tokens_used if trajectory else 0,
                'error': result.get('error'),
                'final_answer_length': len(result.get('final_answer', '')) if result.get('final_answer') else 0
            }
            
            # 4. 计算文本压缩率（压缩后字符数 / 原始抓取字符数）
            if trajectory and trajectory.tool_calls:
                total_original = 0
                total_compressed = 0
                
                for call in trajectory.tool_calls:
                    if call.compressed_result:
                        total_original += call.compressed_result.original_length
                        total_compressed += call.compressed_result.compressed_length
                    elif call.result and call.tool_name == 'search_web':
                        # 无压缩基线：统计原始 JSON 的字符体积
                        content = json.dumps(call.result)
                        total_original += len(content)
                        total_compressed += len(content)
                
                if total_original > 0:
                    metrics['compression_ratio'] = round(total_compressed / total_original, 3)
                    metrics['total_original_size'] = total_original
                    metrics['total_compressed_size'] = total_compressed
                else:
                    metrics['compression_ratio'] = 1.0
                    metrics['total_original_size'] = 0
                    metrics['total_compressed_size'] = 0
            
            # Print summary
            self._print_summary(metrics)
            
            # Store full result
            full_result = {
                'metrics': metrics,
                'final_answer': result.get('final_answer'),
                'timestamp': datetime.now().isoformat()
            }
            
            return full_result
            
        except Exception as e:
            print(f"{Fore.RED}Error during experiment: {str(e)}{Style.RESET_ALL}")
            
            return {
                'metrics': {
                    'strategy': strategy.value,
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print a summary of the metrics"""
        print(f"\n{Fore.GREEN}📊 Results Summary:{Style.RESET_ALL}")
        print(f"  Success: {self._format_bool(metrics['success'])}")
        print(f"  Iterations: {metrics['iterations']}")
        print(f"  Tool Calls: {metrics['tool_calls']}")
        print(f"  Execution Time: {metrics['execution_time']:.2f}s")
        print(f"  Total Tokens: {metrics.get('total_tokens', 0):,}")

        if 'compression_ratio' in metrics:
            print(f"  Compression Ratio: {metrics['compression_ratio']:.1%}")
            print(f"  Original Size: {metrics['total_original_size']:,} chars")
            print(f"  Compressed Size: {metrics['total_compressed_size']:,} chars")
        
        if metrics.get('context_overflows', 0) > 0:
            print(f"  {Fore.YELLOW}Context Overflows: {metrics['context_overflows']}{Style.RESET_ALL}")
        
        if metrics.get('error'):
            print(f"  {Fore.RED}Error: {metrics['error'][:100]}...{Style.RESET_ALL}")
    
    def _format_bool(self, value: bool) -> str:
        """Format boolean value with color"""
        if value:
            return f"{Fore.GREEN}✓ Yes{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}✗ No{Style.RESET_ALL}"
    
    def run_all_strategies(self, strategies: Optional[List[CompressionStrategy]] = None) -> None:
        """
        批量调度运行指定的压缩策略列表（默认运行全部 6 种）
        通过 tqdm 显示进度条，并在每个策略结束后自动落盘保存中间结果。
        """
        if strategies is None:
            strategies = list(ALL_STRATEGIES)

        print(f"\n{Fore.MAGENTA}{'='*70}")
        print(f"{Fore.MAGENTA}CONTEXT COMPRESSION STRATEGIES COMPARISON EXPERIMENT")
        print(f"{Fore.MAGENTA}{'='*70}{Style.RESET_ALL}")
        print(f"\nTesting {len(strategies)} compression strategies...")
        print(f"Task: Research current affiliations of OpenAI co-founders")
        
        # 逐个执行策略，利用 tqdm 动态渲染进度条
        for strategy in tqdm(strategies, desc="Running experiments"):
            result = self.run_single_strategy(strategy)
            self.results.append(result)
            
            # 每跑完一个策略立即保存到磁盘，防止中途异常退出丢失数据
            self._save_results()
            
            # 策略之间预留 2 秒冷却，避免频繁发起 API 请求被服务端限流
            time.sleep(2)
        
        # 全部策略执行完毕，打印最终横向综合对比总表
        self._print_comparison()
    
    def _save_results(self):
        """将运行评测指标和结果持久化保存到 JSON 文件"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {self.results_file}")
    
    def _print_comparison(self):
        """
        打印所有策略的终极横向对比大表：
        输出各策略的【是否成功】、【总耗时】、【Token消耗】、【压缩率】、【上下文溢出次数】
        """
        print(f"\n{Fore.MAGENTA}{'='*70}")
        print(f"{Fore.MAGENTA}FINAL COMPARISON")
        print(f"{Fore.MAGENTA}{'='*70}{Style.RESET_ALL}")
        
        # 打印表头
        print(f"\n{'Strategy':<38} {'Success':<9} {'Time':<9} {'Tokens':<11} {'Compress':<10} {'Overflows':<10}")
        print("-" * 90)

        # 逐行输出对比数据，成功显示绿色，失败显示红色
        for result in self.results:
            metrics = result['metrics']
            strategy = metrics['strategy'][:36]
            success = "✓" if metrics['success'] else "✗"
            time_str = f"{metrics.get('execution_time', 0):.1f}s"
            tokens = f"{metrics.get('total_tokens', 0):,}" if metrics.get('total_tokens') else "N/A"
            compress = f"{metrics.get('compression_ratio', 1.0):.1%}" if 'compression_ratio' in metrics else "N/A"
            overflows = str(metrics.get('context_overflows', 0))

            color = Fore.GREEN if metrics['success'] else Fore.RED
            print(f"{color}{strategy:<38} {success:<9} {time_str:<9} {tokens:<11} {compress:<10} {overflows:<10}{Style.RESET_ALL}")

        print("\n" + "="*90)
        
        # 输出统计与核心实验发现
        self._print_analysis()
    
    def _print_analysis(self):
        """
        分析对比实验结果，提炼核心结论：
        1. 统计成功与失败的策略数量
        2. 找出执行速度最快的策略（Fastest）
        3. 找出压缩率最高、最节省字符的策略（Most Efficient）
        4. 罗列实验总结核心发现（Key Findings）
        """
        print(f"\n{Fore.CYAN}📈 Analysis:{Style.RESET_ALL}")
        
        successful = [r for r in self.results if r['metrics']['success']]
        failed = [r for r in self.results if not r['metrics']['success']]
        
        print(f"\n  Successful Strategies: {len(successful)}/{len(self.results)}")
        
        if successful:
            # 找到最快的策略和最极致压缩的策略
            fastest = min(successful, key=lambda x: x['metrics']['execution_time'])
            most_efficient = min(successful, key=lambda x: x['metrics'].get('total_compressed_size', float('inf')))
            
            print(f"  Fastest: {fastest['metrics']['strategy']} ({fastest['metrics']['execution_time']:.1f}s)")
            print(f"  Most Efficient: {most_efficient['metrics']['strategy']} ({most_efficient['metrics'].get('total_compressed_size', 0):,} chars)")
        
        if failed:
            print(f"\n  Failed Strategies:")
            for r in failed:
                err = r['metrics'].get('error') or 'No final answer within max iterations'
                print(f"    - {r['metrics']['strategy']}: {err[:50]}...")
        
        # 核心实验结论总结（与书本正文论述对应）
        print(f"\n{Fore.CYAN}🔍 Key Findings:{Style.RESET_ALL}")
        print("  1. No Compression (无压缩): 预期必然在长任务中因上下文溢出而崩溃 ✓")
        print("  2. Non-Context-Aware (非目标感知): 容易丢失后续回答所需的关键背景事实")
        print("  3. Context-Aware (目标感知压缩): 兼顾极低 Token 与极高信息保留率，表现最佳")
        print("  4. With Citations (带引用出处): 能为后续事实核实和追问提供可信来源")
        print("  5. Windowed Context (滑动窗口): 虽简单高效，但若窗口过小会遗忘早期关键记忆")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="experiment.py",
        description="上下文压缩策略对比实验（对应《深入理解 AI Agent》实验 2-10）。\n"
                    "对同一个研究任务（追踪 OpenAI 联合创始人的现状）分别运行多种压缩策略，"
                    "输出 token 用量 / 压缩率 / 成功率对比表，并保存 JSON 结果。",
        epilog="示例：\n"
               "  python experiment.py                       # 运行全部 6 种策略并对比\n"
               "  python experiment.py -s context_aware      # 只运行“上下文感知压缩”\n"
               "  python experiment.py -s individual combined # 只对比两种非任务感知策略\n"
               "  python experiment.py --model kimi-k3 -o results/k2.json\n"
               "  python experiment.py --list-strategies     # 查看可选策略名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s", "--strategy", nargs="+", choices=list(STRATEGY_CHOICES.keys()), metavar="NAME",
        help="要运行的压缩策略（可指定多个，默认运行全部 6 种）。可选值："
             + ", ".join(STRATEGY_CHOICES.keys()),
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help=f"覆盖使用的模型名称（默认读取环境变量 MODEL_NAME，当前为 {Config.MODEL_NAME}）",
    )
    parser.add_argument(
        "-o", "--output", default=None, metavar="PATH",
        help="结果 JSON 的保存路径（默认 results/experiment_<时间戳>.json）",
    )
    parser.add_argument(
        "-n", "--max-iterations", type=int, default=None, metavar="N",
        help=f"每个策略允许的最大迭代（工具调用轮数），默认 {Config.MAX_ITERATIONS}",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="实时流式打印模型与压缩过程的输出（默认关闭，以获得更整洁的对比输出）",
    )
    parser.add_argument(
        "--list-strategies", action="store_true",
        help="列出所有可选的压缩策略名称后退出",
    )
    return parser


def main():
    """Main entry point"""
    parser = build_parser()
    args = parser.parse_args()

    if args.list_strategies:
        print("可选的压缩策略（--strategy 的取值）：")
        for alias, strat in STRATEGY_CHOICES.items():
            print(f"  {alias:<16} -> {strat.value}")
        return

    # Apply CLI overrides onto the shared Config
    if args.model:
        Config.MODEL_NAME = args.model
    if args.max_iterations is not None:
        Config.MAX_ITERATIONS = args.max_iterations

    # Resolve which strategies to run
    if args.strategy:
        strategies = [STRATEGY_CHOICES[name] for name in args.strategy]
    else:
        strategies = list(ALL_STRATEGIES)

    # Check configuration
    if not Config.validate():
        print(f"\n{Fore.RED}Configuration validation failed!{Style.RESET_ALL}")
        print("\nPlease set up your .env file with:")
        print("  MOONSHOT_API_KEY=your_api_key_here")
        print("  SERPER_API_KEY=your_api_key_here (optional)")
        sys.exit(1)

    # Print configuration
    Config.print_config()

    # Create runner
    runner = ExperimentRunner(
        Config.MOONSHOT_API_KEY,
        results_file=args.output,
        enable_streaming=args.streaming,
    )

    # Run experiments
    try:
        runner.run_all_strategies(strategies)
        print(f"\n{Fore.GREEN}✅ Experiment completed successfully!{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️ Experiment interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Experiment failed: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
