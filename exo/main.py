# 导入必要的标准库
import argparse  # 用于解析命令行参数
import asyncio   # 用于异步编程支持
import atexit   # 用于注册程序退出时的清理函数
import signal   # 用于处理系统信号
import json     # 用于JSON数据处理
import logging  # 用于日志记录
import platform # 用于获取系统平台信息
import os       # 用于操作系统相关操作
import sys      # 用于系统相关功能
import time     # 用于时间相关操作
import traceback # 用于异常追踪
import uuid     # 用于生成唯一标识符

# 导入项目自定义模块
from exo.networking.manual.manual_discovery import ManualDiscovery  # 手动发现模块
from exo.networking.manual.network_topology_config import NetworkTopology  # 网络拓扑配置
from exo.orchestration.standard_node import StandardNode  # 标准节点实现
from exo.networking.grpc.grpc_server import GRPCServer  # gRPC服务器
from exo.networking.udp.udp_discovery import UDPDiscovery  # UDP发现模块
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery  # Tailscale发现模块
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle  # gRPC对等节点处理
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy  # 环形内存加权分区策略
from exo.api import ChatGPTAPI  # ChatGPT API实现
from exo.download.shard_download import ShardDownloader, RepoProgressEvent, NoopShardDownloader  # 分片下载相关类
from exo.download.hf.hf_shard_download import HFShardDownloader  # HuggingFace分片下载器
from exo.helpers import (  # 辅助函数
    print_yellow_exo, find_available_port, DEBUG, get_system_info,
    get_or_create_node_id, get_all_ip_addresses, terminal_link, shutdown
)
from exo.inference.shard import Shard  # 推理分片类
from exo.inference.inference_engine import get_inference_engine, InferenceEngine  # 推理引擎
from exo.inference.tokenizers import resolve_tokenizer  # 分词器解析
from exo.orchestration.node import Node  # 节点基类
from exo.models import build_base_shard, get_repo  # 模型相关函数
from exo.viz.topology_viz import TopologyViz  # 拓扑可视化
from exo.download.hf.hf_helpers import (  # HuggingFace辅助函数
    has_hf_home_read_access, has_hf_home_write_access,
    get_hf_home, move_models_to_hf
)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")

# 基本命令和模型参数
parser.add_argument("command", nargs="?", choices=["run"], help="要执行的命令")
parser.add_argument("model_name", nargs="?", help="要运行的模型名称")
parser.add_argument("--default-model", type=str, default=None, help="默认使用的模型")
parser.add_argument("--node-id", type=str, default=None, help="节点的唯一标识符")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="节点监听的主机地址")
parser.add_argument("--node-port", type=int, default=None, help="节点监听的端口号")
parser.add_argument("--models-seed-dir", type=str, default=None, help="模型种子目录的路径")

# 网络和发现相关参数
parser.add_argument("--listen-port", type=int, default=5678, help="节点发现服务的监听端口")
parser.add_argument("--broadcast-port", type=int, default=5678, help="节点发现服务的广播端口")
parser.add_argument("--discovery-module", type=str, choices=["udp", "tailscale", "manual"], default="udp", help="使用的节点发现模块类型")
parser.add_argument("--discovery-timeout", type=int, default=30, help="节点发现超时时间（秒）")
parser.add_argument("--discovery-config-path", type=str, default=None, help="手动发现模式的配置文件路径")
parser.add_argument("--wait-for-peers", type=int, default=0, help="启动前等待连接的对等节点数量")

# 下载和性能相关参数
parser.add_argument("--download-quick-check", action="store_true", help="快速检查本地是否已有模型分片")
parser.add_argument("--max-parallel-downloads", type=int, default=4, help="最大并行下载分片数")
parser.add_argument("--prometheus-client-port", type=int, default=None, help="Prometheus监控客户端端口")

# API和推理相关参数
parser.add_argument("--chatgpt-api-port", type=int, default=52415, help="ChatGPT API服务端口")
parser.add_argument("--chatgpt-api-response-timeout", type=int, default=90, help="ChatGPT API响应超时时间（秒）")
parser.add_argument("--max-generate-tokens", type=int, default=10000, help="单次请求最大生成的token数量")
parser.add_argument("--inference-engine", type=str, default=None, help="使用的推理引擎类型（mlx、tinygrad或dummy）")
parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, help="是否禁用终端用户界面")
parser.add_argument("--run-model", type=str, help="直接运行指定的模型")
parser.add_argument("--prompt", type=str, help="使用--run-model时的提示文本", default="Who are you?")
parser.add_argument("--default-temp", type=float, help="token采样的默认温度值", default=0.0)

# Tailscale相关参数
parser.add_argument("--tailscale-api-key", type=str, default=None, help="Tailscale API密钥")
parser.add_argument("--tailnet-name", type=str, default=None, help="Tailscale网络名称")
parser.add_argument("--node-id-filter", type=str, default=None, help="允许连接的节点ID列表（逗号分隔，仅用于UDP和Tailscale发现模式）")
# 解析命令行参数并打印选择的推理引擎
args = parser.parse_args()
print(f"Selected inference engine: {args.inference_engine}")

# 打印程序标识
print_yellow_exo()

# 获取并打印系统信息
system_info = get_system_info()
print(f"Detected system: {system_info}")

# 根据推理引擎类型选择合适的分片下载器
# 如果是dummy引擎则使用空操作下载器，否则使用HuggingFace下载器
shard_downloader: ShardDownloader = HFShardDownloader(
    quick_check=args.download_quick_check,
    max_parallel_downloads=args.max_parallel_downloads
) if args.inference_engine != "dummy" else NoopShardDownloader()

# 确定使用的推理引擎名称
# 如果未指定，则在Apple Silicon Mac上默认使用mlx，其他平台使用tinygrad
inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
print(f"Inference engine name after selection: {inference_engine_name}")

# 初始化推理引擎实例
inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")

# 如果未指定节点端口，则自动查找可用端口
if args.node_port is None:
    args.node_port = find_available_port(args.node_host)
    if DEBUG >= 1: print(f"Using available port: {args.node_port}")

# 获取或创建节点ID
args.node_id = args.node_id or get_or_create_node_id()

# 构建ChatGPT API端点和Web聊天URL列表
chatgpt_api_endpoints = [f"http://{ip}:{args.chatgpt_api_port}/v1/chat/completions" for ip in get_all_ip_addresses()]
web_chat_urls = [f"http://{ip}:{args.chatgpt_api_port}" for ip in get_all_ip_addresses()]

# 打印接口信息（如果调试级别允许）
if DEBUG >= 0:
    print("Chat interface started:")
    for web_chat_url in web_chat_urls:
        print(f" - {terminal_link(web_chat_url)}")
    print("ChatGPT API endpoint served at:")
    for chatgpt_api_endpoint in chatgpt_api_endpoints:
        print(f" - {terminal_link(chatgpt_api_endpoint)}")

# 将节点ID过滤器字符串转换为列表（如果提供）
allowed_node_ids = args.node_id_filter.split(',') if args.node_id_filter else None

# 根据选择的发现模块类型初始化相应的发现服务
if args.discovery_module == "udp":
    # 初始化UDP发现服务
    discovery = UDPDiscovery(
        args.node_id,
        args.node_port,
        args.listen_port,
        args.broadcast_port,
        lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities),
        discovery_timeout=args.discovery_timeout,
        allowed_node_ids=allowed_node_ids
    )
elif args.discovery_module == "tailscale":
    # 初始化Tailscale发现服务
    discovery = TailscaleDiscovery(
        args.node_id,
        args.node_port,
        lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities),
        discovery_timeout=args.discovery_timeout,
        tailscale_api_key=args.tailscale_api_key,
        tailnet=args.tailnet_name,
        allowed_node_ids=allowed_node_ids
    )
elif args.discovery_module == "manual":
    # 初始化手动发现服务
    if not args.discovery_config_path:
        raise ValueError(f"--discovery-config-path is required when using manual discovery. Please provide a path to a config json file.")
    discovery = ManualDiscovery(
        args.discovery_config_path,
        args.node_id,
        create_peer_handle=lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities)
    )

# 初始化拓扑可视化（如果未禁用TUI）
topology_viz = TopologyViz(
    chatgpt_api_endpoints=chatgpt_api_endpoints,
    web_chat_urls=web_chat_urls
) if not args.disable_tui else None

# 创建标准节点实例
node = StandardNode(
    args.node_id,
    None,  # server将在后面设置
    inference_engine,
    discovery,
    partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
    max_generate_tokens=args.max_generate_tokens,
    topology_viz=topology_viz,
    shard_downloader=shard_downloader,
    default_sample_temperature=args.default_temp
)

# 创建gRPC服务器并将其关联到节点
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server

# 初始化ChatGPT API服务
api = ChatGPTAPI(
    node,
    inference_engine.__class__.__name__,
    response_timeout=args.chatgpt_api_response_timeout,
    on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None,
    default_model=args.default_model
)

# 注册token更新的回调函数，用于更新拓扑可视化
node.on_token.register("update_topology_viz").on_next(
    lambda req_id, tokens, __: topology_viz.update_prompt_output(req_id, inference_engine.tokenizer.decode(tokens))
    if topology_viz and hasattr(inference_engine, "tokenizer") else None
)

# 定义预先启动下载的函数
def preemptively_start_download(request_id: str, opaque_status: str):
    """
    预先开始下载模型分片的函数
    Args:
        request_id: 请求ID
        opaque_status: 包含状态信息的JSON字符串
    """
    try:
        status = json.loads(opaque_status)  # 解析状态信息
        if status.get("type") == "node_status" and status.get("status") == "start_process_prompt":
            current_shard = node.get_current_shard(Shard.from_dict(status.get("shard")))  # 获取当前分片
            if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")  # 调试信息
            asyncio.create_task(shard_downloader.ensure_shard(current_shard, inference_engine.__class__.__name__))  # 确保下载分片
    except Exception as e:
        if DEBUG >= 2:
            print(f"Failed to preemptively start download: {e}")  # 错误信息
            traceback.print_exc()  # 打印堆栈跟踪

# 注册状态更新的回调函数
node.on_opaque_status.register("start_download").on_next(preemptively_start_download)  # 注册回调函数

# 如果指定了prometheus_client_port，则启动指标服务器
if args.prometheus_client_port:
    from exo.stats.metrics import start_metrics_server  # 导入指标服务器启动函数
    start_metrics_server(node, args.prometheus_client_port)  # 启动指标服务器

# 用于控制广播频率的时间戳
last_broadcast_time = 0

def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
    """
    节流广播函数，控制广播频率
    Args:
        shard: 模型分片
        event: 仓库进度事件
    """
    global last_broadcast_time  # 使用全局变量
    current_time = time.time()  # 获取当前时间
    # 仅在完成时或距离上次广播超过0.1秒时才广播
    if event.status == "complete" or current_time - last_broadcast_time >= 0.1:
        last_broadcast_time = current_time  # 更新最后广播时间
        asyncio.create_task(node.broadcast_opaque_status("", json.dumps({
            "type": "download_progress",  # 广播类型
            "node_id": node.id,  # 节点ID
            "progress": event.to_dict()  # 进度信息
        })))  # 广播进度信息

# 注册进度广播的回调函数
shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)  # 注册回调函数

async def run_model_cli(node: Node, inference_engine: InferenceEngine, model_name: str, prompt: str):
    inference_class = inference_engine.__class__.__name__  # 获取推理引擎类名
    shard = build_base_shard(model_name, inference_class)  # 构建基础分片
    if not shard:
        print(f"Error: Unsupported model '{model_name}' for inference engine {inference_engine.__class__.__name__}")  # 错误信息
        return
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))  # 解析分片的分词器
    request_id = str(uuid.uuid4())  # 生成请求ID
    callback_id = f"cli-wait-response-{request_id}"  # 回调ID
    callback = node.on_token.register(callback_id)  # 注册回调
    if topology_viz:
        topology_viz.update_prompt(request_id, prompt)  # 更新拓扑可视化
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)  # 应用聊天模板

    try:
        print(f"Processing prompt: {prompt}")  # 打印处理提示
        await node.process_prompt(shard, prompt, request_id=request_id)  # 处理提示

        _, tokens, _ = await callback.wait(lambda _request_id, tokens, is_finished: _request_id == request_id and is_finished, timeout=300)  # 等待回调

        print("\nGenerated response:")  # 打印生成的响应
        print(tokenizer.decode(tokens))  # 解码并打印tokens
    except Exception as e:
        print(f"Error processing prompt: {str(e)}")  # 错误信息
        traceback.print_exc()  # 打印堆栈跟踪
    finally:
        node.on_token.deregister(callback_id)  # 注销回调

def clean_path(path):
    """Clean and resolve path"""
    if path.startswith("Optional("):
        path = path.strip('Optional("').rstrip('")')  # 清理路径
    return os.path.expanduser(path)  # 扩展用户目录

async def main():
    loop = asyncio.get_running_loop()  # 获取事件循环

    # Check HuggingFace directory permissions
    hf_home, has_read, has_write = get_hf_home(), await has_hf_home_read_access(), await has_hf_home_write_access()  # 检查权限
    if DEBUG >= 1: print(f"Model storage directory: {hf_home}")  # 打印模型存储目录
    print(f"{has_read=}, {has_write=}")  # 打印权限信息
    if not has_read or not has_write:
        print(f"""
              WARNING: Limited permissions for model storage directory: {hf_home}.
              This may prevent model downloads from working correctly.
              {"❌ No read access" if not has_read else ""}
              {"❌ No write access" if not has_write else ""}
              """)
    
    if not args.models_seed_dir is None:
        try:
            models_seed_dir = clean_path(args.models_seed_dir)  # 清理模型种子目录路径
            await move_models_to_hf(models_seed_dir)  # 移动模型到HuggingFace目录
        except Exception as e:
            print(f"Error moving models to .cache/huggingface: {e}")  # 错误信息

    def restore_cursor():
        if platform.system() != "Windows":
            os.system("tput cnorm")  # 显示光标

    # Restore the cursor when the program exits
    atexit.register(restore_cursor)  # 注册退出时恢复光标的函数

    # Use a more direct approach to handle signals
    def handle_exit():
        asyncio.ensure_future(shutdown(signal.SIGTERM, loop, node.server))  # 处理退出信号

    if platform.system() != "Windows":
        for s in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(s, handle_exit)  # 注册信号处理

    await node.start(wait_for_peers=args.wait_for_peers)  # 启动节点

    if args.command == "run" or args.run_model:
        model_name = args.model_name or args.run_model  # 获取模型名称
        if not model_name:
            print("Error: Model name is required when using 'run' command or --run-model")  # 错误信息
            return
        await run_model_cli(node, inference_engine, model_name, args.prompt)  # 运行模型CLI
    else:
        asyncio.create_task(api.run(port=args.chatgpt_api_port))  # 启动API服务器作为非阻塞任务
        await asyncio.Event().wait()  # 等待事件

def run():
    loop = asyncio.new_event_loop()  # 创建新的事件循环
    asyncio.set_event_loop(loop)  # 设置事件循环
    try:
        loop.run_until_complete(main())  # 运行主函数
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Shutting down...")  # 捕获键盘中断
    finally:
        loop.run_until_complete(shutdown(signal.SIGTERM, loop, node.server))  # 关闭服务器
        loop.close()  # 关闭事件循环

if __name__ == "__main__":
    run()  # 运行程序