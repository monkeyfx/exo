import uuid
import time
import asyncio
import json
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Literal, Union, Dict
from aiohttp import web
import aiohttp_cors
import traceback
import signal
from exo import DEBUG, VERSION
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import PrefixDict, shutdown
from exo.inference.tokenizers import resolve_tokenizer
from exo.orchestration import Node
from exo.models import build_base_shard, model_cards, get_repo, pretty_name, get_supported_models
from exo.apputil import create_animation_mp4
from typing import Callable, Optional
import tempfile

# 消息类，表示角色和内容
class Message:
    def __init__(self, role: str, content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]):
        self.role = role  # 消息角色（如用户或助手）
        self.content = content  # 消息内容

    def to_dict(self):
        return {"role": self.role, "content": self.content}  # 转换为字典格式


# 聊天完成请求类，包含模型、消息和温度
class ChatCompletionRequest:
    def __init__(self, model: str, messages: List[Message], temperature: float):
        self.model = model  # 使用的模型
        self.messages = messages  # 消息列表
        self.temperature = temperature  # 温度参数，控制生成的随机性

    def to_dict(self):
        return {
            "model": self.model,
            "messages": [message.to_dict() for message in self.messages],
            "temperature": self.temperature
        }  # 转换为字典格式


# 生成聊天完成的函数
def generate_completion(
    chat_request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    request_id: str,
    tokens: List[int],
    stream: bool,
    finish_reason: Union[Literal["length", "stop"], None],
    object_type: Literal["chat.completion", "text_completion"],
) -> dict:
    # 创建完成的基本结构
    completion = {
        "id": f"chatcmpl-{request_id}",
        "object": object_type,
        "created": int(time.time()),
        "model": chat_request.model,
        "system_fingerprint": f"exo_{VERSION}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": tokenizer.decode(tokens)},
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
    }

    # 如果不是流式响应，添加使用情况
    if not stream:
        completion["usage"] = {
            "prompt_tokens": len(tokenizer.encode(prompt)),  # 计算提示的token数量
            "completion_tokens": len(tokens),  # 计算生成的token数量
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokens),  # 计算总token数量
        }

    choice = completion["choices"][0]
    # 根据对象类型设置返回内容
    if object_type.startswith("chat.completion"):
        key_name = "delta" if stream else "message"  # 根据是否流式选择返回的键名
        choice[key_name] = {"role": "assistant", "content": tokenizer.decode(tokens)}  # 设置助手的消息内容
    elif object_type == "text_completion":
        choice["text"] = tokenizer.decode(tokens)  # 设置文本完成的内容
    else:
        ValueError(f"Unsupported response type: {object_type}")  # 抛出不支持的对象类型错误

    return completion  # 返回生成的完成


# 消息重映射函数，处理消息内容
def remap_messages(messages: List[Message]) -> List[Message]:
    remapped_messages = []  # 存储重映射后的消息
    last_image = None  # 存储最后一张图片的引用
    for message in messages:
        if not isinstance(message.content, list):
            remapped_messages.append(message)  # 如果内容不是列表，直接添加
            continue

        remapped_content = []  # 存储重映射后的内容
        for content in message.content:
            if isinstance(content, dict):
                if content.get("type") in ["image_url", "image"]:
                    # 处理图片内容
                    image_url = content.get("image_url", {}).get("url") or content.get("image")
                    if image_url:
                        last_image = {"type": "image", "image": image_url}
                        remapped_content.append({"type": "text", "text": "[An image was uploaded but is not displayed here]"})
                else:
                    remapped_content.append(content)  # 其他类型内容直接添加
            else:
                remapped_content.append(content)  # 其他类型内容直接添加
        remapped_messages.append(Message(role=message.role, content=remapped_content))  # 添加重映射后的消息

    # 替换最后一张图片的占位符
    if last_image:
        for message in reversed(remapped_messages):
            for i, content in enumerate(message.content):
                if isinstance(content, dict):
                    if content.get("type") == "text" and content.get("text") == "[An image was uploaded but is not displayed here]":
                        message.content[i] = last_image
                        return remapped_messages  # 返回重映射后的消息

    return remapped_messages  # 返回重映射后的消息


# 构建提示的函数
def build_prompt(tokenizer, _messages: List[Message]):
    messages = remap_messages(_messages)  # 重映射消息
    prompt = tokenizer.apply_chat_template([m.to_dict() for m in messages], tokenize=False, add_generation_prompt=True)  # 应用聊天模板
    for message in messages:
        if not isinstance(message.content, list):
            continue

    return prompt  # 返回构建的提示


# 解析消息的函数
def parse_message(data: dict):
    if "role" not in data or "content" not in data:
        raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
    return Message(data["role"], data["content"])  # 返回解析后的消息对象


# 解析聊天请求的函数
def parse_chat_request(data: dict, default_model: str):
    # 解析聊天请求数据，构建ChatCompletionRequest对象
    return ChatCompletionRequest(
        data.get("model", default_model),  # 获取模型，若无则使用默认模型
        [parse_message(msg) for msg in data["messages"]],  # 解析消息列表
        data.get("temperature", 0.0),  # 获取温度参数，控制生成的随机性
    )


# 提示会话类，存储请求ID、时间戳和提示
class PromptSession:
    def __init__(self, request_id: str, timestamp: int, prompt: str):
        self.request_id = request_id  # 请求ID
        self.timestamp = timestamp  # 时间戳
        self.prompt = prompt  # 提示内容


# ChatGPT API 类，处理聊天请求
class ChatGPTAPI:
    def __init__(self, node: Node, inference_engine_classname: str, response_timeout: int = 90, on_chat_completion_request: Callable[[str, ChatCompletionRequest, str], None] = None, default_model: Optional[str] = None):
        self.node = node  # 节点
        self.inference_engine_classname = inference_engine_classname  # 推理引擎类名
        self.response_timeout = response_timeout  # 响应超时时间
        self.on_chat_completion_request = on_chat_completion_request  # 完成请求的回调
        self.app = web.Application(client_max_size=100*1024*1024)  # 创建Web应用，支持最大100MB的上传
        self.prompts: PrefixDict[str, PromptSession] = PrefixDict()  # 存储提示会话
        self.prev_token_lens: Dict[str, int] = {}  # 存储之前的token长度
        self.stream_tasks: Dict[str, asyncio.Task] = {}  # 存储流任务
        self.default_model = default_model or "llama-3.2-1b"  # 默认模型

        # 设置CORS
        cors = aiohttp_cors.setup(self.app)
        cors_options = aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
        # 添加路由和CORS选项
        cors.add(self.app.router.add_get("/models", self.handle_get_models), {"*": cors_options})
        cors.add(self.app.router.add_get("/v1/models", self.handle_get_models), {"*": cors_options})
        cors.add(self.app.router.add_post("/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
        cors.add(self.app.router.add_post("/v1/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
        cors.add(self.app.router.add_post("/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
        cors.add(self.app.router.add_post("/v1/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
        cors.add(self.app.router.add_get("/v1/download/progress", self.handle_get_download_progress), {"*": cors_options})
        cors.add(self.app.router.add_get("/modelpool", self.handle_model_support), {"*": cors_options})
        cors.add(self.app.router.add_get("/healthcheck", self.handle_healthcheck), {"*": cors_options})
        cors.add(self.app.router.add_post("/quit", self.handle_quit), {"*": cors_options})
        cors.add(self.app.router.add_post("/create_animation", self.handle_create_animation), {"*": cors_options})

        # 如果没有编译标志，设置静态文件目录
        if "__compiled__" not in globals():
            self.static_dir = Path(__file__).parent.parent/"tinychat"
            self.app.router.add_get("/", self.handle_root)
            self.app.router.add_static("/", self.static_dir, name="static")

        # 添加中间件
        self.app.middlewares.append(self.timeout_middleware)
        self.app.middlewares.append(self.log_request)

    # 处理退出请求
    async def handle_quit(self, request):
        if DEBUG >= 1: print("Received quit signal")
        response = web.json_response({"detail": "Quit signal received"}, status=200)
        await response.prepare(request)
        await response.write_eof()
        await shutdown(signal.SIGINT, asyncio.get_event_loop(), self.node.server)  # 关闭服务器

    # 超时中间件
    async def timeout_middleware(self, app, handler):
        async def middleware(request):
            try:
                return await asyncio.wait_for(handler(request), timeout=self.response_timeout)  # 设置超时
            except asyncio.TimeoutError:
                return web.json_response({"detail": "Request timed out"}, status=408)  # 返回超时响应

        return middleware

    # 日志请求中间件
    async def log_request(self, app, handler):
        async def middleware(request):
            if DEBUG >= 2: print(f"Received request: {request.method} {request.path}")  # 打印请求日志
            return await handler(request)

        return middleware

    # 处理根请求
    async def handle_root(self, request):
        return web.FileResponse(self.static_dir/"index.html")  # 返回静态文件

    # 健康检查请求
    async def handle_healthcheck(self, request):
        return web.json_response({"status": "ok"})  # 返回健康状态

    # 处理模型支持请求
    async def handle_model_support(self, request):
        return web.json_response({
            "model pool": {
                model_name: pretty_name.get(model_name, model_name)
                for model_name in get_supported_models(self.node.topology_inference_engines_pool)
            }
        })

    # 获取模型请求
    async def handle_get_models(self, request):
        return web.json_response([{"id": model_name, "object": "model", "owned_by": "exo", "ready": True} for model_name, _ in model_cards.items()])  # 返回模型列表

    # 处理聊天token编码请求
    async def handle_post_chat_token_encode(self, request):
        data = await request.json()  # 获取请求数据
        shard = build_base_shard(self.default_model, self.inference_engine_classname)  # 构建基础分片
        messages = [parse_message(msg) for msg in data.get("messages", [])]  # 解析消息
        tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))  # 解析tokenizer
        return web.json_response({"length": len(build_prompt(tokenizer, messages)[0])})  # 返回提示长度

    # 获取下载进度请求
    async def handle_get_download_progress(self, request):
        progress_data = {}
        for node_id, progress_event in self.node.node_download_progress.items():
            if isinstance(progress_event, RepoProgressEvent):
                progress_data[node_id] = progress_event.to_dict()  # 转换为字典格式
            else:
                print(f"Unknown progress event type: {type(progress_event)}. {progress_event}")
        return web.json_response(progress_data)  # 返回进度数据

    # 处理聊天完成请求
    async def handle_post_chat_completions(self, request):
        data = await request.json()  # 获取请求数据
        if DEBUG >= 2: print(f"Handling chat completions request from {request.remote}: {data}")
        stream = data.get("stream", False)  # 获取流式参数
        chat_request = parse_chat_request(data, self.default_model)  # 解析聊天请求
        # 检查模型有效性并设置默认模型
        if chat_request.model and chat_request.model.startswith("gpt-:"):  # 兼容gpt-模型请求
            chat_request.model = self.default_model
        if not chat_request.model or chat_request.model not in model_cards:
            if DEBUG >= 1: print(f"Invalid model: {chat_request.model}. Supported: {list(model_cards.keys())}. Defaulting to {self.default_model}")
            chat_request.model = self.default_model  # 设置默认模型
        shard = build_base_shard(chat_request.model, self.inference_engine_classname)  # 构建基础分片
        if not shard:
            supported_models = [model for model, info in model_cards.items() if self.inference_engine_classname in info.get("repo", {})]
            return web.json_response(
                {"detail": f"Unsupported model: {chat_request.model} with inference engine {self.inference_engine_classname}. Supported models for this engine: {supported_models}"},
                status=400,
            )

        tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))  # 解析tokenizer
        if DEBUG >= 4: print(f"Resolved tokenizer: {tokenizer}")

        prompt = build_prompt(tokenizer, chat_request.messages)  # 构建提示
        request_id = str(uuid.uuid4())  # 生成请求ID
        if self.on_chat_completion_request:
            try:
                self.on_chat_completion_request(request_id, chat_request, prompt)  # 调用回调
            except Exception as e:
                if DEBUG >= 2: traceback.print_exc()
        # request_id = None
        # match = self.prompts.find_longest_prefix(prompt)
        # if match and len(prompt) > len(match[1].prompt):
        #     if DEBUG >= 2:
        #       print(f"Prompt for request starts with previous prompt {len(match[1].prompt)} of {len(prompt)}: {match[1].prompt}")
        #     request_id = match[1].request_id
        #     self.prompts.add(prompt, PromptSession(request_id=request_id, timestamp=int(time.time()), prompt=prompt))
        #     # remove the matching prefix from the prompt
        #     prompt = prompt[len(match[1].prompt):]
        # else:
        #   request_id = str(uuid.uuid4())
        #   self.prompts.add(prompt, PromptSession(request_id=request_id, timestamp=int(time.time()), prompt=prompt))

        callback_id = f"chatgpt-api-wait-response-{request_id}"  # 创建回调ID
        callback = self.node.on_token.register(callback_id)  # 注册回调

        if DEBUG >= 2: print(f"Sending prompt from ChatGPT api {request_id=} {shard=} {prompt=}")

        try:
            await asyncio.wait_for(asyncio.shield(asyncio.create_task(self.node.process_prompt(shard, prompt, request_id=request_id))), timeout=self.response_timeout)  # 处理提示

            if DEBUG >= 2: print(f"Waiting for response to finish. timeout={self.response_timeout}s")

            if stream:
                response = web.StreamResponse(
                    status=200,
                    reason="OK",
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                    },
                )
                await response.prepare(request)  # 准备响应

                async def stream_result(_request_id: str, tokens: List[int], is_finished: bool):
                    prev_last_tokens_len = self.prev_token_lens.get(_request_id, 0)  # 获取之前的token长度
                    self.prev_token_lens[_request_id] = max(prev_last_tokens_len, len(tokens))  # 更新token长度
                    new_tokens = tokens[prev_last_tokens_len:]  # 获取新token
                    finish_reason = None
                    eos_token_id = tokenizer.special_tokens_map.get("eos_token_id") if hasattr(tokenizer, "_tokenizer") and isinstance(tokenizer._tokenizer,
                                                                                                                                 AutoTokenizer) else getattr(tokenizer, "eos_token_id", None)
                    if len(new_tokens) > 0 and new_tokens[-1] == eos_token_id:
                        new_tokens = new_tokens[:-1]  # 去掉结束token
                        if is_finished:
                            finish_reason = "stop"  # 设置完成原因
                    if is_finished and not finish_reason:
                        finish_reason = "length"  # 设置完成原因

                    completion = generate_completion(
                        chat_request,
                        tokenizer,
                        prompt,
                        request_id,
                        new_tokens,
                        stream,
                        finish_reason,
                        "chat.completion",
                    )
                    if DEBUG >= 2: print(f"Streaming completion: {completion}")
                    try:
                        await response.write(f"data: {json.dumps(completion)}\n\n".encode())  # 写入响应
                    except Exception as e:
                        if DEBUG >= 2: print(f"Error streaming completion: {e}")
                        if DEBUG >= 2: traceback.print_exc()

                def on_result(_request_id: str, tokens: List[int], is_finished: bool):
                    if _request_id == request_id: self.stream_tasks[_request_id] = asyncio.create_task(stream_result(_request_id, tokens, is_finished))

                    return _request_id == request_id and is_finished  # 返回是否完成

                _, tokens, _ = await callback.wait(on_result, timeout=self.response_timeout)  # 等待结果
                if request_id in self.stream_tasks:  # 如果还有流任务在运行，等待其完成
                    if DEBUG >= 2: print("Pending stream task. Waiting for stream task to complete.")
                    try:
                        await asyncio.wait_for(self.stream_tasks[request_id], timeout=30)  # 等待流任务
                    except asyncio.TimeoutError:
                        print("WARNING: Stream task timed out. This should not happen.")
                await response.write_eof()  # 结束响应
                return response
            else:
                _, tokens, _ = await callback.wait(
                    lambda _request_id, tokens, is_finished: _request_id == request_id and is_finished,
                    timeout=self.response_timeout,
                )

                finish_reason = "length"  # 设置完成原因
                eos_token_id = tokenizer.special_tokens_map.get("eos_token_id") if isinstance(getattr(tokenizer, "_tokenizer", None), AutoTokenizer) else tokenizer.eos_token_id
                if DEBUG >= 2: print(f"Checking if end of tokens result {tokens[-1]=} is {eos_token_id=}")
                if tokens[-1] == eos_token_id:
                    tokens = tokens[:-1]  # 去掉结束token
                    finish_reason = "stop"  # 设置完成原因

                return web.json_response(generate_completion(chat_request, tokenizer, prompt, request_id, tokens, stream, finish_reason, "chat.completion"))  # 返回完成结果
        except asyncio.TimeoutError:
            return web.json_response({"detail": "Response generation timed out"}, status=408)  # 返回超时响应
        except Exception as e:
            if DEBUG >= 2: traceback.print_exc()
            return web.json_response({"detail": f"Error processing prompt (see logs with DEBUG>=2): {str(e)}"}, status=500)  # 返回错误响应
        finally:
            deregistered_callback = self.node.on_token.deregister(callback_id)  # 注销回调
            if DEBUG >= 2: print(f"Deregister {callback_id=} {deregistered_callback=}")

    # 处理创建动画请求
    async def handle_create_animation(self, request):
        try:
            data = await request.json()  # 获取请求数据
            replacement_image_path = data.get("replacement_image_path")  # 获取替换图片路径
            device_name = data.get("device_name", "Local Device")  # 获取设备名称
            prompt_text = data.get("prompt", "")  # 获取提示文本

            if DEBUG >= 2: print(f"Creating animation with params: replacement_image={replacement_image_path}, device={device_name}, prompt={prompt_text}")

            if not replacement_image_path:
                return web.json_response({"error": "replacement_image_path is required"}, status=400)  # 返回错误响应

            # 创建临时目录
            tmp_dir = Path(tempfile.gettempdir())/"exo_animations"
            tmp_dir.mkdir(parents=True, exist_ok=True)  # 创建目录

            # 生成唯一的输出文件名
            output_filename = f"animation_{uuid.uuid4()}.mp4"
            output_path = str(tmp_dir/output_filename)  # 输出路径

            if DEBUG >= 2: print(f"Animation temp directory: {tmp_dir}, output file: {output_path}, directory exists: {tmp_dir.exists()}, directory permissions: {oct(tmp_dir.stat().st_mode)[-3:]}")

            # 创建动画
            create_animation_mp4(
                replacement_image_path,
                output_path,
                device_name,
                prompt_text
            )

            return web.json_response({
                "status": "success",
                "output_path": output_path  # 返回输出路径
            })

        except Exception as e:
            if DEBUG >= 2: traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)  # 返回错误响应

    # 运行API服务
    async def run(self, host: str = "0.0.0.0", port: int = 52415):
        runner = web.AppRunner(self.app)  # 创建应用运行器
        await runner.setup()  # 设置运行器
        site = web.TCPSite(runner, host, port)  # 创建TCP站点
        await site.start()  # 启动站点
