import os
import sys
import asyncio
from typing import Callable, TypeVar, Optional, Dict, Generic, Tuple, List
import socket
import random
import platform
import psutil
import uuid
import netifaces
from pathlib import Path
import tempfile

# 从环境变量中获取调试级别
DEBUG = int(os.getenv("DEBUG", default="0"))
DEBUG_DISCOVERY = int(os.getenv("DEBUG_DISCOVERY", default="0"))
VERSION = "0.0.1"

# ASCII艺术文本
exo_text = r"""
  _____  _____  
 / _ \ \/ / _ \ 
|  __/>  < (_) |
 \___/_/\_\___/ 
    """


def get_system_info():
  """
  获取系统信息，判断当前系统类型
  Returns:
      系统类型字符串
  """
  if psutil.MACOS:
    if platform.machine() == "arm64":
      return "Apple Silicon Mac"
    if platform.machine() in ["x86_64", "i386"]:
      return "Intel Mac"
    return "Unknown Mac architecture"
  if psutil.LINUX:
    return "Linux"
  return "Non-Mac, non-Linux system"


def find_available_port(host: str = "", min_port: int = 49152, max_port: int = 65535) -> int:
  """
  查找可用的网络端口
  Args:
      host: 主机地址
      min_port: 最小端口号
      max_port: 最大端口号
  Returns:
      可用端口号
  Raises:
      RuntimeError: 如果没有可用端口
  """
  used_ports_file = os.path.join(tempfile.gettempdir(), "exo_used_ports")

  def read_used_ports():
    # 读取已使用的端口列表
    if os.path.exists(used_ports_file):
      with open(used_ports_file, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]
    return []

  def write_used_port(port, used_ports):
    # 将新使用的端口写入文件
    with open(used_ports_file, "w") as f:
      print(used_ports[-19:])
      for p in used_ports[-19:] + [port]:
        f.write(f"{p}\n")

  used_ports = read_used_ports()
  available_ports = set(range(min_port, max_port + 1)) - set(used_ports)

  while available_ports:
    port = random.choice(list(available_ports))
    if DEBUG >= 2: print(f"Trying to find available port {port=}")
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
      write_used_port(port, used_ports)
      return port
    except socket.error:
      available_ports.remove(port)

  raise RuntimeError("No available ports in the specified range")


def print_exo():
  """
  打印exo的ASCII艺术文本
  """
  print(exo_text)


def print_yellow_exo():
  """
  打印黄色的exo ASCII艺术文本
  """
  yellow = "\033[93m"  # ANSI转义码，黄色
  reset = "\033[0m"  # ANSI转义码，重置颜色
  print(f"{yellow}{exo_text}{reset}")


def terminal_link(uri, label=None):
  """
  创建终端可点击链接
  Args:
      uri: 链接地址
      label: 链接标签
  Returns:
      格式化的终端链接字符串
  """
  if label is None:
    label = uri
  parameters = ""

  # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
  escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"

  return escape_mask.format(parameters, uri, label)


T = TypeVar("T")
K = TypeVar("K")


class AsyncCallback(Generic[T]):
  """
  异步回调类，用于处理异步事件
  """
  def __init__(self) -> None:
    self.condition: asyncio.Condition = asyncio.Condition()
    self.result: Optional[Tuple[T, ...]] = None
    self.observers: list[Callable[..., None]] = []

  async def wait(self, check_condition: Callable[..., bool], timeout: Optional[float] = None) -> Tuple[T, ...]:
    """
    等待条件满足
    Args:
        check_condition: 检查条件的回调函数
        timeout: 超时时间
    Returns:
        满足条件的结果
    """
    async with self.condition:
      await asyncio.wait_for(self.condition.wait_for(lambda: self.result is not None and check_condition(*self.result)), timeout)
      assert self.result is not None  # for type checking
      return self.result

  def on_next(self, callback: Callable[..., None]) -> None:
    """
    注册观察者回调
    Args:
        callback: 回调函数
    """
    self.observers.append(callback)

  def set(self, *args: T) -> None:
    """
    设置结果并通知观察者
    Args:
        *args: 结果参数
    """
    self.result = args
    for observer in self.observers:
      observer(*args)
    asyncio.create_task(self.notify())

  async def notify(self) -> None:
    """
    通知所有等待的协程
    """
    async with self.condition:
      self.condition.notify_all()


class AsyncCallbackSystem(Generic[K, T]):
  """
  异步回调系统，管理多个异步回调
  """
  def __init__(self) -> None:
    self.callbacks: Dict[K, AsyncCallback[T]] = {}

  def register(self, name: K) -> AsyncCallback[T]:
    """
    注册新的回调
    Args:
        name: 回调名称
    Returns:
        注册的回调对象
    """
    if name not in self.callbacks:
      self.callbacks[name] = AsyncCallback[T]()
    return self.callbacks[name]

  def deregister(self, name: K) -> None:
    """
    注销回调
    Args:
        name: 回调名称
    """
    if name in self.callbacks:
      del self.callbacks[name]

  def trigger(self, name: K, *args: T) -> None:
    """
    触发回调
    Args:
        name: 回调名称
        *args: 回调参数
    """
    if name in self.callbacks:
      self.callbacks[name].set(*args)

  def trigger_all(self, *args: T) -> None:
    """
    触发所有回调
    Args:
        *args: 回调参数
    """
    for callback in self.callbacks.values():
      callback.set(*args)


K = TypeVar('K', bound=str)
V = TypeVar('V')


class PrefixDict(Generic[K, V]):
  """
  前缀字典类，用于存储键值对并支持前缀查找
  """
  def __init__(self):
    self.items: Dict[K, V] = {}

  def add(self, key: K, value: V) -> None:
    """
    添加键值对
    Args:
        key: 键
        value: 值
    """
    self.items[key] = value

  def find_prefix(self, argument: str) -> List[Tuple[K, V]]:
    """
    查找具有指定前缀的键值对
    Args:
        argument: 前缀字符串
    Returns:
        匹配的键值对列表
    """
    return [(key, value) for key, value in self.items.items() if argument.startswith(key)]

  def find_longest_prefix(self, argument: str) -> Optional[Tuple[K, V]]:
    """
    查找具有最长前缀的键值对
    Args:
        argument: 前缀字符串
    Returns:
        匹配的键值对
    """
    matches = self.find_prefix(argument)
    if len(matches) == 0:
      return None

    return max(matches, key=lambda x: len(x[0]))


def is_valid_uuid(val):
  """
  检查字符串是否为有效的UUID
  Args:
      val: 待检查的字符串
  Returns:
      是否为有效UUID
  """
  try:
    uuid.UUID(str(val))
    return True
  except ValueError:
    return False


def get_or_create_node_id():
  """
  获取或创建节点ID
  Returns:
      节点ID字符串
  """
  NODE_ID_FILE = Path(tempfile.gettempdir())/".exo_node_id"
  try:
    if NODE_ID_FILE.is_file():
      with open(NODE_ID_FILE, "r") as f:
        stored_id = f.read().strip()
      if is_valid_uuid(stored_id):
        if DEBUG >= 2: print(f"Retrieved existing node ID: {stored_id}")
        return stored_id
      else:
        if DEBUG >= 2: print("Stored ID is not a valid UUID. Generating a new one.")

    new_id = str(uuid.uuid4())
    with open(NODE_ID_FILE, "w") as f:
      f.write(new_id)

    if DEBUG >= 2: print(f"Generated and stored new node ID: {new_id}")
    return new_id
  except IOError as e:
    if DEBUG >= 2: print(f"IO error creating node_id: {e}")
    return str(uuid.uuid4())
  except Exception as e:
    if DEBUG >= 2: print(f"Unexpected error creating node_id: {e}")
    return str(uuid.uuid4())


def pretty_print_bytes(size_in_bytes: int) -> str:
  """
  格式化输出字节大小
  Args:
      size_in_bytes: 字节数
  Returns:
      格式化的字符串
  """
  if size_in_bytes < 1024:
    return f"{size_in_bytes} B"
  elif size_in_bytes < 1024**2:
    return f"{size_in_bytes / 1024:.2f} KB"
  elif size_in_bytes < 1024**3:
    return f"{size_in_bytes / (1024 ** 2):.2f} MB"
  elif size_in_bytes < 1024**4:
    return f"{size_in_bytes / (1024 ** 3):.2f} GB"
  else:
    return f"{size_in_bytes / (1024 ** 4):.2f} TB"


def pretty_print_bytes_per_second(bytes_per_second: int) -> str:
  """
  格式化输出每秒字节数
  Args:
      bytes_per_second: 每秒字节数
  Returns:
      格式化的字符串
  """
  if bytes_per_second < 1024:
    return f"{bytes_per_second} B/s"
  elif bytes_per_second < 1024**2:
    return f"{bytes_per_second / 1024:.2f} KB/s"
  elif bytes_per_second < 1024**3:
    return f"{bytes_per_second / (1024 ** 2):.2f} MB/s"
  elif bytes_per_second < 1024**4:
    return f"{bytes_per_second / (1024 ** 3):.2f} GB/s"
  else:
    return f"{bytes_per_second / (1024 ** 4):.2f} TB/s"


def get_all_ip_addresses():
  """
  获取所有网络接口的IP地址
  Returns:
      IP地址列表
  """
  try:
    ip_addresses = []
    for interface in netifaces.interfaces():
      ifaddresses = netifaces.ifaddresses(interface)
      if netifaces.AF_INET in ifaddresses:
        for link in ifaddresses[netifaces.AF_INET]:
          ip = link['addr']
          ip_addresses.append(ip)
    return list(set(ip_addresses))
  except:
    if DEBUG >= 1: print("Failed to get all IP addresses. Defaulting to localhost.")
    return ["localhost"]


async def shutdown(signal, loop, server):
  """
  优雅地关闭服务器并关闭异步循环
  Args:
      signal: 接收到的信号
      loop: 异步事件循环
      server: 服务器对象
  """
  print(f"Received exit signal {signal.name}...")
  print("Thank you for using exo.")
  print_yellow_exo()
  server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
  [task.cancel() for task in server_tasks]
  print(f"Cancelling {len(server_tasks)} outstanding tasks")
  await asyncio.gather(*server_tasks, return_exceptions=True)
  await server.stop()


def is_frozen():
  """
  检查程序是否被打包为可执行文件
  Returns:
      是否为打包状态
  """
  return getattr(sys, 'frozen', False) or os.path.basename(sys.executable) == "exo" \
    or ('Contents/MacOS' in str(os.path.dirname(sys.executable))) \
    or '__nuitka__' in globals() or getattr(sys, '__compiled__', False)