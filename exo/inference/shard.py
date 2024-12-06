from dataclasses import dataclass, field


@dataclass(frozen=True)# frozen=True 使这个类变成不可变的,类似于元组
class Shard:
    """
    表示模型的一个分片(部分层)的类
    用于在分布式系统中管理模型的不同部分
    """
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int

    def __hash__(self):
        """
        实现哈希方法,使Shard可以用作字典键或集合元素
        返回分片所有属性的组合哈希值
        """
        return hash((self.model_id, self.start_layer, self.end_layer, self.n_layers))

    def is_first_layer(self) -> bool:
        """
        判断该分片是否包含模型的第一层
        Returns:
            bool: 如果start_layer为0则返回True
        """
        return self.start_layer == 0

    def is_last_layer(self) -> bool:
        """
        判断该分片是否包含模型的最后一层
        Returns:
            bool: 如果end_layer等于总层数减1则返回True
        """
        return self.end_layer == self.n_layers - 1

    def get_layer_count(self) -> int:
        """
        计算该分片包含的层数
        Returns:
            int: 分片中包含的层数
        """
        return self.end_layer - self.start_layer + 1

    def to_dict(self) -> dict:
        """
        将分片对象转换为字典格式,用于序列化
        Returns:
            dict: 包含分片所有属性的字典
        """
        return {
            "model_id": self.model_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "n_layers": self.n_layers,
        }

    def from_dict(data: dict) -> 'Shard':
        """
        从字典创建Shard对象的静态方法
        Args:
            data (dict): 包含分片属性的字典
        Returns:
            Shard: 新创建的Shard对象
        """
        return Shard(**data)

    def overlaps(self, other: 'Shard') -> bool:
        """
        检查此分片是否与另一个分片有重叠
        Args:
            other (Shard): 要检查的另一个分片
        Returns:
            bool: 如果有重叠则返回True
        """
        return shards_overlap(self, other)


def shards_overlap(shard1: Shard, shard2: Shard) -> bool:
    """
    检查两个分片是否有重叠的辅助函数
    Args:
        shard1 (Shard): 第一个分片
        shard2 (Shard): 第二个分片
    Returns:
        bool: 如果两个分片属于同一个模型且层范围有重叠则返回True
    """
    return (shard1.model_id == shard2.model_id and 
            max(shard1.start_layer, shard2.start_layer) <= min(shard1.end_layer, shard2.end_layer))
