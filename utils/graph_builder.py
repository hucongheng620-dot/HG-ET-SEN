import torch


def normalize_features(raw_features: torch.Tensor, delta: float = 1e-6) -> torch.Tensor:
    """
    对原始流量统计特征进行严格的 L2 归一化。
    实现论文公式 (2)：x_i = x_i^{raw} / (||x_i^{raw}||_2 + delta)
    这能严格防止在后续的非欧几里得指数映射中出现梯度爆炸。
    """
    norm = torch.norm(raw_features, p=2, dim=-1, keepdim=True)
    return raw_features / (norm + delta)


def build_weighted_adjacency(features: torch.Tensor, edge_index: torch.Tensor, num_nodes: int,
                             tau: float = 0.2) -> torch.Tensor:
    """
    根据特征相似度和拓扑连通性构建动态加权邻接矩阵。
    实现论文公式 (3)：A_{i,j} = exp(-||x_i - x_j||_2^2 / tau) * I(e_{i,j} \in E)
    tau 为控制拓扑分布平滑度的温度超参数。
    """
    # 初始化全零的稠密邻接矩阵
    A = torch.zeros((num_nodes, num_nodes), device=features.device)

    # 获取通信边的源节点和目的节点索引
    src, dst = edge_index[0], edge_index[1]

    # 计算存在通信边的节点间的欧式距离平方
    dist_sq = torch.norm(features[src] - features[dst], p=2, dim=-1) ** 2

    # 计算特征相似度权重
    weights = torch.exp(-dist_sq / tau)

    # 利用指示函数 I(e_{i,j} \in E) 进行赋值
    A[src, dst] = weights

    # 针对无向/双向流处理，若有需要可进行对称化：A = torch.maximum(A, A.T)
    return A