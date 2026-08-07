import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.hyperbolic_ops import logmap0


class DifferentiableEncodingTree(nn.Module):
    """
    宏观结构熵优化 (Macro-stability Optimization)。
    通过构建可微层次树，实现图结构熵的端到端联合优化。
    """

    def __init__(self, embed_dim: int, tree_depth: int, max_community_width: int, c: float, mu: float = 0.01):
        super(DifferentiableEncodingTree, self).__init__()
        self.tree_depth = tree_depth
        self.max_width = max_community_width
        self.c = c
        self.mu = mu  # 层级分布的体积正则化惩罚系数

        # 为每个层级 l \in {1, ..., H} 初始化多头感知机 phi^{(l)}
        self.community_projectors = nn.ModuleList([
            nn.Linear(embed_dim, max_community_width) for _ in range(tree_depth)
        ])

    def forward(self, h_L: torch.Tensor, degree_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算宏观图结构熵损失 L_{SE}
        h_L: (N, embed_dim) 最后一层的双曲嵌入表示
        degree_matrix: (N, 1) 图的节点度数矩阵 D
        """
        N = h_L.size(0)
        # 将笛卡尔双曲嵌入转换为原点切空间的表示，用以模拟极坐标系下的向径和角坐标分布
        tangent_h = logmap0(h_L, self.c)

        total_vol_G = torch.sum(degree_matrix).clamp_min(1e-6)
        L_SE = 0.0
        Omega_T = 0.0  # L2 体积正则化项

        # 逐层遍历层次树 (l=1 到 H)
        for l in range(self.tree_depth):
            # 公式 (13)：利用多头 softmax 感知机计算软社区分配矩阵 S^{(l)}
            logits = self.community_projectors[l](tangent_h)
            S_l = F.softmax(logits, dim=-1)  # (N, C_l)

            # 聚合层级 l 中每个社区 c 的期望体积 V_c^{(l)}
            # V_c^{(l)} = \sum_{i=1}^N S_{i,c}^{(l)} * D_{ii}
            community_volumes = torch.sum(S_l * degree_matrix, dim=0)  # (C_l,)

            # 计算社区体积的分布比例
            p_c = community_volumes / total_vol_G
            p_c = torch.clamp(p_c, min=1e-7)  # 防止 log(0)

            # 累加标准结构熵： - \sum p_c * log(p_c)
            layer_entropy = -torch.sum(p_c * torch.log(p_c))
            L_SE += layer_entropy

            # 计算体积正则化惩罚，限制过度集中于单一超节点
            Omega_T += torch.sum(p_c ** 2)

        # 公式 (14)：融合正则化惩罚
        L_SE_final = L_SE + self.mu * Omega_T
        return L_SE_final