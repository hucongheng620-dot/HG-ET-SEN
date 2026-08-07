import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入后续将逐一实现的自定义模块
from layers.hyperbolic_ops import expmap0, logmap0
from layers.hyp_attention import HyperbolicGraphAttentionLayer
from utils.structural_entropy import DifferentiableEncodingTree
from utils.topology_ph import PersistentHomologyDenoising


class HGETSEN(nn.Module):
    """
    HG-ET-SEN: Hyperbolic Graph Neural Networks with Entropy of Encoding Trees
    for Network Intrusion Detection.
    """

    def __init__(self,
                 feature_dim: int,
                 embed_dim: int = 64,
                 num_layers: int = 3,
                 c: float = 1.05,
                 tree_depth: int = 3,
                 max_community_width: int = 10,
                 alpha: float = 0.25,
                 gamma_pos: float = 2.0,
                 gamma_neg: float = 2.0):
        super(HGETSEN, self).__init__()

        # Table 2 超参数初始化
        self.c = c  # 庞加莱球的负曲率 (c=1.05)
        self.embed_dim = embed_dim  # 嵌入维度 (64)
        self.num_layers = num_layers  # 网络层数 L=3
        self.tree_depth = tree_depth  # 树深度 H=3

        # Asymmetric Focal Loss 参数
        self.alpha = alpha
        self.gamma_pos = gamma_pos  # 对应论文中的 rho
        self.gamma_neg = gamma_neg  # 对应论文中的 beta

        # 1. 双曲图注意力层 (L 层)
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(HyperbolicGraphAttentionLayer(feature_dim, embed_dim, self.c))
        for _ in range(num_layers - 1):
            self.gnn_layers.append(HyperbolicGraphAttentionLayer(embed_dim, embed_dim, self.c))

        # 2. 宏观稳定性优化：可微编码树
        self.encoding_tree = DifferentiableEncodingTree(
            embed_dim=embed_dim,
            tree_depth=tree_depth,
            max_community_width=max_community_width,
            c=self.c
        )

        # 3. 拓扑去噪模块：持久同调代理梯度计算
        self.ph_denoising = PersistentHomologyDenoising(c=self.c, persistence_threshold=0.40)

        # 4. 异常检测 MLP 分类器 (边预测任务)
        # 将源节点和目的节点的正切空间投影拼接，输入到全连接层中
        self.edge_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),  # dropout rate = 0.5
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        前向传播函数
        x: 初始欧几里得节点特征矩阵 X (L2归一化后)
        adj: 动态加权邻接矩阵 A
        """
        # (b) 双曲流形嵌入：利用 exp_0^c 将欧几里得特征投影到庞加莱球中
        h = expmap0(x, self.c)

        # 执行多层双曲注意力聚合，获取低失真几何表示矩阵 Z_H
        for layer in self.gnn_layers:
            h = layer(h, adj)

        return h

    def detect_intrusion(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        基于学习到的双曲嵌入，计算边 (i, j) 的异常概率。
        """
        # 将节点在庞加莱球中的最终双曲嵌入映射回正切空间
        v_i = logmap0(h_i, self.c)
        v_j = logmap0(h_j, self.c)

        # 拼接正切空间投影
        cat_features = torch.cat([v_i, v_j], dim=-1)

        # MLP 输出预测的异常概率
        prob = self.edge_predictor(cat_features)
        return prob

    def compute_asymmetric_focal_loss(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        非对称焦点交叉熵损失 L_{FL}
        解耦焦点参数，严格惩罚漏报，平滑控制误报率。
        """
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

        # 对应公式 (16)
        loss_pos = -labels * self.alpha * torch.pow(1 - probs, self.gamma_neg) * torch.log(probs)
        loss_neg = -(1 - labels) * (1 - self.alpha) * torch.pow(probs, self.gamma_pos) * torch.log(1 - probs)

        return torch.sum(loss_pos + loss_neg)