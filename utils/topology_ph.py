import torch
import torch.nn as nn
from layers.hyperbolic_ops import poincare_dist


class PersistentHomologyDenoising(nn.Module):
    """
    拓扑一致性引导的去噪机制。
    结合 Vietoris-Rips 过滤，识别并惩罚瞬态拓扑噪声。
    """

    def __init__(self, c: float, persistence_threshold: float = 0.40):
        super(PersistentHomologyDenoising, self).__init__()
        self.c = c
        self.eta = persistence_threshold  # 阈值 eta

    def forward(self, h_embeds: torch.Tensor, noise_cycles: list) -> torch.Tensor:
        """
        计算拓扑去噪损失 L_{topo}。
        h_embeds: (N, embed_dim) 双曲节点嵌入。
        noise_cycles: 这是一个包含多个元组的列表，每个元组记录了瞬态噪声特征 \gamma_k 的关键节点对。
                      元组格式为 ((u_k, v_k), (x_k, y_k))，
                      其中 (u_k, v_k) 是触发拓扑特征 Birth 的节点索引，
                      (x_k, y_k) 是触发拓扑特征 Death 的节点索引。
                      注意：该列表在每次 Forward 之前，由 CPU 端调用 Dionysus2 离散计算得出。
        """
        L_topo = 0.0
        if len(noise_cycles) == 0:
            return torch.tensor(0.0, device=h_embeds.device, requires_grad=True)

        # 遍历由 Dionysus2 定位出的瞬态对抗性拓扑特征 \gamma_k \in D(G)
        for (u, v), (x, y) in noise_cycles:
            # 建立连续松弛：计算触发产生和消亡的节点在庞加莱球中的距离
            h_u, h_v = h_embeds[u], h_embeds[v]
            h_x, h_y = h_embeds[x], h_embeds[y]

            # b_k = d_c(h_u, h_v)
            birth_k = poincare_dist(h_u, h_v, self.c)
            # d_k = d_c(h_x, h_y)
            death_k = poincare_dist(h_x, h_y, self.c)

            persistence = death_k - birth_k

            # 公式 (10)：严格的拓扑惩罚函数
            # 仅对寿命小于等于 eta 的特征施加平方惩罚
            if persistence <= self.eta:
                L_topo += (death_k - birth_k) ** 2

        # 公式 (11)：全局拓扑去噪损失聚合
        return L_topo