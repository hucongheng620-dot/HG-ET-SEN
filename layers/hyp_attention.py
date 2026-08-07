import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hyperbolic_ops import mobius_add, expmap0, logmap0


class HyperbolicGraphAttentionLayer(nn.Module):
    """
    双曲图注意力聚合层 (Hyperbolic Graph Attention Layer)。
    用于在具有规模无标度特性 (scale-free) 的树状网络中无失真地聚合邻居信息。
    """

    def __init__(self, in_features: int, out_features: int, c: float, alpha_leaky: float = 0.2):
        super(HyperbolicGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        # 针对每一层的可训练权重 W 和 偏置 b
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.b = nn.Parameter(torch.zeros(out_features))

        # 可学习的注意力向量 a^(l)
        self.a = nn.Linear(out_features * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha_leaky)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, h_prev: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算。
        h_prev: (N, in_features) 当前层的双曲特征
        adj: (N, N) 动态加权邻接矩阵 A
        """
        N = h_prev.size(0)

        # 1. 线性变换：映射到切空间，进行欧几里得线性变换后，投影回双曲流形并加上偏置
        # 公式 (6): h'_{i} = exp_0^c (W * log_0^c(h_{i})) \oplus_c b
        h_tangent = logmap0(h_prev, self.c)
        h_transformed = self.W(h_tangent)

        h_prime = expmap0(h_transformed, self.c)

        # 为了应用莫比乌斯偏置，需要将偏置参数先指数映射到庞加莱球
        # 注意：此处为简化实现，直接将欧式偏置当作切向量
        bias_hyperbolic = expmap0(self.b.unsqueeze(0).expand(N, -1), self.c)
        h_prime = mobius_add(h_prime, bias_hyperbolic, self.c)

        # 2. 计算双曲注意力分数
        # 将 h_prime 映射回正切空间用于拼接和注意力计算
        v_prime = logmap0(h_prime, self.c)

        # 构建节点对拼接矩阵 (N, N, 2 * out_features)
        v_prime_repeat = v_prime.repeat(1, N).view(N, N, 2 * self.out_features)
        v_prime_repeat_t = v_prime_repeat.transpose(0, 1)

        # 拼接后的特征对
        cat_features = torch.cat([v_prime_repeat, v_prime_repeat_t], dim=-1)

        # 公式 (7): e_{i,j} = LeakyReLU(a^T [log_0^c(h'_i) || log_0^c(h'_j)])
        e = self.leaky_relu(self.a(cat_features).squeeze(2))

        # 利用邻接矩阵进行 Mask，掩盖不存在通信边的注意力分数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # 获取归一化注意力系数 alpha_{i,j}
        attention = F.softmax(attention, dim=1)

        # 防止完全断开的节点在 softmax 后产生 NaN
        attention = torch.nan_to_num(attention, nan=0.0)

        # 3. 邻居聚合：在统一的正切空间中进行信息加权聚合以保证几何有效性
        # 公式 (8): m_i = sum (alpha_{i,j} * log_0^c(h'_j))
        m = torch.matmul(attention, v_prime)

        # 4. 非线性激活与投影回双曲空间
        # h_{i} = exp_0^c( sigma(m_i) )，此处选用 ReLU 激活
        m_activated = F.relu(m)
        h_new = expmap0(m_activated, self.c)

        return h_new