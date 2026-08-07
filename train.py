import torch
import torch.optim as optim
import numpy as np

from models.hg_et_sen import HGETSEN
from utils.graph_builder import normalize_features, build_weighted_adjacency


# 假设 Dionysus2 的离散拓扑特征提取已封装为外部接口
# from external.dionysus_interface import extract_transient_noise_cycles

def train_hg_et_sen(
        raw_node_features: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor,  # 边异常标签，1为异常，0为正常
        epochs: int = 100
):
    """
    HG-ET-SEN 框架的端到端训练程序 (遵循论文 Algorithm 1)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 属性网络流量图构建 (Attributed Network Traffic Graph Construction)
    num_nodes = raw_node_features.size(0)
    # 特征 L2 归一化防梯度爆炸
    X = normalize_features(raw_node_features).to(device)
    # 构建动态加权邻接矩阵，tau=0.2
    A = build_weighted_adjacency(X, edge_index, num_nodes, tau=0.2).to(device)
    labels = labels.to(device)

    # 2. 模型初始化与优化器配置
    # 严格按照表 2 (Table 2) 的最优超参数设置
    model = HGETSEN(
        feature_dim=X.size(1),
        embed_dim=64,
        num_layers=3,
        c=1.05,
        tree_depth=3,
        alpha=0.25,  # Asymmetric focal loss param
        gamma_pos=2.0,  # rho
        gamma_neg=2.0  # beta
    ).to(device)

    # 使用 Adam 优化器，学习率为 0.05
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # 定义联合优化权衡系数
    lambda_1 = 0.1  # 宏观结构熵权重 (Entropy weight)
    lambda_2 = 0.05  # 微观拓扑去噪权重 (Topo weight)

    # 预计算度数矩阵用于结构熵 (简化为无权度数或加权度数)
    degree_matrix = torch.sum(A, dim=1, keepdim=True)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- 步骤 A: 双曲流形嵌入 ---
        # 执行 L 层双曲注意力聚合，获取低失真几何表示矩阵 Z_H
        h_embeds = model(X, A)

        # --- 步骤 B: 拓扑去噪 (Persistent Homology) ---
        # 离散计算 Vietoris-Rips 过滤，获取寿命 <= eta 的瞬态噪声循环
        # 注意: 实际工程中需在 CPU 异步调用 Dionysus2，此处模拟返回的临界节点对
        # noise_cycles = extract_transient_noise_cycles(h_embeds.detach().cpu().numpy(), eta=0.40)
        noise_cycles = []  # 占位符

        L_topo = model.ph_denoising(h_embeds, noise_cycles)

        # --- 步骤 C: 宏观结构熵优化 (Macro-stability Optimization) ---
        L_SE = model.encoding_tree(h_embeds, degree_matrix)

        # --- 步骤 D: 异常检测与预测 (Intrusion Detection) ---
        src, dst = edge_index[0], edge_index[1]
        probs = model.detect_intrusion(h_embeds[src], h_embeds[dst]).squeeze()

        L_FL = model.compute_asymmetric_focal_loss(probs, labels)

        # --- 步骤 E: 总体损失函数联合优化 ---
        # 对应论文公式 (17): L_{Final} = L_{FL} + \lambda_1 L_{SE} + \lambda_2 L_{topo}
        L_final = L_FL + lambda_1 * L_SE + lambda_2 * L_topo

        L_final.backward()

        # 梯度裁剪以进一步保护双曲空间的数值稳定性
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | Total Loss: {L_final.item():.4f} "
                  f"(FL: {L_FL.item():.4f}, SE: {L_SE.item():.4f}, Topo: {L_topo.item():.4f})")

    return model


if __name__ == "__main__":
    # 模拟输入运行测试
    print("Initializing HG-ET-SEN Training Pipeline...")
    dummy_x = torch.rand((100, 42))  # 100个节点，42维原始特征
    dummy_edges = torch.randint(0, 100, (2, 500))  # 500条通信边
    dummy_labels = torch.randint(0, 2, (500,)).float()

    trained_model = train_hg_et_sen(dummy_x, dummy_edges, dummy_labels, epochs=20)
    print("Training Completed Successfully.")