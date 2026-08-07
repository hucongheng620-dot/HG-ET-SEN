import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_performance(model, x_test: torch.Tensor, adj_test: torch.Tensor,
                         edge_index_test: torch.Tensor, y_true: np.ndarray,
                         threshold: float = 0.5):
    """
    评估异常检测性能，计算 Precision, Recall, Micro-F1 以及 FPR。
    """
    model.eval()
    with torch.no_grad():
        h_embeds = model(x_test, adj_test)
        src, dst = edge_index_test[0], edge_index_test[1]

        # 获取预测概率
        probs = model.detect_intrusion(h_embeds[src], h_embeds[dst]).squeeze().cpu().numpy()

    y_pred = (probs >= threshold).astype(int)

    # 基础指标计算
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='micro')  # 使用 Micro-F1

    # 计算 FPR (False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("=== Evaluation Results ===")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"Micro-F1:  {f1 * 100:.2f}%")
    print(f"FPR:       {fpr * 100:.2f}%")

    return precision, recall, f1, fpr


def simulate_structural_noise(edge_index: torch.Tensor, num_nodes: int, noise_ratio: float = 0.15):
    """
    模拟随机拓扑注入与结构噪声 (对应论文 6.1 节鲁棒性分析)。
    通过随机丢弃现有边并添加等量伪造边来破坏图结构。
    """
    num_edges = edge_index.size(1)
    num_noise = int(num_edges * noise_ratio)

    # 1. 随机丢弃 (Drop)
    keep_indices = torch.randperm(num_edges)[num_noise:]
    retained_edges = edge_index[:, keep_indices]

    # 2. 随机添加伪造边 (Add fake connections)
    fake_src = torch.randint(0, num_nodes, (num_noise,))
    fake_dst = torch.randint(0, num_nodes, (num_noise,))
    fake_edges = torch.stack([fake_src, fake_dst], dim=0)

    # 拼接形成受污染的拓扑
    poisoned_edge_index = torch.cat([retained_edges, fake_edges], dim=1)
    return poisoned_edge_index