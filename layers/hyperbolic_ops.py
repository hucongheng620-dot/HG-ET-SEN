import torch

# 为保证数值稳定，定义极小值约束
MIN_NORM = 1e-15
MAX_NORM = 1e7
EPS = 1e-5


def arcosh(x):
    """双曲反余弦函数，增加数值稳定性"""
    x = torch.clamp(x, min=1.0 + EPS)
    return torch.log(x + torch.sqrt(x ** 2 - 1.0))


def artanh(x):
    """双曲反正切函数，增加数值稳定性"""
    x = torch.clamp(x, min=-1.0 + EPS, max=1.0 - EPS)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def project_to_poincare(x, c):
    """
    确保向量严格限制在 c 负曲率的庞加莱球边界内。
    半径需满足 c * ||x||^2 < 1。
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    maxnorm = (1.0 - EPS) / torch.sqrt(torch.tensor(c, dtype=x.dtype, device=x.device))
    cond = norm > maxnorm
    projected = torch.where(cond, x / norm * maxnorm, x)
    return projected


def mobius_add(x, y, c):
    """
    莫比乌斯加法 (Mobius addition)。
    实现论文公式 (4)，保证加法结果严格位于庞加莱球内。
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)

    num_1 = (1 + 2 * c * xy + c * y2) * x
    num_2 = (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c ** 2 * x2 * y2

    res = (num_1 + num_2) / denominator.clamp_min(MIN_NORM)
    return project_to_poincare(res, c)


def expmap0(v, c):
    """
    指数映射 (Exponential Map)。
    将原点切空间上的欧几里得向量 v 投影到庞加莱球中。
    实现论文公式 (5)。
    """
    norm_v = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=v.dtype, device=v.device))

    coef = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    res = coef * v
    return project_to_poincare(res, c)


def logmap0(y, c):
    """
    对数映射 (Logarithmic Map)。
    将庞加莱球中的点 y 映射回原点的欧几里得切空间。
    """
    norm_y = torch.norm(y, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=y.dtype, device=y.device))

    coef = artanh(sqrt_c * norm_y) / (sqrt_c * norm_y)
    return coef * y


def poincare_dist(x, y, c):
    """
    计算两个双曲特征在庞加莱球流形上的等距双曲距离。
    实现论文公式 (1)。
    """
    sqrt_c = torch.sqrt(torch.tensor(c, dtype=x.dtype, device=x.device))
    x_minus_y_sq = torch.norm(x - y, p=2, dim=-1) ** 2

    x_norm_sq = torch.norm(x, p=2, dim=-1) ** 2
    y_norm_sq = torch.norm(y, p=2, dim=-1) ** 2

    numerator = 2 * c * x_minus_y_sq
    denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denominator = torch.clamp(denominator, min=MIN_NORM)

    dist = (1.0 / sqrt_c) * arcosh(1.0 + numerator / denominator)
    return dist