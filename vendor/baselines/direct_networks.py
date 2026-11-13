"""
端到端网络定义：MLP + 卷积解码器
"""
import torch
import torch.nn as nn


class E2ENet(nn.Module):
    """
    端到端温度场预测网络

    架构：TOF(120) → MLP(120→512→1024) → reshape(32×32) → 卷积解码器 → T_pred(32×32)

    输入: TOF向量 (B, 120)
    输出: 温度场 (B, 1, 32, 32)

    注意：输出无激活函数限制，由MSE Loss自动学习正确范围
    """

    def __init__(self, tof_dim=120):
        super().__init__()

        # MLP编码器
        self.mlp = nn.Sequential(
            nn.Linear(tof_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),  # 1024 = 32×32，用于reshape
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 卷积解码器（3×3卷积增强）
        # Conv → BN → ReLU 是标准顺序
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),  # 无激活，输出范围由Loss学习
        )

    def forward(self, tof):
        """
        前向传播

        Args:
            tof: (B, 120) TOF向量

        Returns:
            (B, 1, 32, 32) 预测温度场（归一化值）
        """
        # MLP编码
        x = self.mlp(tof)                    # (B, 1024)

        # Reshape到图像 (1024 = 1×32×32)
        x = x.view(-1, 1, 32, 32)           # (B, 1, 32, 32)

        # 卷积解码
        x = self.decoder(x)                  # (B, 1, 32, 32)

        return x


class E2ENetV1(nn.Module):
    """
    E2E 网络扩展版本 V1
    MLP: 120 → 1024 → 1024 (更宽)
    Conv: 64 通道解码器
    """
    def __init__(self, tof_dim=120):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tof_dim, 1024),  # 512 → 1024
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),      # 1024 = 32×32，用于 reshape
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   # 32 → 64 通道
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),   # 增加一层
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, tof):
        x = self.mlp(tof)              # (B, 1024)
        x = x.view(-1, 1, 32, 32)      # (B, 1, 32, 32)
        return self.decoder(x)         # (B, 1, 32, 32)


class E2ENetV2(nn.Module):
    """
    E2E 网络扩展版本 V2
    MLP: 120 → 1024 → 2048 → 1024 (瓶颈)
    """
    def __init__(self, tof_dim=120):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tof_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),       # 中间层更宽
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),       # 瓶颈层
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, tof):
        x = self.mlp(tof)              # (B, 1024)
        x = x.view(-1, 1, 32, 32)
        return self.decoder(x)


class E2ENetV3(nn.Module):
    """
    E2E 网络扩展版本 V3
    MLP: 120 → 1024 → 2048 (最宽)
    Conv: 更深的解码器
    """
    def __init__(self, tof_dim=120):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tof_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),            # 增加 Dropout
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),      # 输出 1024 = 32×32，用于 reshape
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),  # 更宽
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, tof):
        x = self.mlp(tof)              # (B, 1024)
        x = x.view(-1, 1, 32, 32)
        return self.decoder(x)


class E2ENetV1_Multi(nn.Module):
    """
    E2E 多尺度架构 V1-Multi
    简单的编码器-解码器结构，带跳跃连接

    架构：TOF(120) → MLP(120→1024→1024) → reshape → 编码器 → 跳跃连接 → 解码器 → T_pred

    输入: TOF向量 (B, 120)
    输出: 温度场 (B, 1, 32, 32)

    参数量: ~1.18M
    """
    def __init__(self, tof_dim=120):
        super().__init__()

        # MLP 编码器
        self.mlp = nn.Sequential(
            nn.Linear(tof_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 通道映射：1 → 32
        self.to_features = nn.Sequential(
            nn.Conv2d(1, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 解码器 (stride=1 保持尺寸)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 跳跃连接融合
        self.fusion = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, tof):
        """前向传播"""
        # MLP 编码
        x = self.mlp(tof)  # (B, 1024)
        x = x.view(-1, 1, 32, 32)  # (B, 1, 32, 32)
        x = self.to_features(x)  # (B, 32, 32, 32)

        # 编码
        enc = self.encoder(x)  # (B, 64, 32, 32)

        # 瓶颈
        feat = self.bottleneck(enc)  # (B, 64, 32, 32)

        # 解码
        dec = self.decoder(feat)  # (B, 32, 32, 32)

        # 跳跃连接融合 (Concat)
        combined = torch.cat([dec, enc], dim=1)  # (B, 96, 32, 32)
        out = self.fusion(combined)  # (B, 1, 32, 32)

        return out


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def compare_all_versions():
    """比较所有版本的参数量"""
    versions = {
        'E2ENet (Baseline)': E2ENet(tof_dim=120),
        'E2ENetV1': E2ENetV1(tof_dim=120),
        'E2ENetV2': E2ENetV2(tof_dim=120),
        'E2ENetV3': E2ENetV3(tof_dim=120),
        'E2ENetV1_Multi': E2ENetV1_Multi(tof_dim=120),
    }

    print("=" * 50)
    print("Network Capacity Comparison")
    print("=" * 50)
    for name, model in versions.items():
        n_params = count_parameters(model)
        print(f"{name}: {n_params:,} parameters")
    print("=" * 50)


if __name__ == '__main__':
    # 设置CUDA确定性
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 运行参数量对比
    compare_all_versions()

    # 测试所有网络的前向传播
    print("\n" + "=" * 50)
    print("Testing all network forward passes")
    print("=" * 50)

    test_batch = 4
    tof = torch.randn(test_batch, 120).to(device)

    for name, cls in [('E2ENet', E2ENet), ('E2ENetV1', E2ENetV1),
                      ('E2ENetV2', E2ENetV2), ('E2ENetV3', E2ENetV3),
                      ('E2ENetV1_Multi', E2ENetV1_Multi)]:
        model = cls(tof_dim=120).to(device)
        output = model(tof)
        print(f"{name}: output shape {output.shape}, range [{output.min():.4f}, {output.max():.4f}]")
        assert output.shape == (test_batch, 1, 32, 32), f"{name}: shape mismatch"

    print("\n[PASS] All networks tested successfully")