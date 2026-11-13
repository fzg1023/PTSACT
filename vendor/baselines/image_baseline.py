"""
端到端网络架构：端到端A和端到端B

端到端A：TOF向量 → MLP编码器 → UNet/PINN
端到端B：直接接收图像 → UNet/PINN（与两阶段Stage2相同架构）
"""
import torch
import torch.nn as nn
import sys
import os

# Dynamic path setup (relative to script location)
import os
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NEWEXP_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC_DIR = os.path.join(_NEWEXP_DIR, 'TempFieldReconstructionv2', 'src')
BASE_DIR = _NEWEXP_DIR
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# 添加源码路径
class TOFEncoder(nn.Module):
    """MLP编码器：TOF(120维) → 特征向量 → 32×32特征图

    输出: (B, 64, 32, 32) 特征图，供UNet处理
    """

    def __init__(self, tof_dim=120, hidden_dim=512):
        super().__init__()
        # 120 -> 512 -> 1024
        # 1024 = 1 * 32 * 32
        self.encoder = nn.Sequential(
            nn.Linear(tof_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # 1×1卷积: (B, 1, 32, 32) -> (B, 64, 32, 32)
        self.to_channels = nn.Conv2d(1, 64, kernel_size=1)

    def forward(self, tof):
        # tof: (B, 120)
        x = self.encoder(tof)  # (B, 1024)
        x = x.view(-1, 1, 32, 32)  # reshape: (B, 1, 32, 32)
        x = self.to_channels(x)  # (B, 64, 32, 32)
        return x

class EndToEndUNetA(nn.Module):
    """端到端A-UNet：TOF → MLP编码器 → UNet → T_pred"""

    def __init__(self, tof_dim=120, base_ch=24):
        super().__init__()
        self.encoder = TOFEncoder(tof_dim=tof_dim)
        from nn_prior import AttentionUNet
        self.unet = AttentionUNet(in_channels=64, base_ch=base_ch, dropout_p=0.1)

    def forward(self, tof):
        # tof: (B, 120)
        x = self.encoder(tof)  # (B, 64, 32, 32)
        x = self.unet(x)  # (B, 1, 32, 32)
        return x

class EndToEndPINNA(nn.Module):
    """端到端A-PINN：TOF → MLP编码器 → ResidualPINN → T_pred"""

    def __init__(self, tof_dim=120, base_ch=32):
        super().__init__()
        self.encoder = TOFEncoder(tof_dim=tof_dim)
        from nn_prior import ResidualPINN
        self.pinn = ResidualPINN(in_ch=64, base_ch=base_ch, dropout=0.1)

    def forward(self, tof):
        x = self.encoder(tof)  # (B, 64, 32, 32)
        x = self.pinn(x)  # (B, 1, 32, 32)
        return x

class EndToEndUNetB(nn.Module):
    """端到端B-UNet：T_perfect图像 → UNet → T_pred

    注意：与两阶段的Stage2架构相同，直接接收图像输入
    """

    def __init__(self, base_ch=24):
        super().__init__()
        from nn_prior import AttentionUNet
        self.unet = AttentionUNet(in_channels=1, base_ch=base_ch, dropout_p=0.1)

    def forward(self, x):
        # x: (B, 1, 32, 32) - T_perfect图像
        return self.unet(x)

class EndToEndPINNB(nn.Module):
    """端到端B-PINN：T_perfect图像 → ResidualPINN → T_pred

    注意：与两阶段的Stage2架构相同，直接接收图像输入
    """

    def __init__(self, base_ch=32):
        super().__init__()
        from nn_prior import ResidualPINN
        self.pinn = ResidualPINN(in_ch=1, base_ch=base_ch, dropout=0.1)

    def forward(self, x):
        # x: (B, 1, 32, 32) - T_perfect图像
        return self.pinn(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # 测试端到端A
    tof = torch.randn(2, 120).to(device)

    e2e_a_unet = EndToEndUNetA().to(device)
    out = e2e_a_unet(tof)
    print(f"E2E-A UNet output: {out.shape}")  # 期望: (2, 1, 32, 32)
    assert out.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {out.shape}"

    e2e_a_pinn = EndToEndPINNA().to(device)
    out = e2e_a_pinn(tof)
    print(f"E2E-A PINN output: {out.shape}")  # 期望: (2, 1, 32, 32)
    assert out.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {out.shape}"

    # 测试端到端B
    img = torch.randn(2, 1, 32, 32).to(device)

    e2e_b_unet = EndToEndUNetB().to(device)
    out = e2e_b_unet(img)
    print(f"E2E-B UNet output: {out.shape}")  # 期望: (2, 1, 32, 32)
    assert out.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {out.shape}"

    e2e_b_pinn = EndToEndPINNB().to(device)
    out = e2e_b_pinn(img)
    print(f"E2E-B PINN output: {out.shape}")  # 期望: (2, 1, 32, 32)
    assert out.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {out.shape}"

    print("[PASS] 所有网络输出维度正确")