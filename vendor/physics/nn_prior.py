"""
声学CT温度场重建 —— 深度学习辅助模块（PINN + 数据驱动先验）
=============================================================
最新研究成果集成（2023-2025）：

  1. DeepTempPrior     —— 基于 U-Net 的温度场先验生成器
  2. PINNTemperature   —— Physics-Informed Neural Network
  3. HybridReconstructor —— PINN 初始场 + SART-TV 精化

  【创新算法（2025 版新增）】
  4. AttentionUNet       —— 带注意力门控的 U-Net，自适应筛选跳跃连接特征
     参考：Oktay O et al. "Attention U-Net" MIDL 2018
  5. TransformerReconstructor —— 将 TOF 向量直接映射为温度场的 Transformer
     参考：Dosovitskiy A et al. "An image is worth 16x16 words" ICLR 2021
  6. MCDropoutReconstructor   —— MC-Dropout 不确定性量化
     参考：Gal Y, Ghahramani Z. "Dropout as a Bayesian Approximation" ICML 2016
  7. SelfSupervisedContrastivePrior —— 自监督对比学习先验
     参考：He K et al. "MoCo" CVPR 2020; Chen T et al. "SimCLR" ICML 2020

  注意：本模块在无 PyTorch 时自动降级为高斯过程先验（scikit-learn）。

参考文献：
  [1] Raissi M et al. "Physics-informed neural networks" JCP 2019.
  [2] Cai S et al. "PINNs for heat transfer problems" JHT 2021.
  [3] Wang S et al. "When and why PINNs fail to train" JCP 2022.
  [4] Oktay O et al. "Attention U-Net" MIDL 2018.
  [5] Chung H et al. "Diffusion Posterior Sampling" ICLR 2023.
  [6] Gal Y, Ghahramani Z. "Dropout as a Bayesian Approximation" ICML 2016.
"""

import numpy as np
from typing import Optional, Tuple, Callable
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch 未安装，PINN 模块将降级为高斯过程先验（GP Prior）。\n"
        "安装方式：pip install torch",
        ImportWarning,
        stacklevel=2,
    )


# =============================================================================
# 1. U-Net 先验生成器（PyTorch）
# =============================================================================

if TORCH_AVAILABLE:

    class UNetBlock(nn.Module):
        """U-Net 编解码块"""
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class SmallUNet(nn.Module):
        """
        轻量级 U-Net，用于温度场先验重建
        输入：(1, ny, nx) 换能器位置掩码或初始近似
        输出：(1, ny, nx) 预测温度场
        """
        def __init__(self, in_channels: int = 1, base_ch: int = 32):
            super().__init__()
            c = base_ch

            # 编码器
            self.enc1 = UNetBlock(in_channels, c)
            self.enc2 = UNetBlock(c, c * 2)
            self.enc3 = UNetBlock(c * 2, c * 4)

            self.pool = nn.MaxPool2d(2)

            # 瓶颈
            self.bottleneck = UNetBlock(c * 4, c * 8)

            # 解码器
            self.up3   = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec3  = UNetBlock(c * 8, c * 4)
            self.up2   = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2  = UNetBlock(c * 4, c * 2)
            self.up1   = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1  = UNetBlock(c * 2, c)

            self.out_conv = nn.Conv2d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))

            bn = self.bottleneck(self.pool(e3))

            d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.out_conv(d1)


# =============================================================================
# 2. Physics-Informed Neural Network (PINN)
# =============================================================================

if TORCH_AVAILABLE:

    class TempNet(nn.Module):
        """
        坐标输入网络: f(x, y) → T(x, y)
        使用 Fourier 特征映射改善低频偏好（spectral bias）
        """
        def __init__(
            self,
            n_fourier: int = 128,
            hidden_dim: int = 256,
            n_hidden: int = 5,
            sigma: float = 10.0,
        ):
            super().__init__()
            # 随机 Fourier 特征（固定，不参与训练）
            B = torch.randn(2, n_fourier) * sigma
            self.register_buffer("B", B)
            in_dim = 2 * n_fourier

            layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            for _ in range(n_hidden - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            layers.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, xy: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                xy: shape (N, 2)，每行为 [x, y]（归一化坐标）
            Returns:
                T : shape (N, 1)
            """
            # Fourier 特征映射
            z = xy @ self.B                              # (N, n_fourier)
            z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # (N, 2*n_fourier)
            return self.net(z)

    class PINNTemperature:
        """
        Physics-Informed Neural Network 温度场重建器

        损失函数：
            L = w_data · L_data + w_pde · L_pde + w_bc · L_bc

        其中：
            L_data = ‖A_nn s(θ) − b‖²   （TOF 数据拟合）
            L_pde  = ‖∇²T‖²_Ω           （稳态热传导 PDE 残差）
            L_bc   = ‖T|_∂Ω − T_wall‖²  （边界条件）
        """

        def __init__(
            self,
            A,
            b: np.ndarray,
            positions: np.ndarray,       # 换能器坐标 (N,2)，用于建立 A_nn
            grid_shape: Tuple[int, int],
            domain: Tuple[float, float, float, float],  # (x_min,x_max,y_min,y_max)
            n_epochs: int = 3000,
            lr: float = 1e-3,
            w_data: float = 1.0,
            w_pde: float = 0.01,
            w_bc: float = 0.1,
            T_wall: float = 20.0,        # 边界温度 [℃]
            n_colloc: int = 2000,        # PDE 配点数量
            device: str = "cpu",
        ):
            assert TORCH_AVAILABLE, "PINN 需要 PyTorch，请先 pip install torch"

            self.A        = A
            self.b        = b.astype(np.float64)
            self.grid_shape = grid_shape
            self.domain   = domain
            self.n_epochs = n_epochs
            self.lr       = lr
            self.w_data   = w_data
            self.w_pde    = w_pde
            self.w_bc     = w_bc
            self.T_wall   = T_wall
            self.n_colloc = n_colloc
            self.device   = torch.device(device)

            self.net = TempNet().to(self.device)
            self.losses: list = []

            # 归一化系数
            x0, x1, y0, y1 = domain
            self._xc = 0.5 * (x0 + x1)
            self._xs = 0.5 * (x1 - x0)
            self._yc = 0.5 * (y0 + y1)
            self._ys = 0.5 * (y1 - y0)

        def _normalize(self, xy: np.ndarray) -> "torch.Tensor":
            """物理坐标 → [-1,1] 归一化"""
            xn = (xy[:, 0] - self._xc) / (self._xs + 1e-14)
            yn = (xy[:, 1] - self._yc) / (self._ys + 1e-14)
            return torch.tensor(
                np.column_stack([xn, yn]), dtype=torch.float32, device=self.device
            )

        def _pixel_centers_norm(self) -> "torch.Tensor":
            """返回所有像素中心的归一化坐标"""
            ny, nx = self.grid_shape
            x0, x1, y0, y1 = self.domain
            dx = (x1 - x0) / nx; dy = (y1 - y0) / ny
            xs = x0 + (np.arange(nx) + 0.5) * dx
            ys = y0 + (np.arange(ny) + 0.5) * dy
            XX, YY = np.meshgrid(xs, ys)                      # (ny, nx)
            xy = np.column_stack([XX.ravel(), YY.ravel()])    # (ny*nx, 2)
            return self._normalize(xy)

        def _boundary_points(self, n_per_side: int = 50) -> "torch.Tensor":
            """在域边界均匀采样配点"""
            x0, x1, y0, y1 = self.domain
            t = np.linspace(0, 1, n_per_side)
            pts = np.vstack([
                np.column_stack([x0 + t*(x1-x0), np.full(n_per_side, y0)]),  # 下
                np.column_stack([x0 + t*(x1-x0), np.full(n_per_side, y1)]),  # 上
                np.column_stack([np.full(n_per_side, x0), y0 + t*(y1-y0)]),  # 左
                np.column_stack([np.full(n_per_side, x1), y0 + t*(y1-y0)]),  # 右
            ])
            return self._normalize(pts)

        def _pde_loss(self, xy_norm: "torch.Tensor") -> "torch.Tensor":
            """
            稳态热传导 PDE 残差：∇²T = 0
            使用自动微分计算 ∂²T/∂x² + ∂²T/∂y²
            """
            xy = xy_norm.clone().requires_grad_(True)
            T  = self.net(xy)                           # (N, 1)

            # 一阶偏导
            grad_T = torch.autograd.grad(
                T, xy,
                grad_outputs=torch.ones_like(T),
                create_graph=True,
            )[0]                                         # (N, 2)

            dTdx = grad_T[:, 0:1]
            dTdy = grad_T[:, 1:2]

            # 二阶偏导（Laplacian）
            Txx = torch.autograd.grad(
                dTdx, xy,
                grad_outputs=torch.ones_like(dTdx),
                create_graph=True,
            )[0][:, 0:1]

            Tyy = torch.autograd.grad(
                dTdy, xy,
                grad_outputs=torch.ones_like(dTdy),
                create_graph=True,
            )[0][:, 1:2]

            lap = Txx + Tyy
            return torch.mean(lap ** 2)

        def train(self) -> np.ndarray:
            """训练 PINN，返回重建温度场（展平，shape: n_pixels）"""
            import torch.optim as optim

            optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_epochs, eta_min=1e-5
            )

            # 转换系统矩阵到 torch（仅一次）
            A_dense = torch.tensor(
                self.A.toarray(), dtype=torch.float32, device=self.device
            )
            b_torch = torch.tensor(self.b, dtype=torch.float32, device=self.device)

            # 像素中心坐标（固定）
            xy_pix = self._pixel_centers_norm()
            # 边界点（固定）
            xy_bc  = self._boundary_points()

            # PDE 配点（随机重采样加速训练）
            x0, x1, y0, y1 = self.domain

            T_wall_t = torch.tensor([self.T_wall], dtype=torch.float32,
                                     device=self.device)

            for epoch in range(self.n_epochs):
                optimizer.zero_grad()

                # --- 数据拟合损失 ---
                T_pix = self.net(xy_pix).squeeze()            # (n_pixels,)
                # T [℃] → s [s/m] → TOF
                s_pix = 1.0 / (331.3 * torch.sqrt(1.0 + T_pix / 273.15))
                tof_pred = A_dense @ s_pix
                loss_data = torch.mean((tof_pred - b_torch) ** 2)

                # --- PDE 损失 ---
                # 每 epoch 随机采样配点（避免过拟合特定点）
                xr = np.random.uniform(x0, x1, self.n_colloc)
                yr = np.random.uniform(y0, y1, self.n_colloc)
                xy_colloc = self._normalize(np.column_stack([xr, yr]))
                xy_colloc.requires_grad_(True)
                loss_pde  = self._pde_loss(xy_colloc)

                # --- 边界条件损失 ---
                T_bc     = self.net(xy_bc).squeeze()
                loss_bc  = torch.mean((T_bc - T_wall_t.expand_as(T_bc)) ** 2)

                # --- 总损失 ---
                loss = (self.w_data * loss_data
                        + self.w_pde  * loss_pde
                        + self.w_bc   * loss_bc)

                loss.backward()
                optimizer.step()
                scheduler.step()

                if epoch % 500 == 0 or epoch == self.n_epochs - 1:
                    print(
                        f"  [PINN] epoch {epoch:5d}/{self.n_epochs} | "
                        f"total={loss.item():.4e}  data={loss_data.item():.4e}  "
                        f"pde={loss_pde.item():.4e}  bc={loss_bc.item():.4e}"
                    )
                self.losses.append(float(loss.item()))

            # 提取离散温度场
            with torch.no_grad():
                T_pred = self.net(xy_pix).squeeze().cpu().numpy()
            return T_pred   # shape (n_pixels,)，单位 ℃

        def get_temperature_field(self, xy: np.ndarray) -> np.ndarray:
            """在任意坐标点查询温度（训练后调用）"""
            import torch
            self.net.eval()
            with torch.no_grad():
                xy_norm = self._normalize(xy)
                T = self.net(xy_norm).squeeze().cpu().numpy()
            return T


# =============================================================================
# 3. 降级：高斯过程先验（无 PyTorch 时使用）
# =============================================================================

class GaussianProcessPrior:
    """
    基于高斯过程（GP）的温度场先验

    当 PyTorch 不可用时作为 PINN 的降级替代方案。
    使用 RBF 核拟合换能器路径上的积分约束，输出光滑温度场估计。

    注意：GP 扩展性有限（O(n³) 训练），适合小规模网格（≤ 32×32）。
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        grid_shape: Tuple[int, int],
        domain: Tuple[float, float, float, float],
        alpha: float = 1e-2,
        length_scale: float = 0.05,
    ):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            self._gp_available = True
        except ImportError:
            warnings.warn("scikit-learn 未安装，GP 先验不可用。将返回均匀初始场。")
            self._gp_available = False

        self.A          = A
        self.b          = b
        self.grid_shape = grid_shape
        self.domain     = domain
        self.alpha      = alpha
        self.length_scale = length_scale

    def solve(self) -> np.ndarray:
        """
        返回温度场估计（展平，℃）

        策略：
        · 用 Tikhonov L2 求解粗略慢度场 → 转换为温度
        · 用 GP 对该温度场做平滑插值（引入物理先验）
        """
        from reconstruction import TikhonovReconstructor, slowness_to_temperature

        # 粗略重建
        rec = TikhonovReconstructor(
            self.A, self.b, alpha=self.alpha,
            reg_type="laplacian", grid_shape=self.grid_shape
        )
        s_coarse = rec.solve()
        T_coarse = slowness_to_temperature(s_coarse)

        if not self._gp_available:
            return T_coarse

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        # 网格坐标
        ny, nx = self.grid_shape
        x0, x1, y0, y1 = self.domain
        dx = (x1 - x0) / nx; dy = (y1 - y0) / ny
        xs = x0 + (np.arange(nx) + 0.5) * dx
        ys = y0 + (np.arange(ny) + 0.5) * dy
        XX, YY = np.meshgrid(xs, ys)
        xy_all = np.column_stack([XX.ravel(), YY.ravel()])   # (ny*nx, 2)

        kernel = RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=1e-3)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True)
        gp.fit(xy_all, T_coarse)
        T_smooth = gp.predict(xy_all)
        return T_smooth.ravel()


# =============================================================================
# 4. 混合重建器：PINN初始场 + SART-TV 精化
# =============================================================================

class HybridReconstructor:
    """
    两阶段混合重建：
        阶段 1 — PINN（或 GP 先验）生成连续平滑的温度场估计
        阶段 2 — SART-TV 在离散网格上利用该估计作为初始场精细迭代

    优点：
    · PINN 阶段提供物理一致的全局先验，避免 SART 的局部极小
    · SART-TV 阶段精确拟合测量数据，修正 PINN 的数值误差
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        positions: np.ndarray,
        grid_shape: Tuple[int, int],
        domain: Tuple[float, float, float, float],
        # PINN 参数
        pinn_epochs: int = 2000,
        pinn_w_data: float = 1.0,
        pinn_w_pde: float = 0.01,
        # SART-TV 参数
        n_iter: int = 80,
        relax: float = 0.4,
        tv_weight: float = 1e-6,
        device: str = "cpu",
    ):
        self.A          = A
        self.b          = b
        self.positions  = positions
        self.grid_shape = grid_shape
        self.domain     = domain

        self.pinn_epochs  = pinn_epochs
        self.pinn_w_data  = pinn_w_data
        self.pinn_w_pde   = pinn_w_pde
        self.n_iter       = n_iter
        self.relax        = relax
        self.tv_weight    = tv_weight
        self.device       = device

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行两阶段重建

        Returns:
            T_pinn   : PINN 阶段温度场 (℃)，shape (n_pixels,)
            T_hybrid : 混合精化温度场 (℃)，shape (n_pixels,)
        """
        from ray_tracer import temperature_to_slowness, slowness_to_temperature
        from reconstruction import SARTTVReconstructor

        print("=" * 55)
        print("  阶段 1：PINN 生成平滑初始温度场")
        print("=" * 55)

        if TORCH_AVAILABLE:
            pinn = PINNTemperature(
                A=self.A, b=self.b, positions=self.positions,
                grid_shape=self.grid_shape, domain=self.domain,
                n_epochs=self.pinn_epochs,
                w_data=self.pinn_w_data, w_pde=self.pinn_w_pde,
                device=self.device,
            )
            T_pinn = pinn.train()
        else:
            print("  （PyTorch 不可用，使用 GP 先验代替 PINN）")
            gp = GaussianProcessPrior(
                self.A, self.b, self.grid_shape, self.domain
            )
            T_pinn = gp.solve()

        # PINN 输出的是温度，转换为慢度作为 SART-TV 初始场
        s_pinn_init = temperature_to_slowness(T_pinn)

        print("=" * 55)
        print("  阶段 2：SART-TV 精化迭代")
        print("=" * 55)

        solver = SARTTVReconstructor(
            A=self.A, b=self.b,
            n_iter=self.n_iter, relax=self.relax,
            tv_weight=self.tv_weight,
            grid_shape=self.grid_shape,
            s_init=s_pinn_init,
        )
        s_hybrid = solver.run()
        T_hybrid = slowness_to_temperature(s_hybrid)

        return T_pinn, T_hybrid


# =============================================================================
# 5. Attention U-Net（创新算法）
#    参考：Oktay O et al. "Attention U-Net: Learning where to look" MIDL 2018
# =============================================================================

if TORCH_AVAILABLE:

    class CBAM(nn.Module):
        """
        Convolutional Block Attention Module（CBAM）
        先做通道注意力，再做空间注意力，增强特征表达能力。
        参考：Woo S et al. "CBAM" ECCV 2018
        """
        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            # 通道注意力
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, max(channels // reduction, 1)),
                nn.ReLU(inplace=True),
                nn.Linear(max(channels // reduction, 1), channels),
                nn.Sigmoid(),
            )
            # 空间注意力
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # 通道注意力
            ca = self.channel_attn(x).view(x.size(0), x.size(1), 1, 1)
            x  = x * ca
            # 空间注意力（avg + max 沿通道轴拼接）
            avg_map = x.mean(dim=1, keepdim=True)
            max_map = x.max(dim=1, keepdim=True).values
            sa = self.spatial_attn(torch.cat([avg_map, max_map], dim=1))
            return x * sa

    class AttentionGate(nn.Module):
        """
        注意力门控（Attention Gate）
        在跳跃连接处对编码器特征做软掩码，使解码器专注于相关区域。
        公式：α = σ(Wψ^T · σ(Wg·g + Wx·x + bg) + bψ)
        参考：Oktay O et al. "Attention U-Net" MIDL 2018
        """
        def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
            super().__init__()
            self.Wg = nn.Conv2d(gate_ch,  inter_ch, 1, bias=False)
            self.Wx = nn.Conv2d(skip_ch,  inter_ch, 1, bias=False)
            self.psi = nn.Sequential(
                nn.Conv2d(inter_ch, 1, 1, bias=True),
                nn.Sigmoid(),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g: "torch.Tensor", x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                g : decoder gate signal，shape (B, gate_ch, H, W)
                x : encoder skip feature，shape (B, skip_ch, H, W)
            Returns:
                Attention-weighted skip feature，shape (B, skip_ch, H, W)
            """
            # 如果分辨率不同，对 g 上采样到与 x 相同尺寸
            if g.shape[2:] != x.shape[2:]:
                g = nn.functional.interpolate(
                    g, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            attn = self.psi(self.relu(self.Wg(g) + self.Wx(x)))  # (B,1,H,W)
            return x * attn

    class AttentionUNetBlock(nn.Module):
        """带 CBAM 注意力的卷积块"""
        def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
            super().__init__()
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if dropout_p > 0:
                layers.append(nn.Dropout2d(p=dropout_p))
            self.conv = nn.Sequential(*layers)
            self.cbam = CBAM(out_ch)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.cbam(self.conv(x))

    class AttentionUNet(nn.Module):
        """
        带注意力门控的 U-Net 温度场重建网络

        创新点：
        1. 每个编码器块引入 CBAM（通道+空间注意力）
        2. 跳跃连接使用 AttentionGate 过滤无关特征
        3. 瓶颈层使用深度可分离卷积，减少参数量
        4. 支持 MC-Dropout 不确定性量化

        输入：(B, 1, ny, nx)  换能器位置/TOF 反投影初始图
        输出：(B, 1, ny, nx)  重建温度场（归一化到 [0,1]）
        """

        def __init__(
            self,
            in_channels: int = 1,
            base_ch: int = 32,
            dropout_p: float = 0.1,
            use_attention: bool = True,
        ):
            super().__init__()
            c = base_ch

            # ── 编码器 ──────────────────────────────────────────
            self.enc1 = AttentionUNetBlock(in_channels, c, dropout_p)
            self.enc2 = AttentionUNetBlock(c,     c * 2, dropout_p)
            self.enc3 = AttentionUNetBlock(c * 2, c * 4, dropout_p)

            self.pool = nn.MaxPool2d(2)

            # ── 瓶颈（深度可分离卷积）─────────────────────────
            self.bottleneck = nn.Sequential(
                # 深度可分离卷积 = 逐通道卷积 + 逐点卷积
                nn.Conv2d(c * 4, c * 4, 3, padding=1, groups=c * 4, bias=False),
                nn.Conv2d(c * 4, c * 8, 1, bias=False),
                nn.BatchNorm2d(c * 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_p),
                nn.Conv2d(c * 8, c * 8, 3, padding=1, groups=c * 8, bias=False),
                nn.Conv2d(c * 8, c * 8, 1, bias=False),
                nn.BatchNorm2d(c * 8),
                nn.ReLU(inplace=True),
            )

            # ── 注意力门控 ────────────────────────────────────
            self.use_attention = use_attention
            if use_attention:
                self.ag3 = AttentionGate(c * 8, c * 4, c * 2)
                self.ag2 = AttentionGate(c * 4, c * 2, c)
                self.ag1 = AttentionGate(c * 2, c,     c // 2)

            # ── 解码器 ──────────────────────────────────────────
            self.up3  = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec3 = AttentionUNetBlock(c * 8, c * 4, dropout_p)
            self.up2  = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = AttentionUNetBlock(c * 4, c * 2, dropout_p)
            self.up1  = nn.ConvTranspose2d(c * 2, c,     2, stride=2)
            self.dec1 = AttentionUNetBlock(c * 2, c,     dropout_p)

            # ── 输出层 ──────────────────────────────────────────
            self.out_conv = nn.Conv2d(c, 1, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # 编码路径
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            bn = self.bottleneck(self.pool(e3))

            # 解码路径
            if self.use_attention:
                d3 = self.dec3(torch.cat([self.up3(bn), self.ag3(bn, e3)],  dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), self.ag2(d3, e2)],  dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), self.ag1(d2, e1)],  dim=1))
            else:
                d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.out_conv(d1)


    class PlainUNetBlock(nn.Module):
        """普通卷积块（无 CBAM），用于 PlainUNet 基线"""
        def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
            super().__init__()
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if dropout_p > 0:
                layers.append(nn.Dropout2d(p=dropout_p))
            self.conv = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.conv(x)


    class PlainUNet(nn.Module):
        """
        普通 U-Net 基线模型（无 CBAM，无 AttentionGate）
        用于 Ablation-1 消融实验的公平对比基线。

        与 AttentionUNet 结构完全一致，但去掉了：
        - 编码器/解码器中的 CBAM 注意力
        - 跳跃连接处的 AttentionGate

        输入：(B, 1, ny, nx)  Stage1 重建温度场（归一化）
        输出：(B, 1, ny, nx)  残差修正场
        """

        def __init__(
            self,
            in_channels: int = 1,
            base_ch: int = 24,
            dropout_p: float = 0.1,
        ):
            super().__init__()
            c = base_ch

            # ── 编码器 ──────────────────────────────────────────
            self.enc1 = PlainUNetBlock(in_channels, c, dropout_p)
            self.enc2 = PlainUNetBlock(c,     c * 2, dropout_p)
            self.enc3 = PlainUNetBlock(c * 2, c * 4, dropout_p)

            self.pool = nn.MaxPool2d(2)

            # ── 瓶颈 ────────────────────────────────────────────
            self.bottleneck = nn.Sequential(
                nn.Conv2d(c * 4, c * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(c * 4),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_p),
                nn.Conv2d(c * 4, c * 8, 3, padding=1, bias=False),
                nn.BatchNorm2d(c * 8),
                nn.ReLU(inplace=True),
            )

            # ── 解码器 ──────────────────────────────────────────
            self.up3  = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec3 = PlainUNetBlock(c * 8, c * 4, dropout_p)
            self.up2  = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = PlainUNetBlock(c * 4, c * 2, dropout_p)
            self.up1  = nn.ConvTranspose2d(c * 2, c,     2, stride=2)
            self.dec1 = PlainUNetBlock(c * 2, c,     dropout_p)

            # ── 输出层 ──────────────────────────────────────────
            self.out_conv = nn.Conv2d(c, 1, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # 编码
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            bn = self.bottleneck(self.pool(e3))

            # 解码（普通跳跃连接，无 AttentionGate）
            d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.out_conv(d1)


    # =========================================================================
    # 6. Transformer 序列重建器（创新算法）
    #    将 TOF 向量视为 token 序列，用 Transformer Encoder 编码后
    #    通过 MLP Decoder 映射到展平温度场。
    #    参考：Vaswani A et al. "Attention is All You Need" NeurIPS 2017
    # =========================================================================

    class TransformerReconstructor(nn.Module):
        """
        基于 Transformer Encoder 的声学CT温度场重建网络

        设计思路：
        · TOF 测量值 b ∈ R^{n_rays} 视为变长序列
        · 每条射线的 TOF 值 + 几何编码（src/dst 角度）作为 token
        · Transformer Encoder 学习射线间的全局相关性
        · MLP Decoder 将编码特征映射回温度场

        架构：
            TOF + 几何编码 → Projection → Transformer Encoder
                           → Global Average Pooling → MLP → (ny*nx,) 温度场
        """

        def __init__(
            self,
            n_rays: int,
            n_pixels: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            dim_feedforward: int = 256,
            dropout_p: float = 0.1,
            geo_dim: int = 4,      # 几何编码维度：(src_x, src_y, dst_x, dst_y)
        ):
            """
            Args:
                n_rays    : 射线数（TOF 向量长度）
                n_pixels  : 像素数（输出温度场长度）
                d_model   : Transformer 隐层维度
                n_heads   : 多头注意力头数
                n_layers  : Transformer Encoder 层数
                dim_feedforward : FFN 隐层维度
                dropout_p : Dropout 概率（用于 MC-Dropout 不确定性量化）
                geo_dim   : 几何编码维度
            """
            super().__init__()
            self.n_rays   = n_rays
            self.n_pixels = n_pixels
            self.d_model  = d_model

            # TOF + 几何特征投影到 d_model 维
            self.input_proj = nn.Sequential(
                nn.Linear(1 + geo_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(inplace=True),
            )

            # 正弦位置编码（可学习）
            self.pos_embed = nn.Parameter(torch.zeros(1, n_rays, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout_p,
                batch_first=True,
                norm_first=True,   # Pre-LN（更稳定的训练）
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers,
                norm=nn.LayerNorm(d_model),
            )

            # MLP Decoder：d_model → n_pixels
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(d_model * 2, d_model * 4),
                nn.GELU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(d_model * 4, n_pixels),
            )

        def forward(
            self,
            tof: "torch.Tensor",      # (B, n_rays)
            geo: "torch.Tensor",      # (B, n_rays, geo_dim)
        ) -> "torch.Tensor":
            """
            Returns:
                T_flat : (B, n_pixels)，归一化温度场
            """
            # 拼接 TOF 标量与几何编码
            x = torch.cat([tof.unsqueeze(-1), geo], dim=-1)  # (B, n_rays, 1+geo_dim)
            x = self.input_proj(x)                            # (B, n_rays, d_model)
            x = x + self.pos_embed                            # 加位置编码

            # Transformer Encoder
            x = self.transformer(x)                           # (B, n_rays, d_model)

            # Global Average Pooling（聚合所有射线信息）
            x = x.mean(dim=1)                                 # (B, d_model)

            # MLP 解码
            T_flat = self.decoder(x)                          # (B, n_pixels)
            return T_flat

    # =========================================================================
    # 辅助：构建射线几何编码
    # =========================================================================

    def build_geo_encoding(
        positions: np.ndarray,
        ray_pairs: list,
    ) -> np.ndarray:
        """
        构建射线几何编码：每条射线对应 4 维向量
        [src_x, src_y, dst_x, dst_y]（归一化到 [-1, 1]）

        Args:
            positions : 换能器坐标 (N, 2)
            ray_pairs : 射线对列表 [(i,j), ...]

        Returns:
            geo : (n_rays, 4) float32 数组
        """
        geo = np.array(
            [[positions[i, 0], positions[i, 1],
              positions[j, 0], positions[j, 1]]
             for i, j in ray_pairs],
            dtype=np.float32,
        )
        # 归一化到 [-1, 1]
        geo = (geo - geo.mean(axis=0, keepdims=True)) / (geo.std(axis=0, keepdims=True) + 1e-8)
        return geo


    # =========================================================================
    # 7. MC-Dropout 不确定性量化重建器（创新算法）
    #    参考：Gal Y, Ghahramani Z. "Dropout as a Bayesian Approximation" ICML 2016
    # =========================================================================

    class MCDropoutReconstructor:
        """
        基于 MC-Dropout 的温度场重建与不确定性量化

        推理时保持 Dropout 激活（model.train()），通过 K 次前向传播
        估计预测均值与方差（近似 Bayesian 推断）：

            E[T(x,y)] ≈ (1/K) Σ_k T̂_k(x,y)
            Var[T(x,y)] ≈ (1/K) Σ_k T̂_k^2 - E[T]^2

        不确定性图高亮区域代表模型对该位置温度估计的置信度低，
        可用于：
        · 指导换能器布局优化（在不确定性高处增加传感器）
        · 提供重建结果的可靠性分图（工业安全监控）

        使用 AttentionUNet 作为基础网络（可替换为其他结构）。
        """

        def __init__(
            self,
            A,
            b: np.ndarray,
            positions: np.ndarray,
            ray_pairs: list,
            grid_shape: Tuple[int, int],
            domain: Tuple[float, float, float, float],
            n_mc: int = 50,
            base_ch: int = 32,
            dropout_p: float = 0.15,
            T_ref: float = 20.0,
            T_scale: float = 1200.0,
            device: str = "cpu",
        ):
            """
            Args:
                n_mc      : MC-Dropout 采样次数（越多越准确，但越慢）
                dropout_p : Dropout 概率（影响不确定性幅度）
                T_ref     : 归一化参考温度（均值）[℃]
                T_scale   : 归一化缩放系数（标准差量级）[℃]
            """
            assert TORCH_AVAILABLE, "MC-Dropout 需要 PyTorch"

            self.A          = A
            self.b          = b.astype(np.float64)
            self.positions  = positions
            self.ray_pairs  = ray_pairs
            self.grid_shape = grid_shape
            self.domain     = domain
            self.n_mc       = n_mc
            self.T_ref      = T_ref
            self.T_scale    = T_scale
            self.device     = torch.device(device)

            ny, nx = grid_shape
            self.net = AttentionUNet(
                in_channels=1, base_ch=base_ch, dropout_p=dropout_p
            ).to(self.device)

        def _backproject(self) -> np.ndarray:
            """
            简单反投影（Filtered Back-Projection 近似）：
            将 TOF 测量值通过 A^T 反投影到像素域作为网络输入。
            """
            At_b = self.A.T.dot(self.b)
            # 归一化到 [0, 1]
            At_b = (At_b - At_b.min()) / (At_b.max() - At_b.min() + 1e-14)
            return At_b.astype(np.float32)

        def predict_with_uncertainty(
            self,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            执行 K 次 MC-Dropout 前向传播，返回均值与方差温度场。

            注意：本方法无需训练，适用于：
            · 已预训练的网络载入权重后调用
            · 或结合 SART-TV 粗糙解作为输入

            Returns:
                T_mean : 均值温度场 (ny*nx,) [℃]
                T_var  : 方差温度场 (ny*nx,) [℃²]（取 sqrt 得标准差）
            """
            ny, nx = self.grid_shape

            # 构造输入：反投影图 (1, 1, ny, nx)
            bp = self._backproject().reshape(1, 1, ny, nx)
            x_input = torch.tensor(bp, device=self.device)

            # MC-Dropout：保持 train 模式，多次前向传播
            self.net.train()   # ← 关键：保持 Dropout 激活
            samples = []
            with torch.no_grad():
                for _ in range(self.n_mc):
                    out = self.net(x_input).squeeze().cpu().numpy()  # (ny, nx)
                    # 反归一化到温度域
                    T_sample = out * self.T_scale + self.T_ref
                    samples.append(T_sample.ravel())

            samples = np.stack(samples, axis=0)  # (K, n_pixels)
            T_mean  = samples.mean(axis=0)
            T_var   = samples.var(axis=0)

            return T_mean, T_var

        def train_supervised(
            self,
            T_train_list: list,        # 训练温度场列表（每个 shape = (n_pixels,)）
            n_epochs: int = 500,
            lr: float = 1e-3,
            batch_size: int = 8,
        ) -> None:
            """
            有监督训练（当有标注温度场时使用）

            Args:
                T_train_list : 训练集温度场列表（仿真生成）
                n_epochs     : 训练轮数
                lr           : 学习率
                batch_size   : 批大小
            """
            from ray_tracer import temperature_to_slowness

            ny, nx = self.grid_shape
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=lr * 0.01
            )

            # 归一化目标
            T_arr = np.stack(T_train_list, axis=0)  # (N, n_pixels)
            T_norm = (T_arr - self.T_ref) / self.T_scale

            # 生成对应输入（反投影）
            inputs = []
            for T_i in T_train_list:
                s_i  = temperature_to_slowness(T_i)
                tof_i = self.A.dot(s_i).astype(np.float32)
                At_b  = self.A.T.dot(tof_i)
                At_b  = (At_b - At_b.min()) / (At_b.max() - At_b.min() + 1e-14)
                inputs.append(At_b.reshape(1, ny, nx))

            inputs_t = torch.tensor(np.stack(inputs, axis=0)[:, np.newaxis, :, :].squeeze(2),
                                    dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(T_norm.reshape(-1, 1, ny, nx),
                                     dtype=torch.float32, device=self.device)

            N = len(T_train_list)
            self.net.train()
            for epoch in range(n_epochs):
                idx = np.random.choice(N, min(batch_size, N), replace=False)
                x_b = inputs_t[idx]
                y_b = targets_t[idx]

                optimizer.zero_grad()
                pred = self.net(x_b)
                loss = nn.functional.mse_loss(pred, y_b)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if epoch % 100 == 0 or epoch == n_epochs - 1:
                    print(f"  [AttentionUNet] epoch {epoch:4d}/{n_epochs} | loss={loss.item():.4e}")


    # =========================================================================
    # 8. 自监督对比学习先验（创新算法）
    #    参考：Chen T et al. "SimCLR" ICML 2020
    #         He K et al. "MoCo" CVPR 2020
    # =========================================================================

    class TemperatureAugmenter:
        """
        温度场物理增强策略（用于自监督对比学习）

        生成正样本对：对同一温度场施加物理等效变换，
        使变换前后 TOF 测量结果具有内在一致性约束。

        增强策略（保持物理合理性）：
        1. 空间旋转（仅 90°/180°/270°，保持整数像素对齐）
        2. 高斯噪声（模拟温度场测量不确定性）
        3. 局部平滑（模拟热扩散）
        4. 全局温升/降（均匀偏置，不改变梯度结构）
        """

        def __init__(
            self,
            grid_shape: Tuple[int, int],
            noise_std: float = 2.0,      # 温度噪声标准差 [℃]
            smooth_sigma: float = 0.5,   # 平滑核 sigma [像素]
            bias_range: Tuple[float, float] = (-20.0, 20.0),  # 全局偏置范围 [℃]
        ):
            self.grid_shape   = grid_shape
            self.noise_std    = noise_std
            self.smooth_sigma = smooth_sigma
            self.bias_range   = bias_range

        def augment(self, T_flat: np.ndarray) -> np.ndarray:
            """
            对展平温度场施加随机增强

            Args:
                T_flat : 展平温度场 (n_pixels,)
            Returns:
                T_aug  : 增强后的展平温度场 (n_pixels,)
            """
            ny, nx = self.grid_shape
            T = T_flat.reshape(ny, nx).copy()

            # 1. 随机 90° 旋转
            k = np.random.randint(0, 4)
            T = np.rot90(T, k=k)

            # 2. 高斯噪声
            T += np.random.randn(ny, nx) * self.noise_std

            # 3. 全局温度偏置
            bias = np.random.uniform(*self.bias_range)
            T += bias

            # 4. 局部平滑（可选）
            if self.smooth_sigma > 0 and np.random.rand() < 0.5:
                from scipy.ndimage import gaussian_filter
                T = gaussian_filter(T, sigma=self.smooth_sigma)

            return T.ravel()

        def generate_positive_pair(
            self, T_flat: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """生成一对正样本（同一温度场的两个增强视图）"""
            return self.augment(T_flat), self.augment(T_flat)


    class SimCLRProjectionHead(nn.Module):
        """SimCLR 投影头：将特征映射到对比学习空间"""
        def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    class SelfSupervisedContrastivePrior:
        """
        自监督对比学习温度场先验

        训练阶段（无需标注）：
        · 用 AttentionUNet 编码器提取温度场特征
        · 对同一温度场的两个增强视图构造正样本对
        · 最小化 NT-Xent 对比损失
        · 学到的表示编码了温度场的结构特征

        推理阶段：
        · 冻结编码器，仅微调解码器（线性探测或全微调）
        · 在标注样本极少时仍有良好性能

        损失函数（NT-Xent，即 SimCLR 损失）：
            L = -(1/N) Σ_i log[exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ)]
        其中 τ 为温度超参数，sim 为余弦相似度。
        """

        def __init__(
            self,
            grid_shape: Tuple[int, int],
            base_ch: int = 32,
            proj_dim: int = 64,
            temperature: float = 0.1,     # 对比损失温度超参数 τ
            dropout_p: float = 0.1,
            device: str = "cpu",
        ):
            assert TORCH_AVAILABLE, "自监督对比学习需要 PyTorch"

            self.grid_shape  = grid_shape
            self.temperature = temperature
            self.device      = torch.device(device)

            ny, nx = grid_shape
            n_pixels = ny * nx

            # 主干网络（使用 AttentionUNet 编码器部分）
            self.backbone = AttentionUNet(
                in_channels=1, base_ch=base_ch, dropout_p=dropout_p
            ).to(self.device)

            # 投影头
            feat_dim = base_ch  # 编码器最后层特征维度
            self.proj_head = SimCLRProjectionHead(
                feat_dim, feat_dim * 2, proj_dim
            ).to(self.device)

            # 数据增强器
            self.augmenter = TemperatureAugmenter(grid_shape)
            self.losses: list = []

        @staticmethod
        def _nt_xent_loss(
            z1: "torch.Tensor",
            z2: "torch.Tensor",
            tau: float = 0.1,
        ) -> "torch.Tensor":
            """
            NT-Xent 对比损失（SimCLR Loss）

            Args:
                z1, z2 : 正样本对特征 (B, proj_dim)，已 L2 归一化
                tau    : 温度超参数
            Returns:
                loss : 标量损失值
            """
            B = z1.shape[0]
            z  = torch.cat([z1, z2], dim=0)          # (2B, D)
            z  = nn.functional.normalize(z, dim=1)

            # 余弦相似度矩阵 (2B, 2B)
            sim = torch.mm(z, z.T) / tau
            # 对角线置 -inf（排除自身）
            sim.fill_diagonal_(float("-inf"))

            # 正样本标签：i 的正样本为 i+B（或 i-B）
            labels = torch.cat([
                torch.arange(B, 2 * B),
                torch.arange(0, B),
            ]).to(z.device)

            loss = nn.functional.cross_entropy(sim, labels)
            return loss

        def _extract_features(
            self, T_flat: np.ndarray
        ) -> "torch.Tensor":
            """将展平温度场转换为网络特征（全局平均池化后的向量）"""
            ny, nx = self.grid_shape
            T_norm = (T_flat - T_flat.mean()) / (T_flat.std() + 1e-8)
            x = torch.tensor(
                T_norm.reshape(1, 1, ny, nx), dtype=torch.float32, device=self.device
            )
            # 使用 backbone 的编码器部分提取特征
            e1 = self.backbone.enc1(x)
            feat = e1.mean(dim=[2, 3])    # 全局平均池化 (1, base_ch)
            return feat

        def train_contrastive(
            self,
            T_unlabeled: list,   # 无标注温度场列表（仿真或实测）
            n_epochs: int = 300,
            lr: float = 1e-3,
            batch_size: int = 16,
        ) -> None:
            """
            自监督对比预训练

            Args:
                T_unlabeled : 无标注温度场列表
                n_epochs    : 训练轮数
                lr          : 学习率
            """
            optimizer = torch.optim.Adam(
                list(self.backbone.parameters()) + list(self.proj_head.parameters()),
                lr=lr,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=lr * 0.01
            )

            N = len(T_unlabeled)
            for epoch in range(n_epochs):
                idx = np.random.choice(N, min(batch_size, N), replace=False)
                batch = [T_unlabeled[i] for i in idx]

                z1_list, z2_list = [], []
                for T_flat in batch:
                    aug1, aug2 = self.augmenter.generate_positive_pair(T_flat)
                    feat1 = self._extract_features(aug1)
                    feat2 = self._extract_features(aug2)
                    z1_list.append(self.proj_head(feat1))
                    z2_list.append(self.proj_head(feat2))

                z1 = torch.cat(z1_list, dim=0)
                z2 = torch.cat(z2_list, dim=0)

                optimizer.zero_grad()
                loss = self._nt_xent_loss(z1, z2, tau=self.temperature)
                loss.backward()
                optimizer.step()
                scheduler.step()

                self.losses.append(float(loss.item()))

                if epoch % 100 == 0 or epoch == n_epochs - 1:
                    print(
                        f"  [SimCLR] epoch {epoch:4d}/{n_epochs} | "
                        f"contrastive_loss={loss.item():.4f}"
                    )

    # =========================================================================
    # 9. 残差PINN网络（用于Stage 2温度场精化）
    #    图像到图像的残差学习 + PDE物理约束（Laplacian=0）
    # =========================================================================

    class ResidualPINN(nn.Module):
        """
        图像到图像的残差PINN网络（CNN编码器-解码器）
        输入: (B, 1, 16, 16) Stage1归一化温度场
        输出: (B, 1, 16, 16) 残差 R_pred
        """

        def __init__(self, in_ch=1, base_ch=32, dropout=0.1):
            super().__init__()
            c = base_ch

            # 编码器
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_ch, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
            self.enc2 = nn.Sequential(
                nn.Conv2d(c, c * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(c * 2),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.MaxPool2d(2)

            # 瓶颈
            self.bottleneck = nn.Sequential(
                nn.Conv2d(c * 2, c * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(c * 4),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            )

            # 解码器（无注意力门控，直接concat）
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(c * 4, c * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(c * 2),
                nn.ReLU(inplace=True),
            )
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(c * 2, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )

            # 输出（无激活，残差可为正可负）
            self.out_conv = nn.Conv2d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            bn = self.bottleneck(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(bn), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out_conv(d1)

        def pde_loss(self, R):
            """
            计算残差场的Laplacian=0损失
            L_pde = mean((∂²R/∂x² + ∂²R/∂y²)²)
            使用有限差分计算二阶导数（Neumann边界条件）
            R: (B, 1, H, W)
            """
            # 有限差分二阶导数
            d2R_dx2 = R[:, :, 1:-1, 2:] - 2 * R[:, :, 1:-1, 1:-1] + R[:, :, 1:-1, :-2]
            d2R_dy2 = R[:, :, 2:, 1:-1] - 2 * R[:, :, 1:-1, 1:-1] + R[:, :, :-2, 1:-1]

            lap = d2R_dx2 + d2R_dy2
            return torch.mean(lap ** 2)

    class PINNStage2Trainer:
        """
        PINN Stage2 训练器
        完全复用 0328best.py 的训练数据生成和评估逻辑
        """

        def __init__(
            self,
            stage1_X,      # list of Stage1 temperature fields (ny, nx)
            stage1_Y,      # list of ground truth temperature fields (ny, nx)
            GLOBAL_T_MIN,
            GLOBAL_T_MAX,
            lambda_pde=0.01,
            lr=1e-3,
            weight_decay=1e-5,
            device='cpu',
        ):
            self.X = [np.array(x) for x in stage1_X]
            self.Y = [np.array(y) for y in stage1_Y]
            self.GLOBAL_T_MIN = GLOBAL_T_MIN
            self.GLOBAL_T_MAX = GLOBAL_T_MAX
            self.scale = GLOBAL_T_MAX - GLOBAL_T_MIN + 1e-14
            self.lambda_pde = lambda_pde
            self.device = device

            # 计算归一化残差标签
            self.X_n = [(x - GLOBAL_T_MIN) / self.scale for x in self.X]
            self.Y_n = [(y - x) / self.scale for x, y in zip(self.X, self.Y)]

        def to_torch_dataset(self, indices=None):
            """将数据转换为 PyTorch Dataset"""
            if indices is None:
                X_n, Y_n = self.X_n, self.Y_n
            else:
                X_n = [self.X_n[i] for i in indices]
                Y_n = [self.Y_n[i] for i in indices]

            class _DS:
                def __init__(ds, x_list, y_list):
                    ds.x = [torch.FloatTensor(x).unsqueeze(0) for x in x_list]
                    ds.y = [torch.FloatTensor(y).unsqueeze(0) for y in y_list]

                def __len__(ds):
                    return len(ds.x)

                def __getitem__(ds, idx):
                    return ds.x[idx], ds.y[idx]

            return _DS(X_n, Y_n)

        def train(
            self,
            train_indices,
            val_indices,
            model,
            n_epochs=300,
            batch_size=16,
            verbose=True,
        ):
            """
            训练 PINN 模型
            L_total = MSE(R_pred, R_true) + lambda_pde * L_pde
            """
            import torch.optim as optim
            from torch.utils.data import DataLoader

            train_ds = self.to_torch_dataset(train_indices)
            val_ds = self.to_torch_dataset(val_indices)
            train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

            criterion = nn.MSELoss()
            train_losses, val_rmses = [], []
            best_rmse = float('inf')
            best_state = None

            for epoch in range(n_epochs):
                model.train()
                epoch_loss = 0.0
                for x, y in train_ld:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()

                    R_pred = model(x)
                    L_mse = criterion(R_pred, y)
                    L_pde = model.pde_loss(R_pred)
                    loss = L_mse + self.lambda_pde * L_pde

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * x.size(0)

                epoch_loss /= len(train_ds)
                train_losses.append(epoch_loss)

                if (epoch + 1) % 20 == 0 or epoch == 0:
                    model.eval()
                    vp, vt = [], []
                    with torch.no_grad():
                        for x, y in DataLoader(val_ds, batch_size=batch_size):
                            x, y = x.to(self.device), y.to(self.device)
                            r = model(x)
                            vp_denorm = (x.cpu().numpy()[:, 0] + r.cpu().numpy()[:, 0]) * self.scale + self.GLOBAL_T_MIN
                            vt_denorm = y.cpu().numpy()[:, 0] * self.scale + self.GLOBAL_T_MIN
                            for p_row, t_row in zip(vp_denorm, vt_denorm):
                                vp.append(p_row.ravel())
                                vt.append(t_row.ravel())

                    vp = np.concatenate(vp).ravel()
                    vt = np.concatenate(vt).ravel()

                    from visualization import compute_metrics
                    m = compute_metrics(vt, vp)
                    val_rmses.append(m['RMSE(%)'])

                    if m['RMSE(%)'] < best_rmse:
                        best_rmse = m['RMSE(%)']
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                    if verbose:
                        print(f"  Epoch {epoch+1:4d}/{n_epochs} | "
                              f"Loss={epoch_loss:.6f} | "
                              f"Val RMSE={m['RMSE(%)']:.2f}% | "
                              f"best={best_rmse:.2f}%")

                scheduler.step()

            if best_state:
                model.load_state_dict(best_state)

            return {
                'train_losses': train_losses,
                'val_rmses': val_rmses,
                'best_rmse': best_rmse,
                'best_state': best_state,
            }

        def predict(self, model, T_s1, device='cpu'):
            """
            用训练好的 PINN 模型预测
            T_pred = T_s1 + PINN(T_s1_norm) * scale
            """
            model.eval()
            x_n = (T_s1 - self.GLOBAL_T_MIN) / self.scale
            x_t = torch.FloatTensor(x_n).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                R = model(x_t).cpu().numpy().ravel()
            T_pred = T_s1.ravel() + R * self.scale
            return T_pred


# =============================================================================
# 9. RVFLN —— 随机向量函数链接网络（不依赖 PyTorch）
#    参考：Li Y, Lei J, Ren S. Applied Thermal Engineering 2025, 277:127113
#         Malik A K et al. Appl. Soft Comput. 2023.
#
#    网络结构（单层前馈 + 直连）：
#        K = [K1 | K2]     K1: 增强节点特征，K2: 直连输入特征
#        K f = g            解：f = (K^T K + α I)^{-1} K^T g  （岭回归）
#
#    创新（MORVFLN）：
#        将权重训练问题转化为多目标优化：
#        min { ||Kf - g||² / (||f||²+1) ,  L1_2(f) }
#        其中 L1_2(f) = ||f||_1 - ||f||_2  （L1-2 稀疏先验）
#        使用帕累托加权扫描近似多目标前沿（轻量级替代 NSGA-II）
# =============================================================================

class RVFLN:
    """
    随机向量函数链接网络（Random Vector Functional Link Network）

    单层前馈网络 + 直连（skip connection），无需迭代训练。
    仅需拟合输出权重（岭回归或 LSQR），训练极快。

    网络结构：
        输入层 s ∈ R^n_in
        增强层（随机映射）: K1 ∈ R^{N×δ}，K1_{ij} = σ(a_j^T s_i + c_j)
        直连层: K2 ∈ R^{N×n_in}，K2 = S（输入矩阵本身）
        输出: f = [K1 | K2] w，其中 w 通过岭回归求得

    典型用法（MOTDRM 第二阶段）：
        1. 以粗网格温度场的空间坐标 (x, y) 为输入
        2. 以对应温度值 T 为目标
        3. 训练后以细网格坐标为输入，推理精细温度场
    """

    def __init__(
        self,
        n_hidden: int = 300,
        alpha: float  = 1e-3,      # 岭正则化参数
        activation: str = "tanh",  # 激活函数：'tanh','sigmoid','relu','triangular'
        random_state: Optional[int] = None,
    ):
        """
        Args:
            n_hidden     : 增强节点数（随机隐层神经元数）
            alpha        : 岭回归正则化参数
            activation   : 激活函数名称（'triangular' 对应三角基函数）
            random_state : 随机种子（保证可重复性）
        """
        self.n_hidden    = int(n_hidden)
        self.alpha       = float(alpha)
        self.activation  = activation
        self.random_state = random_state

        self._rng  : Optional[np.random.RandomState] = None
        self._W    : Optional[np.ndarray] = None  # 随机权重 (n_in, n_hidden)
        self._b    : Optional[np.ndarray] = None  # 随机偏置 (n_hidden,)
        self._beta : Optional[np.ndarray] = None  # 输出权重
        self._n_in : int = 0

    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """应用激活函数"""
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == "relu":
            return np.maximum(x, 0.0)
        elif self.activation == "triangular":
            # 三角基函数：max(1 - |x|, 0)（如 Li et al. 2025 所用）
            return np.maximum(1.0 - np.abs(x), 0.0)
        else:
            return np.tanh(x)

    def _build_feature_matrix(self, S: np.ndarray) -> np.ndarray:
        """
        构建增强特征矩阵 K = [K1 | K2]

        Args:
            S : 输入矩阵 (N, n_in)
        Returns:
            K : 特征矩阵 (N, n_hidden + n_in)
        """
        K1 = self._activation_fn(S @ self._W + self._b)  # (N, n_hidden)
        K2 = S                                             # (N, n_in) 直连
        return np.hstack([K1, K2])                        # (N, n_hidden + n_in)

    def fit(self, S: np.ndarray, g: np.ndarray) -> "RVFLN":
        """
        训练 RVFLN（岭回归求输出权重）

        Args:
            S : 训练输入矩阵 (N, n_in)，如空间坐标或反投影特征
            g : 训练目标向量 (N,) 或矩阵 (N, n_out)，如温度值
        Returns:
            self（支持链式调用）
        """
        rng = np.random.RandomState(self.random_state)
        self._rng = rng
        N, n_in = S.shape
        self._n_in = n_in

        # 初始化随机权重（Gaussian 分布）
        self._W = rng.randn(n_in, self.n_hidden)   # (n_in, n_hidden)
        self._b = rng.randn(self.n_hidden)          # (n_hidden,)

        K = self._build_feature_matrix(S)           # (N, n_hidden + n_in)
        d_feat = K.shape[1]

        # 岭回归：β = (K^T K + α I)^{-1} K^T g
        KtK = K.T @ K
        KtK_reg = KtK + self.alpha * np.eye(d_feat)
        Ktg = K.T @ g

        try:
            self._beta = np.linalg.solve(KtK_reg, Ktg)
        except np.linalg.LinAlgError:
            self._beta, _, _, _ = np.linalg.lstsq(KtK_reg, Ktg, rcond=None)

        return self

    def predict(self, S: np.ndarray) -> np.ndarray:
        """
        推理

        Args:
            S : 测试输入矩阵 (N_test, n_in)
        Returns:
            预测输出 (N_test,) 或 (N_test, n_out)
        """
        if self._beta is None:
            raise RuntimeError("RVFLN 尚未训练，请先调用 fit()")
        K = self._build_feature_matrix(S)
        return K @ self._beta

    def compute_objective(
        self, S: np.ndarray, g: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算两个目标函数值（用于多目标评估）

        J1 = ||Kf - g||² / (||f||² + 1)   （加权数据拟合）
        J2 = L1_2(f) = ||f||_1 - ||f||_2  （L1-2 稀疏先验）

        Returns:
            (J1, J2) : 两个目标值的元组
        """
        if self._beta is None:
            raise RuntimeError("RVFLN 尚未训练")

        K = self._build_feature_matrix(S)
        f = self._beta
        Kf = K @ f

        J1 = float(np.sum((Kf - g) ** 2) / (np.dot(f, f) + 1.0))
        J2 = float(np.sum(np.abs(f)) - np.linalg.norm(f))  # L1 - L2

        return J1, J2


class MORVFLN:
    """
    多目标随机向量函数链接网络（Multi-Objective RVFLN）
    对应 MOTDRM 第二阶段（Li et al. Applied Thermal Engineering 2025）

    创新点：
    1. 将 RVFLN 训练问题转化为多目标优化：
           min { J1(f), J2(f) }
           J1 = ||Kf - g||² / (||f||² + 1)   （数据拟合 + 模型不确定性）
           J2 = ||f||_1 - ||f||_2             （L1-2 稀疏先验，防过拟合）
    2. 使用帕累托权重扫描生成近似帕累托前沿（避免 NSGA-II 的高计算成本）
    3. 用最大化超体积指标（HV proxy）选择最优解

    使用流程（Algorithm 2 in Li et al. 2025）：
    Step 1: 收集 LowRankReconstructor 粗网格重建温度场
    Step 2: 用粗网格空间坐标 + 温度作为训练样本 (s_i, g_i)
    Step 3: 训练 MORVFLN（帕累托权重扫描）
    Step 4: 用细网格坐标推理精细温度分布
    """

    def __init__(
        self,
        n_hidden: int = 300,
        activation: str = "triangular",
        n_weights: int = 20,     # 帕累托权重扫描数量
        alpha_min: float = 1e-6,
        alpha_max: float = 1.0,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Args:
            n_hidden     : RVFLN 增强节点数
            activation   : 激活函数（'triangular' 为论文默认）
            n_weights    : 帕累托权重扫描步数（替代 NSGA-II，轻量高效）
            alpha_min    : 正则化参数搜索下界
            alpha_max    : 正则化参数搜索上界
            random_state : 随机种子
            verbose      : 是否打印训练过程
        """
        self.n_hidden    = n_hidden
        self.activation  = activation
        self.n_weights   = n_weights
        self.alpha_min   = alpha_min
        self.alpha_max   = alpha_max
        self.random_state = random_state
        self.verbose     = verbose

        self._best_model : Optional[RVFLN] = None
        self._pareto_front : list = []  # [(J1, J2, alpha, model), ...]

    def _is_dominated(self, p: Tuple[float, float], front: list) -> bool:
        """
        判断解 p = (J1, J2) 是否被前沿中某解支配
        （支配条件：另一解在两个目标上均不差，至少一个更好）
        """
        for q in front:
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                return True
        return False

    def fit(
        self,
        S_coarse: np.ndarray,     # 粗网格坐标 (N_coarse, n_in)
        T_coarse: np.ndarray,     # 粗网格温度 (N_coarse,)
    ) -> "MORVFLN":
        """
        多目标训练 RVFLN

        对正则化参数 α 进行对数均匀扫描，每个 α 对应一个 RVFLN 解，
        构建近似帕累托前沿。最终选择最接近帕累托前沿"肘部"的解。

        Args:
            S_coarse : 粗网格空间坐标 (N_coarse, 2) 或特征矩阵
            T_coarse : 粗网格温度向量 (N_coarse,)
        Returns:
            self（支持链式调用）
        """
        alphas = np.logspace(
            np.log10(self.alpha_min),
            np.log10(self.alpha_max),
            self.n_weights,
        )

        candidates = []  # [(J1, J2, alpha, model)]

        for i, alpha in enumerate(alphas):
            model = RVFLN(
                n_hidden=self.n_hidden,
                alpha=alpha,
                activation=self.activation,
                random_state=self.random_state,
            )
            model.fit(S_coarse, T_coarse)
            J1, J2 = model.compute_objective(S_coarse, T_coarse)

            if self.verbose:
                print(f"  [MORVFLN] alpha={alpha:.2e} | J1={J1:.4e} | J2={J2:.4e}")

            candidates.append((J1, J2, alpha, model))

        # ── 构建帕累托前沿 ──────────────────────────────────────
        pareto_front = []
        for cand in candidates:
            p = (cand[0], cand[1])
            if not self._is_dominated(p, [(c[0], c[1]) for c in pareto_front]):
                # 移除被当前解支配的解
                pareto_front = [
                    c for c in pareto_front
                    if not (cand[0] <= c[0] and cand[1] <= c[1]
                            and (cand[0] < c[0] or cand[1] < c[1]))
                ]
                pareto_front.append(cand)

        self._pareto_front = pareto_front

        if self.verbose:
            print(f"  [MORVFLN] 帕累托前沿解数量：{len(pareto_front)}")

        # ── 选择最优解（拐角点：最接近理想点的解）──────────────
        #   理想点 = (min J1, min J2)，选取 L2 距离最小的帕累托解
        J1_vals = np.array([c[0] for c in pareto_front])
        J2_vals = np.array([c[1] for c in pareto_front])

        # 归一化（防止量纲差异）
        J1_norm = (J1_vals - J1_vals.min()) / (np.ptp(J1_vals) + 1e-14)
        J2_norm = (J2_vals - J2_vals.min()) / (np.ptp(J2_vals) + 1e-14)
        dist_to_ideal = np.sqrt(J1_norm ** 2 + J2_norm ** 2)

        best_idx = int(np.argmin(dist_to_ideal))
        self._best_model = pareto_front[best_idx][3]

        if self.verbose:
            best_alpha = pareto_front[best_idx][2]
            print(f"  [MORVFLN] 最优 alpha={best_alpha:.2e} | "
                  f"J1={pareto_front[best_idx][0]:.4e} | "
                  f"J2={pareto_front[best_idx][1]:.4e}")

        return self

    def predict(self, S_fine: np.ndarray) -> np.ndarray:
        """
        用最优 RVFLN 模型推理细网格温度分布

        Args:
            S_fine : 细网格空间坐标 (N_fine, n_in)
        Returns:
            T_fine : 细网格温度预测 (N_fine,)
        """
        if self._best_model is None:
            raise RuntimeError("MORVFLN 尚未训练，请先调用 fit()")
        return self._best_model.predict(S_fine)

    def get_pareto_front(self) -> list:
        """
        返回帕累托前沿解列表，每项为 (J1, J2, alpha, RVFLN model)
        """
        return list(self._pareto_front)


class MOTDRMReconstructor:
    """
    两阶段多目标温度分布重建方法（MOTDRM）
    对应 Li Y et al. Applied Thermal Engineering 2025, 277:127113

    阶段 1：LowRankReconstructor（低秩正则化，ADMM 求解）
             → 粗网格温度场（整体趋势）
    阶段 2：MORVFLN（多目标 RVFLN，空间坐标→精细温度）
             → 细网格温度场（局部细节）

    特点：
    · 无需预收集训练数据，无需预训练
    · 自动平衡数据拟合和正则化（多目标优化）
    · 处理模型误差和测量噪声双重不确定性
    · 集成低秩先验（温度场通常低秩）
    · L1-2 稀疏先验（RVFLN 权重稀疏，抑制过拟合）

    使用示例：
        reconstructor = MOTDRMReconstructor(
            A=A, b=tof_noisy, grid_shape=(ny, nx),
            coarse_shape=(8, 8), domain=(x0, x1, y0, y1)
        )
        T_coarse, T_fine = reconstructor.run()
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        grid_shape: Tuple[int, int],
        domain: Tuple[float, float, float, float],
        coarse_shape: Optional[Tuple[int, int]] = None,
        # 第一阶段参数
        lr_alpha: float = 5e-5,
        lr_rho: float   = 0.01,
        lr_n_iter: int  = 100,
        # 第二阶段参数
        rvfln_hidden: int  = 300,
        rvfln_activation: str = "triangular",
        rvfln_n_weights: int  = 20,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Args:
            A            : 系统矩阵
            b            : TOF 测量向量
            grid_shape   : 精细网格 (ny, nx)
            domain       : 测量域 (x_min, x_max, y_min, y_max) [m]
            coarse_shape : 粗网格形状（默认为 fine 的 1/2）
            lr_alpha     : 低秩正则化权重（第一阶段）
            lr_rho       : ADMM 惩罚参数（第一阶段）
            lr_n_iter    : ADMM 最大迭代数（第一阶段）
            rvfln_hidden : RVFLN 增强节点数（第二阶段）
            rvfln_activation : 激活函数（第二阶段）
            rvfln_n_weights  : 帕累托权重扫描数（第二阶段）
            random_state : 随机种子
            verbose      : 是否打印过程信息
        """
        from reconstruction import LowRankReconstructor

        self.A           = A
        self.b           = b.astype(np.float64)
        self.grid_shape  = grid_shape
        self.domain      = domain
        self.verbose     = verbose

        ny, nx = grid_shape
        if coarse_shape is None:
            coarse_shape = (max(ny // 2, 4), max(nx // 2, 4))
        self.coarse_shape = coarse_shape

        # 第一阶段：低秩重建器
        self._stage1 = LowRankReconstructor(
            A=A, b=b, grid_shape=grid_shape,
            alpha=lr_alpha, rho=lr_rho, n_iter=lr_n_iter,
            verbose=verbose,
        )

        # 第二阶段：MORVFLN
        self._stage2 = MORVFLN(
            n_hidden=rvfln_hidden,
            activation=rvfln_activation,
            n_weights=rvfln_n_weights,
            random_state=random_state,
            verbose=verbose,
        )

    def _make_grid_coords(
        self, shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        生成归一化空间坐标网格

        Args:
            shape : (ny, nx)
        Returns:
            coords : (ny*nx, 2)，归一化到 [-1, 1]
        """
        ny, nx = shape
        x_min, x_max, y_min, y_max = self.domain
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xx, yy = np.meshgrid(xs, ys)

        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
        # 归一化到 [-1, 1]
        coords[:, 0] = 2 * (coords[:, 0] - x_min) / (x_max - x_min + 1e-14) - 1
        coords[:, 1] = 2 * (coords[:, 1] - y_min) / (y_max - y_min + 1e-14) - 1
        return coords.astype(np.float64)

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行两阶段重建

        Returns:
            T_coarse : 粗网格温度场（第一阶段），shape = (ny*nx,)
                       注：由低秩重建器在 fine 网格上的结果下采样后再在 coarse 网格坐标上重建
            T_fine   : 精细温度场（第二阶段），shape = (ny*nx,)
        """
        from ray_tracer import slowness_to_temperature

        ny, nx = self.grid_shape
        ny_c, nx_c = self.coarse_shape

        if self.verbose:
            print("=" * 55)
            print("  MOTDRM 第一阶段：低秩正则化重建（粗网格）")
            print("=" * 55)

        # 第一阶段：在 fine 网格上低秩重建（获得粗尺度温度）
        s_stage1 = self._stage1.solve()
        T_stage1 = slowness_to_temperature(s_stage1)   # shape (ny*nx,)

        # 将 fine 网格温度下采样至 coarse 网格
        T_fine_img = T_stage1.reshape(ny, nx)

        # 双线性下采样（scipy.ndimage.zoom）
        try:
            from scipy.ndimage import zoom
            zoom_y = ny_c / ny
            zoom_x = nx_c / nx
            T_coarse_img = zoom(T_fine_img, (zoom_y, zoom_x), order=1)
        except Exception:
            # 降级：简单均匀采样
            iy = np.round(np.linspace(0, ny - 1, ny_c)).astype(int)
            ix = np.round(np.linspace(0, nx - 1, nx_c)).astype(int)
            T_coarse_img = T_fine_img[np.ix_(iy, ix)]

        T_coarse_flat = T_coarse_img.ravel()

        if self.verbose:
            print("=" * 55)
            print("  MOTDRM 第二阶段：MORVFLN 精细推理")
            print("=" * 55)

        # 构造训练样本：粗网格空间坐标 → 对应温度
        coarse_coords = self._make_grid_coords(self.coarse_shape)  # (ny_c*nx_c, 2)

        # 第二阶段训练
        self._stage2.fit(coarse_coords, T_coarse_flat)

        # 在 fine 网格坐标上推理
        fine_coords = self._make_grid_coords(self.grid_shape)       # (ny*nx, 2)
        T_fine_pred = self._stage2.predict(fine_coords)             # (ny*nx,)

        if self.verbose:
            print("  MOTDRM 重建完成。")

        return T_stage1, T_fine_pred
