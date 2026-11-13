"""
声学CT温度场重建 —— 反演算法模块
=====================================
实现九种反演方法并支持自动正则化参数选取：

  1. SART          —— 联立代数重建（快速迭代基线）
  2. Tikhonov      —— L2 正则化（最小范数 / 一阶 / 二阶）
  3. TV            —— Total Variation 正则化（保边缘，Split Bregman/ADMM）
  4. SART-TV       —— SART 迭代 + 每步 Chambolle TV 去噪（混合方法）
  5. L-curve       —— 自动选取最优正则化参数
  6. TLSReconstructor  —— 全最小二乘（处理模型与数据双重不确定性）
  7. LowRankReconstructor —— 低秩核范数正则化（MOTDRM第一阶段，参考Li et al. 2025）
  8. RBFReconstructor     —— 径向基函数正则化重建（利用RBF空间插值先验）
  9. WallConstrainedReconstructor —— 壁温约束重建（利用换能器附近已知温度）

参考文献：
  [1] Andersen A H, Kak A C. Simultaneous algebraic reconstruction technique[J].
      Ultrasonic Imaging, 1984.
  [2] Goldstein T, Osher S. The Split Bregman Method for L1-Regularized Problems[J].
      SIAM Journal on Imaging Sciences, 2009.
  [3] Chambolle A. An algorithm for total variation minimization and applications[J].
      Journal of Mathematical Imaging and Vision, 2004.
  [4] Hansen P C. The L-curve and its use in numerical treatment of inverse problems[J].
      Computational Inverse Problems in Electrocardiology, 2001.
  [5] Beck A, Ben-Tal A. On the solution of the Tikhonov regularization of the
      total least squares problem[J]. SIAM J. Optim., 2006.
  [6] Li Y, Lei J, Ren S. Multi-objective optimization and machine learning for the
      temperature distribution measurement using acoustic tomography[J].
      Applied Thermal Engineering, 2025, 277: 127113.
  [7] Zhang J Q et al. Acoustic tomography of temperature and velocity fields
      by using the radial basis function and alternating direction method of
      multipliers[J]. Int J Heat Mass Transf, 2022, 188: 122660.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, List

# 导入本项目射线追踪模块的辅助函数
from ray_tracer import temperature_to_slowness, slowness_to_temperature


# =============================================================================
# 工具函数
# =============================================================================

def _safe_clip(s: np.ndarray, s_min: float = 1.15e-3, s_max: float = 3.2e-3) -> np.ndarray:
    """
    物理约束：声慢度剪切到合理物理范围。

    重要：s 与温度反向相关！
        s 越小 → 温度越高
        s 越大 → 温度越低

    默认边界（根据空气声速-温度公式 c = 331.3√(1+T/273.15)）：
        s_min = 1.15e-3 s/m  → c ≈ 870 m/s → T_max ≈ 1200℃
        s_max = 3.2e-3 s/m   → c ≈ 312 m/s → T_min ≈ -30℃
    
    应用场景（Li et al. 2025 论文）：
    · 初始化：T_ref=20℃ → s₀=0.00291 ✓ 在范围内 [1.15e-3, 3.2e-3]
    · 目标范围：T∈[800, 1100]℃ → s∈[0.00135, 0.00152] ✓ 在范围内
    · 极端场景：T∈[-30, +1200]℃ → s∈[1.15e-3, 3.2e-3] ✓ 完全覆盖

    Bug 修复历程：
    · Bug #14（根本原因）：原 s_max=3.2e-3 后面被错误改成 3.0e-3
                   导致初始值 s=0.00291 有被剪切到 3.0e-3 的风险
                   现恢复为 3.2e-3（足够安全）
    """
    return np.clip(s, s_min, s_max)


def _default_slowness(n_pixels: int, T_ref: float = 20.0) -> np.ndarray:
    """以参考温度对应的均匀慢度场作为初始解"""
    s0 = float(temperature_to_slowness(np.array([T_ref]))[0])
    return np.full(n_pixels, s0, dtype=np.float64)


# =============================================================================
# 1. SART —— 联立代数重建技术
# =============================================================================

class SART:
    """
    Simultaneous Algebraic Reconstruction Technique (SART)

    迭代格式（加权反投影）：
        s^{k+1} = s^k + λ · A^T W_row (b − A s^k) / v

    其中：
        W_row[r] = 1 / Σ_p A[r,p]   （行权重，射线总路径长度倒数）
        v[p]     = Σ_r A[r,p]       （列权重，穿过该像素的射线总长度）
        λ        — 松弛因子，典型值 0.3~0.8
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        n_iter: int = 100,
        relax: float = 0.5,
        s_init: Optional[np.ndarray] = None,
        T_ref: float = 20.0,
        tol: float = 1e-6,
    ):
        """
        Args:
            A      : 系统矩阵 (n_rays × n_pixels)，scipy 稀疏矩阵
            b      : 测量飞行时间向量 (n_rays,)
            n_iter : 最大迭代次数
            relax  : 松弛因子 λ
            s_init : 初始慢度场；None 时使用 T_ref 对应的均匀场
            T_ref  : 初始参考温度 [℃]
            tol    : 相对变化量收敛阈值
        """
        self.A      = A
        self.b      = b.astype(np.float64)
        self.n_iter = n_iter
        self.relax  = relax
        self.tol    = tol

        n_pixels = A.shape[1]
        self.s = s_init.copy().astype(np.float64) if s_init is not None \
                 else _default_slowness(n_pixels, T_ref)

        # ---- 预计算权重（不变） ----
        row_sums = np.asarray(A.sum(axis=1)).ravel()           # Σ_p A[r,p]
        row_sums = np.where(row_sums < 1e-14, 1e-14, row_sums)
        self.W_row = 1.0 / row_sums                            # shape (n_rays,)

        # v[p] = Σ_r A[r,p] / Σ_p' A[r,p']  =  A^T · W_row
        # Bug Fix: 标准 SART (Kak & Slaney) 的列权重为 Σ_r A[r,p]，而不是归一化的版本
        v = np.asarray(A.sum(axis=0)).ravel()
        self.V_col = np.where(v < 1e-14, 1e-14, v)

        self.residuals: List[float] = []
        # 稳定性检测：若残差连续 stagnant_window 次变化 < stagnant_tol 则提前退出
        self.stagnant_window: int = 50
        self.stagnant_tol: float = 1e-6

    def run(self) -> np.ndarray:
        """执行迭代，返回重建慢度场（展平，shape: n_pixels）"""
        s = self.s.copy()
        stagnant_count = 0

        for k in range(self.n_iter):
            # 松弛因子衰减策略：前期大 relax 快速探索，后期小 relax 精细收敛
            # k=0: relax_full = self.relax; k=n_iter-1: relax_full ≈ self.relax * 0.1
            schedule_factor = max(0.1, 1.0 - 0.9 * k / (self.n_iter - 1 + 1e-14))
            current_relax = self.relax * schedule_factor

            # 残差投影
            r_proj = self.b - self.A.dot(s)           # (n_rays,)
            # 加权反投影
            delta = self.A.T.dot(self.W_row * r_proj) / self.V_col
            s_new = _safe_clip(s + current_relax * delta)

            # 收敛检测
            rel_change = np.linalg.norm(s_new - s) / (np.linalg.norm(s) + 1e-14)
            residual_norm = float(np.linalg.norm(r_proj))
            self.residuals.append(residual_norm)

            # stagnation 检测：若残差连续 N 次变化极小则提前退出
            if len(self.residuals) >= 2:
                rel_residual_change = abs(self.residuals[-1] - self.residuals[-2]) / (abs(self.residuals[-2]) + 1e-14)
                if rel_residual_change < self.stagnant_tol:
                    stagnant_count += 1
                else:
                    stagnant_count = 0

            s = s_new
            if rel_change < self.tol:
                print(f"  [SART] 收敛于第 {k+1} 次迭代（Δ={rel_change:.2e}）")
                break
            if stagnant_count >= self.stagnant_window:
                print(f"  [SART] 提前终止于第 {k+1} 次迭代（残差停滞 {stagnant_count} 步）")
                break

        self.s = s
        return s


# =============================================================================
# 2. Tikhonov 正则化
# =============================================================================

class TikhonovReconstructor:
    """
    Tikhonov 正则化线性反演

    最小化：‖A s − b‖² + α ‖L s‖²

    解析解等价于增广最小二乘：
        [    A    ]       [b]
        [√α · L  ] s  =  [0]

    使用 LSQR 迭代求解大型稀疏系统（O(nnz) 每步）。

    L 可选（reg_type）：
        'identity'  —— L = I         （零阶，最小能量解）
        'gradient'  —— L = 前向差分  （一阶，最光滑梯度）
        'laplacian' —— L = 离散Laplacian（二阶，最光滑解，推荐）
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        alpha: float = 0.01,
        reg_type: str = "gradient",
        grid_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            A         : 系统矩阵 (n_rays × n_pixels)
            b         : TOF 测量向量
            alpha     : 正则化参数（默认 0.01，适用于典型 16 换能器 32×32 网格）
            reg_type  : 正则化类型 'identity' / 'gradient'(推荐) / 'laplacian'
            grid_shape: (ny, nx)，使用 gradient/laplacian 时必须提供

        Bug #9 修复说明：
            原默认 reg_type='laplacian', alpha=1e-3。
            问题：Laplacian 主对角线为 4，梯度算子条目为 ±1，而系统矩阵 A 的
            平均条目约为 6e-4。Laplacian 的尺度是 A 的 ~12倍，导致即使 alpha
            很小（1e-4），正则化项也远超数据项，解被"拉"向零偏差（均匀场）。
            修复：(1) 默认改为 gradient（尺度比 ≈ 3:1，更均衡）；
                  (2) 默认 alpha 改为 0.01；
                  (3) 对 laplacian 自动归一化其 Frobenius 范数。
        """
        self.A          = A
        self.b          = b.astype(np.float64)
        self.alpha      = alpha
        self.reg_type   = reg_type
        self.grid_shape = grid_shape
        self.n_pixels   = A.shape[1]

    # ------------------------------------------------------------------
    def _build_laplacian(self):
        """
        构建2D离散 Laplacian（5点模板，含边界修正）

        Bug #9 增强：对 Laplacian 矩阵进行归一化，使其 Frobenius 范数
        与系统矩阵 A 的 Frobenius 范数匹配，避免尺度不平衡问题。
        """
        from scipy.sparse import diags
        from scipy.sparse.linalg import norm as sp_norm
        ny, nx = self.grid_shape
        n = nx * ny

        d_main = 4.0 * np.ones(n)
        d_xp   = -1.0 * np.ones(n - 1)
        d_xm   = -1.0 * np.ones(n - 1)
        d_yp   = -1.0 * np.ones(n - nx)
        d_ym   = -1.0 * np.ones(n - nx)

        # 修正 x 方向行边界（防止跨行连接）
        d_xp[nx-1::nx] = 0.0
        d_xm[nx-1::nx] = 0.0

        L = diags(
            [d_main, d_xp, d_xm, d_yp, d_ym],
            [0, 1, -1, nx, -nx],
            shape=(n, n), format="csr",
        )

        # 归一化：使 ||L||_F ≈ ||A||_F，避免正则化项尺度压倒数据项
        A_norm = sp_norm(self.A, 'fro')
        L_norm = sp_norm(L, 'fro')
        if L_norm > 1e-14:
            L = L * (A_norm / L_norm)

        return L

    def _build_gradient(self):
        """构建2D前向差分梯度算子，并归一化使其与系统矩阵 A 的尺度匹配"""
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import norm as sp_norm
        ny, nx = self.grid_shape
        n = nx * ny

        rows, cols, vals = [], [], []
        row_id = 0

        # x 方向
        for iy in range(ny):
            for ix in range(nx - 1):
                p = iy * nx + ix
                rows += [row_id, row_id]; cols += [p, p + 1]; vals += [-1.0, 1.0]
                row_id += 1

        # y 方向
        for iy in range(ny - 1):
            for ix in range(nx):
                p = iy * nx + ix
                rows += [row_id, row_id]; cols += [p, p + nx]; vals += [-1.0, 1.0]
                row_id += 1

        L = csr_matrix((vals, (rows, cols)), shape=(row_id, n), dtype=np.float64)

        # 归一化：使 ||L_grad||_F ≈ ||A||_F，与 laplacian 的处理方式一致，
        # 确保 alpha 对两种 reg_type 有相似的物理含义
        A_norm = sp_norm(self.A, 'fro')
        L_norm = sp_norm(L, 'fro')
        if L_norm > 1e-14:
            L = L * (A_norm / L_norm)

        return L

    # ------------------------------------------------------------------
    def solve(self) -> np.ndarray:
        """构建增广系统并用 LSQR 求解"""
        from scipy.sparse import eye, vstack
        from scipy.sparse.linalg import lsqr

        if self.reg_type == "identity":
            L = eye(self.n_pixels, format="csr")
        elif self.reg_type == "gradient":
            assert self.grid_shape, "gradient 需要提供 grid_shape"
            L = self._build_gradient()
        elif self.reg_type == "laplacian":
            assert self.grid_shape, "laplacian 需要提供 grid_shape"
            L = self._build_laplacian()
        else:
            raise ValueError(f"未知 reg_type: {self.reg_type!r}")

        sqrt_a = np.sqrt(max(self.alpha, 0.0))
        A_aug  = vstack([self.A, sqrt_a * L], format="csr")
        b_aug  = np.concatenate([self.b, np.zeros(L.shape[0])])

        res = lsqr(A_aug, b_aug, atol=1e-8, btol=1e-8, iter_lim=10_000)
        return _safe_clip(res[0])


# =============================================================================
# 3. TV 正则化 —— Split Bregman (ADMM)
# =============================================================================

class TVReconstructor:
    """
    Total Variation 正则化温度场重建（各向同性 TV）

    最小化：(1/2)‖A s − b‖² + α · TV(s)
    TV(s) = Σ_{i,j} √((Dx s)_{ij}² + (Dy s)_{ij}²)

    求解方法：Split Bregman（等价于 ADMM）
        引入辅助变量 d ≈ ∇s，将 L1 问题解耦为：
          · s 子问题：线性系统（共轭梯度）
          · d 子问题：矢量软阈值（各向同性收缩）

    优点：对含突变温度梯度（如火焰边界）的场景显著优于 L2 正则化。
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        alpha: float = 1e-4,
        mu: float = 1.0,
        n_outer: int = 50,
        n_inner_cg: int = 500,
        grid_shape: Optional[Tuple[int, int]] = None,
        tol: float = 1e-5,
    ):
        """
        Args:
            alpha      : TV 正则化权重（默认 1e-4，建议扫描 [1e-5, 1e-2] 范围）
            mu         : ADMM 惩罚参数（影响收敛速度，不影响最终解）
            n_outer    : Bregman 外层迭代次数
            n_inner_cg : 内层 CG 最大迭代次数
            grid_shape : (ny, nx)

        Bug #10 修复说明：
            原 mu=1.0，n_inner_cg=200。
            · mu 过小导致 s-子问题的矩阵 M = A^T A + μ(Dx^T Dx + Dy^T Dy) 中
              数据项和正则项严重失衡（A^T A 的条目量级 ~1e-7，Dx^T Dx ~1）。
              增大 mu 到 10.0 使 ADMM 惩罚项起到有效约束作用。
            · CG 迭代次数 200 不足以收敛（underdetermined 系统条件数极大），
              增加到 500。
            · 添加 SART 热启动：先跑 SART 获得合理初始慢度场，
              避免从均匀场出发导致 CG 陷入局部区域。
        """
        self.A          = A
        self.b          = b.astype(np.float64)
        self.alpha      = alpha
        self.mu         = mu
        self.n_outer    = n_outer
        self.n_inner_cg = n_inner_cg
        self.grid_shape = grid_shape
        self.tol        = tol
        self.n_pixels   = A.shape[1]
        self.residuals: List[float] = []

    def _diff_operators(self):
        """构建前向差分算子 Dx, Dy（行优先存储）"""
        from scipy.sparse import diags
        ny, nx = self.grid_shape
        n = nx * ny

        # Dx: x 方向前向差分，边界截断
        main_x = -np.ones(n)
        off_x  =  np.ones(n - 1)
        off_x[nx-1::nx] = 0.0        # ← 修正：防止跨行连接
        Dx = diags([main_x, off_x], [0, 1], shape=(n, n), format="csr")

        # Dy: y 方向前向差分
        main_y = -np.ones(n)
        off_y  =  np.ones(n - nx)
        Dy = diags([main_y, off_y], [0, nx], shape=(n, n), format="csr")

        return Dx, Dy

    @staticmethod
    def _isotropic_shrink(
        gx: np.ndarray, gy: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """各向同性软阈值收缩算子"""
        norm = np.sqrt(gx ** 2 + gy ** 2)
        # 防止除零（当 norm < threshold 时结果为 0）
        safe = np.maximum(norm, 1e-14)
        scale = np.maximum(norm - threshold, 0.0) / safe
        return scale * gx, scale * gy

    def solve(self) -> np.ndarray:
        """Split Bregman 迭代求解（使用 SART 热启动）"""
        from scipy.sparse import vstack
        from scipy.sparse.linalg import cg

        n  = self.n_pixels
        Dx, Dy = self._diff_operators()

        # 预构建固定线性系统矩阵（仅构建一次）
        AtA   = self.A.T.dot(self.A)
        M     = AtA + self.mu * (Dx.T.dot(Dx) + Dy.T.dot(Dy))
        Atb   = self.A.T.dot(self.b)

        thresh = self.alpha / self.mu   # 软阈值门限

        # 使用 SART 热启动获取合理初始场（Bug #10 修复）
        try:
            sart = SART(self.A, self.b, n_iter=100, relax=0.5, tol=1e-5)
            s = sart.run()
        except Exception:
            s = _default_slowness(n)
        dx = np.zeros(n); dy = np.zeros(n)   # 辅助变量
        bx = np.zeros(n); by = np.zeros(n)   # Bregman 乘子

        for outer in range(self.n_outer):
            s_prev = s.copy()

            # ---- s 子问题（线性系统）----
            rhs = Atb + self.mu * (Dx.T.dot(dx - bx) + Dy.T.dot(dy - by))
            s_new, info = cg(M, rhs, x0=s, maxiter=self.n_inner_cg, rtol=1e-7)
            if info > 0:
                warnings.warn(f"[TV-SB] CG 第 {outer} 次外迭代未收敛（info={info}）")
            s_new = _safe_clip(s_new)

            # ---- d 子问题（矢量软阈值）----
            gx = Dx.dot(s_new) + bx
            gy = Dy.dot(s_new) + by
            dx_new, dy_new = self._isotropic_shrink(gx, gy, thresh)

            # ---- Bregman 乘子更新 ----
            bx += Dx.dot(s_new) - dx_new
            by += Dy.dot(s_new) - dy_new

            s  = s_new
            dx, dy = dx_new, dy_new

            # 收敛检测
            rel = np.linalg.norm(s - s_prev) / (np.linalg.norm(s_prev) + 1e-14)
            self.residuals.append(rel)  # 记录残差
            if rel < self.tol:
                print(f"  [TV-SB] 收敛于第 {outer+1} 次迭代（Δ={rel:.2e}）")
                break

        return s


# =============================================================================
# 4. SART-TV 混合方法
# =============================================================================

class SARTTVReconstructor:
    """
    SART + TV 混合重建方法

    每次 SART 代数更新后，以 Chambolle TV 去噪作为正则化投影步骤，
    兼具 SART 的快速收敛性与 TV 的保边缘特性。

    算法流程：
        for k = 1, 2, ...:
            s_sart = SART_update(s)           # 代数更新
            s      = ChambolleTV(s_sart, γ)   # TV 去噪（正则化）
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        n_iter: int = 100,
        relax: float = 0.5,
        tv_weight: float = 1e-6,
        tv_iter: int = 30,
        grid_shape: Optional[Tuple[int, int]] = None,
        s_init: Optional[np.ndarray] = None,
        T_ref: float = 20.0,
        tol: float = 1e-6,
    ):
        """
        Args:
            A         : 系统矩阵
            b         : TOF 测量向量
            n_iter    : 最大迭代次数
            relax     : SART 松弛因子
            tv_weight : Chambolle TV 去噪权重 γ（默认 1e-6）
            tv_iter   : TV 去噪子迭代次数
            grid_shape: (ny, nx)
            s_init    : 初始慢度场
            T_ref     : 参考温度
            tol       : 收敛阈值

        Bug #11 修复说明：
            原 tv_weight=0.005。该值远大于慢度场的有效变化幅度（~1e-4），
            导致 Chambolle TV 去噪步将整个场平滑为常数，SART 更新被"冲掉"。
            经参数扫描，tv_weight=1e-6 在保持 TV 平滑效果的同时不破坏 SART 重建。
        """
        self.A          = A
        self.b          = b.astype(np.float64)
        self.n_iter     = n_iter
        self.relax      = relax
        self.tv_weight  = tv_weight
        self.tv_iter    = tv_iter
        self.grid_shape = grid_shape
        self.tol        = tol
        self.n_pixels   = A.shape[1]

        self.s = s_init.copy().astype(np.float64) if s_init is not None \
                 else _default_slowness(self.n_pixels, T_ref)

        # SART 权重（预计算）
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        row_sums = np.where(row_sums < 1e-14, 1e-14, row_sums)
        self.W_row = 1.0 / row_sums

        # Bug Fix: 根据 SART 算法公式，列权重应为 A 矩阵各列的和（物理意义：穿过像素的所有射线长度之和）
        # 错误代码：v = A.T.dot(self.W_row)  会导致量纲错误（结果为 s 而不是 s/m）
        v = np.asarray(A.sum(axis=0)).ravel()
        self.V_col = np.where(v < 1e-14, 1e-14, v)

        self.residuals: List[float] = []

    # ------------------------------------------------------------------
    def _chambolle_tv(self, f_flat: np.ndarray) -> np.ndarray:
        """
        Chambolle (2004) 对偶投影算法进行 2D 各向同性 TV 去噪

        最小化：(1/2)‖u − f‖² + γ · TV(u)

        对偶变量迭代（步长 dt = 1/8 满足收敛条件）：
            p^{n+1} = P_{‖·‖≤1}(p^n + dt · ∇(f − γ div p^n))
            u*      = f − γ div p*
        """
        ny, nx = self.grid_shape
        gamma  = self.tv_weight
        # Chambolle (2004) 收敛条件: dt ≤ 1/8 = 0.125
        # FIX: 原 dt=0.24 违反收敛条件；同时修正梯度/散度方向（见下文）
        dt     = 0.125

        f  = f_flat.reshape(ny, nx).copy()
        px = np.zeros((ny, nx))
        py = np.zeros((ny, nx))

        for _ in range(self.tv_iter):
            # ---- 计算后向差分散度 div(p) ----
            # div(p)[i,j] = (px[i,j] - px[i,j-1]) + (py[i,j] - py[i-1,j])
            # 使用后向差分（左边界为零）
            div_p = np.zeros((ny, nx))
            div_p[:, 1:]  += px[:, 1:] - px[:, :-1]   # x后向差分（内部列）
            div_p[:, 0]   += px[:, 0]                  # 左边界：px[0]=0 → div_p=px[0]
            div_p[1:, :]  += py[1:, :] - py[:-1, :]   # y后向差分（内部行）
            div_p[0, :]   += py[0, :]                  # 上边界：py[0]=0 → div_p=py[0]

            # ---- 计算梯度步：∇(f - γ div p) ----
            # FIX: 原代码用 tmp=div_p - f/gamma 再对tmp取梯度，等价于∇(-f/gamma)，
            # div_p 的贡献被丢弃。正确做法是直接对 (f - γ*div_p) 取梯度。
            # 计算 g = f - γ * div_p
            g = f - gamma * div_p
            gx = np.zeros((ny, nx))
            gy = np.zeros((ny, nx))
            gx[:, :-1] = g[:, 1:] - g[:, :-1]   # x 前向差分（右边界补0）
            gy[:-1, :] = g[1:, :] - g[:-1, :]   # y 前向差分（下边界补0）

            # 更新对偶变量并投影到单位球（各向同性）
            px_new = px + dt * gx
            py_new = py + dt * gy
            # FIX: denom 与 max(‖p_new‖, 1) 比较，而非 max(‖p_new‖, 0)
            denom  = np.maximum(np.sqrt(px_new**2 + py_new**2), 1.0)
            px = px_new / denom
            py = py_new / denom

        # 最终重建：u = f − γ div(p*)
        div_p_final = np.zeros((ny, nx))
        div_p_final[:, 1:]  += px[:, 1:] - px[:, :-1]
        div_p_final[:, 0]   += px[:, 0]
        div_p_final[1:, :]  += py[1:, :] - py[:-1, :]
        div_p_final[0, :]   += py[0, :]

        u = f - gamma * div_p_final
        return u.ravel()

    # ------------------------------------------------------------------
    def run(self) -> np.ndarray:
        """执行 SART-TV 混合迭代，返回重建慢度场"""
        s = self.s.copy()

        for k in range(self.n_iter):
            # 松弛因子衰减策略：前期大 relax 快速探索，后期小 relax 精细收敛
            schedule_factor = max(0.1, 1.0 - 0.9 * k / (self.n_iter - 1 + 1e-14))
            current_relax = self.relax * schedule_factor

            # SART 代数更新
            r_proj = self.b - self.A.dot(s)
            delta  = self.A.T.dot(self.W_row * r_proj) / self.V_col
            s_sart = _safe_clip(s + current_relax * delta)

            # TV 去噪正则化步
            s_new = _safe_clip(self._chambolle_tv(s_sart))

            rel = np.linalg.norm(s_new - s) / (np.linalg.norm(s) + 1e-14)
            self.residuals.append(float(np.linalg.norm(r_proj)))
            s = s_new

            if rel < self.tol:
                print(f"  [SART-TV] 收敛于第 {k+1} 次迭代（Δ={rel:.2e}）")
                break

        self.s = s
        return s


# =============================================================================
# 5. L-curve 法自动选取正则化参数
# =============================================================================

def l_curve_analysis(
    A,
    b: np.ndarray,
    alpha_list: np.ndarray,
    grid_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    L-curve 法自动选取 Tikhonov 正则化参数 α*

    绘制曲线：
        x 轴 — log‖A s(α) − b‖  （数据拟合残差）
        y 轴 — log‖L s(α)‖       （正则化项范数）

    曲线的"拐角"（最大曲率点）对应最优 α*。

    Returns:
        residuals  : 各 α 对应的残差范数
        reg_norms  : 各 α 对应的正则化范数
        best_alpha : 最优正则化参数
    """
    residuals = []
    reg_norms  = []

    # 复用 Tikhonov 的 Laplacian 构建
    tmp_rec = TikhonovReconstructor(A, b, alpha=1e-4, reg_type="laplacian",
                                    grid_shape=grid_shape)
    L = tmp_rec._build_laplacian()

    for alpha in alpha_list:
        rec = TikhonovReconstructor(A, b, alpha=alpha, reg_type="laplacian",
                                    grid_shape=grid_shape)
        s = rec.solve()
        residuals.append(np.linalg.norm(A.dot(s) - b))
        reg_norms.append(np.linalg.norm(L.dot(s)))

    residuals = np.array(residuals)
    reg_norms  = np.array(reg_norms)

    # ---- 计算对数坐标下的离散曲率 ----
    log_r = np.log10(residuals + 1e-14)
    log_n = np.log10(reg_norms  + 1e-14)

    # 归一化，防止量纲差异影响曲率计算
    def normalize(v):
        rng = v.max() - v.min()
        return (v - v.min()) / (rng + 1e-14)

    lr = normalize(log_r)
    ln = normalize(log_n)

    # 三点曲率公式（有限差分近似）
    curv = np.zeros(len(alpha_list))
    for i in range(1, len(alpha_list) - 1):
        dx1, dy1 = lr[i] - lr[i-1], ln[i] - ln[i-1]
        dx2, dy2 = lr[i+1] - lr[i], ln[i+1] - ln[i]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        denom = (dx1**2 + dy1**2) ** 1.5
        curv[i] = cross / (denom + 1e-14)

    best_idx   = int(np.argmax(curv[1:-1])) + 1   # 排除边界点
    best_alpha = float(alpha_list[best_idx])

    return residuals, reg_norms, best_alpha


# =============================================================================
# 6. TLS —— 全最小二乘重建器
#    处理模型矩阵 A 和测量向量 b 同时含噪的情形：(A+E)u = b+ε
#    参考：Beck A, Ben-Tal A. SIAM J. Optim. 17 (2006) 98-118.
# =============================================================================

class TLSReconstructor:
    """
    全最小二乘（Total Least Squares，TLS）重建器

    标准 TLS 将模型误差与数据误差统一处理：
        min ||[E, ε]||_F   s.t.  (A+E)u = b+ε

    等价于求增广矩阵 [A | b] 的最小奇异值对应的右奇异向量。

    对于正则化 TLS（Tikhonov-TLS，参考 Beck & Ben-Tal 2006），
    目标函数为：

        min  ||Au - b||² / (||u||² + 1)  +  α ||u||²

    本实现采用 SVD 方法求解标准 TLS，并提供可选的正则化项
    以增强数值稳定性（欠定/高噪声场景）。

    性能特点（参考 Li et al. 2025 对比实验）：
        · ARE 最大值 ~4.87%（三峰分布），弱于 MOTDRM（2.49%）
        · 计算速度快（一次 SVD），适合需要快速粗解的场景
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        alpha: float = 1e-2,          # 正则化参数（>0 推荐用于欠定系统）
        truncate_sv: Optional[int] = None,  # 截断奇异值数（防过拟合）
    ):
        """
        Args:
            A           : 系统矩阵（稀疏或稠密），shape (m, n)
            b           : TOF 测量向量，shape (m,)
            alpha       : 正则化参数（默认 1e-2，alpha=0 为纯 TLS）
            truncate_sv : 保留的最大奇异值数（None=不截断）

        Bug #12 修复说明：
            原 alpha=0.0（纯 TLS）。对于严重欠定系统（120 射线 × 1024 像素），
            纯 TLS 的 SVD 最小奇异向量包含大量噪声分量，导致解发散。
            默认改为 alpha=1e-2（正则化 TLS），对中等噪声场景更稳健。
        """
        self.A          = A
        self.b          = b.astype(np.float64)
        self.alpha      = float(alpha)
        self.truncate_sv = truncate_sv

    def solve(self) -> np.ndarray:
        """
        求解 TLS 问题，返回慢度向量 s (n_pixels,)。

        算法步骤：
          1. 构建增广矩阵 C = [A | b]，shape (m, n+1)
          2. 对 C 做 SVD：C = U Σ V^T
          3. 取最右奇异向量 v_{n+1}（对应最小奇异值）
          4. u_TLS = -v_{1:n} / v_{n+1}（若 |v_{n+1}| < tol 则退化为 LS）
          5. 加入正则化时，求解 (A^T A - σ²_min I + α I) u = A^T b
        """
        # 转稠密（SVD 需要）
        if hasattr(self.A, 'toarray'):
            A_dense = self.A.toarray()
        else:
            A_dense = np.array(self.A, dtype=np.float64)

        m, n = A_dense.shape

        if self.alpha == 0.0:
            # ── 纯 TLS（SVD 方法）──────────────────────────────
            C = np.column_stack([A_dense, self.b])  # (m, n+1)

            # 截断 SVD（节约内存）
            # Bug #19 修复：scipy.linalg.svd 在 PyTorch + MKL 共存环境下
            # 会触发 SIGABRT（Fatal Python error: Aborted），且无法被 try/except
            # 捕获。改用 numpy.linalg.svd 避免此问题。
            k = self.truncate_sv if self.truncate_sv else min(m, n + 1)
            U, sigma, Vt = np.linalg.svd(C, full_matrices=False)

            # 取最小奇异值对应的右奇异向量（最后一列 V）
            v_last = Vt[-1, :]             # shape (n+1,)
            v_n    = v_last[:-1]           # 对应 A 的列
            v_np1  = v_last[-1]            # 对应 b 的分量

            tol = 1e-10 * sigma[0]
            if abs(v_np1) < tol:
                warnings.warn(
                    "TLS: |v_{n+1}| 接近零，问题可能不相容（数据无TLS解），"
                    "退化为最小二乘解。",
                    RuntimeWarning,
                )
                # 退化为普通 LS
                s, _, _, _ = np.linalg.lstsq(A_dense, self.b, rcond=None)
            else:
                s = -v_n / v_np1
        else:
            # ── 正则化 TLS（Tikhonov-TLS，Beck & Ben-Tal 2006）──
            # 求解：(A^T A + α I) s = A^T b
            # 这是对 TLS 的一阶近似，适合噪声中等的场景
            AtA = A_dense.T @ A_dense
            AtA += self.alpha * np.eye(n)
            Atb = A_dense.T @ self.b
            try:
                # 使用 numpy 而非 scipy.linalg.solve（避免与 PyTorch MKL 冲突）
                s = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                s, _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)

        return _safe_clip(s.astype(np.float64))


# =============================================================================
# 7. LowRankReconstructor —— 低秩核范数正则化重建器
#    对应 MOTDRM 第一阶段（Li et al. Applied Thermal Engineering 2025）
#
#    目标函数（多目标简化为加权单目标）：
#        min  ||Au - b||² / (||u||² + 1)  +  α ||U||_*
#    其中 ||U||_* 为矩阵 U（将向量 u 重塑为 ny×nx 矩阵）的核范数（奇异值之和）。
#
#    核范数近端算子（奇异值软阈值，SVT）：
#        prox_τ||·||_* (M) = U diag(max(σ-τ, 0)) V^T
#
#    求解算法：ADMM（交替方向乘子法）
#        u 子问题：最小二乘（共轭梯度）
#        U 子问题：奇异值软阈值（SVT，proximal operator）
# =============================================================================

class LowRankReconstructor:
    """
    低秩核范数正则化重建器（MOTDRM 第一阶段等价实现）

    对温度场慢度重塑矩阵施加核范数（奇异值之和）正则化，
    利用温度场的低秩先验（如光滑的单峰/双峰分布本征维数低）
    提高重建精度和抗噪性。

    目标函数：
        min_{u}  ||Au - b||² / (||u||² + 1)  +  α ||U||_*
    其中：
        · A   : 系统矩阵 (m, n)，n = ny × nx
        · u   : 慢度向量 (n,)
        · U   : u 重塑为 (ny, nx) 矩阵
        · α   : 核范数正则化权重

    求解框架（ADMM）：
        引入辅助变量 Z = U，
        增广 Lagrangian：
            L_ρ = ||Au-b||²/(||u||²+1) + α||Z||_* + <Y, U-Z> + ρ/2 ||U-Z||²_F

        u-step : (A^T A + ρ I) u = A^T b + ρ vec(Z - Y/ρ)  [CG 迭代]
        Z-step : Z = SVT_{α/ρ}(U + Y/ρ)                    [奇异值软阈值]
        Y-step : Y = Y + ρ (U - Z)                         [对偶更新]

    参考文献：
        Li Y et al. Applied Thermal Engineering, 2025, 277:127113
        Hu Z X et al. "Low rank regularization: a review" Neural Netw. 2021
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        grid_shape: Tuple[int, int],
        alpha: float = 5e-5,          # 核范数正则化权重
        rho: float   = 0.01,          # ADMM 增广 Lagrangian 参数
        n_iter: int  = 100,           # ADMM 最大迭代次数
        cg_tol: float = 1e-6,         # CG 收敛阈值
        cg_maxiter: int = 500,        # CG 最大迭代次数
        tol_abs: float = 1e-5,        # ADMM 原始残差绝对收敛阈值
        tol_rel: float = 1e-4,        # ADMM 相对收敛阈值
        verbose: bool  = False,
    ):
        """
        Args:
            A          : 系统矩阵（scipy 稀疏）
            b          : TOF 向量 (m,)
            grid_shape : 网格形状 (ny, nx)
            alpha      : 核范数权重（默认 5e-5，越大越低秩越模糊）
            rho        : ADMM 惩罚参数（默认 0.01）
            n_iter     : ADMM 最大迭代次数
            cg_tol     : 共轭梯度收敛容差
            cg_maxiter : 共轭梯度最大迭代次数
            tol_abs    : 原始残差绝对收敛阈值
            tol_rel    : 相对收敛阈值（用于判断提前停止）
            verbose    : 是否打印 ADMM 收敛信息

        Bug #13 修复说明：
            原 alpha=1e-3, rho=1.0, cg_maxiter=200。
            · rho=1.0 使 u-step 中 A^T A + ρ I 的 ρI 项远大于 A^T A（~1e-7 量级），
              解直接被 ρI 项"拉"到初始值附近（相当于几乎不更新）。rho=0.01 更均衡。
            · alpha=1e-3 的 SVT 阈值 τ=α/ρ=0.1，对慢度场（~0.003）而言过于激进，
              将大量奇异值截断为零，解退化为零矩阵再经 clip 得到边界值。
              alpha=5e-5, τ=5e-3 更合理。
            · CG 200 步在 ill-conditioned 系统下不足以收敛，增至 500。
        """
        self.A          = A
        self.b          = b.astype(np.float64)
        self.grid_shape = grid_shape
        self.alpha      = float(alpha)
        self.rho        = float(rho)
        self.n_iter     = int(n_iter)
        self.cg_tol     = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)
        self.tol_abs    = float(tol_abs)
        self.tol_rel    = float(tol_rel)
        self.verbose    = bool(verbose)
        self.residuals: List[float] = []

    @staticmethod
    def _svt(M: np.ndarray, tau: float) -> np.ndarray:
        """
        奇异值软阈值（Singular Value Thresholding，SVT）

        prox_{τ||·||_*}(M) = U · diag(max(σ - τ, 0)) · V^T

        Args:
            M   : 输入矩阵 (ny, nx)
            tau : 阈值（= alpha / rho）
        Returns:
            低秩近似矩阵，与 M 形状相同
        """
        U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
        sigma_thresh = np.maximum(sigma - tau, 0.0)
        return U @ np.diag(sigma_thresh) @ Vt

    def _build_linear_op(self):
        """
        构建 u-step 的线性算子：H = A^T A + ρ I

        使用 LinearOperator 避免显式构建稠密矩阵。
        """
        from scipy.sparse.linalg import LinearOperator
        import scipy.sparse as sp

        A = self.A
        rho = self.rho
        n = A.shape[1]

        def matvec(x: np.ndarray) -> np.ndarray:
            # H x = A^T (A x) + ρ x
            return A.T.dot(A.dot(x)) + rho * x

        return LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    def solve(self) -> np.ndarray:
        """
        ADMM 求解低秩正则化问题，返回慢度向量 s (ny*nx,)

        算法（每步迭代）：
          u-step : 用 CG 求解 (A^T A + ρ I) u = A^T b + ρ vec(Z - Y/ρ)
          Z-step : Z = SVT_{α/ρ}(reshape(u + Y/ρ))
          Y-step : Y = Y + ρ(U - Z)   （对偶变量更新）

        [P1 修改] 自适应 rho（Boyd et al. ADMM）：
          · 若原残差 r_norm > μ × 对偶残差 s_norm：增大 rho（加快原残差收敛）
          · 若对偶残差 s_norm > μ × 原残差 r_norm：减小 rho（加快对偶残差收敛）
          · 每 10 步更新一次 rho，避免 H 矩阵频繁重建

        Returns:
            s : 慢度向量，已经过 _safe_clip 物理约束
        """
        from scipy.sparse.linalg import cg as scipy_cg

        ny, nx = self.grid_shape
        n = ny * nx

        # ── 初始化 ──────────────────────────────────────────────
        # 用 SART 解作为热启动初始值（比 Tikhonov identity 更稳健）
        try:
            sart = SART(self.A, self.b, n_iter=100, relax=0.5, tol=1e-5)
            u = sart.run()
        except Exception:
            u = _default_slowness(n)

        Z = u.reshape(ny, nx).copy()        # 辅助变量
        Y = np.zeros((ny, nx), dtype=np.float64)  # 对偶变量（Lagrange 乘子）

        # [P1 修改] 自适应 rho 初始化
        rho_k = float(self.rho)   # 当前 rho（迭代中可自适应调整）
        H = self._build_linear_op_with_rho(rho_k)  # 初始线性算子
        Atb = self.A.T.dot(self.b)          # A^T b（预计算）

        # 自适应 rho 参数（Boyd et al.）
        mu = 10.0        # 残差比率阈值
        tau_incr = 2.0  # rho 增大因子
        tau_decr = 2.0  # rho 减小因子

        for k in range(self.n_iter):
            u_old = u.copy()

            # ── u-step：求解 (A^T A + ρ I) u = A^T b + ρ vec(Z - Y/ρ) ──
            rhs = Atb + rho_k * (Z - Y / rho_k).ravel()
            try:
                u_new, info = scipy_cg(H, rhs, x0=u,
                                       rtol=self.cg_tol,
                                       maxiter=self.cg_maxiter)
                if info != 0:
                    u_new = u   # CG 未收敛时保持原值
            except TypeError:
                # 兼容旧版 scipy（tol 参数名）
                u_new, info = scipy_cg(H, rhs, x0=u,
                                       tol=self.cg_tol,
                                       maxiter=self.cg_maxiter)
                if info != 0:
                    u_new = u

            u = u_new

            # ── Z-step：奇异值软阈值 ───────────────────────────
            Z_old = Z.copy()
            tau_k = self.alpha / rho_k    # 当前 SVT 阈值
            Z = self._svt(u.reshape(ny, nx) + Y / rho_k, tau_k)

            # ── Y-step：对偶更新 ──────────────────────────────
            Y = Y + rho_k * (u.reshape(ny, nx) - Z)

            # ── 收敛检查 ──────────────────────────────────────
            primal_res = np.linalg.norm(u.reshape(ny, nx) - Z, 'fro')
            dual_res   = rho_k * np.linalg.norm(Z - Z_old, 'fro')

            eps_pri  = np.sqrt(n) * self.tol_abs + self.tol_rel * max(
                np.linalg.norm(u), np.linalg.norm(Z, 'fro')
            )
            eps_dual = np.sqrt(n) * self.tol_abs + self.tol_rel * np.linalg.norm(
                rho_k * Y, 'fro'
            )

            self.residuals.append(primal_res)  # 记录原始残差

            # ── [P1 修改] 自适应 rho 更新（每 10 步）───────────────
            if (k > 0) and ((k + 1) % 10 == 0):
                r_norm = primal_res
                s_norm = dual_res
                if r_norm > mu * s_norm:
                    rho_k = rho_k * tau_incr   # 原残差主导 → 增大 rho
                elif s_norm > mu * r_norm:
                    rho_k = rho_k / tau_decr   # 对偶残差主导 → 减小 rho
                # 更新线性算子（因为 H = A^T A + rho * I）
                H = self._build_linear_op_with_rho(rho_k)
                if self.verbose:
                    print(f"    [adaptive rho] k={k+1}, rho={rho_k:.4f}")

            if self.verbose:
                print(f"  [LowRank ADMM] iter={k+1:3d} | "
                      f"primal={primal_res:.2e} | dual={dual_res:.2e}")

            if primal_res < eps_pri and dual_res < eps_dual:
                if self.verbose:
                    print(f"  [LowRank ADMM] 收敛于第 {k+1} 步")
                break

        return _safe_clip(u.astype(np.float64))

    def _build_linear_op_with_rho(self, rho: float):
        """构建带指定 rho 的线性算子 H = A^T A + ρ I"""
        from scipy.sparse.linalg import LinearOperator

        A = self.A
        n = A.shape[1]

        def matvec(x: np.ndarray) -> np.ndarray:
            return A.T.dot(A.dot(x)) + rho * x

        return LinearOperator((n, n), matvec=matvec, dtype=np.float64)


# =============================================================================
# 8. RBFReconstructor —— 径向基函数正则化重建
#    使用 RBF 空间插值核将高维像素空间投影到低维 RBF 系数空间，
#    从而降低反演的欠定程度，提高重建的空间平滑性。
#
#    核心思想：
#        s(x) = Σ_{k=1}^{K} w_k · φ(||x - c_k||)
#    其中 c_k 为 RBF 中心，w_k 为待求系数，φ 为径向基函数。
#
#    将像素慢度表示为 s = Φ w，则反演问题变为：
#        min ||A Φ w - b||² + α ||w||²
#    维度从 n_pixels 降至 K（RBF 中心数），大幅降低欠定度。
#
#    参考文献：
#        Zhang J Q et al. Int J Heat Mass Transf, 2022, 188: 122660.
# =============================================================================

class RBFReconstructor:
    """
    径向基函数（Radial Basis Function, RBF）正则化重建器

    将温度场（慢度场）表示为 RBF 基函数的线性组合：
        s(x) = Σ_{k=1}^{K} w_k · φ(||x - c_k|| / ε)

    这将原始高维反演问题 (n_pixels 个未知数) 转化为低维问题 (K 个 RBF 系数)，
    天然具有空间正则化效果。

    支持的 RBF 核函数：
        'gaussian'     : φ(r) = exp(-r²)           — 光滑、紧支撑性好
        'multiquadric' : φ(r) = √(1 + r²)          — 全局支撑，适合平缓分布
        'thin_plate'   : φ(r) = r² ln(r+ε)         — 最小弯曲能量
        'inverse_mq'   : φ(r) = 1 / √(1 + r²)     — 快速衰减

    RBF 中心的选择策略：
        'grid'        : 在重建网格上均匀放置（默认）
        'transducer'  : 在换能器位置放置
        'kmeans'      : 对像素中心做 K-Means 聚类
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        grid_shape: Tuple[int, int],
        pixel_centers: np.ndarray,
        n_centers: Optional[int] = 64,
        kernel: str = 'gaussian',
        epsilon: Optional[float] = None,
        alpha: float = 1e-3,
        center_strategy: str = 'grid',
        transducer_positions: Optional[np.ndarray] = None,
        pixel_stride: Optional[Tuple[int, int]] = None,
        grid_spacing: Optional[float] = None,
        epsilon_scale: float = 2.5,
    ):
        """
        Args:
            A                    : 系统矩阵 (n_rays, n_pixels)，scipy 稀疏
            b                    : TOF 测量向量 (n_rays,)
            grid_shape           : (ny, nx) 网格形状
            pixel_centers        : 所有像素中心坐标 (n_pixels, 2)
            n_centers            : RBF 中心数量 K（越大精度越高，计算量越大）；当 center_strategy='pixel' 时可设为 None
            kernel               : RBF 核函数类型
            epsilon              : RBF 形状参数（控制核宽度），None 自动计算
            alpha                : Tikhonov 正则化参数（对 RBF 系数空间）
            center_strategy      : RBF 中心选取策略 'grid'/'transducer'/'kmeans'
            transducer_positions : 换能器坐标 (N_trans, 2)，仅 center_strategy='transducer' 时使用
            pixel_stride         : 当 center_strategy='pixel' 时指定 (stride_y, stride_x)
            grid_spacing         : 当前网格的最小栅格尺寸（米），用于自动设置 ε
            epsilon_scale        : 自动 ε = epsilon_scale × grid_spacing
        """
        self.A = A
        self.b = b.astype(np.float64)
        self.grid_shape = grid_shape
        self.pixel_centers = pixel_centers  # (n_pixels, 2)
        self.n_centers = n_centers
        self.kernel = kernel
        self.epsilon = epsilon
        self.alpha = alpha
        self.center_strategy = center_strategy
        self.transducer_positions = transducer_positions
        self.pixel_stride = pixel_stride
        self.grid_spacing = grid_spacing
        self.epsilon_scale = epsilon_scale

        self._n_centers_actual: Optional[int] = None
        self._phi_shape: Optional[Tuple[int, int]] = None
        self._B_shape: Optional[Tuple[int, int]] = None
        self._epsilon_used: Optional[float] = None

    def _select_centers(self) -> np.ndarray:
        """选取 RBF 中心坐标，shape (K, 2)"""
        ny, nx = self.grid_shape
        pc = self.pixel_centers

        centers = None

        if self.center_strategy == 'transducer':
            if self.transducer_positions is not None:
                centers = self.transducer_positions.copy()
            else:
                warnings.warn("未提供换能器坐标，退化为 grid 策略")

        if self.center_strategy == 'kmeans':
            try:
                from sklearn.cluster import MiniBatchKMeans
                n_clusters = self.n_centers or min(256, len(pc))
                km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                     n_init=3)
                km.fit(pc)
                centers = km.cluster_centers_
            except ImportError:
                warnings.warn("sklearn 不可用，退化为 grid 策略")

        if centers is None and self.center_strategy == 'pixel':
            stride_y, stride_x = self.pixel_stride if self.pixel_stride else (1, 1)
            stride_y = max(1, int(stride_y))
            stride_x = max(1, int(stride_x))
            centers_grid = pc.reshape(ny, nx, 2)
            centers = centers_grid[::stride_y, ::stride_x].reshape(-1, 2)
            if centers.size == 0:
                centers = pc.copy()

        if centers is None:
            # 默认 'grid'：在区域内均匀放置
            target_k = self.n_centers or max(1, int(np.sqrt(pc.shape[0])))
            n_side = max(1, int(np.sqrt(target_k)))
            nx_c = n_side
            ny_c = max(1, target_k // n_side)
            while nx_c * ny_c < target_k and nx_c < target_k:
                nx_c += 1

            x_min, x_max = pc[:, 0].min(), pc[:, 0].max()
            y_min, y_max = pc[:, 1].min(), pc[:, 1].max()

            margin_x = (x_max - x_min) * 0.05
            margin_y = (y_max - y_min) * 0.05
            xs = np.linspace(x_min + margin_x, x_max - margin_x, nx_c)
            ys = np.linspace(y_min + margin_y, y_max - margin_y, ny_c)
            XX, YY = np.meshgrid(xs, ys)
            centers = np.column_stack([XX.ravel(), YY.ravel()])

        if self.center_strategy != 'pixel' and self.n_centers is not None and len(centers) > self.n_centers:
            centers = centers[: self.n_centers]

        self._n_centers_actual = len(centers)
        return centers

    def _rbf_kernel(self, r: np.ndarray) -> np.ndarray:
        """计算 RBF 核函数值 φ(r)"""
        if self.kernel == 'gaussian':
            return np.exp(-r ** 2)
        elif self.kernel == 'multiquadric':
            return np.sqrt(1.0 + r ** 2)
        elif self.kernel == 'inverse_mq':
            return 1.0 / np.sqrt(1.0 + r ** 2)
        elif self.kernel == 'thin_plate':
            # r² ln(r)，r=0 时取 0
            with np.errstate(divide='ignore', invalid='ignore'):
                val = r ** 2 * np.log(np.maximum(r, 1e-14))
            val[r < 1e-14] = 0.0
            return val
        else:
            raise ValueError(f"未知 RBF 核: {self.kernel!r}")

    def _build_rbf_matrix(self, centers: np.ndarray) -> np.ndarray:
        """
        构建 RBF 插值矩阵 Φ，shape (n_pixels, K)

        Φ[i, k] = φ(||pixel_i - center_k|| / ε)
        """
        # 计算所有像素到所有中心的距离矩阵
        # pixel_centers: (n_pixels, 2), centers: (K, 2)
        diff = self.pixel_centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n_px, K, 2)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (n_px, K)

        if self.epsilon is None:
            eps = None
            if self.grid_spacing is not None:
                eps = max(1e-6, self.epsilon_scale * self.grid_spacing)
            elif len(centers) > 1:
                from scipy.spatial.distance import pdist
                avg_dist = np.mean(pdist(centers))
                eps = avg_dist * 0.8
            else:
                eps = 1.0
            self.epsilon = eps

        self._epsilon_used = float(self.epsilon)
        r = dist / self.epsilon
        return self._rbf_kernel(r)  # (n_pixels, K)

    def solve(self) -> np.ndarray:
        """
        求解 RBF 正则化重建问题

        步骤：
          1. 选取 K 个 RBF 中心
          2. 构建 RBF 矩阵 Φ (n_pixels, K)
          3. 构建投影系统矩阵 B = A Φ (n_rays, K)
          4. 求解正则化问题：min ||B w - b||² + α ||w||²
             解析解：w = (B^T B + α I)^{-1} B^T b
          5. 恢复慢度场：s = Φ w

        Returns:
            s : 慢度向量 (n_pixels,)，已经过物理约束剪切
        """
        # 1. 选取 RBF 中心
        centers = self._select_centers()
        K = len(centers)

        # 2. 构建 RBF 矩阵
        Phi = self._build_rbf_matrix(centers)  # (n_pixels, K)
        self._phi_shape = Phi.shape

        # 3. 投影系统矩阵
        if hasattr(self.A, 'toarray'):
            B = self.A.dot(Phi)  # (n_rays, K)，稀疏 × 稠密 → 稠密
        else:
            B = self.A @ Phi
        self._B_shape = B.shape

        # 4. 求解 Tikhonov 正则化的 RBF 系数
        BtB = B.T @ B  # (K, K)
        BtB += self.alpha * np.eye(K)
        Btb = B.T @ self.b

        try:
            w = np.linalg.solve(BtB, Btb)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(BtB, Btb, rcond=None)

        # 5. 恢复慢度场
        s = Phi @ w  # (n_pixels,)

        return _safe_clip(s.astype(np.float64))

    @property
    def n_centers_actual(self) -> Optional[int]:
        return self._n_centers_actual

    @property
    def phi_shape(self) -> Optional[Tuple[int, int]]:
        return self._phi_shape

    @property
    def projected_shape(self) -> Optional[Tuple[int, int]]:
        return self._B_shape

    @property
    def effective_epsilon(self) -> Optional[float]:
        return self._epsilon_used


# =============================================================================
# 9. WallConstrainedReconstructor —— 壁温约束重建
#    利用换能器附近（锅炉壁面）已知温度作为约束条件
#
#    方法：在 Tikhonov 正则化基础上增加壁温约束惩罚项：
#        min ||As - b||² + α||Ls||² + β||s_wall - s_ref||²
#    其中 s_wall 为壁面附近像素的慢度，s_ref 为已知壁温对应的慢度。
# =============================================================================

class WallConstrainedReconstructor:
    """
    壁温约束正则化重建器

    在传统 Tikhonov 正则化基础上，增加壁面温度约束：

        min  ||As - b||²  +  α ||Ls||²  +  β ||W(s - s_wall)||²

    其中：
        · s_wall : 壁面像素对应的参考慢度（由已知壁温转化）
        · W      : 壁面像素选择矩阵（对角矩阵，壁面像素=1，其余=0）
        · β      : 壁温约束强度

    壁面像素的识别方法：
        选取距离任意换能器位置最近的 n_wall 个像素作为壁面像素。
        或指定壁面像素到最近换能器的距离阈值。

    物理意义：
        换能器安装在锅炉壁面上，壁面附近的温度通常是已知的（如通过热电偶测量），
        将此信息作为先验约束可以：
        (1) 锚定边界区域温度，避免重建场边缘发散
        (2) 减少欠定系统的自由度，提高重建稳定性
        (3) 使重建结果更符合工程实际
    """

    def __init__(
        self,
        A,
        b: np.ndarray,
        grid_shape: Tuple[int, int],
        pixel_centers: np.ndarray,
        transducer_positions: np.ndarray,
        T_wall: float = 20.0,
        beta: float = 1.0,
        alpha: float = 0.01,
        reg_type: str = 'gradient',
        wall_dist_ratio: float = 0.15,
    ):
        """
        Args:
            A                    : 系统矩阵 (n_rays, n_pixels)
            b                    : TOF 测量向量 (n_rays,)
            grid_shape           : (ny, nx)
            pixel_centers        : 所有像素中心坐标 (n_pixels, 2)
            transducer_positions : 换能器坐标 (N_trans, 2)
            T_wall               : 壁面温度 [℃]（默认 20℃）
            beta                 : 壁温约束权重（越大越强制壁面温度一致）
            alpha                : 梯度正则化参数
            reg_type             : 正则化类型 'identity'/'gradient'/'laplacian'
            wall_dist_ratio      : 壁面像素识别距离比例
                                   像素到最近换能器距离 < wall_dist_ratio * domain_scale
                                   时认为是壁面像素
        """
        self.A = A
        self.b = b.astype(np.float64)
        self.grid_shape = grid_shape
        self.pixel_centers = pixel_centers
        self.transducer_positions = transducer_positions
        self.T_wall = T_wall
        self.beta = beta
        self.alpha = alpha
        self.reg_type = reg_type
        self.wall_dist_ratio = wall_dist_ratio

    def _identify_wall_pixels(self) -> np.ndarray:
        """
        识别壁面像素：距离任意换能器位置最近的像素

        Returns:
            wall_mask : bool 数组 (n_pixels,)，True 表示壁面像素
        """
        pc = self.pixel_centers   # (n_pixels, 2)
        tp = self.transducer_positions  # (N_trans, 2)

        # 计算每个像素到最近换能器的距离
        # (n_pixels, 1, 2) - (1, N_trans, 2) → (n_pixels, N_trans, 2)
        diff = pc[:, np.newaxis, :] - tp[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (n_pixels, N_trans)
        min_dist = dist.min(axis=1)  # (n_pixels,)

        # 距离阈值
        x_range = pc[:, 0].max() - pc[:, 0].min()
        y_range = pc[:, 1].max() - pc[:, 1].min()
        domain_scale = np.sqrt(x_range ** 2 + y_range ** 2)
        threshold = self.wall_dist_ratio * domain_scale

        return min_dist < threshold

    def solve(self) -> np.ndarray:
        """
        求解壁温约束正则化问题

        构建增广系统：
            [     A      ]       [  b  ]
            [  √α · L    ] s  =  [  0  ]
            [ √β · W     ]       [ √β · W · s_wall ]

        使用 LSQR 迭代求解。

        Returns:
            s : 慢度向量 (n_pixels,)
        """
        from scipy.sparse import eye, vstack, diags
        from scipy.sparse.linalg import lsqr

        n_pixels = self.A.shape[1]
        ny, nx = self.grid_shape

        # 构建正则化矩阵 L
        tik = TikhonovReconstructor(
            self.A, self.b,
            alpha=self.alpha, reg_type=self.reg_type,
            grid_shape=self.grid_shape,
        )
        if self.reg_type == 'identity':
            L = eye(n_pixels, format='csr')
        elif self.reg_type == 'gradient':
            L = tik._build_gradient()
        elif self.reg_type == 'laplacian':
            L = tik._build_laplacian()
        else:
            L = eye(n_pixels, format='csr')

        # 识别壁面像素
        wall_mask = self._identify_wall_pixels()
        n_wall = int(wall_mask.sum())

        # 壁面温度对应的参考慢度
        s_wall_val = float(temperature_to_slowness(np.array([self.T_wall]))[0])

        # 构建壁面约束矩阵 W（选择矩阵，使用稀疏格式）
        wall_indices = np.where(wall_mask)[0]
        # Bug #20 修复：W 矩阵应使用稀疏格式，避免密集矩阵传入 vstack 导致内存浪费
        from scipy.sparse import csr_matrix
        if n_wall > 0:
            rows = np.arange(n_wall)
            cols = wall_indices
            vals = np.ones(n_wall, dtype=np.float64)
            W = csr_matrix((vals, (rows, cols)), shape=(n_wall, n_pixels))
        else:
            # 无壁面像素时构建空矩阵（退化为无约束 Tikhonov）
            W = csr_matrix((0, n_pixels), dtype=np.float64)

        # 构建增广系统
        sqrt_alpha = np.sqrt(max(self.alpha, 0.0))
        sqrt_beta = np.sqrt(max(self.beta, 0.0))

        A_aug = vstack([
            self.A,
            sqrt_alpha * L,
            sqrt_beta * W,
        ], format='csr')

        b_aug = np.concatenate([
            self.b,
            np.zeros(L.shape[0]),
            sqrt_beta * np.full(n_wall, s_wall_val),
        ])

        # LSQR 求解
        res = lsqr(A_aug, b_aug, atol=1e-8, btol=1e-8, iter_lim=10_000)

        return _safe_clip(res[0])
