"""
声学CT温度场重建 —— 仿真数据生成模块
========================================
提供多种标准测试温度场（phantom），并模拟声学CT测量过程：
  · 构建系统矩阵 A
  · 正向计算无噪 TOF
  · 叠加高斯白噪声（模拟测量误差）

标准测试场（phantom）：
  1. gaussian_hotspot   —— 单个高斯热点（经典验证场）
  2. multi_hotspot      —— 多高斯热点叠加
  3. flame_profile      —— 火焰横截面剖面（不对称、含边缘突变）
  4. step_field         —— 阶跃场（测试保边缘能力）
  5. dual_peak_phantom  —— 双峰温度分布（参考 Li et al. 2025 Eq.19）
  6. triple_peak_phantom—— 三峰温度分布（参考 Li et al. 2025 Eq.20）
  7. custom_callable    —— 用户自定义函数 T(x, y)
"""

import numpy as np
from typing import Callable, Optional, Tuple, List

from ray_tracer import (
    TransducerConfig, GridConfig, SiddonRayTracer,
    temperature_to_slowness, compute_tof,
)


# =============================================================================
# 标准测试温度场 (Phantom)
# =============================================================================

def gaussian_hotspot(
    grid: GridConfig,
    centers: Optional[List[Tuple[float, float]]] = None,
    amplitudes: Optional[List[float]] = None,
    sigmas: Optional[List[float]] = None,
    T_bg: float = 20.0,
) -> np.ndarray:
    """
    高斯热点叠加温度场

    T(x,y) = T_bg + Σ_k  A_k · exp(-((x-cx_k)²+(y-cy_k)²) / (2σ_k²))

    Args:
        centers   : 热点中心列表 [(cx, cy), ...]
        amplitudes: 各热点幅值 [℃]（相对于背景温度的升温量）
        sigmas    : 各热点标准差 [m]
        T_bg      : 背景温度 [℃]

    Returns:
        T_flat : 展平温度场，shape (ny*nx,)
    """
    xx, yy = grid.pixel_centers()   # (ny, nx)

    cx_grid, cy_grid = grid.center_xy()
    scale = min(grid.lx, grid.ly)

    if centers is None:
        centers = [(cx_grid + 0.15 * scale, cy_grid + 0.10 * scale)]
    if amplitudes is None:
        amplitudes = [80.0]
    if sigmas is None:
        sigmas = [0.08 * scale]

    T = np.full_like(xx, T_bg)
    for (cx, cy), amp, sig in zip(centers, amplitudes, sigmas):
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        T += amp * np.exp(-r2 / (2.0 * sig ** 2))

    return T.ravel()


def flame_profile(grid: GridConfig, T_bg: float = 20.0) -> np.ndarray:
    """
    模拟炉膛截面火焰温度场（不对称分布）

    包含：
    · 中央高温核心区（最高 1200 ℃）
    · 不对称侧翼（模拟偏心火焰）
    · 含边缘突变（测试 TV 算法保边缘能力）

    Returns:
        T_flat : 展平温度场，shape (ny*nx,)
    """
    xx, yy = grid.pixel_centers()
    cx, cy = grid.center_xy()
    lx, ly = grid.lx, grid.ly

    T = np.full_like(xx, T_bg)

    # 主火焰核心
    r2_core = ((xx - cx - 0.05 * lx) / (0.18 * lx)) ** 2 \
            + ((yy - cy + 0.03 * ly) / (0.12 * ly)) ** 2
    T += 1180.0 * np.exp(-r2_core)

    # 左侧余焰
    r2_left = ((xx - cx + 0.22 * lx) / (0.10 * lx)) ** 2 \
            + ((yy - cy + 0.08 * ly) / (0.15 * ly)) ** 2
    T += 400.0 * np.exp(-r2_left)

    # 右侧冷区（局部降温，模拟冷气流）
    r2_right = ((xx - cx - 0.20 * lx) / (0.12 * lx)) ** 2 \
             + ((yy - cy - 0.12 * ly) / (0.10 * ly)) ** 2
    T -= 10.0 * np.exp(-r2_right)  # 轻微降温

    # 阶跃边界（增加边缘突变）
    mask_step = (xx > cx + 0.28 * lx) & (yy < cy - 0.15 * ly)
    T[mask_step] = T_bg + 5.0

    return T.ravel()


def step_field(grid: GridConfig, T_lo: float = 20.0, T_hi: float = 200.0) -> np.ndarray:
    """
    阶跃温度场（矩形高温区）

    用于测试保边缘算法（TV 正则化）的分辨能力。
    """
    xx, yy = grid.pixel_centers()
    cx, cy = grid.center_xy()
    lx, ly = grid.lx, grid.ly

    T = np.full_like(xx, T_lo)
    mask = (np.abs(xx - cx) < 0.25 * lx) & (np.abs(yy - cy) < 0.20 * ly)
    T[mask] = T_hi
    return T.ravel()


def multi_hotspot(grid: GridConfig, T_bg: float = 20.0) -> np.ndarray:
    """三个高斯热点（验证算法分辨多目标的能力）"""
    cx, cy = grid.center_xy()
    lx, ly = grid.lx, grid.ly
    r = 0.22

    centers    = [
        (cx - r * lx, cy - r * ly),
        (cx + r * lx, cy),
        (cx,           cy + r * ly),
    ]
    amplitudes = [100.0, 60.0, 80.0]
    sigmas     = [0.07 * lx, 0.06 * lx, 0.05 * lx]

    return gaussian_hotspot(grid, centers, amplitudes, sigmas, T_bg)


def dual_peak_phantom(
    grid: GridConfig,
    centers: Optional[List[Tuple[float, float]]] = None,
    amplitudes: Optional[List[float]] = None,
    sigma_params: Optional[List[Tuple[float, float]]] = None,
    T_bg: float = 800.0,
) -> np.ndarray:
    """
    严格按论文复现：双峰温度分布（Li et al. 2025, Eq.(19)）
    
    T(x,y) = 300·exp((-30·(x-3)² - 17·(y-3)²)/500)
           + 200·exp((-20·(x-10)² - 30·(y-18)²)/500) + 800
    
    物理坐标系：[0, 20] × [0, 20] m，中心位置为绝对坐标。

    Args:
        grid          : 重建网格配置
        centers       : 两个峰的中心坐标 [(cx1,cy1), (cx2,cy2)]，单位m
                        默认参考论文：[(3, 3), (10, 18)]（20m×20m域）
        amplitudes    : 两个峰的温度幅值 [A1, A2]（℃），默认 [300, 200]
        sigma_params  : 两个峰的高斯形状参数 [(a1,b1,d1), (a2,b2,d2)]
                        参数意义：exp((-a*(x-cx)² - b*(y-cy)²)/d)
                        默认参考论文 Eq.19：[(30, 17, 500), (20, 30, 500)]
        T_bg          : 背景温度 [℃]，默认 800℃（论文配置）

    Returns:
        T_flat : 展平温度场，shape (ny*nx,)
    """
    xx, yy = grid.pixel_centers()
    
    if centers is None:
        # 论文 Eq.19 的标准配置：物理坐标 (3, 3) 和 (10, 18)
        centers = [(3.0, 3.0), (10.0, 18.0)]
    
    if amplitudes is None:
        # 论文 Eq.19：A1=300℃, A2=200℃
        amplitudes = [300.0, 200.0]
    
    if sigma_params is None:
        # 论文 Eq.19 的参数：exp((-a*(x-cx)² - b*(y-cy)²)/d)
        # 峰1: a1=30, b1=17, d1=500
        # 峰2: a2=20, b2=30, d2=500
        sigma_params = [(30.0, 17.0, 500.0), (20.0, 30.0, 500.0)]
    
    T = np.full_like(xx, T_bg)
    for (cx, cy), amp, (a, b, d) in zip(centers, amplitudes, sigma_params):
        T += amp * np.exp((-a * (xx - cx) ** 2 - b * (yy - cy) ** 2) / d)

    return T.ravel()


def triple_peak_phantom(
    grid: GridConfig,
    centers: Optional[List[Tuple[float, float]]] = None,
    amplitudes: Optional[List[float]] = None,
    sigma_params: Optional[List[Tuple[float, float]]] = None,
    T_bg: float = 850.0,
) -> np.ndarray:
    """
    严格按论文复现：三峰温度分布（Li et al. 2025, Eq.(20)）
    
    T(x,y) = 400·exp((-30·(x-8)² - 45·(y-15)²)/1000)
           + 500·exp((-30·(x-7)² - 50·(y-6)²)/400)
           + 500·exp((-30·(x-12)² - 60·(y-10)²)/100) + 850
    
    物理坐标系：[0, 20] × [0, 20] m，中心位置为绝对坐标。

    Args:
        grid          : 重建网格配置
        centers       : 三个峰的中心坐标 [(cx1,cy1), (cx2,cy2), (cx3,cy3)]，单位m
                        默认参考论文：[(8, 15), (7, 6), (12, 10)]（20m×20m域）
        amplitudes    : 三个峰的温度幅值 [A1, A2, A3]（℃），默认 [400, 500, 500]
        sigma_params  : 三个峰的高斯形状参数 [(a1,b1,d1), (a2,b2,d2), (a3,b3,d3)]
                        参数意义：exp((-a*(x-cx)² - b*(y-cy)²)/d)
                        默认参考论文 Eq.20：[(30,45,1000), (30,50,400), (30,60,100)]
        T_bg          : 背景温度 [℃]，默认 850℃（论文配置）

    Returns:
        T_flat : 展平温度场，shape (ny*nx,)
    """
    xx, yy = grid.pixel_centers()
    
    if centers is None:
        # 论文 Eq.20 的标准配置：物理坐标 (8, 15), (7, 6), (12, 10)
        centers = [(8.0, 15.0), (7.0, 6.0), (12.0, 10.0)]
    
    if amplitudes is None:
        # 论文 Eq.20：A1=400℃, A2=500℃, A3=500℃
        amplitudes = [400.0, 500.0, 500.0]
    
    if sigma_params is None:
        # 论文 Eq.20 的参数：exp((-a*(x-cx)² - b*(y-cy)²)/d)
        # 峰1: a1=30, b1=45, d1=1000
        # 峰2: a2=30, b2=50, d2=400
        # 峰3: a3=30, b3=60, d3=100
        sigma_params = [(30.0, 45.0, 1000.0), (30.0, 50.0, 400.0), (30.0, 60.0, 100.0)]
    
    T = np.full_like(xx, T_bg)
    for (cx, cy), amp, (a, b, d) in zip(centers, amplitudes, sigma_params):
        T += amp * np.exp((-a * (xx - cx) ** 2 - b * (yy - cy) ** 2) / d)

    return T.ravel()


def single_peak_phantom(
    grid: GridConfig,
    center: Optional[Tuple[float, float]] = None,
    amp1: float = 100.0,
    sigma1_sq: float = 150.0,
    amp2: float = 700.0,
    sigma2_sq: float = 80.0,
    T_bg: float = 720.0,
) -> np.ndarray:
    """
    严格按论文复现：单峰温度分布（Li et al. 2025, Eq.(21)）
    
    双层同心高斯混合：低幅值宽幅（σ²=150）+ 高幅值窄幅（σ²=80）
    
    T(x,y) = 100·exp(-((x-8)² + (y-12)²)/150)
           + 700·exp(-((x-8)² + (y-12)²)/80) + 720
    
    物理坐标系：[0, 20] × [0, 20] m，中心位置为绝对坐标 (8, 12)。

    Args:
        grid       : 重建网格配置
        center     : 单峰中心坐标 (cx, cy)，单位m
                     默认参考论文：(8, 12)（20m×20m域）
        amp1       : 低幅值高斯的振幅 [℃]，默认 100℃（论文）
        sigma1_sq  : 低幅值高斯的 σ²（控制宽度），默认 150（论文）
        amp2       : 高幅值高斯的振幅 [℃]，默认 700℃（论文）
        sigma2_sq  : 高幅值高斯的 σ²（控制宽度），默认 80（论文，窄幅）
        T_bg       : 背景温度 [℃]，默认 720℃（论文配置）

    Returns:
        T_flat : 展平温度场，shape (ny*nx,)
    """
    xx, yy = grid.pixel_centers()
    
    if center is None:
        # 论文 Eq.21 的标准配置：物理坐标 (8, 12)（20m×20m域）
        center = (8.0, 12.0)
    
    cx, cy = center
    rr = (xx - cx) ** 2 + (yy - cy) ** 2
    
    # 按论文 Eq.21 构造：双层高斯叠加
    # 外层：低幅宽幅（代表外侧热影响区）
    # 内层：高幅窄幅（代表高温核心）
    T = amp1 * np.exp(-rr / sigma1_sq) + amp2 * np.exp(-rr / sigma2_sq) + T_bg

    return T.ravel()


def custom_phantom(
    grid: GridConfig,
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    用户自定义温度场

    Args:
        func : 函数 T = func(xx, yy)，xx/yy 为网格坐标数组 (ny, nx)
    Returns:
        T_flat : 展平温度场
    """
    xx, yy = grid.pixel_centers()
    return func(xx, yy).ravel()


# =============================================================================
# 声学CT测量仿真器
# =============================================================================

class AcousticCTSimulator:
    """
    声学CT温度场测量仿真器

    完整模拟流程：
        1. 初始化换能器阵列 & 重建网格
        2. 构建系统矩阵 A（Siddon 射线追踪）
        3. 生成目标温度场 T_true
        4. 正向计算无噪 TOF = A · s(T_true)
        5. 叠加高斯白噪声 → 测量 TOF b

    所有返回的 TOF 单位均为秒 [s]。
    """

    def __init__(
        self,
        transducer: TransducerConfig,
        grid: GridConfig,
        noise_level: float = 0.001,   # 相对噪声水平（0.001 = 0.1%）
        random_seed: int = 42,
    ):
        """
        Args:
            transducer  : 换能器阵列配置
            grid        : 重建网格配置
            noise_level : 噪声水平，设 >0 则使用论文基准 σ=3e-5 [秒]
            random_seed : 随机种子（保证可重复性）
        """
        self.transducer  = transducer
        self.grid        = grid
        self.noise_level = noise_level
        self.rng         = np.random.default_rng(random_seed)

        # 构建换能器坐标与射线对
        self.positions = transducer.get_positions()   # (N, 2)
        self.ray_pairs = transducer.get_ray_pairs()   # List[(i,j)]

        # 构建系统矩阵（计算可能耗时，仅构建一次）
        print(f"  构建系统矩阵 A "
              f"({len(self.ray_pairs)} 条射线 × {grid.n_pixels} 个像素)...")
        tracer = SiddonRayTracer(grid)
        self.A = tracer.build_system_matrix(self.positions, self.ray_pairs)
        print(f"  A 矩阵稀疏度: {self.A.nnz / (self.A.shape[0]*self.A.shape[1]):.4%}")

    # ------------------------------------------------------------------
    def simulate(self, T_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        正向仿真：给定真实温度场，生成含噪 TOF 测量值

        Args:
            T_true : 真实温度场（℃），shape (ny*nx,) 或 (ny, nx)

        Returns:
            tof_clean : 无噪飞行时间 [s]，shape (n_rays,)
            tof_noisy : 含噪飞行时间 [s]，shape (n_rays,)
        """
        T_flat = T_true.ravel().astype(np.float64)
        s_true = temperature_to_slowness(T_flat)

        tof_clean = compute_tof(self.A, s_true)

        if self.noise_level > 0:
            # 论文（Li et al. 2025）基准噪声: b_C = b_O + ω, ω ~ N(0, 3e-5)
            # 对应约 0.1% 相对噪声（800°C时 TOF≈30ms, σ=3e-5）
            sigma_paper = 3e-5  # 论文基准 σ [秒]
            noise = self.rng.normal(0.0, sigma_paper, size=tof_clean.shape)
            tof_noisy = tof_clean + noise
        else:
            tof_noisy = tof_clean.copy()

        # 确保 TOF 为正值（物理约束）
        tof_noisy = np.maximum(tof_noisy, 1e-12)

        return tof_clean, tof_noisy

    # ------------------------------------------------------------------
    def run_full_pipeline(
        self,
        phantom_name: str = "flame",
        T_bg: float = 20.0,
    ) -> dict:
        """
        一键运行完整仿真流水线

        Args:
            phantom_name : 测试场名称
              "gaussian"    — 单高斯热点
              "multi"       — 多高斯热点
              "flame"       — 火焰剖面（默认）
              "step"        — 阶跃场
              "dual_peak"   — 双峰温度分布（参考 Li et al. 2025 Eq.19）
              "triple_peak" — 三峰温度分布（参考 Li et al. 2025 Eq.20）
            T_bg         : 背景温度 [℃]

        Returns:
            包含以下键的字典：
              grid, transducer, positions, ray_pairs, A,
              T_true, tof_clean, tof_noisy
        """
        g = self.grid

        phantom_map = {
            "gaussian":     lambda: gaussian_hotspot(g, T_bg=T_bg),
            "multi":        lambda: multi_hotspot(g, T_bg),
            "flame":        lambda: flame_profile(g, T_bg),
            "step":         lambda: step_field(g, T_lo=T_bg, T_hi=T_bg + 200.0),
            "single_peak":  lambda: single_peak_phantom(g),
            "dual_peak":    lambda: dual_peak_phantom(g),
            "triple_peak":  lambda: triple_peak_phantom(g),
        }

        if phantom_name not in phantom_map:
            raise ValueError(
                f"未知 phantom_name={phantom_name!r}，"
                f"可选值: {list(phantom_map.keys())}"
            )

        T_true = phantom_map[phantom_name]()
        tof_clean, tof_noisy = self.simulate(T_true)

        print(f"  温度范围: {T_true.min():.1f} ~ {T_true.max():.1f} ℃")
        print(f"  TOF 范围: {tof_clean.min():.6f} ~ {tof_clean.max():.6f} s")
        print(f"  噪声水平: {self.noise_level*100:.2f}%")

        return {
            "grid":        self.grid,
            "transducer":  self.transducer,
            "positions":   self.positions,
            "ray_pairs":   self.ray_pairs,
            "A":           self.A,
            "T_true":      T_true,
            "tof_clean":   tof_clean,
            "tof_noisy":   tof_noisy,
        }


# =============================================================================
# 训练数据生成器（用于深度学习模型训练）
# =============================================================================

class TrainingDataGenerator:
    """
    深度学习训练数据批量生成器

    为需要训练数据的深度学习重建方法（AttentionUNet、Transformer、MC-Dropout 等）
    批量生成 (输入, 标签) 对：
        输入：含噪 TOF 向量 b ∈ R^{n_rays}
        标签：真实温度场 T ∈ R^{ny*nx}

    支持多种 phantom 类型的随机化：
        · 随机高斯热点（位置、幅值、宽度随机）
        · 随机双峰 / 三峰分布
        · 随机混合（从多种类型中随机选择）

    使用方法示例：
        >>> gen = TrainingDataGenerator(sim, n_samples=500)
        >>> tof_data, T_data = gen.generate()
        >>> # tof_data.shape = (500, n_rays)
        >>> # T_data.shape   = (500, ny*nx)
    """

    def __init__(
        self,
        simulator: 'AcousticCTSimulator',
        n_samples: int = 500,
        phantom_types: Optional[List[str]] = None,
        T_bg_range: Tuple[float, float] = (15.0, 30.0),
        amp_range: Tuple[float, float] = (30.0, 200.0),
        n_peaks_range: Tuple[int, int] = (1, 4),
        random_seed: int = 123,
    ):
        """
        Args:
            simulator     : 已初始化的 AcousticCTSimulator 实例（含系统矩阵 A）
            n_samples     : 生成样本数
            phantom_types : 使用的 phantom 类型列表
                           可选: 'random_gaussian', 'random_dual', 'random_triple', 'mixed'
                           默认: ['mixed']
            T_bg_range    : 背景温度随机范围 [℃]
            amp_range     : 高斯峰幅值随机范围 [℃]
            n_peaks_range : 随机高斯热点数量范围 (min, max)
            random_seed   : 随机种子
        """
        self.sim = simulator
        self.n_samples = n_samples
        self.phantom_types = phantom_types or ['mixed']
        self.T_bg_range = T_bg_range
        self.amp_range = amp_range
        self.n_peaks_range = n_peaks_range
        self.rng = np.random.default_rng(random_seed)

    def _random_gaussian_phantom(self) -> np.ndarray:
        """生成随机高斯热点温度场"""
        grid = self.sim.grid
        lx, ly = grid.lx, grid.ly

        T_bg = self.rng.uniform(*self.T_bg_range)
        n_peaks = self.rng.integers(self.n_peaks_range[0], self.n_peaks_range[1] + 1)

        centers = []
        amplitudes = []
        sigmas = []
        for _ in range(n_peaks):
            cx = self.rng.uniform(grid.x0 + 0.15 * lx, grid.x0 + 0.85 * lx)
            cy = self.rng.uniform(grid.y0 + 0.15 * ly, grid.y0 + 0.85 * ly)
            centers.append((cx, cy))
            amplitudes.append(self.rng.uniform(*self.amp_range))
            sigmas.append(self.rng.uniform(0.04 * min(lx, ly), 0.15 * min(lx, ly)))

        return gaussian_hotspot(grid, centers, amplitudes, sigmas, T_bg)

    def _random_dual_peak_phantom(self) -> np.ndarray:
        """生成随机双峰温度场"""
        grid = self.sim.grid
        lx, ly = grid.lx, grid.ly

        centers = []
        for _ in range(2):
            cx = self.rng.uniform(grid.x0 + 0.15 * lx, grid.x0 + 0.85 * lx)
            cy = self.rng.uniform(grid.y0 + 0.15 * ly, grid.y0 + 0.85 * ly)
            centers.append((cx, cy))

        amplitudes = [self.rng.uniform(100, 400), self.rng.uniform(100, 300)]
        T_bg = self.rng.uniform(700, 900)

        return dual_peak_phantom(grid, centers=centers, amplitudes=amplitudes, T_bg=T_bg)

    def _random_triple_peak_phantom(self) -> np.ndarray:
        """生成随机三峰温度场"""
        grid = self.sim.grid
        lx, ly = grid.lx, grid.ly

        centers = []
        for _ in range(3):
            cx = self.rng.uniform(grid.x0 + 0.15 * lx, grid.x0 + 0.85 * lx)
            cy = self.rng.uniform(grid.y0 + 0.15 * ly, grid.y0 + 0.85 * ly)
            centers.append((cx, cy))

        amplitudes = [self.rng.uniform(200, 500) for _ in range(3)]
        T_bg = self.rng.uniform(750, 900)

        return triple_peak_phantom(grid, centers=centers, amplitudes=amplitudes, T_bg=T_bg)

    def _random_single_peak_phantom(self) -> np.ndarray:
        """生成随机单峰温度场（参考 Li et al. 2025 Eq.21）"""
        grid = self.sim.grid
        lx, ly = grid.lx, grid.ly

        # 随机选择中心位置
        cx = self.rng.uniform(grid.x0 + 0.2 * lx, grid.x0 + 0.8 * lx)
        cy = self.rng.uniform(grid.y0 + 0.2 * ly, grid.y0 + 0.8 * ly)
        center = (cx, cy)

        # 随机化幅值和宽度参数
        amp1 = self.rng.uniform(50, 150)      # 低幅值高斯
        sigma1 = self.rng.uniform(100, 200)   # 宽 sigma
        amp2 = self.rng.uniform(500, 900)     # 高幅值高斯
        sigma2 = self.rng.uniform(50, 120)    # 窄 sigma
        T_bg = self.rng.uniform(650, 800)

        return single_peak_phantom(grid, center=center, amp1=amp1, sigma1_sq=sigma1,
                                   amp2=amp2, sigma2_sq=sigma2, T_bg=T_bg)

    def generate(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量生成训练数据

        Returns:
            tof_data : 含噪 TOF 数据，shape (n_samples, n_rays)
            T_data   : 真实温度场，shape (n_samples, ny*nx)
        """
        n_rays = self.sim.A.shape[0]
        n_pixels = self.sim.grid.n_pixels

        tof_data = np.zeros((self.n_samples, n_rays), dtype=np.float64)
        T_data = np.zeros((self.n_samples, n_pixels), dtype=np.float64)

        generators = {
            'random_gaussian': self._random_gaussian_phantom,
            'random_dual':     self._random_dual_peak_phantom,
            'random_triple':   self._random_triple_peak_phantom,
            'random_single':   self._random_single_peak_phantom,
        }

        for i in range(self.n_samples):
            # 选择 phantom 类型
            if 'mixed' in self.phantom_types:
                ptype = self.rng.choice(['random_gaussian', 'random_dual', 'random_triple'])
            else:
                ptype = self.rng.choice(self.phantom_types)

            T_true = generators[ptype]()
            _, tof_noisy = self.sim.simulate(T_true)

            tof_data[i] = tof_noisy
            T_data[i] = T_true

            if verbose and (i + 1) % max(1, self.n_samples // 10) == 0:
                print(f"  训练数据生成进度: {i+1}/{self.n_samples} "
                      f"({(i+1)/self.n_samples*100:.0f}%)")

        if verbose:
            print(f"  [OK] Generate done: TOF {tof_data.shape}, Temp {T_data.shape}")
            print(f"    温度范围: {T_data.min():.1f} ~ {T_data.max():.1f} ℃")

        return tof_data, T_data
