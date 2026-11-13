# phantom_generators.py
"""
随机 Phantom 生成函数（各向同性高斯形式）
修正了旧代码中双/三峰形状参数不随机的问题

单峰：双层同心各向同性高斯
双峰：两个独立各向同性高斯
三峰：三个独立各向同性高斯
"""
import numpy as np
from typing import Optional


def single_peak_isotropic(
    nx: int, ny: int,
    seed: Optional[int] = None,
    cx: Optional[float] = None, cy: Optional[float] = None,
    amp_wide: Optional[float] = None, sigma_wide_sq: Optional[float] = None,
    amp_narrow: Optional[float] = None, sigma_narrow_sq: Optional[float] = None,
    T_bg: Optional[float] = None,
    x0: float = 0.0, y0: float = 0.0, lx: float = 20.0, ly: float = 20.0,
) -> np.ndarray:
    """
    单峰：双层同心各向同性高斯

    T = amp_wide·exp(-r²/σ_wide²) + amp_narrow·exp(-r²/σ_narrow²) + T_bg
    r² = (x - cx)² + (y - cy)²

    参数：
        cx, cy           : 峰中心坐标，若为 None 则随机
        amp_wide         : 外层（宽）高斯幅值，若为 None 则随机
        sigma_wide_sq    : 外层宽度（σ²），若为 None 则随机
        amp_narrow       : 内层（窄）高斯幅值，若为 None 则随机
        sigma_narrow_sq  : 内层宽度（σ²），若为 None 则随机
        T_bg             : 背景温度，若为 None 则随机
        x0, y0, lx, ly  : 物理域定义，默认 20m×20m

    返回：
        T_flat : 展平温度场，shape (nx*ny,)
    """
    rng = np.random.default_rng(seed)

    xx = np.linspace(x0, x0 + lx, nx)
    yy = np.linspace(y0, y0 + ly, ny)
    X, Y = np.meshgrid(xx, yy, indexing='ij')

    if cx is None:
        cx = rng.uniform(0.20 * lx, 0.80 * lx)
    if cy is None:
        cy = rng.uniform(0.20 * ly, 0.80 * ly)
    if amp_wide is None:
        amp_wide = rng.uniform(50, 150)
    if sigma_wide_sq is None:
        sigma_wide_sq = rng.uniform(100, 200)
    if amp_narrow is None:
        amp_narrow = rng.uniform(500, 900)
    if sigma_narrow_sq is None:
        sigma_narrow_sq = rng.uniform(50, 120)
    if T_bg is None:
        T_bg = rng.uniform(650, 800)

    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    T = (amp_wide * np.exp(-r2 / sigma_wide_sq)
       + amp_narrow * np.exp(-r2 / sigma_narrow_sq)
       + T_bg)
    return T.ravel()


def dual_peak_isotropic(
    nx: int, ny: int,
    seed: Optional[int] = None,
    cx1: Optional[float] = None, cy1: Optional[float] = None,
    cx2: Optional[float] = None, cy2: Optional[float] = None,
    A1: Optional[float] = None, A2: Optional[float] = None,
    sigma1: Optional[float] = None, sigma2: Optional[float] = None,
    T_bg: Optional[float] = None,
    x0: float = 0.0, y0: float = 0.0, lx: float = 20.0, ly: float = 20.0,
) -> np.ndarray:
    """
    双峰：两个独立各向同性高斯

    T = A1·exp(-r1²/(2σ1²)) + A2·exp(-r2²/(2σ2²)) + T_bg
    ri² = (x - cxi)² + (y - cyi)²

    注意：旧代码中形状参数(a,b,d)是固定的，这是BUG。
    新实现中每个峰的中心、幅值、宽度均独立随机。

    参数：
        cx1,cy1 / cx2,cy2 : 两峰中心，若为 None 则随机
        A1, A2            : 两峰幅值，若为 None 则随机
        sigma1, sigma2     : 两峰宽度(σ)，若为 None 则随机
        T_bg               : 背景温度，若为 None 则随机

    返回：
        T_flat : 展平温度场，shape (nx*ny,)
    """
    rng = np.random.default_rng(seed)

    xx = np.linspace(x0, x0 + lx, nx)
    yy = np.linspace(y0, y0 + ly, ny)
    X, Y = np.meshgrid(xx, yy, indexing='ij')

    if cx1 is None:
        cx1 = rng.uniform(0.15 * lx, 0.85 * lx)
    if cy1 is None:
        cy1 = rng.uniform(0.15 * ly, 0.85 * ly)
    if cx2 is None:
        cx2 = rng.uniform(0.15 * lx, 0.85 * lx)
    if cy2 is None:
        cy2 = rng.uniform(0.15 * ly, 0.85 * ly)
    if A1 is None:
        A1 = rng.uniform(100, 400)
    if A2 is None:
        A2 = rng.uniform(100, 400)
    if sigma1 is None:
        sigma1 = rng.uniform(0.8, 3.0)
    if sigma2 is None:
        sigma2 = rng.uniform(0.8, 3.0)
    if T_bg is None:
        T_bg = rng.uniform(700, 900)

    r1_sq = (X - cx1) ** 2 + (Y - cy1) ** 2
    r2_sq = (X - cx2) ** 2 + (Y - cy2) ** 2

    T = (A1 * np.exp(-r1_sq / (2 * sigma1 ** 2))
       + A2 * np.exp(-r2_sq / (2 * sigma2 ** 2))
       + T_bg)
    return T.ravel()


def triple_peak_isotropic(
    nx: int, ny: int,
    seed: Optional[int] = None,
    T_bg: Optional[float] = None,
    x0: float = 0.0, y0: float = 0.0, lx: float = 20.0, ly: float = 20.0,
) -> np.ndarray:
    """
    三峰：三个独立各向同性高斯

    T = Σᵢ₌₁³ Aᵢ·exp(-rᵢ²/(2σᵢ²)) + T_bg

    参数：
        T_bg : 背景温度，若为 None 则随机
        其他参数均在函数内部随机生成

    返回：
        T_flat : 展平温度场，shape (nx*ny,)
    """
    rng = np.random.default_rng(seed)

    xx = np.linspace(x0, x0 + lx, nx)
    yy = np.linspace(y0, y0 + ly, ny)
    X, Y = np.meshgrid(xx, yy, indexing='ij')

    if T_bg is None:
        T_bg = rng.uniform(750, 900)

    T = np.full((ny, nx), T_bg, dtype=np.float64)

    for _ in range(3):
        cx = rng.uniform(0.15 * lx, 0.85 * lx)
        cy = rng.uniform(0.15 * ly, 0.85 * ly)
        A  = rng.uniform(150, 500)
        sigma = rng.uniform(0.8, 3.0)
        r_sq = (X - cx) ** 2 + (Y - cy) ** 2
        T += A * np.exp(-r_sq / (2 * sigma ** 2))

    return T.ravel()
