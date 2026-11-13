"""
声学CT温度场重建 —— 射线追踪模块
=====================================
物理基础:
    空气中声速与温度的关系: c(T) = 331.3 * sqrt(1 + T/273.15)  [m/s]
    声慢度定义:             s(T) = 1 / c(T)                      [s/m]
    飞行时间沿射线积分:     TOF  = ∫ s(T(r)) dr  ≈ Σ_p A[r,p]*s[p]

系统矩阵 A[r,p]:
    第 r 条射线穿过第 p 个像素的路径长度 [m]，
    由 Siddon 算法精确计算。

参考文献:
    Siddon R L. Fast calculation of the exact radiological path[J].
    Medical Physics, 1985, 12(2): 252-255.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 换能器与网格配置
# ---------------------------------------------------------------------------

@dataclass
class TransducerConfig:
    """换能器阵列配置（支持圆形、矩形和正方形布局）

    geometry 取值：
        'circle' — 圆形锅炉：换能器均匀分布在半径为 radius 的圆周上
        'rect'   — 矩形锅炉：换能器均匀分布在宽 width × 高 height 的矩形边界上
        'square' — 正方形锅炉(论文实验配置)：换能器均匀分布在边长为 side_length 的正方形边界上

    对于 'circle'，可配置 radius（直径 = 2 * radius）。
    对于 'rect'，可配置 width 和 height（长和宽）。
    对于 'square'，可配置 side_length（边长）。
    端口数 n_transducers 始终可自由配置。
    
    论文参考：
        Li et al. (2025), Applied Thermal Engineering, 277: 127113
        - 实验场景：20 m × 20 m 正方形（Furnace chamber）
        - 换能器数量：16个，均匀分布在正方形边界上
        - 测量方式：旋转扫描（一个发射，其余接收）
    """
    n_transducers: int                            # 换能器总数
    radius: float = 0.5                           # 圆形阵列半径 [m]
    center: Tuple[float, float] = (0.0, 0.0)     # 阵列中心坐标
    geometry: str = "circle"                      # 'circle', 'rect' 或 'square'
    width: float  = 1.0                           # 矩形宽度 [m]（仅 rect）
    height: float = 1.0                           # 矩形高度 [m]（仅 rect）
    side_length: float = 1.0                      # 正方形边长 [m]（仅 square）

    def get_positions(self) -> np.ndarray:
        """
        计算各换能器坐标

        circle: 均匀分布在圆周上
        rect:   均匀分布在矩形四条边上（按周长等分）
        square: 均匀分布在正方形四条边上（按周长等分）

        Returns:
            positions: shape (n_transducers, 2)，每行为 [x, y]
        """
        if self.geometry == "circle":
            return self._circle_positions()
        elif self.geometry == "rect":
            return self._rect_positions()
        elif self.geometry == "square":
            return self._square_positions()
        else:
            raise ValueError(
                f"未知 geometry={self.geometry!r}，"
                f"可选 'circle'、'rect' 或 'square'"
            )

    def _circle_positions(self) -> np.ndarray:
        """圆形阵列换能器坐标"""
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_transducers, endpoint=False)
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        return np.column_stack([x, y])

    def _rect_positions(self) -> np.ndarray:
        """
        矩形阵列换能器坐标

        在矩形四条边上按周长等距分布换能器。
        矩形中心为 self.center，宽度 self.width，高度 self.height。

        布局顺序：底边→右边→顶边→左边（逆时针）
        """
        cx, cy = self.center
        w, h = self.width, self.height
        x0 = cx - w / 2.0   # 左下角
        y0 = cy - h / 2.0

        perimeter = 2.0 * (w + h)
        N = self.n_transducers
        positions = []

        for i in range(N):
            # 沿周长的归一化位置 [0, perimeter)
            d = (i / N) * perimeter

            if d < w:
                # 底边：从左到右
                positions.append([x0 + d, y0])
            elif d < w + h:
                # 右边：从下到上
                positions.append([x0 + w, y0 + (d - w)])
            elif d < 2 * w + h:
                # 顶边：从右到左
                positions.append([x0 + w - (d - w - h), y0 + h])
            else:
                # 左边：从上到下
                positions.append([x0, y0 + h - (d - 2 * w - h)])

        return np.array(positions, dtype=np.float64)

    def _square_positions(self) -> np.ndarray:
        """
        正方形阵列换能器坐标（论文实验配置）

        在正方形四条边上按周长等距分布换能器。
        正方形中心为 self.center，边长为 self.side_length。

        布局顺序：底边→右边→顶边→左边（逆时针）
        
        参考：
            Li et al. (2025) Sec.6.1 - Simulation details
            实验场景：20m × 20m正方形，16个换能器均匀分布在边界上
            应用场景：炉膛（Furnace chamber）温度场重建
        
        Returns:
            positions: shape (n_transducers, 2)，每行为 [x, y]
        """
        cx, cy = self.center
        s = self.side_length
        x0 = cx - s / 2.0   # 左下角 x 坐标
        y0 = cy - s / 2.0   # 左下角 y 坐标

        perimeter = 4.0 * s
        N = self.n_transducers
        positions = []

        for i in range(N):
            # 沿周长的归一化位置 [0, perimeter)
            d = (i / N) * perimeter

            if d < s:
                # 底边：从左到右，y = y0，x 从 x0 到 x0+s
                positions.append([x0 + d, y0])
            elif d < 2 * s:
                # 右边：从下到上，x = x0+s，y 从 y0 到 y0+s
                positions.append([x0 + s, y0 + (d - s)])
            elif d < 3 * s:
                # 顶边：从右到左，y = y0+s，x 从 x0+s 到 x0
                positions.append([x0 + s - (d - 2 * s), y0 + s])
            else:
                # 左边：从上到下，x = x0，y 从 y0+s 到 y0
                positions.append([x0, y0 + s - (d - 3 * s)])

        return np.array(positions, dtype=np.float64)

    def get_ray_pairs(self) -> List[Tuple[int, int]]:
        """
        返回所有发射-接收换能器对（不重复，不含自身）
        
        在旋转扫描模式下，系统矩阵的行数为 n_transducers * (n_transducers - 1) / 2
        (因为每对换能器只计算一条射线，TOF 测量是对称的)
        
        Returns:
            pairs: List of (src_idx, dst_idx)，其中 src_idx < dst_idx
        """
        N = self.n_transducers
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


@dataclass
class GridConfig:
    """二维重建网格配置"""
    nx: int             # x 方向像素数
    ny: int             # y 方向像素数
    lx: float           # x 方向物理长度 [m]
    ly: float           # y 方向物理长度 [m]
    x0: float = 0.0     # 网格左下角 x 坐标 [m]
    y0: float = 0.0     # 网格左下角 y 坐标 [m]

    @property
    def dx(self) -> float:
        """单像素 x 方向宽度"""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """单像素 y 方向高度"""
        return self.ly / self.ny

    @property
    def n_pixels(self) -> int:
        return self.nx * self.ny

    def pixel_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回所有像素中心坐标的网格 (xx, yy)
        shape: (ny, nx)
        """
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        return np.meshgrid(x, y)

    def center_xy(self) -> Tuple[float, float]:
        """返回网格物理中心坐标"""
        return (self.x0 + self.lx / 2.0, self.y0 + self.ly / 2.0)


# ---------------------------------------------------------------------------
# Siddon 射线追踪器
# ---------------------------------------------------------------------------

class SiddonRayTracer:
    """
    基于 Siddon 算法的精确射线-像素交叉长度计算器

    系统矩阵 A 满足:
        TOF[r] = Σ_p  A[r,p] * s[p]

    其中 s[p] 为第 p 个像素的声慢度 [s/m]。
    """

    def __init__(self, grid: GridConfig):
        self.grid = grid
        # 预计算网格边界平面坐标（不变量）
        g = grid
        self._x_planes = g.x0 + np.arange(g.nx + 1) * g.dx   # shape (nx+1,)
        self._y_planes = g.y0 + np.arange(g.ny + 1) * g.dy   # shape (ny+1,)

    # ------------------------------------------------------------------
    def trace_ray(
        self,
        src: np.ndarray,
        dst: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        追踪单条射线，返回穿过像素的索引与对应路径长度。

        Args:
            src: 射线起点 [x, y]（换能器A坐标）
            dst: 射线终点 [x, y]（换能器B坐标）

        Returns:
            pixel_indices : shape (K,)  穿过像素的一维索引（行优先 iy*nx+ix）
            lengths       : shape (K,)  对应路径长度 [m]
        """
        g = self.grid
        x1, y1 = float(src[0]), float(src[1])
        x2, y2 = float(dst[0]), float(dst[1])

        dx_ray = x2 - x1
        dy_ray = y2 - y1
        ray_len = np.hypot(dx_ray, dy_ray)

        if ray_len < 1e-14:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

        # ---- 计算射线与各平面的参数化交点 t ∈ [0,1] ----
        def plane_intersections(planes: np.ndarray, p: float, dp: float) -> np.ndarray:
            if abs(dp) < 1e-14:
                return np.empty(0)
            return (planes - p) / dp

        alpha_x = plane_intersections(self._x_planes, x1, dx_ray)
        alpha_y = plane_intersections(self._y_planes, y1, dy_ray)

        # 合并、裁剪、去重
        alphas = np.concatenate([alpha_x, alpha_y, [0.0, 1.0]])
        alphas = np.unique(np.clip(alphas, 0.0, 1.0))

        if len(alphas) < 2:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

        # ---- 各线段中点确定像素 ----
        alpha_mid = 0.5 * (alphas[:-1] + alphas[1:])
        seg_len   = np.diff(alphas) * ray_len          # 各线段物理长度

        xm = x1 + alpha_mid * dx_ray
        ym = y1 + alpha_mid * dy_ray

        # 转换为像素下标
        ix = np.floor((xm - g.x0) / g.dx).astype(np.int32)
        iy = np.floor((ym - g.y0) / g.dy).astype(np.int32)

        # 筛选在网格内的像素
        # Bug fix #15: 处理浮点误差导致的边界问题
        # 当坐标恰好在网格边界（如 x=20.0）时，floor 可能返回 g.nx（=64）
        # 这超出像素索引范围 [0, g.nx-1]。需要用 clip 而不是硬排除
        # 或者更好的做法：先 clip 到有效范围，再检查
        ix = np.clip(ix, 0, g.nx - 1)
        iy = np.clip(iy, 0, g.ny - 1)
        
        # 现在所有坐标都被强制到有效范围内，只需检查线段长度
        valid = seg_len > 1e-14

        if ix.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

        pix = (iy * g.nx + ix).astype(np.int32)

        # 合并同一像素的重复线段（数值噪声可能导致同一像素出现两次）
        unique_pix, inv = np.unique(pix, return_inverse=True)
        merged = np.zeros(len(unique_pix), dtype=np.float64)
        np.add.at(merged, inv, seg_len)

        keep = merged > 1e-14
        return unique_pix[keep], merged[keep]

    # ------------------------------------------------------------------
    def build_system_matrix(
        self,
        positions: np.ndarray,
        ray_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        构建稀疏系统矩阵 A  (shape: n_rays × n_pixels)

        Args:
            positions : 换能器坐标，shape (N, 2)
            ray_pairs : [(src_idx, dst_idx), ...]；None 则取所有无重复对

        Returns:
            A : scipy.sparse.csr_matrix
        """
        from scipy.sparse import csr_matrix

        N = len(positions)
        if ray_pairs is None:
            ray_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

        n_rays   = len(ray_pairs)
        n_pixels = self.grid.n_pixels

        rows, cols, data = [], [], []
        for r_idx, (i, j) in enumerate(ray_pairs):
            pix, lengths = self.trace_ray(positions[i], positions[j])
            rows.extend([r_idx] * len(pix))
            cols.extend(pix.tolist())
            data.extend(lengths.tolist())

        A = csr_matrix(
            (data, (rows, cols)),
            shape=(n_rays, n_pixels),
            dtype=np.float64,
        )
        return A


# ---------------------------------------------------------------------------
# 声速 ↔ 温度 转换
# ---------------------------------------------------------------------------

def temperature_to_slowness(T: np.ndarray) -> np.ndarray:
    """
    温度 T [℃] → 声慢度 s [s/m]
    物理公式: c(T) = 331.3 × √(1 + T/273.15)
    """
    T = np.asarray(T, dtype=np.float64)
    c = 331.3 * np.sqrt(1.0 + T / 273.15)
    return 1.0 / c


def slowness_to_temperature(s: np.ndarray) -> np.ndarray:
    """
    声慢度 s [s/m] → 温度 T [℃]
    逆变换: T = 273.15 × ((c/331.3)² - 1)，c = 1/s
    """
    s = np.asarray(s, dtype=np.float64)
    # 防止除零
    s_safe = np.where(np.abs(s) < 1e-14, 1e-14, s)
    c = 1.0 / s_safe
    return 273.15 * ((c / 331.3) ** 2 - 1.0)


def compute_tof(A, slowness_field: np.ndarray) -> np.ndarray:
    """
    由声慢度场正向计算飞行时间向量。

    Args:
        A             : 系统矩阵 (n_rays × n_pixels)，稀疏或密集
        slowness_field: 展平慢度向量，shape (n_pixels,)

    Returns:
        tof : 飞行时间，shape (n_rays,)
    """
    return A.dot(slowness_field)
