"""
声学CT温度场重建 —— 可视化与定量评估模块
============================================
功能：
  1. 温度场伪彩色图（带换能器标注）
  2. 多算法重建结果对比图（子图布局）
  3. 误差场可视化（绝对误差 / 相对误差）
  4. 收敛曲线绘制
  5. L-curve 图
    6. 定量评估指标：RMSE / ARE
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Dict, List, Tuple
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置中文字体（Windows 环境）
plt.rcParams.update({
    "font.family":       "SimHei",
    "axes.unicode_minus": False,
    "figure.dpi":        120,
})


# =============================================================================
# 定量评估指标
# =============================================================================

def compute_metrics(T_true: np.ndarray, T_pred: np.ndarray) -> Dict[str, float]:
    """
    计算温度场重建定量评估指标（完全遵循论文公式）

    论文中的定义（第6.2节）：
        ARE(%) = (1/n) × Σ|Ti,o - Ti| / T_MEAN × 100%
        RMSE(%) = √((1/n) × Σ(Ti,o - Ti)²) / T_MEAN × 100%

    Args:
        T_true : 真实温度场（℃），任意形状
        T_pred : 重建温度场（℃），与 T_true 形状相同

        Returns:
            包含以下键的字典：
                RMSE(%) — 均方根误差（%），遵循论文公式
                ARE(%)  — 平均相对误差（%），遵循论文公式
                MaxAE   — 最大绝对误差 [℃]
                MAE     — 平均绝对误差 [℃]（辅助指标）
                RE(%)   — 相对误差（%），‖T_pred−T_true‖/‖T_true‖×100（兼容既有代码）
                PSNR(dB) — 峰值信噪比（dB），20*log10(T_max / RMSE_abs)
    """
    t = T_true.ravel().astype(np.float64)
    p = T_pred.ravel().astype(np.float64)

    diff = p - t
    t_mean = float(np.mean(t))
    t_max = float(np.max(t))
    
    # 论文公式：RMSE(%) = sqrt(mean(diff²)) / T_MEAN × 100%
    rmse_absolute = float(np.sqrt(np.mean(diff ** 2)))
    rmse_percent = float(rmse_absolute / (t_mean + 1e-14) * 100.0)

    # 论文公式：ARE(%) = mean(|diff|) / T_MEAN × 100%
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    are = float(mae / (t_mean + 1e-14) * 100.0)
    max_ae = float(np.max(abs_diff))

    # 兼容既有代码：RE(%) = ‖diff‖ / ‖T_true‖ × 100%
    denom = np.linalg.norm(t)
    re = float(np.linalg.norm(diff) / (denom + 1e-14) * 100.0)

    # PSNR = 20*log10(T_max / RMSE_abs)
    psnr = 20.0 * np.log10(t_max / (rmse_absolute + 1e-14)) if rmse_absolute > 0 else 100.0

    metrics = {
        "RMSE(%)": rmse_percent,
        "RMSE": rmse_percent,  # 向后兼容性：与RMSE(%)相同
        "ARE(%)": are,
        "MaxAE": max_ae,
        "MAE": mae,
        "RE(%)": re,
        "PSNR(dB)": float(psnr),
    }

    return metrics


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    title: str = "重建精度评估",
) -> None:
    """打印各算法评估指标对比表"""
    keys = list(next(iter(results.values())).keys())
    if "RMSE(%)" in keys and "ARE(%)" in keys:
        keys = ["RMSE(%)", "ARE(%)"]
    col_w = max(12, max(len(k) for k in results.keys()) + 2)
    metric_w = 12

    header = f"{'算法':<{col_w}}" + "".join(f"{k:>{metric_w}}" for k in keys)
    print(f"\n{'='*(col_w + metric_w*len(keys))}")
    print(f"  {title}")
    print(f"{'='*(col_w + metric_w*len(keys))}")
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<{col_w}}" + "".join(f"{m.get(k, float('nan')):>{metric_w}.4f}"
                                            for k in keys)
        print(row)
    print(f"{'='*(col_w + metric_w*len(keys))}\n")


# =============================================================================
# 单幅温度场可视化
# =============================================================================

def plot_temperature_field(
    T: np.ndarray,
    grid_shape: Tuple[int, int],
    domain: Tuple[float, float, float, float],
    title: str = "温度场",
    transducer_pos: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
) -> plt.Axes:
    """
    绘制单幅温度场伪彩色图

    Args:
        T              : 温度场（℃），shape (ny, nx) 或展平 (ny*nx,)
        grid_shape     : (ny, nx)
        domain         : (x_min, x_max, y_min, y_max) [m]
        transducer_pos : 换能器坐标 (N, 2)，None 则不绘制
        vmin/vmax      : 颜色范围
        ax             : 已有的 Axes；None 则新建
    """
    ny, nx = grid_shape
    x0, x1, y0, y1 = domain
    T2d = T.reshape(ny, nx)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        T2d,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap=cmap,
        vmin=vmin if vmin is not None else T2d.min(),
        vmax=vmax if vmax is not None else T2d.max(),
        aspect="equal",
        interpolation="bilinear",
    )

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        plt.colorbar(im, cax=cax, label="温度 [℃]")

    if transducer_pos is not None:
        ax.scatter(
            transducer_pos[:, 0], transducer_pos[:, 1],
            s=60, c="white", marker="^", edgecolors="black", linewidths=0.8,
            zorder=5, label="换能器",
        )
        ax.legend(fontsize=8, loc="upper right")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return ax


# =============================================================================
# 多算法对比图
# =============================================================================

def plot_comparison(
    T_true: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    grid_shape: Tuple[int, int],
    domain: Tuple[float, float, float, float],
    transducer_pos: Optional[np.ndarray] = None,
    suptitle: str = "温度场重建对比",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    多算法结果对比图（真值 + N 种重建 + 误差场）

    布局（每列为一个算法）：
        第1行：重建温度场
        第2行：绝对误差场  |T_pred − T_true|
    """
    names  = list(reconstructions.keys())
    n_algs = len(names)
    n_cols = n_algs + 1              # +1 列放真值
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 8))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)

    #vmin = min(T_true.min(), *(r.min() for r in reconstructions.values()))
    #vmax = max(T_true.max(), *(r.max() for r in reconstructions.values()))
        # ✅ 使用真值范围（与单个结果图的颜色风格一致）
    vmin = T_true.min()
    vmax = T_true.max()

    # 真值列
    ny, nx = grid_shape
    x0, x1, y0, y1 = domain

    def _imshow(ax, data2d, title, vmin_, vmax_, cmap="jet"):
        im = ax.imshow(
            data2d, origin="lower", extent=[x0, x1, y0, y1],
            cmap=cmap, vmin=vmin_, vmax=vmax_, aspect="equal",
            interpolation="bilinear",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label="[℃]")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        if transducer_pos is not None:
            ax.scatter(transducer_pos[:, 0], transducer_pos[:, 1],
                       s=40, c="white", marker="^", edgecolors="k",
                       linewidths=0.7, zorder=5)

    # 第0列：真值
    _imshow(axes[0, 0], T_true.reshape(ny, nx), "真实温度场", vmin, vmax)
    axes[1, 0].axis("off")   # 真值无误差列
    axes[1, 0].text(0.5, 0.5, "（基准）", ha="center", va="center",
                    transform=axes[1, 0].transAxes, fontsize=12)

    # 各算法列
    err_vmax = 0.0
    err_maps = {}
    for name, T_rec in reconstructions.items():
        err_maps[name] = np.abs(T_rec - T_true)
        err_vmax = max(err_vmax, err_maps[name].max())

    for col_i, name in enumerate(names):
        col = col_i + 1
        T_rec = reconstructions[name]
        m = compute_metrics(T_true, T_rec)

        title_rec = f"{name}\nRMSE={m['RMSE(%)']:.2f}%  ARE={m['ARE(%)']:.1f}%"
        _imshow(axes[0, col], T_rec.reshape(ny, nx), title_rec, vmin, vmax)

        err_title = f"绝对误差 | ARE={m['ARE(%)']:.1f}% | MaxAE={m['MaxAE']:.2f}℃"
        _imshow(axes[1, col], err_maps[name].reshape(ny, nx),
                err_title, 0.0, err_vmax, cmap="hot_r")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  图像已保存至: {save_path}")

    return fig


# =============================================================================
# 收敛曲线
# =============================================================================

def plot_convergence(
    residuals_dict: Dict[str, List[float]],
    title: str = "迭代收敛曲线",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """绘制各算法残差收敛曲线"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    colors = plt.cm.tab10.colors
    for i, (name, res) in enumerate(residuals_dict.items()):
        ax.semilogy(res, label=name, color=colors[i % 10], linewidth=1.8)

    ax.set_xlabel("迭代次数")
    ax.set_ylabel("残差范数 ||A s - b||")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.35)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# =============================================================================
# L-curve 图
# =============================================================================

def plot_l_curve(
    residuals: np.ndarray,
    reg_norms: np.ndarray,
    alpha_list: np.ndarray,
    best_alpha: float,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """绘制 L-curve，并标注最优正则化参数"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.loglog(residuals, reg_norms, "b.-", linewidth=1.5, markersize=5)

    # 标注最优点
    best_idx = np.argmin(np.abs(alpha_list - best_alpha))
    ax.loglog(residuals[best_idx], reg_norms[best_idx],
              "r*", markersize=15, label=f"最优 α={best_alpha:.1e}")

    # 标注各 α 值
    for i, (r, n, a) in enumerate(zip(residuals, reg_norms, alpha_list)):
        if i % max(1, len(alpha_list) // 6) == 0:
            ax.annotate(f"{a:.0e}", (r, n), fontsize=7,
                        xytext=(6, 2), textcoords="offset points")

    ax.set_xlabel("残差范数 ||A s - b||")
    ax.set_ylabel("正则化范数 ‖L s‖")
    ax.set_title("L-curve 正则化参数选取")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
