import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import os
import pandas as pd

# 导入q1.py中的模型函数
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'solve'))
from q1 import ball_dynamics, parse_coordinate, feature_points, frames_data

def run_trajectory_simulation(params):
    """
    使用给定的参数运行轨迹模拟，返回进球点的高度
    params: [vx0, vy0, vz0, wx, wy, wz, Cd, x0, y0]
    """
    vx0, vy0, vz0, wx, wy, wz, Cd, x0, y0 = params
    z0 = 0.0  # 固定起始高度

    state0 = [x0, y0, z0, vx0, vy0, vz0]
    omega_vec = np.array([wx, wy, wz])

    # 时间设置
    dt_frame = 0.04
    max_frame = max(d['t_idx'] for d in frames_data)
    t_eval = np.array([d['t_idx'] * dt_frame for d in frames_data])

    try:
        # 求解ODE
        sol = solve_ivp(ball_dynamics, [0, max_frame*dt_frame + 0.1], state0,
                        t_eval=t_eval, args=(Cd, 0.0, omega_vec), rtol=1e-5)

        if not sol.success:
            return None

        # 找到x=52.5m（球门线）处的高度
        sim_positions = sol.y[:3, :].T
        goal_line_idx = np.argmin(np.abs(sim_positions[:, 0] - 52.5))

        if goal_line_idx < len(sim_positions):
            return sim_positions[goal_line_idx, 2]  # z坐标（高度）
        else:
            return None

    except:
        return None

def sensitivity_analysis(base_params, param_names=None):
    """
    对足球轨迹模型进行灵敏度分析

    Parameters:
    base_params: 基准参数 [vx0, vy0, vz0, wx, wy, wz, Cd, x0, y0]
    param_names: 参数名称列表
    """
    if param_names is None:
        param_names = ['vx0', 'vy0', 'vz0', 'wx', 'wy', 'wz', 'Cd', 'x0', 'y0']

    # 扰动范围：-5% 到 +5%
    perturbations = np.linspace(0.95, 1.05, 21)
    percent_changes = (perturbations - 1.0) * 100

    # 计算基准情况的高度
    base_height = run_trajectory_simulation(base_params)
    if base_height is None:
        print("基准参数模拟失败")
        return

    print(f"基准参数下进球高度: {base_height:.3f} m")
    print("正在进行灵敏度分析...")

    # 创建单一大图显示所有参数的灵敏度曲线
    fig, ax = plt.subplots(figsize=(16, 10))

    # 学术风格配色和形状
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']  # matplotlib默认颜色方案
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+']  # 专业标记形状
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    # 对每个参数进行分析并绘制在同一图上
    for i, (param_name, base_value) in enumerate(zip(param_names, base_params)):
        heights = []

        for p in perturbations:
            # 修改当前参数
            test_params = base_params.copy()
            test_params[i] = base_value * p

            # 运行模拟
            height = run_trajectory_simulation(test_params)
            if height is not None:
                heights.append(height)
            else:
                heights.append(np.nan)  # 模拟失败

        # 移除NaN值
        valid_indices = ~np.isnan(heights)
        valid_changes = percent_changes[valid_indices]
        valid_heights = np.array(heights)[valid_indices]

        if len(valid_heights) > 0:
            # 绘制结果
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            ax.plot(valid_changes, valid_heights, marker=marker, linestyle='-',
                   color=color, linewidth=2, markersize=6, alpha=0.8,
                   label=f'{param_name} (Base: {base_value:.3f})')

    # 添加参考线
    ax.axhline(y=base_height, color='#666666', linestyle='--', linewidth=2, alpha=0.8,
              label=f'Base Height: {base_height:.1f}m')
    ax.axhline(y=2.44, color='#d62728', linestyle=':', linewidth=2,
              label='Crossbar (2.44m)')
    ax.axhline(y=0, color='#2ca02c', linestyle='-', linewidth=1,
              label='Ground (0m)')

    # 设置图表属性
    ax.set_title('Parameter Sensitivity Analysis - All Parameters', fontsize=16, fontweight='bold')
    ax.set_xlabel('Parameter Change (%)', fontsize=12)
    ax.set_ylabel('Goal Height (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # 设置坐标轴范围
    ax.set_xlim(-6, 6)  # -5% 到 +5%
    ax.set_ylim(-1, 4)  # 合理的高度范围

    plt.tight_layout()
    plt.savefig('/home/ubuntu/projects/mcm/training2/analysis/sensitivity_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # 创建放大部分的图 (1.4-2.5米高度范围)
    fig2, ax2 = plt.subplots(figsize=(16, 10))

    # 重新绘制所有曲线，但只显示1.4-2.5米范围
    for i, (param_name, base_value) in enumerate(zip(param_names, base_params)):
        heights = []

        for p in perturbations:
            test_params = base_params.copy()
            test_params[i] = base_value * p
            height = run_trajectory_simulation(test_params)
            if height is not None:
                heights.append(height)
            else:
                heights.append(np.nan)

        valid_indices = ~np.isnan(heights)
        valid_changes = percent_changes[valid_indices]
        valid_heights = np.array(heights)[valid_indices]

        if len(valid_heights) > 0:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            ax2.plot(valid_changes, valid_heights, marker=marker, linestyle='-',
                    color=color, linewidth=2, markersize=6, alpha=0.8,
                    label=f'{param_name} (Base: {base_value:.3f})')

    # 添加参考线
    ax2.axhline(y=base_height, color='#666666', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Base Height: {base_height:.1f}m')
    ax2.axhline(y=2.44, color='#d62728', linestyle=':', linewidth=2,
               label='Crossbar (2.44m)')
    ax2.axhline(y=0, color='#2ca02c', linestyle='-', linewidth=1,
               label='Ground (0m)')

    # 设置放大图的属性
    ax2.set_title('Parameter Sensitivity Analysis - Zoomed View (1.4-2.5m)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Parameter Change (%)', fontsize=12)
    ax2.set_ylabel('Goal Height (m)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # 设置放大的坐标轴范围
    ax2.set_xlim(-6, 6)  # -5% 到 +5%
    ax2.set_ylim(1.4, 2.5)  # 放大到1.4-2.5米范围

    # 保存放大图
    plt.savefig('/home/ubuntu/projects/mcm/training2/analysis/sensitivity_analysis_zoomed.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("灵敏度分析完成，结果已保存为 sensitivity_analysis.png")

    # 生成参数重要性总结
    print("\n=== 参数重要性总结 ===")
    print("每个参数1%变化对进球高度的影响：")

    summary_data = []
    for i, (param_name, base_value) in enumerate(zip(param_names, base_params)):
        heights = []
        for p in [0.99, 1.01]:  # -1% 和 +1% 的变化
            test_params = base_params.copy()
            test_params[i] = base_value * p
            height = run_trajectory_simulation(test_params)
            if height is not None:
                heights.append(height)

        if len(heights) == 2:
            height_change = heights[1] - heights[0]  # +1% 变化导致的高度变化
            importance = abs(height_change)
            summary_data.append((param_name, importance, height_change))

    # 按重要性排序
    summary_data.sort(key=lambda x: x[1], reverse=True)

    for param_name, importance, height_change in summary_data:
        direction = "+" if height_change > 0 else "-"
        print(".3f")

def load_optimized_params():
    """
    从q1.py的输出中加载优化后的参数
    或者使用手动设置的参数进行演示
    """
    # 这里可以从q1.py的结果中读取，或者使用典型的参数值
    # 基于之前的运行结果，使用近似值
    optimized_params = [
        27.7,   # vx0
        -5.1,   # vy0
        10.5,   # vz0
        0.15,   # wx
        0.06,   # wy
        0.14,   # wz
        0.48,   # Cd
        31.2,   # x0
        -2.2    # y0
    ]

    return optimized_params

if __name__ == "__main__":
    # 加载优化参数
    params = load_optimized_params()

    # 定义参数名称
    param_names = ['vx0', 'vy0', 'vz0', 'wx', 'wy', 'wz', 'Cd', 'x0', 'y0']

    # 运行灵敏度分析
    sensitivity_analysis(params, param_names)