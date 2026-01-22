import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import os

# ==========================================
# 统一物理参数（与q1、q2保持一致）
# ==========================================

# 足球场坐标系（基于q1.py）
GOAL_LINE_X = 52.5  # 球门线位置 (m)
PENALTY_ARC_X = GOAL_LINE_X - 9.15  # 罚球弧位置 ≈ 43.35m
FREE_KICK_X = 31.22  # 从q1优化结果获取起始位置 (m)

# 物理常数（与q1、q2一致）
m = 0.436  # kg (FIFA标准足球质量)
radius = 0.11  # m (足球半径)
rho = 1.225  # kg/m³ (空气密度，15°C)
g = 9.81  # m/s² (重力加速度)
mu = 1.8e-5  # Pa·s (空气动力粘度)

# 计算参数
A = np.pi * radius**2  # 截面积
V_vol = 4/3 * np.pi * radius**3  # 体积

# 马格努斯力系数
C_M = 0.2

# 预计算系数
alpha_base = rho * A / (2 * m)
beta = rho * V_vol * C_M / (2 * m)

# 阻力系数（使用q1优化结果的固定值）
def get_drag_coefficient(v_magnitude):
    """使用q1优化结果的阻力系数"""
    return 0.476  # 从q1优化结果获取

# 改进的足球动力学模型（与q2一致）
def ball_dynamics(t, state, omega):
    """
    足球动力学模型（电梯球）
    state: [x, y, z, vx, vy, vz]
    omega: 角速度向量 [ωx, ωy, ωz]
    """
    x, y, z, vx, vy, vz = state
    v_vec = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v_vec)

    if v_mag < 1e-6:
        return [vx, vy, vz, 0, 0, -g]

    # 1. 重力
    F_g = np.array([0, 0, -m * g])

    # 2. 阻力（与q1一致）
    Cd = get_drag_coefficient(v_mag)
    F_d = -0.5 * rho * A * Cd * v_mag * v_vec

    # 3. 马格努斯力（与q1完全一致）
    F_m = np.cross(omega, v_vec)

    # 总加速度
    acc = (F_g + F_d + F_m) / m
    return [vx, vy, vz, acc[0], acc[1], acc[2]]

def simulate_shot(vx, vy, vz, omega, start_x=FREE_KICK_X, start_y=0.0, start_z=0.0):
    """
    模拟任意球轨迹
    vx, vy, vz: 笛卡尔坐标速度分量 (m/s)
    omega: 角速度向量 (rad/s)
    """
    v0 = np.sqrt(vx**2 + vy**2 + vz**2)  # 计算总速度（用于阻力计算）

    # Wall plane event (near penalty arc)
    def wall_plane(t, y, *args):
        return y[0] - PENALTY_ARC_X
    wall_plane.direction = 1

    # 球门线事件
    def goal_line(t, y, *args):
        return y[0] - GOAL_LINE_X
    goal_line.terminal = True
    goal_line.direction = 1

    # 积分求解
    sol = solve_ivp(ball_dynamics, [0, 2.0], [start_x, start_y, start_z, vx, vy, vz],
                   args=(omega,), events=[wall_plane, goal_line], rtol=1e-4, max_step=0.01)

    return sol

# ==========================================
# 改进的防守策略优化器
# ==========================================

def load_ronaldo_params():
    """
    从q1优化结果加载罗纳尔多任意球参数
    或者使用近似值进行演示
    """
    # 基于q1.py的优化结果
    params = {
        'v0': 30.09,      # 初速度 (m/s)
        'vx0': 27.75,     # x方向速度
        'vy0': -5.10,     # y方向速度
        'vz0': 10.47,     # z方向速度
        'omega': np.array([0.149, 0.056, 0.140]),  # 角速度 (rad/s)
        'start_pos': np.array([31.22, -2.25, 0.0])   # 起始位置
    }
    return params

def generate_ronaldo_shots(n_samples=1000):
    """
    基于q1结果生成罗纳尔多式射门样本
    考虑测量误差和技能变异性
    """
    base_params = load_ronaldo_params()

    shots = []
    for _ in range(n_samples):
        # 添加测量误差和技能变异（减小误差范围以确保进球）
        # 速度误差：±1 m/s
        v0 = np.random.normal(base_params['v0'], 0.5)
        v0 = np.clip(v0, 28, 32)

        # 角度误差：基于初速度计算仰角
        v_total = np.linalg.norm([base_params['vx0'], base_params['vy0'], base_params['vz0']])
        base_angle = np.degrees(np.arcsin(base_params['vz0'] / v_total))
        angle = np.random.normal(base_angle, 1.0)  # ±1度误差
        angle = np.clip(angle, 18, 22)

        # 侧向角误差 - 减小误差，让球更容易进门
        base_side_angle = np.degrees(np.arctan2(base_params['vy0'], base_params['vx0']))
        side_angle = np.random.normal(base_side_angle, 1.0)  # ±1度误差
        side_angle = np.clip(side_angle, base_side_angle - 2, base_side_angle + 2)

        # 角速度误差：±20%
        omega_scale = np.random.uniform(0.8, 1.2)
        omega = base_params['omega'] * omega_scale

        # 起始位置误差：±0.3m
        start_pos = base_params['start_pos'] + np.random.normal(0, 0.2, 3)
        start_pos[2] = 0  # 地面约束

        # 计算速度分量（考虑角度误差）
        angle_rad = np.radians(angle)
        side_angle_rad = np.radians(side_angle)
        vx = v0 * np.cos(angle_rad) * np.cos(side_angle_rad)
        vy = v0 * np.cos(angle_rad) * np.sin(side_angle_rad)
        vz = v0 * np.sin(angle_rad)

        # 模拟射门
        sol = simulate_shot(vx, vy, vz, omega,
                          start_pos[0], start_pos[1], start_pos[2])

        if sol.status == 1 and len(sol.t_events[1]) > 0:  # 到达球门线
            goal_y = sol.y_events[1][0][1]
            goal_z = sol.y_events[1][0][2]
            goal_t = sol.t_events[1][0]

            # 检查是否在球门范围内
            in_goal = (abs(goal_y) <= 3.66 and goal_z >= 0 and goal_z <= 2.44)

            shot_data = {
                'goal_y': goal_y,
                'goal_z': goal_z,
                'goal_t': goal_t,
                'valid': in_goal,
                'start_pos': start_pos,
                'v0': v0,
                'angle': angle,
                'side_angle': side_angle,
                'omega': omega
            }

            # 人墙处信息（罚球弧）
            if len(sol.t_events[0]) > 0:
                shot_data['wall_y'] = sol.y_events[0][0][1]
                shot_data['wall_z'] = sol.y_events[0][0][2]
                shot_data['wall_t'] = sol.t_events[0][0]
            else:
                shot_data['wall_y'] = None
                shot_data['wall_z'] = None
                shot_data['wall_t'] = None

            shots.append(shot_data)

    return shots

def optimize_wall_strategy(shots, n_players_range=[3,4,5], wall_positions=None):
    """
    Optimize wall strategy
    n_players_range: number of players range
    wall_positions: wall center position search range
    """
    if wall_positions is None:
        wall_positions = np.linspace(-2, 2, 21)  # -2m 到 2m

    # 人墙参数
    player_width = 0.8  # 球员宽度 (m)
    player_gap = 0.2     # 球员间距 (m)
    effective_width_per_player = player_width + player_gap

    # Jump parameters
    jump_time_to_peak = 0.35  # Time to reach peak jump height (s)
    jump_height = 2.3         # Maximum jump height (m)
    g_jump = 9.81             # Gravity for jump calculation

    # Calculate optimal jump timing for wall
    wall_jump_times = np.linspace(0, 0.5, 11)  # Possible jump start times (0-0.5s before ball arrival)

    best_strategy = None
    min_score_prob = 1.0

    for n_players in n_players_range:
        wall_width = n_players * effective_width_per_player

        for y_center in wall_positions:
            y_left = y_center - wall_width/2
            y_right = y_center + wall_width/2

            # Test different jump timings
            for jump_start_offset in wall_jump_times:  # Time before ball arrival to start jump
                blocked_shots = []
                passed_shots = []

                for shot in shots:
                    if shot['wall_y'] is None:
                        passed_shots.append(shot)
                        continue

                    wall_y, wall_z, wall_t = shot['wall_y'], shot['wall_z'], shot['wall_t']

                    # Check lateral position
                    in_wall_width = (y_left <= wall_y <= y_right)

                    if not in_wall_width:
                        passed_shots.append(shot)
                        continue

                    # Calculate wall height at ball arrival time
                    # Wall starts jumping at: wall_t - jump_start_offset
                    # Time from jump start to ball arrival: jump_start_offset
                    time_since_jump_start = jump_start_offset

                    if time_since_jump_start <= jump_time_to_peak:
                        # Rising phase of jump
                        current_wall_height = 0.5 * g_jump * time_since_jump_start**2
                    else:
                        # Falling phase of jump
                        time_in_fall = time_since_jump_start - jump_time_to_peak
                        current_wall_height = jump_height - 0.5 * g_jump * time_in_fall**2

                    # Wall can block if current height >= ball height
                    blocked = current_wall_height >= wall_z

                    if blocked:
                        blocked_shots.append(shot)
                    else:
                        passed_shots.append(shot)

                # Calculate score probability for this configuration
                score_prob = sum(1 for s in passed_shots if s['valid']) / len(shots)

                if score_prob < min_score_prob:
                    min_score_prob = score_prob
                    best_strategy = {
                        'n_players': n_players,
                        'y_center': y_center,
                        'y_left': y_left,
                        'y_right': y_right,
                        'wall_width': wall_width,
                        'jump_start_offset': jump_start_offset,
                        'blocked_count': len(blocked_shots),
                        'passed_count': len(passed_shots),
                        'score_prob': score_prob
                    }

    return best_strategy

def optimize_keeper_strategy(passed_shots):
    """
    Optimize goalkeeper strategy with dive direction
    Goalkeeper chooses optimal starting position and dive direction to cover most threatening shots
    """
    # Goalkeeper parameters
    reaction_time = 0.45  # Reaction time (s)
    avg_speed = 6.0        # Average movement speed (m/s)
    arm_reach = 0.8        # Arm reach (m)

    # Possible starting positions (near goal center)
    start_positions = np.linspace(-1.0, 1.0, 9)  # -1m to 1m from center

    # Possible dive angles (directions goalkeeper can dive)
    dive_angles = np.linspace(-60, 60, 25)  # -60° to 60° from forward

    best_strategy = None
    min_expected_goals = float('inf')

    for start_y in start_positions:
        for dive_angle_deg in dive_angles:
            dive_angle_rad = np.radians(dive_angle_deg)

            # Calculate dive direction vector
            dive_direction = np.array([np.sin(dive_angle_rad), np.cos(dive_angle_rad)])

            saves = 0
            total_threats = 0

            for shot in passed_shots:
                if shot['valid']:
                    total_threats += 1

                    shot_y, shot_z, shot_t = shot['goal_y'], shot['goal_z'], shot['goal_t']
                    available_time = shot_t - reaction_time

                    if available_time <= 0:
                        continue  # Cannot react in time

                    # Maximum distance goalkeeper can move
                    max_distance = available_time * avg_speed

                    # Vector from start position to shot
                    shot_vector = np.array([shot_y - start_y, 0])  # 2D movement
                    shot_distance = np.linalg.norm(shot_vector)

                    if shot_distance <= max_distance:
                        # Can reach the shot position
                        # Calculate angle between dive direction and required movement
                        if shot_distance > 0:
                            required_direction = shot_vector / shot_distance
                            angle_diff = np.arccos(np.clip(np.dot(dive_direction, required_direction), -1, 1))
                            angle_diff_deg = np.degrees(angle_diff)

                            # Save probability based on angle accuracy and distance
                            if angle_diff_deg <= 30:  # Within 30° of optimal dive direction
                                # Distance factor (closer is better)
                                distance_factor = 1 / (1 + shot_distance / arm_reach)

                                # Height factor
                                height_factor = 1.0
                                if shot_z > 1.8:  # High ball
                                    height_factor = 0.7
                                elif shot_z < 0.3:  # Low ball
                                    height_factor = 0.9

                                save_prob = distance_factor * height_factor * (1 - angle_diff_deg/30)
                                saves += min(save_prob, 0.95)
                            # If angle difference is too large, assume no save

            if total_threats > 0:
                expected_goals = total_threats - saves
            else:
                expected_goals = 0

            if expected_goals < min_expected_goals:
                min_expected_goals = expected_goals
                best_strategy = {
                    'start_y': start_y,
                    'dive_angle': dive_angle_deg,
                    'expected_goals': expected_goals,
                    'total_threats': total_threats,
                    'expected_saves': saves,
                    'save_rate': saves / total_threats if total_threats > 0 else 0
                }

    return best_strategy

def solve_strategy():
    """
    完整的防守策略优化
    """
    print("="*60)
    print("Free Kick Defense Strategy Optimization")
    print("="*60)

    # 1. Generate Ronaldo-style shot samples
    print("\n1. Generating shot samples...")
    shots = generate_ronaldo_shots(n_samples=1500)
    valid_shots = [s for s in shots if s['valid']]
    print(f"   Total samples: {len(shots)}")
    print(f"   Valid scoring shots: {len(valid_shots)}")
    print(".1%")

    # 2. Optimize wall strategy
    print("\n2. Optimizing wall strategy...")
    wall_strategy = optimize_wall_strategy(shots)

    print("   Best wall configuration:")
    print(f"   - Players: {wall_strategy['n_players']}")
    print(f"   - Center position: Y = {wall_strategy['y_center']:.2f} m")
    print(f"   - Width: {wall_strategy['wall_width']:.2f} m")
    print(f"   - Score probability: {wall_strategy['score_prob']:.1%}")
    print(f"   - Blocked shots: {wall_strategy['blocked_count']}")
    print(f"   - Remaining threats: {wall_strategy['passed_count']}")

    # 3. Get remaining shots after wall
    passed_shots = []
    for shot in shots:
        if shot['wall_y'] is None:
            passed_shots.append(shot)
            continue

        wall_y = shot['wall_y']
        in_wall = (wall_strategy['y_left'] <= wall_y <= wall_strategy['y_right'])
        blocked_by_height = shot['wall_z'] <= 2.3  # jump height

        if not (in_wall and blocked_by_height):
            passed_shots.append(shot)

    threat_shots = [s for s in passed_shots if s['valid']]
    print(f"\n   Threatening shots after wall: {len(threat_shots)}")

    # 4. Optimize goalkeeper strategy
    print("\n3. Optimizing goalkeeper strategy...")
    keeper_strategy = optimize_keeper_strategy(threat_shots)

    print("   Best goalkeeper strategy:")
    print(f"   - Start position: Y = {keeper_strategy['start_y']:.2f} m")
    print(f"   - Dive angle: {keeper_strategy['dive_angle']:.1f}°")
    print(f"   - Expected goals: {keeper_strategy['expected_goals']:.2f}")
    print(f"   - Save success rate: {keeper_strategy['save_rate']:.1%}")
    print(f"   - Expected saves: {keeper_strategy['expected_saves']:.2f}")
    # 5. Final results
    final_score_prob = wall_strategy['score_prob'] * (1 - keeper_strategy['save_rate'])
    print("\n4. Final defensive effectiveness:")
    print(f"   - Goal probability: {final_score_prob:.1%}")
    print(f"   - Defense efficiency: {(1-final_score_prob)*100:.3f}%")
    return wall_strategy, keeper_strategy, final_score_prob
    
    # 2. 优化人墙 (遍历人数和位置)
    print("优化人墙策略...")
    player_w = 0.5
    h_jump_peak = 2.45 # 起跳后头部极限高度
    t_jump_peak = 0.3  # 起跳到最高点耗时
    # 假设人墙在球罚出瞬间起跳，符合 h(t) 抛物线
    # 简化：假设球到达时，人墙必须处于有效阻挡高度内
    
    best_wall = None
    min_prob = 1.0
    
    # 搜索人数 [3, 4, 5]
    for n in [3, 4, 5]:
        w_width = n * player_w
        # Search center position
        for y_c in np.linspace(0, 2.5, 30):
            y_min, y_max = y_c - w_width/2, y_c + w_width/2
            
            blocked_indices = []
            for i, s in enumerate(shots):
                # 判定: 
                # 1. 横向在人墙内
                # 2. 高度 < 2.45 (假设人墙时机完美，能覆盖最高点)
                #    这里更严谨应该用 s['wall_t'] 校验跳跃抛物线，但假设职业球员能预判
                if (y_min <= s['wall_y'] <= y_max) and (s['wall_z'] <= h_jump_peak):
                    blocked_indices.append(i)
            
            # 3. 优化门将 (在当前人墙配置下)
            # 门将只处理剩下的球
            remaining_shots = [s for i, s in enumerate(shots) if i not in blocked_indices]
            
            if len(remaining_shots) == 0:
                score_prob = 0
            else:
                # 门将参数
                t_react = 0.3
                acc = 8.0 # m/s^2 爆发力
                arm = 0.9
                
                # 搜索门将最佳位置 (基于剩余球的重心)
                if len(remaining_shots) > 0:
                    avg_y = np.mean([s['goal_y'] for s in remaining_shots])
                else:
                    avg_y = 0
                
                # 计算门将能不能扑救
                saves = 0
                for s in remaining_shots:
                    t_avail = s['goal_t'] - t_react
                    if t_avail > 0:
                        # 最大扑救距离 d = 0.5 * a * t^2 + arm
                        d_max = 0.5 * acc * (t_avail**2) + arm
                        if abs(s['goal_y'] - avg_y) <= d_max and s['goal_z'] <= 2.44:
                            saves += 1
                
                conceded = len(remaining_shots) - saves
                score_prob = conceded / len(shots)
            
            if score_prob < min_prob:
                min_prob = score_prob
                best_wall = {
                    'n': n, 'y': y_c, 'rect': [y_min, y_max, h_jump_peak],
                    'keeper_y': avg_y if len(remaining_shots)>0 else 0
                }
    
def create_visualization(shots, wall_strategy, keeper_strategy):
    """
    创建防守策略可视化
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Goal area shot distribution
    valid_shots = [s for s in shots if s['valid']]
    gy = [s['goal_y'] for s in valid_shots]
    gz = [s['goal_z'] for s in valid_shots]

    ax1.scatter(gy, gz, c='red', alpha=0.6, s=20, label='Scoring shots')
    ax1.scatter([s['goal_y'] for s in shots if not s['valid']],
               [s['goal_z'] for s in shots if not s['valid']],
               c='gray', alpha=0.3, s=10, label='Non-scoring shots')

    # Goal frame
    ax1.plot([-3.66, 3.66], [0, 0], 'k-', lw=2)
    ax1.plot([-3.66, 3.66], [2.44, 2.44], 'k-', lw=2)
    ax1.plot([-3.66, -3.66], [0, 2.44], 'k-', lw=2)
    ax1.plot([3.66, 3.66], [0, 2.44], 'k-', lw=2)

    ax1.set_title('Shot Distribution (Goal Area)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Y Position (m)')
    ax1.set_ylabel('Height (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Wall blocking effect
    wall_y_positions = [s['wall_y'] for s in shots if s['wall_y'] is not None]
    wall_z_positions = [s['wall_z'] for s in shots if s['wall_z'] is not None]

    if wall_y_positions:
        ax2.scatter(wall_y_positions, wall_z_positions, c='blue', alpha=0.5, s=15, label='Trajectory at wall')

        # Wall area
        wall_rect = plt.Rectangle((wall_strategy['y_left'], 0),
                                 wall_strategy['wall_width'], 2.3,
                                 color='red', alpha=0.3, label='Wall blocking zone')
        ax2.add_patch(wall_rect)

        ax2.axhline(y=2.3, color='red', linestyle='--', alpha=0.7, label='Jump height limit')

    ax2.set_title('Wall Blocking Analysis', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Y Position (m)')
    ax2.set_ylabel('Height (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(0, 3)

    # 3. Goalkeeper save analysis
    threat_shots = [s for s in shots if s['valid']]  # 简化：所有进球威胁
    if threat_shots:
        threat_y = [s['goal_y'] for s in threat_shots]
        threat_z = [s['goal_z'] for s in threat_shots]

        ax3.scatter(threat_y, threat_z, c='orange', alpha=0.7, s=20, label='Threatening shots')

        # Goalkeeper start position
        ax3.plot(keeper_strategy['start_y'], 1.0, 'go', markersize=15,
                label=f'GK start (Y={keeper_strategy["start_y"]:.2f}m)')

        # Show dive direction arrow
        dive_angle_rad = np.radians(keeper_strategy['dive_angle'])
        arrow_length = 1.5
        arrow_dx = arrow_length * np.sin(dive_angle_rad)
        arrow_dy = arrow_length * np.cos(dive_angle_rad)
        ax3.arrow(keeper_strategy['start_y'], 1.0, arrow_dx, arrow_dy,
                 head_width=0.1, head_length=0.1, fc='green', ec='green',
                 label=f'Dive direction ({keeper_strategy["dive_angle"]:.1f}°)')

        # Goalkeeper arm reach range (at final position)
        final_x = keeper_strategy['start_y'] + arrow_dx
        final_y = 1.0 + arrow_dy
        arm_reach = 0.8
        circle = plt.Circle((final_x, final_y), arm_reach,
                          color='green', alpha=0.2, label='Arm reach range')
        ax3.add_patch(circle)

    # Goal frame
    ax3.plot([-3.66, 3.66], [0, 0], 'k-', lw=2)
    ax3.plot([-3.66, 3.66], [2.44, 2.44], 'k-', lw=2)
    ax3.plot([-3.66, -3.66], [0, 2.44], 'k-', lw=2)
    ax3.plot([3.66, 3.66], [0, 2.44], 'k-', lw=2)

    ax3.set_title('Goalkeeper Defense Analysis', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Y Position (m)')
    ax3.set_ylabel('Height (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # 4. 策略效果统计
    ax4.axis('off')

    stats_text = f"""Wall Strategy Stats:
Players: {wall_strategy['n_players']}
Position: Y={wall_strategy['y_center']:.1f}m
Width: {wall_strategy['wall_width']:.1f}m
Jump Time: {wall_strategy['jump_start_offset']:.2f}s before ball
Blocked: {wall_strategy['blocked_count']}

GK Strategy Stats:
Start Pos: Y={keeper_strategy['start_y']:.2f}m
Dive Angle: {keeper_strategy['dive_angle']:.1f}°
Exp. Goals: {keeper_strategy['expected_goals']:.2f}
Save Rate: {keeper_strategy['save_rate']:.1%}
Total Threats: {keeper_strategy['total_threats']}"""

    ax4.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))

    plt.tight_layout()
    plt.savefig('/home/ubuntu/projects/mcm/training2/solve/q3/strategy_solution.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("防守策略可视化已保存为 strategy_solution.png")

def main():
    """
    主函数
    """
    # 运行策略优化
    wall_strategy, keeper_strategy, final_score_prob = solve_strategy()

    # 生成射门样本用于可视化
    shots = generate_ronaldo_shots(n_samples=800)  # 较少样本用于可视化

    # Create visualization
    print("\n5. Generating strategy visualization...")
    create_visualization(shots, wall_strategy, keeper_strategy)

    # Output final recommendations
    print("\n" + "="*60)
    print("Defense Strategy Recommendations Summary")
    print("="*60)
    print("Defense strategy against Cristiano Ronaldo's elevator ball free kicks:")
    print()
    print("1. Wall Setup:")
    print(f"   - Position: Y = {wall_strategy['y_center']:.1f} m")
    print(f"   - Players: {wall_strategy['n_players']}")
    print(f"   - Width: {wall_strategy['wall_width']:.2f} m")
    print(f"   - Jump timing: {wall_strategy['jump_start_offset']:.2f}s before ball arrival")
    print("   - Purpose: Block low shots and near-post attempts")
    print()
    print("2. Goalkeeper Strategy:")
    print(f"   - Start position: Y = {keeper_strategy['start_y']:.2f} m")
    print(f"   - Dive angle: {keeper_strategy['dive_angle']:.1f}° from forward")
    print("   - Direction chosen to cover maximum threat shots")
    print("   - Considering reaction time and movement limitations")
    print()
    print("3. Expected Effectiveness:")
    print(f"   - Goal rate: {final_score_prob:.1%}")
    print("   - Significant improvement over no defense")
    print()
    print("4. Key Tactical Points:")
    print("   - Wall timing synchronized with ball trajectory")
    print("   - Goalkeeper dive direction optimized for coverage")
    print("   - Focus on defending elevator ball's sudden drop characteristics")

if __name__ == "__main__":
    main()