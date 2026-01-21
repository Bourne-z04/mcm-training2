import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import os

# 1. 重新加载数据和辅助函数
# 使用绝对路径以确保从任何工作目录运行都能找到数据文件
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'soccer_field_features.csv')
df = pd.read_csv(data_path)

def parse_coordinate(coord_str):
    if pd.isna(coord_str) or str(coord_str).strip() == '/' or 'source' in str(coord_str):
        return None
    try:
        clean_str = coord_str.split('source')[0].strip()
        u, v = map(float, clean_str.split())
        return np.array([u, v])
    except:
        return None

# 提取特征点 3D 坐标
feature_points = {}
for idx, row in df.iterrows():
    fid = row['id']
    if fid == 'soccer': continue
    try:
        feature_points[fid] = np.array([float(row['x']), float(row['y']), float(row['z'])])
    except:
        continue

# 2. 计算每帧的相机参数 (P矩阵, 相机中心, 足球的视线向量)
frames_data = []
frame_cols = [str(i) for i in range(1, 30)]

for frame_idx, frame in enumerate(frame_cols):
    if frame not in df.columns: continue
    
    p3d_list = []
    p2d_list = []
    
    # 收集特征点
    for idx, row in df.iterrows():
        fid = row['id']
        if fid == 'soccer': continue
        p2d = parse_coordinate(row[frame])
        if p2d is not None and fid in feature_points:
            p3d_list.append(feature_points[fid])
            p2d_list.append(p2d)
    
    if len(p3d_list) < 6: continue
    
    # DLT
    p3d_arr = np.array(p3d_list)
    p2d_arr = np.array(p2d_list)
    N = len(p3d_arr)
    A = []
    for i in range(N):
        X, Y, Z = p3d_arr[i]
        u, v = p2d_arr[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    U, S, Vh = np.linalg.svd(np.array(A))
    P = Vh[-1].reshape(3, 4)
    
    # 相机中心
    U_p, S_p, Vh_p = np.linalg.svd(P)
    C = Vh_p[-1]
    C = C[:3] / C[3]
    
    # 足球数据
    soccer_row = df[df['id'] == 'soccer']
    if soccer_row.empty: continue
    ball_uv = parse_coordinate(soccer_row.iloc[0][frame])
    
    if ball_uv is not None:
        # 计算视线方向向量 D = inv(M) * [u, v, 1] - C (approximation)
        # 更准确的是: M = P[:, :3], p4 = P[:, 3]. Ray direction d = inv(M) * pixel_homo
        # 但 P 是 3x4，我们用伪逆或直接求解方向
        # 简单方法: 找两个点，C 和 远平面上的点
        # inv P 方法:
        # [u, v, 1]^T = P * [X, Y, Z, 1]^T
        # 设 Z=0 和 Z=100 的点? 不, 视线是确定的
        # 使用 H = P[:, [0,1,3]] 找 Z=0 的点 (Ground intersection) 作为射线上一点 P_ground
        # Ray = P_ground - C
        
        # 使用之前的逻辑找 ground intersection
        H = P[:, [0, 1, 3]]
        try:
            H_inv = np.linalg.inv(H)
            pixel_vec = np.array([ball_uv[0], ball_uv[1], 1.0])
            ground_pt = np.dot(H_inv, pixel_vec)
            ground_pt = ground_pt / ground_pt[2]
            ground_pt_3d = np.array([ground_pt[0], ground_pt[1], 0.0])
            
            ray_dir = ground_pt_3d - C
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            frames_data.append({
                'frame_idx': frame_idx, # 0-based index
                't_idx': frame_idx,      # relative time step
                'C': C,
                'ray_dir': ray_dir,
                'uv': ball_uv,
                'P': P
            })
        except:
            pass

# 3. 物理模型与优化
# 常数
g = 9.81
m = 0.436  # kg
rho = 1.225 # kg/m^3
radius = 0.11 # m
A = np.pi * radius**2

# 微分方程
def ball_dynamics(t, state, Cd, Cl, omega_vec):
    # state = [x, y, z, vx, vy, vz]
    x, y, z, vx, vy, vz = state
    v_vec = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v_vec)
    
    if v_mag == 0: return [vx, vy, vz, 0, 0, 0]
    
    # Drag force
    Fd = -0.5 * rho * A * Cd * v_mag * v_vec
    
    # Magnus force
    # Fm = 0.5 * rho * A * Cl * (omega x v) / |omega| * |v| ? 
    # 或者 Fm = S * (omega x v). 通常 Cl 已经包含了相关系数. 
    # 简化模型: Fm = 0.5 * rho * A * Cl * (omega_unit x v_unit) * v^2 ?
    # 这里的Cl通常指升力系数. 标准公式 Fm = 1/2 rho A Cl |v|^2 * (wxv / |wxv|)
    # 为了简化优化，我们可以把 Magnus 力作为一个未知向量参数，或者假设 omega 恒定
    # 假设 omega 恒定，方向未知。
    # Let's use: Fm = 0.5 * rho * A * (S_vec x v_vec) where S_vec absorbs Cl and omega direction
    # 为了物理意义明确，使用: Fm = 0.5 * rho * A * Cl * cross(omega_norm, v)
    # 但 omega 方向未知. 让 params 优化 effective_spin_vector = Cl * omega_unit
    
    # 使用简化向量形式: F_total = m*g + F_drag + F_magnus
    # F_magnus_vec = np.cross(omega_vec, v_vec) # 这里 omega_vec 包含了系数
    
    F_g = np.array([0, 0, -m*g])
    F_m = np.cross(omega_vec, v_vec) # omega_vec 这里是 "Effective Spin Vector"
    
    F_total = F_g + Fd + F_m
    acc = F_total / m
    
    return [vx, vy, vz, acc[0], acc[1], acc[2]]

# 误差函数
def objective(params):
    # params: [vx0, vy0, vz0, wx, wy, wz, Cd, start_x, start_y]
    # start_z is fixed to 0
    # fps is fixed or optimized? Assuming 30fps for now, dt = 1/30
    
    vx0, vy0, vz0, wx, wy, wz, Cd, x0, y0 = params
    z0 = 0.0 # 题目约束
    
    state0 = [x0, y0, z0, vx0, vy0, vz0]
    omega_vec = np.array([wx, wy, wz])
    
    # 积分时间点
    # frames_data 中的 t_idx 是帧号. 真实时间 t = t_idx * dt
    dt_frame = 0.04 # 25fps = 0.04s, 30fps = 0.033s. 尝试 30fps
    
    # 获取最大帧号
    max_frame = max(d['t_idx'] for d in frames_data)
    t_eval = np.array([d['t_idx'] * dt_frame for d in frames_data])
    
    # 求解 ODE
    try:
        sol = solve_ivp(ball_dynamics, [0, max_frame*dt_frame + 0.1], state0, 
                        t_eval=t_eval, args=(Cd, 0.0, omega_vec), rtol=1e-5) # Cl is absorbed in omega_vec
    except:
        return 1e9
        
    if not sol.success:
        return 1e9
        
    # 计算误差: 3D 点到视线的距离
    # Distance from point P to line (C, D): || (P-C) x D || / || D ||
    # D is already normalized in frames_data
    total_error = 0
    
    sim_positions = sol.y[:3, :].T # (N, 3)
    
    if len(sim_positions) != len(frames_data):
        return 1e9
        
    for i, data in enumerate(frames_data):
        C = data['C']
        D = data['ray_dir']
        P_sim = sim_positions[i]
        
        # 向量 CP
        CP = P_sim - C
        cross_prod = np.cross(CP, D)
        dist = np.linalg.norm(cross_prod) # ||D|| is 1
        
        total_error += dist**2
        
    return total_error

# 4. 执行优化
# 初始猜测
# 根据第一帧的地面投影和大致方向猜测
first_frame = frames_data[0]
last_frame = frames_data[-1]
# 粗略速度: (pos_end - pos_start) / time
# 由于不知道深度，很难猜。
# 假设球速 25 m/s (C罗任意球通常很快)
# 方向：从 (-3.66~3.66) 附近的球门射出? 不，任意球通常是攻方。
# 假设初始位置在 25m 处 (x=25, y=10?), 射向球门 (x=52.5, y=0)
# 初始 x, y 可以从第一帧的 Ray 与 Z=0 交点获取
f0_C = first_frame['C']
f0_D = first_frame['ray_dir']
# Ray方程: P = C + k * D. P.z = 0 => C.z + k * D.z = 0 => k = -C.z / D.z
k0 = -f0_C[2] / f0_D[2]
start_pos_guess = f0_C + k0 * f0_D
x0_guess, y0_guess = start_pos_guess[0], start_pos_guess[1]

# C罗起脚时，球速约为26.8米/秒
v_guess = 26.8

# 计算到达球门的时间和方向
goal_center = np.array([52.5, 0, 2])
distance_to_goal = np.linalg.norm(goal_center[:2] - start_pos_guess[:2])
time_to_goal = 1.2  # 目标时间1.2秒

# 计算需要的水平速度
horizontal_speed = distance_to_goal / time_to_goal

# 如果需要的水平速度超过总速度，使用最大可能速度
if horizontal_speed > v_guess:
    horizontal_speed = v_guess * 0.95  
    time_to_goal = distance_to_goal / horizontal_speed

# 计算垂直速度分量
vz = np.sqrt(max(0, v_guess**2 - horizontal_speed**2))

# 构造方向向量
horizontal_dir = (goal_center[:2] - start_pos_guess[:2]) / distance_to_goal
dir_vec = np.array([horizontal_dir[0], horizontal_dir[1], vz / v_guess])
dir_vec = dir_vec / np.linalg.norm(dir_vec)  # 重新归一化
v0_vec = dir_vec * v_guess

# Spin 猜测 (C罗电梯球/落叶球，Top spin or Knuckle)
# 旋转速度约为5转/秒 = 5 * 2π ≈ 31.4 rad/s
spin_rate = 5 * 2 * np.pi  # 5转/秒 to rad/s
# Top spin: axis perpendicular to velocity direction
velocity_direction = dir_vec / np.linalg.norm(dir_vec)
# 找到垂直于速度的方向 (这里假设是侧旋或顶旋)
# 对于任意球，通常是顶旋转或侧旋转，这里使用顶旋转
spin_axis = np.cross(velocity_direction, np.array([0, 0, 1]))  # 与z轴的叉积得到垂直方向
if np.linalg.norm(spin_axis) < 1e-6:  # 如果速度垂直于z轴
    spin_axis = np.cross(velocity_direction, np.array([1, 0, 0]))
spin_axis = spin_axis / np.linalg.norm(spin_axis)
w_guess = spin_axis * spin_rate

# Params: [vx0, vy0, vz0, wx, wy, wz, Cd, x0, y0]
initial_params = [v0_vec[0], v0_vec[1], v0_vec[2], 0.1, 0.1, 0.1, 0.25, x0_guess, y0_guess]

# 边界
bounds = [
    (-50, 50), (-50, 50), (0, 50), # v
    (-50, 50), (-50, 50), (-50, 50), # w (effective)
    (0.1, 0.5), # Cd
    (x0_guess-5, x0_guess+5), (y0_guess-5, y0_guess+5) # start pos
]

res = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

print("=" * 60)
print("足球轨迹优化结果")
print("=" * 60)

print(f"优化成功: {res.success}")
print(".6f")
print(f"最终误差: {res.fun:.6f}")

# 解析优化结果
vx0, vy0, vz0, wx, wy, wz, Cd, x0, y0 = res.x
v0_magnitude = np.sqrt(vx0**2 + vy0**2 + vz0**2)

print("\n" + "-" * 40)
print("优化后的物理参数:")
print("-" * 40)
print("初速度参数:")
print(f"  vx0: {vx0:.3f} m/s")
print(f"  vy0: {vy0:.3f} m/s")
print(f"  vz0: {vz0:.3f} m/s")
print(f"  |v0|: {v0_magnitude:.3f} m/s")
print("旋转参数 (rad/s):")
print(f"  ωx: {wx:.3f}")
print(f"  ωy: {wy:.3f}")
print(f"  ωz: {wz:.3f}")
print(f"  |ω|: {np.sqrt(wx**2 + wy**2 + wz**2):.3f}")
print("空气阻力系数 Cd:")
print(f"  Cd: {Cd:.3f}")
print("起始位置 (m):")
print(f"  x0: {x0:.3f}")
print(f"  y0: {y0:.3f}")

# 5. 生成最终轨迹用于输出
params = res.x
vx0, vy0, vz0, wx, wy, wz, Cd, x0, y0 = params
z0 = 0.0
state0 = [x0, y0, z0, vx0, vy0, vz0]
omega_vec = np.array([wx, wy, wz])
dt_frame = 0.04
t_eval = np.linspace(0, 1.2, 120) # 模拟 1.2 秒
sol = solve_ivp(ball_dynamics, [0, 1.5], state0, t_eval=t_eval, args=(Cd, 0.0, omega_vec))

trajectory = sol.y[:3, :].T
print("Trajectory Shape:", trajectory.shape)

# 计算到达球门的时间和位置
goal_x = 52.5
goal_idx = np.argmin(np.abs(trajectory[:, 0] - goal_x))
goal_time = goal_idx * 0.01  # 每0.01秒一个点 (120点/1.2秒)
goal_height = trajectory[goal_idx, 2]

print("\n球门区域分析:")
print("-" * 40)
print(f"到达球门时间: {goal_time:.3f} 秒")
print(f"到达球门高度: {goal_height:.3f} 米")
print(f"球门线位置: x = {goal_x:.1f} 米")

# 输出 csv
# 使用绝对路径保存轨迹文件到solve目录
output_path = os.path.join(script_dir, 'optimized_trajectory.csv')
traj_df = pd.DataFrame(trajectory, columns=['x', 'y', 'z'])
traj_df.to_csv(output_path, index=False)