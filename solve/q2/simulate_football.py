import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数
m = 0.43
r = 0.11
S = math.pi * r * r
V = 4.0/3.0 * math.pi * r**3
rho = 1.29
g = 9.8
mu = 1.8e-5
k = 0.07
C_M = 0.2

# 简化系数函数
def alpha(C_D):
    return rho * S * C_D / (2.0 * m)

beta = rho * V * C_M / (2.0 * m)

# 阻力系数随 Re
def C_D_from_v(v):
    Re = rho * v * r / mu
    return 0.2 if Re > 1e5 else 0.5

# 向量叉乘
def cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1],
                     a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]])

# 通用微分方程（返回 dx/dt 和 dv/dt）
def deriv(t, x, v, omega0, cd_func, add_leaf_magnus=False):
    # x: position (3,), v: velocity (3,)
    speed = np.linalg.norm(v)
    C_D = cd_func(speed)
    a = -alpha(C_D) * speed * v  # 阻力贡献
    omega = omega0 * math.exp(-k * t)
    Fm = beta * cross(omega, v)
    a = a + Fm
    a[2] = a[2] - g
    # 落叶球的经验向下修正（额外项）
    if add_leaf_magnus:
        # 经验项: -K * omega_x * v_x  (K==beta)
        a[2] = a[2] - beta * omega[0] * v[0]
    return v, a

# RK4 单步
def rk4_step(t, x, v, dt, omega0, cd_func, add_leaf_magnus=False):
    # state: (x,v)
    k1x, k1v = deriv(t, x, v, omega0, cd_func, add_leaf_magnus)
    k2x, k2v = deriv(t+dt/2, x + k1x*dt/2, v + k1v*dt/2, omega0, cd_func, add_leaf_magnus)
    k3x, k3v = deriv(t+dt/2, x + k2x*dt/2, v + k2v*dt/2, omega0, cd_func, add_leaf_magnus)
    k4x, k4v = deriv(t+dt, x + k3x*dt, v + k3v*dt, omega0, cd_func, add_leaf_magnus)
    x_new = x + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return x_new, v_new

# 集成器
def integrate(omega0, v0_vec, x0_vec, cd_func, add_leaf_magnus=False, dt=0.01, tmax=3.0):
    t = 0.0
    x = np.array(x0_vec, dtype=float)
    v = np.array(v0_vec, dtype=float)
    traj = [x.copy()]
    vel = [v.copy()]
    times = [t]
    while t < tmax and x[2] > 0 and x[0] < 16.5:
        x, v = rk4_step(t, x, v, dt, omega0, cd_func, add_leaf_magnus)
        t += dt
        traj.append(x.copy())
        vel.append(v.copy())
        times.append(t)
    return np.array(times), np.array(traj), np.array(vel)

# 初始条件示例
# 香蕉球
v0_b = 25.0
vz0_b = 6.0
vx0_b = math.sqrt(max(0.0, v0_b**2 - vz0_b**2))
omega0_b = np.array([0.0, 0.0, 150.0])

# 落叶球
v0_l = 25.0
vz0_l = 8.0
vx0_l = math.sqrt(max(0.0, v0_l**2 - vz0_l**2))
omega0_l = np.array([75.0, 0.0, 0.0])

# 电梯球
v0_e = 40.0
vz0_e = 3.0
vx0_e = math.sqrt(max(0.0, v0_e**2 - vz0_e**2))
omega0_e = np.array([20.0, 0.0, 0.0])

# 集成
dt = 0.01

# banana
times_b, traj_b, vel_b = integrate(omega0_b, [vx0_b, 0.0, vz0_b], [0.0,0.0,1.0], lambda v: 0.5, False, dt)
# leaf (add downward magnus correction)
times_l, traj_l, vel_l = integrate(omega0_l, [vx0_l, 0.0, vz0_l], [0.0,0.0,1.0], lambda v: 0.5, True, dt)
# elevator (C_D depends on Re)
times_e, traj_e, vel_e = integrate(omega0_e, [vx0_e, 0.0, vz0_e], [0.0,0.0,1.0], C_D_from_v, False, dt)

# 绘图
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(traj_b[:,0], traj_b[:,1], label='Banana')
plt.plot(traj_l[:,0], traj_l[:,1], label='Leaf')
plt.plot(traj_e[:,0], traj_e[:,1], label='Elevator')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.axis('equal')

plt.subplot(1,2,2)
plt.plot(traj_b[:,0], traj_b[:,2], label='Banana')
plt.plot(traj_l[:,0], traj_l[:,2], label='Leaf')
plt.plot(traj_e[:,0], traj_e[:,2], label='Elevator')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.legend()
plt.tight_layout()

out_path = 'd:\\我的文档\\桌面\\第二次训练\\trajectories.png'
plt.savefig(out_path, dpi=200)
print('Saved', out_path)

# 打印落地或终止信息
def report(name, times, traj):
    t_end = times[-1]
    x_end = traj[-1,0]
    z_end = traj[-1,2]
    print(f"{name}: t_end={t_end:.3f}s, x_end={x_end:.3f}m, z_end={z_end:.3f}m")

report('Banana', times_b, traj_b)
report('Leaf', times_l, traj_l)
report('Elevator', times_e, traj_e)

# 生成三维轨迹图
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj_b[:,0], traj_b[:,1], traj_b[:,2], label='Banana')
ax.plot(traj_l[:,0], traj_l[:,1], traj_l[:,2], label='Leaf')
ax.plot(traj_e[:,0], traj_e[:,1], traj_e[:,2], label='Elevator')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.legend()
plt.tight_layout()
out_path3 = 'd:\\我的文档\\桌面\\第二次训练\\trajectories_3d.png'
plt.savefig(out_path3, dpi=200)
print('Saved', out_path3)
