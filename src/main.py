import numpy as np
import matplotlib.pyplot as plt
from func import *
from characteristic_quantity import *

# 参数设置
N = 64
nu = 0.001
dx = 1.0 / N
dt = 0.1 * min(dx**2/nu, dx/1.0)  # CFL条件
T = 10.0
max_step = int(T / dt)

# 初始化
u, v, p = initialize_grid(N)
u, v = apply_velocity_bc(u, v, dx, N)

# 时间迭代
for step in range(max_step):
    # 1. 同时计算u和v的中间速度
    conv_u, conv_v = compute_convection(u, v, dx, N)
    laplacian_u, laplacian_v = laplacian(u, v, dx)
    
    # 预测步（隐式处理粘性项）
    u_star = u + dt*(-conv_u + nu*laplacian_u)
    v_star = v + dt*(-conv_v + nu*laplacian_v)
    
    # 2. 压力修正（基于u_star和v_star的散度）
    p = pressure_poisson(p, u_star, v_star, dx, dt, N)
    
    # 3. 同时修正u和v的速度场
    u, v = project(u_star, v_star, p, dx, dt)
    u, v = apply_velocity_bc(u, v, dx, N)

# 计算特征量
psi = compute_streamfunction(u, v, dx, N)
vortex_centers = find_vortex_centers(psi, dx)
u_profile = velocity_profile(u, axis='x', pos=0.5, dx=dx)
v_profile = velocity_profile(v, axis='y', pos=0.5, dx=dx)

# 可视化
plt.figure(figsize=(15,5))

# 子图1：流线图与涡心
plt.subplot(131)
X, Y = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
plt.streamplot(X, Y, u.T, v.T, density=2, color='k')
for (x, y) in vortex_centers:
    plt.scatter(x, y, c='r', s=50)
plt.title("Streamlines & Vortex Centers")

# 子图2：x方向速度剖面（y=0.5）
plt.subplot(132)
y_center = np.linspace(0,1,N)
plt.plot(u_profile, y_center)
plt.xlabel("u velocity")
plt.ylabel("y")
plt.title("Vertical Velocity Profile (x=0.5)")

# 子图3：y方向速度剖面（x=0.5）
plt.subplot(133)
x_center = np.linspace(0,1,N)
plt.plot(x_center, v_profile)
plt.xlabel("x")
plt.ylabel("v velocity")
plt.title("Horizontal Velocity Profile (y=0.5)")

plt.tight_layout()
plt.savefig('results.png', dpi=300)