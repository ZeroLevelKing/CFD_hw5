import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from func import *

# ================== 主程序 ==================
def main():
    # 参数设置
    N = 101
    h = 1.0 / (N-1)
    nu = 0.001
    dt = 0.0001
    total_time = 5.0
    max_steps = int(total_time/dt)
    
    # 初始化
    u, v, p = initialize_fields(N)
    u, v = apply_velocity_bc(u, v)
    
    # 主循环
    for step in range(max_steps):
        # 中间速度场
        u_star, v_star = compute_convection_diffusion(u, v, nu, h, dt)
        
        # 压力修正
        p = solve_pressure(u_star, v_star, p, h, dt)
        
        # 速度修正
        u[1:-1,1:-1] -= dt*(p[2:,1:-1]-p[:-2,1:-1])/(2*h)
        v[1:-1,1:-1] -= dt*(p[1:-1,2:]-p[1:-1,:-2])/(2*h)
        
        # 边界条件
        u, v = apply_velocity_bc(u, v)
        
    # 后处理
    psi = compute_streamfunction(u, v, h)
    
    # 可视化
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x)
    plt.contour(X, Y, psi, levels=20, cmap='viridis')
    plt.streamplot(X, Y, u.T, v.T, color='k', density=2)
    plt.savefig('combined_result.png')

if __name__ == "__main__":
    main()