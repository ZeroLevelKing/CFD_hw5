import numpy as np
from scipy.ndimage import gaussian_filter

def initialize_fields(N):
    """初始化场变量（非交错网格版）"""
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    
    # 设置上边界条件
    x = np.linspace(0, 1, N)
    u[:, -1] = np.sin(np.pi * x)**2  # 修正后的边界条件
    return u, v, p

def apply_velocity_bc(u, v):
    """应用速度边界条件（非交错网格版）"""
    # 上边界
    u[:, -1] = u[:, -2]  # 无滑移修正
    v[:, -1] = 0
    
    # 其他边界
    u[0, :] = u[-1, :] = 0  # 左右
    v[:, 0] = 0             # 下边界
    return u, v

def compute_convection_diffusion(u, v, nu, h, dt):
    """计算对流-扩散项（中心差分）"""
    u_new = u.copy()
    v_new = v.copy()
    
    # 水平动量方程
    u_new[1:-1,1:-1] += dt * (
        nu*(u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] -4*u[1:-1,1:-1])/h**2
        - u[1:-1,1:-1]*(u[2:,1:-1]-u[:-2,1:-1])/(2*h)
        - v[1:-1,1:-1]*(u[1:-1,2:]-u[1:-1,:-2])/(2*h)
    )
    
    # 垂直动量方程
    v_new[1:-1,1:-1] += dt * (
        nu*(v[2:,1:-1] + v[:-2,1:-1] + v[1:-1,2:] + v[1:-1,:-2] -4*v[1:-1,1:-1])/h**2 
        - u[1:-1,1:-1]*(v[2:,1:-1]-v[:-2,1:-1])/(2*h)
        - v[1:-1,1:-1]*(v[1:-1,2:]-v[1:-1,:-2])/(2*h)
    )
    
    return u_new, v_new

def solve_pressure(u, v, p, h, dt, max_iter=1000, tol=1e-5):
    """压力泊松方程求解器（带残差监控）"""
    for _ in range(max_iter):
        p_old = p.copy()
        
        # 离散压力方程
        p[1:-1,1:-1] = 0.25*(p[2:,1:-1]+p[:-2,1:-1]+p[1:-1,2:]+p[1:-1,:-2] 
                           - h**2/(4*dt)*(
                               (u[2:,1:-1]-u[:-2,1:-1])/(2*h) + 
                               (v[1:-1,2:]-v[1:-1,:-2])/(2*h)
                           ))
        
        # Neumann边界条件
        p = apply_pressure_bc(p)
        
        # 检查收敛性
        residual = np.max(np.abs(p - p_old))
        if residual < tol:
            break
    return p

def apply_pressure_bc(p):
    """压力场边界条件"""
    p[0, :] = p[1, :]     # 左
    p[-1, :] = p[-2, :]   # 右
    p[:, 0] = p[:, 1]     # 下
    p[:, -1] = p[:, -2]   # 上
    return p

def compute_streamfunction(u, v, h):
    """流函数计算（您的特征值分析部分）"""
    # 计算涡量ω = dv/dx - du/dy
    omega = np.zeros_like(u)
    omega[1:-1,1:-1] = (v[1:-1,2:] - v[1:-1,:-2])/(2*h) - (u[2:,1:-1] - u[:-2,1:-1])/(2*h)
    omega = gaussian_filter(omega, sigma=1)  # 平滑处理
    
    # 迭代求解ψ
    psi = np.zeros_like(u)
    for _ in range(10000):
        psi_old = psi.copy()
        psi[1:-1,1:-1] = 0.25*(psi[2:,1:-1]+psi[:-2,1:-1]+psi[1:-1,2:]+psi[1:-1,:-2] 
                              + h**2*omega[1:-1,1:-1])
        if np.max(np.abs(psi - psi_old)) < 1e-6:
            break
    return psi
