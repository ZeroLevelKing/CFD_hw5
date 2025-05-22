import numpy as np

def initialize_grid(N):
    """初始化交错网格"""
    # 速度场存储位置 (u: Nx+1 × Ny, v: Nx × Ny+1)
    u = np.zeros((N+1, N))    # 水平速度 (i+1/2,j)
    v = np.zeros((N, N+1))    # 垂直速度 (i,j+1/2)
    p = np.zeros((N, N))      # 压力 (i,j)
    return u, v, p

def apply_velocity_bc(u, v, dx, N):
    """应用速度边界条件"""
    # 上边界: u = sin²(πx), v=0
    for i in range(N):
        x = (i + 0.5) * dx
        u[i, -1] = np.sin(np.pi * x)**2  # 上边界u位于第N层
    v[:, -1] = 0.0
    
    # 左/右/下边界: u=0, v=0
    u[0, :] = 0.0       # 左边界
    u[-1, :] = 0.0      # 右边界
    v[:, 0] = 0.0       # 下边界
    return u, v

def compute_convection(u, v, dx, N):
    """计算u和v的对流项（二阶迎风）"""
    # 水平速度u的对流项
    conv_u = np.zeros_like(u)
    for i in range(1, N):
        for j in range(1, N-1):
            u_face = 0.5*(u[i,j] + u[i+1,j])  # 面心速度
            v_face = 0.5*(v[i,j] + v[i+1,j])
            # x方向对流
            if u_face > 0:
                dudx = (u[i,j] - u[i-1,j])/dx
            else:
                dudx = (u[i+1,j] - u[i,j])/dx
            # y方向对流
            if v_face > 0:
                dudy = (u[i,j] - u[i,j-1])/dx
            else:
                dudy = (u[i,j+1] - u[i,j])/dx
            conv_u[i,j] = u_face*dudx + v_face*dudy
    
    # 垂直速度v的对流项
    conv_v = np.zeros_like(v)
    for i in range(1, N-1):
        for j in range(1, N):
            u_face = 0.5*(u[i,j] + u[i,j+1])
            v_face = 0.5*(v[i,j] + v[i,j-1])
            # x方向对流
            if u_face > 0:
                dvdx = (v[i,j] - v[i-1,j])/dx
            else:
                dvdx = (v[i+1,j] - v[i,j])/dx
            # y方向对流
            if v_face > 0:
                dvdy = (v[i,j] - v[i,j-1])/dx
            else:
                dvdy = (v[i,j+1] - v[i,j])/dx
            conv_v[i,j] = u_face*dvdx + v_face*dvdy
    
    return conv_u, conv_v

def laplacian(u, v, dx):
    """同时计算u和v的拉普拉斯项"""
    laplacian_u = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2 \
                + (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2
    laplacian_v = (np.roll(v, -1, axis=0) - 2*v + np.roll(v, 1, axis=0)) / dx**2 \
                + (np.roll(v, -1, axis=1) - 2*v + np.roll(v, 1, axis=1)) / dx**2
    return laplacian_u, laplacian_v

def pressure_poisson(p, u, v, dx, dt, N, max_iter=1000, tol=1e-5):
    """SOR求解压力泊松方程"""
    beta = 1.7
    for _ in range(max_iter):
        p_old = p.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                div = (u[i+1,j] - u[i,j] + v[i,j+1] - v[i,j]) / dx
                p[i,j] = (1-beta)*p[i,j] + beta*0.25*(p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1] - dx**2 * div / dt)
        # 压力Neumann边界（∂p/∂n=0）
        p[:, 0] = p[:, 1]     # 下边界
        p[:, -1] = p[:, -2]   # 上边界
        p[0, :] = p[1, :]     # 左边界
        p[-1, :] = p[-2, :]   # 右边界
        if np.max(np.abs(p - p_old)) < tol:
            break
    return p

def project(u_star, v_star, p, dx, dt):
    """速度修正"""
    u = u_star.copy()
    v = v_star.copy()
    # 水平速度修正
    u[1:-1, 1:-1] -= dt * (p[1:, 1:-1] - p[:-1, 1:-1]) / dx
    # 垂直速度修正
    v[1:-1, 1:-1] -= dt * (p[1:-1, 1:] - p[1:-1, :-1]) / dx
    return u, v