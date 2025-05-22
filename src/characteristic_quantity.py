import numpy as np
from scipy.ndimage import gaussian_filter

def compute_streamfunction(u, v, dx, N):
    """求解流函数∇²ψ = -ω（Gauss-Seidel迭代）"""
    omega = (v[1:, :] - v[:-1, :])/dx - (u[:, 1:] - u[:, :-1])/dx
    omega = gaussian_filter(omega, sigma=1)  # 平滑涡量
    
    psi = np.zeros((N, N))
    max_iter = 10000
    for _ in range(max_iter):
        psi_old = psi.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                psi[i,j] = 0.25*(psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] + dx**2 * omega[i,j])
        if np.max(np.abs(psi - psi_old)) < 1e-6:
            break
    return psi

def find_vortex_centers(psi, dx, threshold=0.1):
    """定位主涡、二次涡中心（寻找极值点）"""
    grad_x, grad_y = np.gradient(psi)
    magnitude = grad_x**2 + grad_y**2
    maxima = (magnitude < threshold) & (psi > np.roll(psi,1,axis=0)) & (psi > np.roll(psi,-1,axis=0)) & \
             (psi > np.roll(psi,1,axis=1)) & (psi > np.roll(psi,-1,axis=1))
    y_idx, x_idx = np.where(maxima)
    return [(x*dx, y*dx) for x, y in zip(x_idx, y_idx)]

def velocity_profile(u, axis='x', pos=0.5, dx=0.01):
    """提取中心线速度剖面"""
    N = int(1/dx)
    idx = int(pos/dx)
    if axis == 'x':
        return u[idx, :]  # y方向速度沿x=0.5
    else:
        return u[:, idx]  # x方向速度沿y=0.5