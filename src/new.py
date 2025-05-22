import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SimulationParameters:
    N: int
    nu: float
    dt: float
    max_iter: int
    max_p_iter: int
    p_tol: float
    check_interval: int
    velocity_tol: float
    
    @property
    def h(self):
        return 1.0 / (self.N - 1)

def initialize_fields(params):
    """Initialize velocity and pressure fields with boundary conditions"""
    u = np.zeros((params.N, params.N))
    v = np.zeros((params.N, params.N))
    p = np.zeros((params.N, params.N))
    
    # Set top boundary velocity
    x = np.linspace(0, 1, params.N)
    u_top = np.sin(np.pi * x) ** 2
    u[:, -1] = u_top
    
    return u, v, p, u_top

def apply_velocity_boundary_conditions(u, v, u_top):
    """Apply Dirichlet boundary conditions to velocity fields"""
    # Top boundary
    u[:, -1] = u_top
    v[:, -1] = 0
    
    # Bottom boundary
    u[:, 0] = 0
    v[:, 0] = 0
    
    # Left boundary
    u[0, :] = 0
    v[0, :] = 0
    
    # Right boundary
    u[-1, :] = 0
    v[-1, :] = 0

def apply_pressure_boundary_conditions(p):
    """Apply Neumann boundary conditions to pressure field"""
    p[0, :] = p[1, :]      # Left
    p[-1, :] = p[-2, :]    # Right
    p[:, 0] = p[:, 1]      # Bottom
    p[:, -1] = p[:, -2]    # Top

def compute_intermediate_velocity(u, v, params):
    """Calculate intermediate velocity using explicit scheme"""
    u_prev = u.copy()
    v_prev = v.copy()
    h_sq = params.h ** 2
    
    # Update u component
    u[1:-1, 1:-1] += params.dt * (
        params.nu * (u_prev[2:, 1:-1] + u_prev[:-2, 1:-1] + 
                     u_prev[1:-1, 2:] + u_prev[1:-1, :-2] - 
                     4 * u_prev[1:-1, 1:-1]) / h_sq -
        (u_prev[1:-1, 1:-1] * (u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) / (2*params.h) +
         v_prev[1:-1, 1:-1] * (u_prev[1:-1, 2:] - u_prev[1:-1, :-2]) / (2*params.h))
    )
    
    # Update v component
    v[1:-1, 1:-1] += params.dt * (
        params.nu * (v_prev[2:, 1:-1] + v_prev[:-2, 1:-1] + 
                     v_prev[1:-1, 2:] + v_prev[1:-1, :-2] - 
                     4 * v_prev[1:-1, 1:-1]) / h_sq -
        (u_prev[1:-1, 1:-1] * (v_prev[2:, 1:-1] - v_prev[:-2, 1:-1]) / (2*params.h) +
         v_prev[1:-1, 1:-1] * (v_prev[1:-1, 2:] - v_prev[1:-1, :-2]) / (2*params.h))
    )

def solve_pressure(u, v, p, params):
    """Solve pressure Poisson equation using iterative method"""
    p_residual = 1.0
    for p_iter in range(params.max_p_iter):
        p_old = p.copy()
        
        # Update interior points
        p[1:-1, 1:-1] = (
            (p[1:-1, 2:] + p[1:-1, :-2] + 
             p[2:, 1:-1] + p[:-2, 1:-1]) -
            params.h**2 / (4 * params.dt) * (
                (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*params.h) +
                (v[1:-1, 2:] - v[1:-1, :-2]) / (2*params.h)
            )
        ) / 4
        
        apply_pressure_boundary_conditions(p)
        p_residual = np.max(np.abs(p[1:-1, 1:-1] - p_old[1:-1, 1:-1]))
        
        if p_residual < params.p_tol:
            break
            
    return p_residual

def apply_pressure_correction(u, v, p, params):
    """Correct velocity field using pressure gradient"""
    h_term = 2 * params.h
    dt = params.dt
    
    # Correct u component
    u[1:-1, 1:-1] -= dt * (p[2:, 1:-1] - p[:-2, 1:-1]) / h_term
    
    # Correct v component
    v[1:-1, 1:-1] -= dt * (p[1:-1, 2:] - p[1:-1, :-2]) / h_term

def check_convergence(u, v, prev_u, prev_v, params):
    """Check if velocity fields have converged"""
    du_max = np.max(np.abs(u - prev_u))
    dv_max = np.max(np.abs(v - prev_v))
    return (du_max < params.velocity_tol) and (dv_max < params.velocity_tol)

def plot_analysis_results(u, v):
    """Visualize simulation results"""
    N = u.shape[0]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # Streamline plot
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, u.T, v.T, density=2, color='lightgray')
    plt.title('Vortex Core Locations')
    plt.savefig('1_streamlines.jpg', bbox_inches='tight', dpi=300)
    plt.close()

    # Horizontal velocity profile
    plt.figure(figsize=(10, 4))
    mid_y = N // 2
    velocity = np.sqrt(u[mid_y, :]**2 + v[mid_y, :]**2)
    plt.plot(x, velocity, 'r-', linewidth=2)
    plt.xlabel('x coordinate')
    plt.ylabel('Velocity')
    plt.title(f'Horizontal Velocity Profile @ y={y[mid_y]:.2f}')
    plt.grid(True)
    plt.savefig('2_horizontal_profile.jpg', bbox_inches='tight', dpi=300)
    plt.close()

    # Vertical velocity profile
    plt.figure(figsize=(6, 8))
    mid_x = N // 2
    velocity = np.sqrt(u[:, mid_x]**2 + v[:, mid_x]**2)
    plt.plot(velocity, y, 'b-', linewidth=2)
    plt.ylabel('y coordinate')
    plt.xlabel('Velocity')
    plt.title(f'Vertical Velocity Profile @ x={x[mid_x]:.2f}')
    plt.grid(True)
    plt.savefig('3_vertical_profile.jpg', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Simulation parameters
    params = SimulationParameters(
        N=101,
        nu=0.001,
        dt=0.0001,
        max_iter=1000000,
        max_p_iter=10000,
        p_tol=1e-5,
        check_interval=100,
        velocity_tol=1e-7
    )
    
    # Initialize fields
    u, v, p, u_top = initialize_fields(params)
    prev_u, prev_v = np.zeros_like(u), np.zeros_like(v)
    converged = False
    
    # Main simulation loop
    for iter in range(params.max_iter):
        # Store previous velocities for convergence check
        if iter % params.check_interval == 0:
            prev_u[:], prev_v[:] = u, v
        
        # Velocity prediction step
        compute_intermediate_velocity(u, v, params)
        apply_velocity_boundary_conditions(u, v, u_top)
        
        # Pressure correction step
        p_residual = solve_pressure(u, v, p, params)
        if p_residual > params.p_tol:
            print(f"Pressure residual {p_residual:.3e} at iteration {iter}")
        
        # Velocity correction
        apply_pressure_correction(u, v, p, params)
        apply_velocity_boundary_conditions(u, v, u_top)
        
        # Convergence check
        if iter % params.check_interval == 0 and iter > 0:
            if check_convergence(u, v, prev_u, prev_v, params):
                converged = True
                break
    
    # Post-processing
    print("Simulation converged" if converged else "Simulation did not converge")
    plot_analysis_results(u, v)

if __name__ == "__main__":
    main()