import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from func import *
from param import SimulationParameters

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