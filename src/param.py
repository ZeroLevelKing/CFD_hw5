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