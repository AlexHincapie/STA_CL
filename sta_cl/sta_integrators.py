import torch
from .sta_tensor import STATensor

class GeometricEulerCromer:
    """
    Relativistic Dynamic Integration Engine in Cl(1,3) - v1.2. 
        Includes:
        1. step_lorentz: First-order symplectic Euler-Cromer (fast). 
        2. step_lorentz_rotor: Exact exponential integrator on the Spin+(1,3) Lie group
        that preserves the Minkowski norm and kinetic energy without numerical drift. 
        3. step_geodesic: Evolution of reference frames and parallel transport.
    """
    def __init__(self, dt):
        self.dt = float(dt)

    def step_lorentz(self, pos, vel, F_field, q_m_ratio):
        """
        Integrador Simpléctico de Euler-Cromer (Aproximación tangente rápida).
        Ecuación de Hestenes: dv/dtau = (q/m) * (F . v)
        """
        aceleracion = F_field.dot(vel) * q_m_ratio
        vel_nueva = vel + (aceleracion * self.dt)
        pos_nueva = pos + (vel_nueva * self.dt)
        return pos_nueva, vel_nueva

    def step_lorentz_rotor(self, pos, vel, F_field, q_m_ratio):
        """
        Integrador Exponencial de Rotores de Hestenes (Conservación Exacta en Grupo de Lie).
        Resuelve dv/dtau = (q/m) * (F . v) mediante transformación de sándwich isométrica:
            R = exp( + (q / 2m) * F * dt )
            v(t + dt) = R * v(t) * ~R
        Usa integración de punto medio (Verlet geométrico) para la posición.
        """
        # 1. Generador bivectorial del semigiro de Lorentz
        generador_B = F_field * (0.5 * q_m_ratio * self.dt)
        rotor = generador_B.exp()
        
        # 2. Rotación / Boost exacto del cuadrivector de velocidad: R * v * ~R
        vel_nueva = (rotor * vel * (~rotor)).grade_project(1)
        
        # 3. Actualización simpléctica de segundo orden para la posición (Trapecio/Verlet)
        pos_nueva = pos + ((vel + vel_nueva) * (0.5 * self.dt))
        
        return pos_nueva, vel_nueva

    def step_geodesic(self, pos, rotor_R, omega_B):
        """
        Actualiza trayectorias astrofísicas y espín usando rotores cinemáticos (Gauge Theory Gravity).
        """
        delta_rotor = (omega_B * (0.5 * self.dt)).exp()
        R_nuevo = (delta_rotor * rotor_R).even_part()
        pos_nueva = (R_nuevo * pos * (~R_nuevo)).grade_project(1)
        return pos_nueva, R_nuevo
