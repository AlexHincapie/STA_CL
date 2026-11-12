import torch
from .sta_tensor import STATensor

class GeometricEulerCromer:
    """
    Integration engine in Cl(1,3) - v1.2.
    Includes:
      1. step_lorentz: Euler-Cromer finite-difference scheme.
      2. step_lorentz_rotor: Exponential integrator on the Spin+(1,3) Lie group.
      3. step_geodesic: Evolution of reference frames via rotors.
    """
    def __init__(self, dt):
        self.dt = float(dt)

    def step_lorentz(self, pos, vel, F_field, q_m_ratio):
        """
        Euler-Cromer integrator.
        Hestenes equation: dv/dtau = (q/m) * (F . v)
        """
        aceleracion = F_field.dot(vel) * q_m_ratio
        vel_nueva = vel + (aceleracion * self.dt)
        pos_nueva = pos + (vel_nueva * self.dt)
        return pos_nueva, vel_nueva

    def step_lorentz_rotor(self, pos, vel, F_field, q_m_ratio):
        """
        Hestenes rotor exponential integrator on the Spin+(1,3) Lie group.
        Solves dv/dtau = (q/m) * (F . v) via sandwich transformation:
            R = exp( + (q / 2m) * F * dt )
            v(t + dt) = R * v(t) * ~R
        Applies midpoint integration for position updates.
        """
        # 1. Bivector generator of the Lorentz transformation
        generador_B = F_field * (0.5 * q_m_ratio * self.dt)
        rotor = generador_B.exp()
        
        # 2. Transformation of the four-velocity vector: R * v * ~R
        vel_nueva = (rotor * vel * (~rotor)).grade_project(1)
        
        # 3. Midpoint update for the four-position vector
        pos_nueva = pos + ((vel + vel_nueva) * (0.5 * self.dt))
        
        return pos_nueva, vel_nueva

    def step_geodesic(self, pos, rotor_R, omega_B):
        """
        Updates trajectories and spin using rotors (Gauge Theory Gravity).
        """
        delta_rotor = (omega_B * (0.5 * self.dt)).exp()
        R_nuevo = (delta_rotor * rotor_R).even_part()
        pos_nueva = (R_nuevo * pos * (~R_nuevo)).grade_project(1)
        return pos_nueva, R_nuevo
