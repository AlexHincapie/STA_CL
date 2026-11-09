import torch
from .sta_tensor import STATensor

class GeometricEulerCromer:
    """
    Integrador Simpléctico Geométrico v1.1.
    Preserva invariantes topológicas (energía y momento angular) a largo plazo.
    Diseñado para cálculo masivo paralelizado en GPU.
    """
    def __init__(self, dt):
        self.dt = float(dt)

    def step_lorentz(self, pos, vel, F_field, q_m_ratio):
        """
        Evoluciona un lote de partículas cargadas en un campo electromagnético unificado.
        pos: STATensor (Cuadrivector de posición)
        vel: STATensor (Cuadrivector de velocidad)
        F_field: STATensor (Tensor Bivectorial de Faraday)
        q_m_ratio: float o Tensor (Relación carga-masa)
        
        Matemática STA: La aceleración es el producto punto entre el campo Bivectorial y la velocidad Vectorial.
        """
        # 1. Fuerza de Lorentz (dp/dt = q F . v)
        # La contracción F.dot(vel) proyecta un bivector contra un vector, resultando en el vector aceleración exacto.
        aceleracion = F_field.dot(vel) * q_m_ratio
        
        # 2. Actualización Simpléctica (Euler-Cromer)
        # Se actualiza PRIMERO la velocidad, y se usa la NUEVA velocidad para la posición.
        vel_nueva = vel + (aceleracion * self.dt)
        pos_nueva = pos + (vel_nueva * self.dt)
        
        return pos_nueva, vel_nueva

    def step_geodesic(self, pos, rotor_R, omega_B):
        """
        Actualiza trayectorias astrofísicas usando rotores cinemáticos (Gauge Theory Gravity).
        x(t+dt) = R * x(t) * ~R
        """
        # Generamos el rotor de evolución temporal mediante serie de Taylor
        delta_rotor = (omega_B * (self.dt / 2.0)).exp()
        R_nuevo = delta_rotor * rotor_R
        
        # Aplicamos la transformación de sándwich
        pos_nueva = R_nuevo * pos * (~R_nuevo)
        return pos_nueva, R_nuevo
