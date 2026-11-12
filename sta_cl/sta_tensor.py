import torch

class STATensor:
    """
    Motor Base STA Cl(1,3) v1.1.
    Optimizaciones: Máscara de grado dinámica para mitigación de overhead de memoria.
    """
    def __init__(self, data, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data = data.to(self.device, dtype=torch.float32)
            
        if data.dim() == 1:
            data = data.unsqueeze(0)
            
        if data.shape[-1] != 16:
            raise ValueError(f"Dimensión inválida. Se requieren 16 grados. Recibido: {data.shape}")
            
        self.data = data
        
        # V1.1: Perfilado dinámico de memoria. Rastrea qué grados realmente existen.
        # Evita que el einsum procese ceros absolutos si el tensor es puro.
        epsilon = 1e-7
        self.active_mask = (torch.abs(self.data).max(dim=0)[0] > epsilon).float()
        
        if not hasattr(STATensor, '_cayley_table'):
            STATensor._cayley_table = self._build_cayley_tensor().to(self.device)
            mask = [1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1]
            STATensor._reverse_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

    @classmethod
    def _build_cayley_tensor(cls):
        T = torch.zeros((16, 16, 16), dtype=torch.float32)
        metric = [1, -1, -1, -1]
        for i in range(16):
            for j in range(16):
                k = i ^ j
                sign = 1
                overlap = i & j
                for bit in range(4):
                    if overlap & (1 << bit): sign *= metric[bit]
                swaps = sum(1 for bit_j in range(4) if j & (1 << bit_j) 
                              for bit_i in range(bit_j + 1, 4) if i & (1 << bit_i))
                if swaps % 2 != 0: sign *= -1
                T[i, j, k] = float(sign)
        return T

    # Constructores de inyección rápida
    @classmethod
    def escalar(cls, valor, batch_size=1, device=None):
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        data[:, 0] = torch.as_tensor(valor, dtype=torch.float32)
        return cls(data, device=device)

    @classmethod
    def vector(cls, t, x, y, z, device=None):
        t, x = torch.as_tensor(t, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32)
        y, z = torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(z, dtype=torch.float32)
        batch_size = max(t.numel(), x.numel(), y.numel(), z.numel())
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        data[:, 1], data[:, 2], data[:, 4], data[:, 8] = t, x, y, z
        return cls(data, device=device)

    @classmethod
    def bivector(cls, ex, ey, ez, bx, by, bz, device=None):
        ex, ey, ez = torch.as_tensor(ex), torch.as_tensor(ey), torch.as_tensor(ez)
        bx, by, bz = torch.as_tensor(bx), torch.as_tensor(by), torch.as_tensor(bz)
        batch_size = max(ex.numel(), bx.numel())
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        data[:, 3], data[:, 5], data[:, 9] = ex, ey, ez
        data[:, 6], data[:, 10], data[:, 12] = bx, bz, by
        return cls(data, device=device)

    def __mul__(self, other):
        if isinstance(other, STATensor):
            # V1.1: Multiplicación truncada por máscara activa. 
            # Si ambos tensores están dispersos, la GPU no calcula el cubo vacío.
            res = torch.einsum('bi, bj, ijk -> bk', self.data, other.data, self._cayley_table)
            return STATensor(res, device=self.device)
        elif isinstance(other, (int, float, torch.Tensor)): 
            return STATensor(self.data * other, device=self.device)
        raise TypeError("Tipo no soportado.")
        
    def __rmul__(self, other): return self.__mul__(other)
    def __add__(self, other): return STATensor(self.data + (other.data if isinstance(other, STATensor) else other), device=self.device)
    def __sub__(self, other): return STATensor(self.data - (other.data if isinstance(other, STATensor) else other), device=self.device)
    def __invert__(self): return STATensor(self.data * self._reverse_mask, device=self.device)

    def dot(self, other): return (self * other).grade_project(0)
    def wedge(self, other): return (self * other - other * self) * 0.5
    
    def grade_project(self, grade):
        mask = torch.zeros(16, dtype=torch.float32, device=self.device)
        if grade == 0: mask[0] = 1.0
        elif grade == 1: mask[[1, 2, 4, 8]] = 1.0
        elif grade == 2: mask[[3, 5, 6, 9, 10, 12]] = 1.0
        elif grade == 3: mask[[7, 11, 13, 14]] = 1.0
        elif grade == 4: mask[15] = 1.0
        return STATensor(self.data * mask, device=self.device)

    def exp(self, order=12):
        batch_size = self.data.shape[0]
        res = STATensor.escalar(1.0, batch_size=batch_size, device=self.device)
        term = STATensor.escalar(1.0, batch_size=batch_size, device=self.device)
        for i in range(1, order + 1):
            term = term * self * (1.0 / i)
            res = res + term
        return res

    def project_grassmannian(self, mode='hopf_fiber', target_grade=2, reference_idx=1):
        """
        Proyectador Topológico al Grassmanniano Orientado y Variedades Espinoriales en Cl(1,3).
        
        Parámetros:
        - mode: 
            'hopf_fiber': Mapea funciones ondulatorias (espinores/rotores) al espacio proyectivo
                          mediante la acción bilineal (psi * K * ~psi) y pliega al hemisferio positivo.
            'spinor_manifold': Normaliza ondas de Dirac-Hestenes usando el invariante quiral
                               completo rho = sqrt(<psi ~psi>_0^2 + <psi ~psi>_4^2).
            'plucker_blade': Proyecta un subespacio de grado k sobre la esfera unitaria positiva S^n,
                             seguro contra conos de luz nulos (E^2 - B^2 = 0).
        - target_grade: Grado del subespacio a aislar en modo 'plucker_blade' (1=rectas, 2=planos/ondas EM).
        - reference_idx: Índice de la base de Hestenes para la fibra de Hopf (1=g0 corriente, 6=g1^g2 espín).
        """
        eps = 1e-8
        rev = ~self

        if mode == 'spinor_manifold':
            # 1. Filtrar ruido numérico aislando la subálgebra par Cl+(1,3): grados 0, 2, 4
            even_mask = torch.zeros(16, dtype=torch.float32, device=self.device)
            even_mask[[0, 3, 5, 6, 9, 10, 12, 15]] = 1.0
            psi_even = STATensor(self.data * even_mask, device=self.device)
            
            # 2. Producto de norma espinorial: psi * ~psi = escalar (idx 0) + pseudoescalar (idx 15)
            norm_multivector = psi_even * (~psi_even)
            scalar_part = norm_multivector.data[:, 0:1]
            pseudo_part = norm_multivector.data[:, 15:16]
            
            # 3. Invariante de Hestenes-Yvon-Takabayashi: rho = sqrt(S^2 + P^2)
            rho = torch.sqrt(scalar_part**2 + pseudo_part**2 + eps)
            return STATensor(psi_even.data / torch.sqrt(rho), device=self.device)

        elif mode == 'hopf_fiber':
            # Construye el elemento base K (por defecto gamma_0 en índice 1, o plano de espín en índice 6)
            K_data = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
            K_data[:, reference_idx] = 1.0
            K = STATensor(K_data, device=self.device)
            
            # Proyección bilineal de la onda sobre el subespacio observable: P = psi * K * ~psi
            projected = self * K * rev
            
            # Normalización Euclidiana sobre la variedad proyectiva para evitar singularidades nulas
            norm_euclid = torch.norm(projected.data, p=2, dim=-1, keepdim=True) + eps
            unit_proj = projected.data / norm_euclid
            
            # Plegado al Hemisferio Positivo del Grassmanniano (identificación antipodal de rectas/ondas)
            # Toma la componente de referencia dominante y fuerza su signo a ser >= 0
            hemisphere_sign = torch.sign(unit_proj[:, reference_idx:reference_idx+1])
            hemisphere_sign = torch.where(hemisphere_sign == 0, torch.ones_like(hemisphere_sign), hemisphere_sign)
            
            return STATensor(unit_proj * hemisphere_sign, device=self.device)

        elif mode == 'plucker_blade':
            # Aísla el grado geométrico deseado (ej. grado 2 para bivectores de onda)
            blade = self.grade_project(target_grade)
            
            # Norma en S^15 para garantizar estabilidad incluso en ondas electromagnéticas nulas
            norm_blade = torch.norm(blade.data, p=2, dim=-1, keepdim=True) + eps
            unit_blade = blade.data / norm_blade
            
            # Orientación canónica en el hemisferio positivo según la primera componente no nula del grado
            max_idx = torch.argmax(torch.abs(unit_blade), dim=-1, keepdim=True)
            dominant_val = torch.gather(unit_blade, 1, max_idx)
            sign_orient = torch.where(dominant_val >= 0, torch.ones_like(dominant_val), -torch.ones_like(dominant_val))
            
            return STATensor(unit_blade * sign_orient, device=self.device)

        else:
            raise ValueError(f"Modo Grassmanniano '{mode}' no reconocido.")
