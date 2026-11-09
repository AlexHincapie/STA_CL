import torch
import math

class STATensor:
    """
    Motor Base para Álgebra de Espacio-Tiempo Cl(1,3).
    Evaluación paralelizada en GPU mediante isomorfismo binario de Grassmann.
    Firma Métrica: (+, -, -, -)
    """
    def __init__(self, data, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data = data.to(self.device, dtype=torch.float32)
        
        if data.dim() == 1:
            data = data.unsqueeze(0)
            
        if data.shape[-1] != 16:
            raise ValueError(f"Dimensión inválida. Se requieren 16 grados de libertad. Recibido: {data.shape}")
            
        self.data = data
        
        # Inyección de la Métrica en Caché VRAM
        if not hasattr(STATensor, '_cayley_table'):
            STATensor._cayley_table = self._build_cayley_tensor().to(self.device)
            
        # Máscara estática de reversión espacial (~)
        if not hasattr(STATensor, '_reverse_mask'):
            # Grados: [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
            # La reversión cambia el signo a los grados 2 y 3.
            mask = [1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1]
            STATensor._reverse_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

    @classmethod
    def _build_cayley_tensor(cls):
        """ Construcción algorítmica de la tabla geométrica usando reglas XOR de Grassmann. """
        T = torch.zeros((16, 16, 16), dtype=torch.float32)
        metric = [1, -1, -1, -1] # Firma de Hestenes
        
        for i in range(16):
            for j in range(16):
                k = i ^ j
                sign = 1
                overlap = i & j
                for bit in range(4):
                    if overlap & (1 << bit):
                        sign *= metric[bit]
                swaps = 0
                for bit_j in range(4):
                    if j & (1 << bit_j):
                        for bit_i in range(bit_j + 1, 4):
                            if i & (1 << bit_i):
                                swaps += 1
                if swaps % 2 != 0:
                    sign *= -1
                T[i, j, k] = float(sign)
        return T

    # ==========================================
    # CONSTRUCTORES FÍSICOS (Factory Methods)
    # ==========================================

    @classmethod
    def escalar(cls, valor, batch_size=1, device=None):
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        data[:, 0] = torch.as_tensor(valor, dtype=torch.float32)
        return cls(data, device=device)

    @classmethod
    def vector(cls, t, x, y, z, device=None):
        """ Cuadrivector espacio-temporal (t, x, y, z) """
        t, x = torch.as_tensor(t, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32)
        y, z = torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(z, dtype=torch.float32)
        batch_size = max(t.numel(), x.numel(), y.numel(), z.numel())
        
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        data[:, 1], data[:, 2], data[:, 4], data[:, 8] = t, x, y, z
        return cls(data, device=device)

    @classmethod
    def bivector(cls, ex, ey, ez, bx, by, bz, device=None):
        """ Tensor de Faraday unificado. E: Boosts espacio-temporales. B: Campos espaciales. """
        ex, ey, ez = torch.as_tensor(ex), torch.as_tensor(ey), torch.as_tensor(ez)
        bx, by, bz = torch.as_tensor(bx), torch.as_tensor(by), torch.as_tensor(bz)
        batch_size = max(ex.numel(), bx.numel())
        
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        # Bivectores de Boost (Espacio-Temporales): g0g1=3, g0g2=5, g0g3=9
        data[:, 3], data[:, 5], data[:, 9] = ex, ey, ez
        # Bivectores de Pauli (Espaciales): g1g2=6, g3g1=10, g2g3=12
        data[:, 6], data[:, 10], data[:, 12] = bx, bz, by
        return cls(data, device=device)

    @classmethod
    def trivector(cls, tx, ty, tz, t_xyz, device=None):
        """ Pseudo-vectores cuánticos """
        tx, ty, tz = torch.as_tensor(tx), torch.as_tensor(ty), torch.as_tensor(tz)
        t_xyz = torch.as_tensor(t_xyz)
        batch_size = max(tx.numel(), t_xyz.numel())
        
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        data[:, 7], data[:, 11], data[:, 13], data[:, 14] = tx, ty, tz, t_xyz
        return cls(data, device=device)

    @classmethod
    def pseudoescalar(cls, I, batch_size=1, device=None):
        """ Volumen Fundamental de 4 Dimensiones """
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        data[:, 15] = torch.as_tensor(I, dtype=torch.float32)
        return cls(data, device=device)

    # ==========================================
    # ÁLGEBRA GEOMÉTRICA (Operadores Centrales)
    # ==========================================

    def __mul__(self, other):
        """ Producto Geométrico Paralelizado (AB = A.B + A^B) """
        if isinstance(other, STATensor):
            res = torch.einsum('bi, bj, ijk -> bk', self.data, other.data, self._cayley_table)
            return STATensor(res, device=self.device)
        elif isinstance(other, (int, float, torch.Tensor)): 
            return STATensor(self.data * other, device=self.device)
        raise TypeError("Tipo no soportado para producto geométrico.")
        
    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, STATensor):
            return STATensor(self.data + other.data, device=self.device)
        return STATensor(self.data + other, device=self.device)

    def __sub__(self, other):
        if isinstance(other, STATensor):
            return STATensor(self.data - other.data, device=self.device)
        return STATensor(self.data - other, device=self.device)

    def __invert__(self):
        """ Reversión Espacial (~A): Aísla e invierte bivectores y trivectores. """
        return STATensor(self.data * self._reverse_mask, device=self.device)

    # ==========================================
    # PROYECCIONES DE GRADO
    # ==========================================

    def dot(self, other):
        """ Producto Interior: <AB>_0 para vectores, o contracción de menor grado general. """
        res = self * other
        return res.grade_project(0)

    def wedge(self, other):
        """ Producto Exterior (A^B): Captura áreas, volúmenes e hipervolúmenes. """
        # Para dos cuadrivectores A y B, A^B = 0.5 * (AB - BA)
        return (self * other - other * self) * 0.5

    def grade_project(self, grade):
        """ Aísla un grado geométrico específico (0 a 4) """
        mask = torch.zeros(16, dtype=torch.float32, device=self.device)
        if grade == 0: mask[0] = 1.0
        elif grade == 1: mask[[1, 2, 4, 8]] = 1.0
        elif grade == 2: mask[[3, 5, 6, 9, 10, 12]] = 1.0
        elif grade == 3: mask[[7, 11, 13, 14]] = 1.0
        elif grade == 4: mask[15] = 1.0
        
        return STATensor(self.data * mask, device=self.device)

    # ==========================================
    # DINÁMICA Y CÁLCULO
    # ==========================================

    def exp(self, order=12):
        """ 
        Exponenciación Tensorial por Serie de Taylor. 
        Calcula R = e^{B/2} sin divergencia condicional en GPU.
        Soporta bivectores simples, generales y campos nulos.
        """
        batch_size = self.data.shape[0]
        res = STATensor.escalar(1.0, batch_size=batch_size, device=self.device)
        term = STATensor.escalar(1.0, batch_size=batch_size, device=self.device)
        
        for i in range(1, order + 1):
            term = term * self * (1.0 / i)
            res = res + term
        return res

    @staticmethod
    def nabla(F, coords):
        """
        Derivada Topológica Espacio-Temporal (nabla F).
        Utiliza el grafo computacional de PyTorch para calcular derivadas de campo exactas.
        F: Campo Tensor Evaluado (STATensor)
        coords: Cuadrivector de Coordenadas [t, x, y, z] con requires_grad=True
        """
        assert coords.requires_grad, "Las coordenadas deben tener requires_grad=True"
        
        # Calcular los gradientes para cada uno de los 16 grados de libertad del campo F
        # respecto a las coordenadas (t, x, y, z)
        jacobiano = torch.zeros((*F.data.shape, 4), device=F.device)
        
        for k in range(16):
            grad_k = torch.autograd.grad(outputs=F.data[:, k].sum(), inputs=coords, 
                                         create_graph=True, retain_graph=True)[0]
            jacobiano[:, k, :] = grad_k
            
        # Vectores contravariantes de la base recíproca (gamma^0, gamma^1, gamma^2, gamma^3)
        # En Cl(1,3): g^0 = g_0, g^1 = -g_1, g^2 = -g_2, g^3 = -g_3
        g_0 = STATensor.vector(1, 0, 0, 0, device=F.device)
        g_1 = STATensor.vector(0, -1, 0, 0, device=F.device)
        g_2 = STATensor.vector(0, 0, -1, 0, device=F.device)
        g_3 = STATensor.vector(0, 0, 0, -1, device=F.device)
        
        # dF/dt * gamma^0 + dF/dx * gamma^1 + dF/dy * gamma^2 + dF/dz * gamma^3
        dF_dt = STATensor(jacobiano[:, :, 0], device=F.device)
        dF_dx = STATensor(jacobiano[:, :, 1], device=F.device)
        dF_dy = STATensor(jacobiano[:, :, 2], device=F.device)
        dF_dz = STATensor(jacobiano[:, :, 3], device=F.device)
        
        return g_0 * dF_dt + g_1 * dF_dx + g_2 * dF_dy + g_3 * dF_dz

    def __repr__(self):
        return f"STATensor(batch_size={self.data.shape[0]}, device={self.device})"
