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
