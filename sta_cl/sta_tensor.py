import torch

class STATensor:
    """
    Tensor Base para Álgebra de Espacio-Tiempo Cl(1,3).
    Mapeo de los 16 grados de libertad (Firma de Hestenes: +, -, -, -):
    [0]     : Escalar
    [1-4]   : Vectores (g0, g1, g2, g3)
    [5-10]  : Bivectores (Álgebra de Pauli / Faraday)
    [11-14] : Trivectores (Pseudo-vectores)
    [15]    : Pseudoescalar (I)
    """
    def __init__(self, data, device=None):
        # 1. Anclaje de Hardware Automático
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 2. Ingesta y Estandarización del Tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data = data.to(self.device, dtype=torch.float32)
        
        # Forzar estructura batched: [N, 16] o [16] -> [1, 16]
        if data.dim() == 1:
            data = data.unsqueeze(0)
            
        if data.shape[-1] != 16:
            raise ValueError(f"Violación topológica: Un multivector Cl(1,3) requiere 16 dimensiones. Recibido: {data.shape}")
            
        self.data = data
        
        # 3. Caché del Tensor Multiplicativo (Tabla de Cayley 16x16x16)
        # Se aloja en VRAM estáticamente para que todas las instancias compartan el mismo mapa
        if not hasattr(STATensor, '_cayley_table'):
            STATensor._cayley_table = self._build_cayley_tensor().to(self.device)

    @classmethod
    def vector(cls, t, x, y, z, device=None):
        """
        Crea un Cuadrivector espacio-temporal: V = t*g0 + x*g1 + y*g2 + z*g3.
        Soporta escalares sueltos o tensores 1D para procesamiento por lotes (batches).
        """
        # Convertir a tensores si son escalares para asegurar compatibilidad
        t = torch.as_tensor(t, dtype=torch.float32)
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        z = torch.as_tensor(z, dtype=torch.float32)
        
        # Determinar el tamaño del lote (batch size)
        batch_size = max(t.numel(), x.numel(), y.numel(), z.numel())
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        
        data[:, 1] = t
        data[:, 2] = x
        data[:, 3] = y
        data[:, 4] = z
        return cls(data, device=device)

    @classmethod
    def bivector_pauli(cls, bx, by, bz, device=None):
        """
        Crea un Bivector Espacial Puro (Álgebra de Pauli 3D / Campo Magnético).
        Representa los planos g1^g2, g2^g3, g3^g1.
        """
        bx = torch.as_tensor(bx, dtype=torch.float32)
        by = torch.as_tensor(by, dtype=torch.float32)
        bz = torch.as_tensor(bz, dtype=torch.float32)
        
        batch_size = max(bx.numel(), by.numel(), bz.numel())
        data = torch.zeros((batch_size, 16), dtype=torch.float32, device=device)
        
        # Índices exactos de los bivectores espaciales en la base de 16D
        data[:, 8] = bx  # eej. gamma_1 * gamma_2
        data[:, 9] = by  # eej. gamma_2 * gamma_3
        data[:, 10] = bz # eej. gamma_3 * gamma_1
        return cls(data, device=device)

    @classmethod
    def _build_cayley_tensor(cls):
        """
        Construye automáticamente la Tabla de Cayley [16, 16, 16] para Cl(1,3)
        utilizando manipulación de bits (Grassmann-Cayley algebra mapping).
        """
        # Tensor 16x16x16 inicializado en ceros
        T = torch.zeros((16, 16, 16), dtype=torch.float32)
        
        # Firma métrica de Hestenes (+, -, -, -)
        # Bit 0 (gamma_0) -> +1
        # Bit 1 (gamma_1) -> -1
        # Bit 2 (gamma_2) -> -1
        # Bit 3 (gamma_3) -> -1
        metric = [1, -1, -1, -1]
        
        # Iterar sobre las 16 bases de i y las 16 bases de j
        for i in range(16):
            for j in range(16):
                # 1. El índice resultante es el XOR de los bits base
                k = i ^ j
                
                sign = 1
                
                # 2. Métrica (Aniquilación de cuadrados)
                # Encontramos los vectores que chocan consigo mismos (overlap)
                overlap = i & j
                for bit in range(4):
                    if overlap & (1 << bit):
                        sign *= metric[bit]
                        
                # 3. Anti-conmutatividad (Permutaciones espaciales)
                # Contamos cuántos saltos debe dar cada vector en 'j' para pasar a 
                # través de los vectores de 'i' hasta su posición lexicográfica correcta.
                swaps = 0
                for bit_j in range(4):
                    if j & (1 << bit_j):
                        # Revisar bits en 'i' que estén a la izquierda (mayor peso)
                        for bit_i in range(bit_j + 1, 4):
                            if i & (1 << bit_i):
                                swaps += 1
                                
                if swaps % 2 != 0:
                    sign *= -1
                    
                # 4. Inyectar el escalar resultante en la coordenada del hipercubo
                T[i, j, k] = float(sign)
                
        return T

    def __mul__(self, other):
        """
        Sobrescribe el operador (*) para ejecutar el Producto Geométrico.
        Esta es la línea más importante de toda la librería.
        """
        if isinstance(other, STATensor):
            # Contracción tensorial masiva en GPU: 
            # 'bi' (Batch 1, 16D) * 'bj' (Batch 2, 16D) mapeado por 'ijk' (Reglas Cl1,3) -> 'bk' (Resultado)
            res_data = torch.einsum('bi, bj, ijk -> bk', self.data, other.data, self._cayley_table)
            return STATensor(res_data, device=self.device)
            
        elif isinstance(other, (int, float)): 
            # Producto escalar estándar
            return STATensor(self.data * other, device=self.device)
            
        raise TypeError(f"El producto geométrico no soporta el tipo {type(other)}")
        
    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, STATensor):
            return STATensor(self.data + other.data, device=self.device)
        return STATensor(self.data + other, device=self.device)

    def __repr__(self):
        return f"STATensor(batch_size={self.data.shape[0]}, device={self.device})"
