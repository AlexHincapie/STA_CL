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
    def _build_cayley_tensor(cls):
        """
        Construye el hiper-cubo de coeficientes multiplicativos [16, 16, 16].
        C_{ijk} dicta cómo el grado 'i' multiplicado por el grado 'j' contribuye al grado 'k'.
        """
        # Inicializamos en ceros. (El llenado con la firma +, -, -, - lo haremos programáticamente)
        T = torch.zeros((16, 16, 16), dtype=torch.float32)
        
        # Regla trivial (placeholder): escalar * escalar = escalar
        T[0, 0, 0] = 1.0 
        
        return T

    # --- SOBRECARGA DE OPERADORES MÁGICOS ---

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
