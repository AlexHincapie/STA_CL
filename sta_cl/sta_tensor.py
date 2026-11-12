import torch

class STATensor:
    """
    Motor de Álgebra de Espacio-Tiempo de Hestenes Cl(1,3) - v1.2.
    Firma Métrica: (+, -, -, -).
    Base Binaria de Grassmann (16D):
      Grado 0 (Escalar):       [0]
      Grado 1 (Vectores):      [1:g0, 2:g1, 4:g2, 8:g3]
      Grado 2 (Bivectores):    [3:g01, 5:g02, 6:g12, 9:g03, 10:g13, 12:g23]
      Grado 3 (Trivectores):   [7:g012, 11:g013, 13:g023, 14:g123]
      Grado 4 (Pseudoescalar): [15:I=g0123]
    """
    _cache = {}

    def __init__(self, data, device=None, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            self.device = device if device is not None else data.device
            self.dtype = dtype if dtype is not None else data.dtype
            data = data.to(device=self.device, dtype=self.dtype)
        else:
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dtype = dtype
            data = torch.tensor(data, dtype=self.dtype, device=self.device)

        if data.dim() == 1:
            data = data.unsqueeze(0)

        if data.shape[-1] != 16:
            raise ValueError(f"Dimensión inválida. Se requieren 16 grados en Cl(1,3). Recibido: {data.shape}")

        self.data = data
        self._ensure_cache(self.device, self.dtype)

    @classmethod
    def _ensure_cache(cls, device, dtype):
        """Inicializa y almacena en caché VRAM la tabla de Cayley y las máscaras topológicas por dispositivo/precisión."""
        key = (device, dtype)
        if key not in cls._cache:
            cayley = cls._build_cayley_tensor().to(device=device, dtype=dtype)
            # Grados de cada uno de los 16 índices binarios (popcount)
            grades = [bin(i).count('1') for i in range(16)]
            
            # Reversión (~A): invierte grados 2 y 3 -> (-1)^(k*(k-1)/2)
            rev_mask = torch.tensor([(-1)**(k*(k-1)//2) for k in grades], dtype=dtype, device=device)
            # Involución de grado (^A): invierte grados impares 1 y 3 -> (-1)^k
            inv_mask = torch.tensor([(-1)**k for k in grades], dtype=dtype, device=device)
            # Conjugación de Clifford: combina ambas -> (-1)^(k*(k+1)/2)
            conj_mask = rev_mask * inv_mask

            # Máscaras precomputadas de proyección por grado (0 a 4)
            grade_masks = {
                g: torch.tensor([1.0 if grades[i] == g else 0.0 for i in range(16)], dtype=dtype, device=device)
                for g in range(5)
            }
            even_mask = grade_masks[0] + grade_masks[2] + grade_masks[4]
            odd_mask = grade_masks[1] + grade_masks[3]

            cls._cache[key] = {
                'cayley': cayley,
                'rev': rev_mask,
                'inv': inv_mask,
                'conj': conj_mask,
                'grades': grade_masks,
                'even': even_mask,
                'odd': odd_mask
            }

    @classmethod
    def _build_cayley_tensor(cls):
        """Generador exacto del tensor estructural 16x16x16 mediante isomorfismo binario XOR/AND."""
        T = torch.zeros((16, 16, 16), dtype=torch.float32)
        metric = [1, -1, -1, -1]
        for i in range(16):
            for j in range(16):
                k = i ^ j
                sign = 1
                overlap = i & j
                for bit in range(4):
                    if overlap & (1 << bit):
                        sign *= metric[bit]
                swaps = sum(
                    1 for bit_j in range(4) if j & (1 << bit_j)
                    for bit_i in range(bit_j + 1, 4) if i & (1 << bit_i)
                )
                if swaps % 2 != 0:
                    sign *= -1
                T[i, j, k] = float(sign)
        return T

    @property
    def cache(self):
        return self._cache[(self.device, self.dtype)]

    @property
    def active_mask(self):
        """Perfilado dinámico perezoso (Lazy Evaluation) para auditar qué grados están activos en el lote."""
        return (torch.abs(self.data).max(dim=0)[0] > 1e-7).to(self.dtype)

    # ==========================================
    # CONSTRUCTORES FÍSICOS (Factory Methods)
    # ==========================================

    @classmethod
    def escalar(cls, valor, batch_size=1, device=None, dtype=torch.float32):
        dev = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v = torch.as_tensor(valor, dtype=dtype, device=dev)
        b_size = max(batch_size, v.numel())
        data = torch.zeros((b_size, 16), dtype=dtype, device=dev)
        data[:, 0] = v.view(-1) if v.numel() > 1 else v
        return cls(data, device=dev, dtype=dtype)

    @classmethod
    def vector(cls, t, x, y, z, device=None, dtype=torch.float32):
        """Cuadrivector de Minkowski: V = t*g0 + x*g1 + y*g2 + z*g3 (Índices: 1, 2, 4, 8)."""
        dev = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t, x = torch.as_tensor(t, dtype=dtype, device=dev), torch.as_tensor(x, dtype=dtype, device=dev)
        y, z = torch.as_tensor(y, dtype=dtype, device=dev), torch.as_tensor(z, dtype=dtype, device=dev)
        batch_size = max(t.numel(), x.numel(), y.numel(), z.numel())
        data = torch.zeros((batch_size, 16), dtype=dtype, device=dev)
        data[:, 1], data[:, 2], data[:, 4], data[:, 8] = t, x, y, z
        return cls(data, device=dev, dtype=dtype)

    @classmethod
    def bivector(cls, ex, ey, ez, bx, by, bz, device=None, dtype=torch.float32):
        """
        Tensor Bivectorial Unificado de Faraday F = E + I*B.
        Boosts Eléctricos (g01=3, g02=5, g03=9). Planos Magnéticos de Pauli (g12=6, g31=10, g23=12).
        Nota: g31 = -g13, por lo que el índice 10 (g13) recibe -by o la convención ciclotrónica directa.
        """
        dev = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ex, ey, ez = torch.as_tensor(ex, dtype=dtype, device=dev), torch.as_tensor(ey, dtype=dtype, device=dev), torch.as_tensor(ez, dtype=dtype, device=dev)
        bx, by, bz = torch.as_tensor(bx, dtype=dtype, device=dev), torch.as_tensor(by, dtype=dtype, device=dev), torch.as_tensor(bz, dtype=dtype, device=dev)
        batch_size = max(ex.numel(), ey.numel(), ez.numel(), bx.numel(), by.numel(), bz.numel())
        data = torch.zeros((batch_size, 16), dtype=dtype, device=dev)
        data[:, 3], data[:, 5], data[:, 9] = ex, ey, ez
        data[:, 6], data[:, 10], data[:, 12] = bx, bz, by
        return cls(data, device=dev, dtype=dtype)

    @classmethod
    def trivector(cls, tx, ty, tz, t_xyz, device=None, dtype=torch.float32):
        """Pseudovectores de Espín / Duales de Corriente Axial (Índices: 7, 11, 13, 14)."""
        dev = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tx, ty = torch.as_tensor(tx, dtype=dtype, device=dev), torch.as_tensor(ty, dtype=dtype, device=dev)
        tz, t_xyz = torch.as_tensor(tz, dtype=dtype, device=dev), torch.as_tensor(t_xyz, dtype=dtype, device=dev)
        batch_size = max(tx.numel(), ty.numel(), tz.numel(), t_xyz.numel())
        data = torch.zeros((batch_size, 16), dtype=dtype, device=dev)
        data[:, 7], data[:, 11], data[:, 13], data[:, 14] = tx, ty, tz, t_xyz
        return cls(data, device=dev, dtype=dtype)

    @classmethod
    def pseudoescalar(cls, val=1.0, batch_size=1, device=None, dtype=torch.float32):
        """Volumen Fundamental de 4D: I = g0*g1*g2*g3 (Índice 15, I^2 = -1)."""
        dev = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v = torch.as_tensor(val, dtype=dtype, device=dev)
        b_size = max(batch_size, v.numel())
        data = torch.zeros((b_size, 16), dtype=dtype, device=dev)
        data[:, 15] = v.view(-1) if v.numel() > 1 else v
        return cls(data, device=dev, dtype=dtype)

    # ==========================================
    # ÁLGEBRA GEOMÉTRICA Y OPERADORES
    # ==========================================

    def __mul__(self, other):
        if isinstance(other, STATensor):
            res = torch.einsum('bi, bj, ijk -> bk', self.data, other.data, self.cache['cayley'])
            return STATensor(res, device=self.device, dtype=self.dtype)
        elif isinstance(other, (int, float, torch.Tensor)):
            return STATensor(self.data * other, device=self.device, dtype=self.dtype)
        raise TypeError("Tipo no soportado para el producto geométrico.")

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return STATensor(other * self.data, device=self.device, dtype=self.dtype)
        raise TypeError("Tipo no soportado para multiplicación izquierda.")

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return STATensor(self.data / other, device=self.device, dtype=self.dtype)
        elif isinstance(other, STATensor):
            return self * other.inv()
        raise TypeError("División no soportada.")

    def __add__(self, other):
        return STATensor(self.data + (other.data if isinstance(other, STATensor) else other), device=self.device, dtype=self.dtype)

    def __sub__(self, other):
        return STATensor(self.data - (other.data if isinstance(other, STATensor) else other), device=self.device, dtype=self.dtype)

    def __neg__(self):
        return STATensor(-self.data, device=self.device, dtype=self.dtype)

    def __invert__(self):
        """Reversión (~A): Invierte el orden de los factores vectoriales (signo negativo en grados 2 y 3)."""
        return STATensor(self.data * self.cache['rev'], device=self.device, dtype=self.dtype)

    def grade_involution(self):
        """Involución de Grado (^A): Cambia el signo de todos los grados impares (1 y 3)."""
        return STATensor(self.data * self.cache['inv'], device=self.device, dtype=self.dtype)

    def clifford_conjugate(self):
        """Conjugación de Clifford: Combinación de reversión e involución de grado."""
        return STATensor(self.data * self.cache['conj'], device=self.device, dtype=self.dtype)

    # ==========================================
    # PROYECCIONES Y CONTRACCIONES DE HESTENES
    # ==========================================

    def grade_project(self, grade):
        """Aísla un grado geométrico específico k en {0, 1, 2, 3, 4}."""
        return STATensor(self.data * self.cache['grades'][grade], device=self.device, dtype=self.dtype)

    def even_part(self):
        """Proyecta a la subálgebra par Cl+(1,3) (Espinores de Dirac-Hestenes: grados 0, 2, 4)."""
        return STATensor(self.data * self.cache['even'], device=self.device, dtype=self.dtype)

    def odd_part(self):
        """Proyecta al subespacio impar (Vectores y Trivectores: grados 1, 3)."""
        return STATensor(self.data * self.cache['odd'], device=self.device, dtype=self.dtype)

    def dot(self, other):
        """
        Contracción Interior Inteligente de Hestenes:
        - Si 'self' es Bivector (F) y 'other' es Vector (v) [o viceversa]: calcula 0.5*(F*v - v*F) proyectado a Grado 1 (Fuerza de Lorentz exacta).
        - Si ambos son Vectores: calcula 0.5*(a*b + b*a) proyectado a Grado 0 (Métrica de Minkowski).
        - Caso general: parte simétrica/antisimétrica de menor grado no nulo.
        """
        ab = self * other
        ba = other * self
        # Verificar si es una interacción Bivector-Vector (Lorentz)
        commutator_g1 = ((ab - ba) * 0.5).grade_project(1)
        anticommutator_g0 = ((ab + ba) * 0.5).grade_project(0)
        
        # Si hay resultado vectorial en grado 1 (contracción F . v), retornarlo; si no, retornar escalar grado 0
        has_g1 = (torch.abs(commutator_g1.data).sum(dim=-1, keepdim=True) > 1e-12)
        res_data = torch.where(has_g1, commutator_g1.data, anticommutator_g0.data)
        return STATensor(res_data, device=self.device, dtype=self.dtype)

    def wedge(self, other):
        """Producto Exterior de Grassmann (A ^ B) para vectores y formas antisimétricas."""
        return (self * other - other * self) * 0.5

    def commutator(self, other):
        """Producto Conmutador de Lie para Bivectores: A x B = 0.5 * (AB - BA)."""
        return (self * other - other * self) * 0.5

    def inv(self, eps=1e-8):
        """
        Inverso Geométrico de Hestenes-Shirokov (Exacto para vectores, cuchillas y espinores pares).
        Para espinores: psi * ~psi = S + P*I -> psi^-1 = ~psi * (S - P*I) / (S^2 + P^2).
        """
        rev = ~self
        prod = self * rev
        S = prod.data[:, 0:1]
        P = prod.data[:, 15:16]
        denom = S**2 + P**2 + eps
        
        # Construir el conjugado complejo-quiral (S - P*I)
        conj_sp = torch.zeros_like(self.data)
        conj_sp[:, 0:1] = S / denom
        conj_sp[:, 15:16] = -P / denom
        return rev * STATensor(conj_sp, device=self.device, dtype=self.dtype)

    # ==========================================
    # DINÁMICA, CÁLCULO TOPOLÓGICO Y GRASSMANNIANO
    # ==========================================

    def exp(self, order=8, squarings=4, normalize_rotor=True):
        """
        Exponenciación Tensorial con Scaling & Squaring + Proyección Simpléctica a Spin+(1,3):
        1. Reduce la norma dividiendo por 2^squarings: X_scaled = X / 16.
        2. Evalúa Serie de Taylor de orden 8 en zona de convergencia absoluta.
        3. Eleva al cuadrado 'squarings' veces mediante einsum.
        4. Si normalize_rotor=True, elimina la deriva de float32 proyectando R*~R = 1.
        """
        scale = float(2 ** squarings)
        x_scaled = self * (1.0 / scale)
        
        batch_size = self.data.shape[0]
        res = STATensor.escalar(1.0, batch_size=batch_size, device=self.device, dtype=self.dtype)
        term = STATensor.escalar(1.0, batch_size=batch_size, device=self.device, dtype=self.dtype)
        
        for i in range(1, order + 1):
            term = term * x_scaled * (1.0 / i)
            res = res + term
            
        for _ in range(squarings):
            res = res * res

        if normalize_rotor:
            # Proyección exacta al grupo Spin+(1,3) usando el inverso de la raíz espinorial
            norm_sq = (res * (~res)).data[:, 0:1]
            valid = norm_sq > 1e-7
            inv_norm = torch.where(valid, torch.rsqrt(torch.abs(norm_sq) + 1e-12), torch.ones_like(norm_sq))
            res = STATensor(res.data * inv_norm, device=self.device, dtype=self.dtype)

        return res

    @staticmethod
    def nabla(F, coords):
        """
        Operador Derivada Vectorial de Dirac-Hestenes: nabla F = g^mu d_mu F.
        Evalúa las 4 ecuaciones de Maxwell unificadas (nabla F = J) o la ec. de Dirac usando torch.autograd.
        """
        assert coords.requires_grad, "El tensor 'coords' [Batch, 4] debe tener requires_grad=True."
        jacobiano = torch.zeros((*F.data.shape, 4), device=F.device, dtype=F.dtype)
        
        for k in range(16):
            if F.data[:, k].requires_grad:
                grad_k = torch.autograd.grad(
                    outputs=F.data[:, k].sum(), inputs=coords,
                    create_graph=True, retain_graph=True, allow_unused=True
                )[0]
                if grad_k is not None:
                    jacobiano[:, k, :] = grad_k

        # Base recíproca contravariante: g^0 = g_0, g^i = -g_i
        g_0 = STATensor.vector(1, 0, 0, 0, device=F.device, dtype=F.dtype)
        g_1 = STATensor.vector(0, -1, 0, 0, device=F.device, dtype=F.dtype)
        g_2 = STATensor.vector(0, 0, -1, 0, device=F.device, dtype=F.dtype)
        g_3 = STATensor.vector(0, 0, 0, -1, device=F.device, dtype=F.dtype)

        dF_dt = STATensor(jacobiano[:, :, 0], device=F.device, dtype=F.dtype)
        dF_dx = STATensor(jacobiano[:, :, 1], device=F.device, dtype=F.dtype)
        dF_dy = STATensor(jacobiano[:, :, 2], device=F.device, dtype=F.dtype)
        dF_dz = STATensor(jacobiano[:, :, 3], device=F.device, dtype=F.dtype)

        return g_0 * dF_dt + g_1 * dF_dx + g_2 * dF_dy + g_3 * dF_dz

    def project_grassmannian(self, mode='hopf_fiber', target_grade=2, reference_idx=1):
        """
        Proyectador Topológico al Grassmanniano Orientado y Variedades Espinoriales en Cl(1,3).
        Soporta: 'hopf_fiber', 'spinor_manifold', y 'plucker_blade'.
        """
        eps = 1e-8
        rev = ~self

        if mode == 'spinor_manifold':
            psi_even = self.even_part()
            norm_multivector = psi_even * (~psi_even)
            scalar_part = norm_multivector.data[:, 0:1]
            pseudo_part = norm_multivector.data[:, 15:16]
            rho = torch.sqrt(scalar_part**2 + pseudo_part**2 + eps)
            return STATensor(psi_even.data / torch.sqrt(rho), device=self.device, dtype=self.dtype)

        elif mode == 'hopf_fiber':
            K_data = torch.zeros((1, 16), dtype=self.dtype, device=self.device)
            K_data[:, reference_idx] = 1.0
            K = STATensor(K_data, device=self.device, dtype=self.dtype)
            
            projected = self * K * rev
            norm_euclid = torch.norm(projected.data, p=2, dim=-1, keepdim=True) + eps
            unit_proj = projected.data / norm_euclid
            
            hemisphere_sign = torch.sign(unit_proj[:, reference_idx:reference_idx+1])
            hemisphere_sign = torch.where(hemisphere_sign == 0, torch.ones_like(hemisphere_sign), hemisphere_sign)
            return STATensor(unit_proj * hemisphere_sign, device=self.device, dtype=self.dtype)

        elif mode == 'plucker_blade':
            blade = self.grade_project(target_grade)
            norm_blade = torch.norm(blade.data, p=2, dim=-1, keepdim=True) + eps
            unit_blade = blade.data / norm_blade
            
            max_idx = torch.argmax(torch.abs(unit_blade), dim=-1, keepdim=True)
            dominant_val = torch.gather(unit_blade, 1, max_idx)
            sign_orient = torch.where(dominant_val >= 0, torch.ones_like(dominant_val), -torch.ones_like(dominant_val))
            return STATensor(unit_blade * sign_orient, device=self.device, dtype=self.dtype)

        else:
            raise ValueError(f"Modo Grassmanniano '{mode}' no reconocido.")

    def __repr__(self):
        return f"STATensor(batch_size={self.data.shape[0]}, dtype={self.dtype}, device={self.device})"
