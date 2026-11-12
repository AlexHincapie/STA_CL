import torch

class STATensor:
    """
    Hestenes Spacetime Algebra Cl(1,3) Engine - v1.2.
    Metric Signature: (+, -, -, -).
    Grassmann Binary Basis (16D):
      Grade 0 (Scalar):       [0]
      Grade 1 (Vectors):      [1:g0, 2:g1, 4:g2, 8:g3]
      Grade 2 (Bivectors):    [3:g01, 5:g02, 6:g12, 9:g03, 10:g13, 12:g23]
      Grade 3 (Trivectors):   [7:g012, 11:g013, 13:g023, 14:g123]
      Grade 4 (Pseudoscalar): [15:I=g0123]
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
            raise ValueError(f"Dimension mismatch. 16 components are required in Cl(1,3). Received: {data.shape}")

        self.data = data
        self._ensure_cache(self.device, self.dtype)

    @classmethod
    def _ensure_cache(cls, device, dtype):
        """Allocates and stores the Cayley tensor and sign masks in device memory per device and data type."""
        key = (device, dtype)
        if key not in cls._cache:
            cayley = cls._build_cayley_tensor().to(device=device, dtype=dtype)
            # Grade of each of the 16 binary indices via bit population count
            grades = [bin(i).count('1') for i in range(16)]
            
            # Reversion (~A): changes sign on grades 2 and 3 -> (-1)^(k*(k-1)/2)
            rev_mask = torch.tensor([(-1)**(k*(k-1)//2) for k in grades], dtype=dtype, device=device)
            # Grade involution (^A): changes sign on grades 1 and 3 -> (-1)^k
            inv_mask = torch.tensor([(-1)**k for k in grades], dtype=dtype, device=device)
            # Clifford conjugation: product of reversion and grade involution -> (-1)^(k*(k+1)/2)
            conj_mask = rev_mask * inv_mask

            # Grade projection masks for grades 0 to 4
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
        """Generates the 16x16x16 structure tensor via bitwise XOR and AND operations."""
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
        """Evaluates populated components across the batch."""
        return (torch.abs(self.data).max(dim=0)[0] > 1e-7).to(self.dtype)

    # ==========================================
    # SUBSPACE CONSTRUCTORS
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
        """Minkowski four-vector: V = t*g0 + x*g1 + y*g2 + z*g3 (Indices: 1, 2, 4, 8)."""
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
        Faraday bivector F = E + I*B.
        Electric components (g01=3, g02=5, g03=9). Magnetic planes (g12=6, g31=10, g23=12).
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
        """Axial four-vectors and spin trivectors (Indices: 7, 11, 13, 14)."""
        dev = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tx, ty = torch.as_tensor(tx, dtype=dtype, device=dev), torch.as_tensor(ty, dtype=dtype, device=dev)
        tz, t_xyz = torch.as_tensor(tz, dtype=dtype, device=dev), torch.as_tensor(t_xyz, dtype=dtype, device=dev)
        batch_size = max(tx.numel(), ty.numel(), tz.numel(), t_xyz.numel())
        data = torch.zeros((batch_size, 16), dtype=dtype, device=dev)
        data[:, 7], data[:, 11], data[:, 13], data[:, 14] = tx, ty, tz, t_xyz
        return cls(data, device=dev, dtype=dtype)

    @classmethod
    def pseudoescalar(cls, val=1.0, batch_size=1, device=None, dtype=torch.float32):
        """Four-dimensional volume element: I = g0*g1*g2*g3 (Index 15, I^2 = -1)."""
        dev = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v = torch.as_tensor(val, dtype=dtype, device=dev)
        b_size = max(batch_size, v.numel())
        data = torch.zeros((b_size, 16), dtype=dtype, device=dev)
        data[:, 15] = v.view(-1) if v.numel() > 1 else v
        return cls(data, device=dev, dtype=dtype)

    # ==========================================
    # ALGEBRAIC OPERATORS
    # ==========================================

    def __mul__(self, other):
        if isinstance(other, STATensor):
            res = torch.einsum('bi, bj, ijk -> bk', self.data, other.data, self.cache['cayley'])
            return STATensor(res, device=self.device, dtype=self.dtype)
        elif isinstance(other, (int, float, torch.Tensor)):
            return STATensor(self.data * other, device=self.device, dtype=self.dtype)
        raise TypeError("Operand type is not supported for geometric multiplication.")

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return STATensor(other * self.data, device=self.device, dtype=self.dtype)
        raise TypeError("Operand type is not supported for left multiplication.")

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return STATensor(self.data / other, device=self.device, dtype=self.dtype)
        elif isinstance(other, STATensor):
            return self * other.inv()
        raise TypeError("Operand type is not supported for division.")

    def __add__(self, other):
        return STATensor(self.data + (other.data if isinstance(other, STATensor) else other), device=self.device, dtype=self.dtype)

    def __sub__(self, other):
        return STATensor(self.data - (other.data if isinstance(other, STATensor) else other), device=self.device, dtype=self.dtype)

    def __neg__(self):
        return STATensor(-self.data, device=self.device, dtype=self.dtype)

    def __invert__(self):
        """Reversion (~A): Reverses the order of vector factors (changes sign on grades 2 and 3)."""
        return STATensor(self.data * self.cache['rev'], device=self.device, dtype=self.dtype)

    def grade_involution(self):
        """Grade involution (^A): Changes sign on grades 1 and 3."""
        return STATensor(self.data * self.cache['inv'], device=self.device, dtype=self.dtype)

    def clifford_conjugate(self):
        """Clifford conjugation: Product of reversion and grade involution."""
        return STATensor(self.data * self.cache['conj'], device=self.device, dtype=self.dtype)

    # ==========================================
    # PROJECTIONS AND CONTRACTIONS
    # ==========================================

    def grade_project(self, grade):
        """Extracts grade k in {0, 1, 2, 3, 4}."""
        return STATensor(self.data * self.cache['grades'][grade], device=self.device, dtype=self.dtype)

    def even_part(self):
        """Projects onto the even subalgebra Cl+(1,3) (grades 0, 2, 4)."""
        return STATensor(self.data * self.cache['even'], device=self.device, dtype=self.dtype)

    def odd_part(self):
        """Projects onto the odd subspace (grades 1, 3)."""
        return STATensor(self.data * self.cache['odd'], device=self.device, dtype=self.dtype)

    def dot(self, other):
        """
        Hestenes inner contraction:
        - For a bivector (F) and a vector (v): computes 0.5*(F*v - v*F) projected onto grade 1.
        - For two vectors: computes 0.5*(a*b + b*a) projected onto grade 0.
        """
        ab = self * other
        ba = other * self
        # Evaluate grade-1 commutator and grade-0 anticommutator
        commutator_g1 = ((ab - ba) * 0.5).grade_project(1)
        anticommutator_g0 = ((ab + ba) * 0.5).grade_project(0)
        
        # Select grade-1 output when present; otherwise select grade-0 output
        has_g1 = (torch.abs(commutator_g1.data).sum(dim=-1, keepdim=True) > 1e-12)
        res_data = torch.where(has_g1, commutator_g1.data, anticommutator_g0.data)
        return STATensor(res_data, device=self.device, dtype=self.dtype)

    def wedge(self, other):
        """Exterior product: 0.5 * (A*B - B*A)."""
        return (self * other - other * self) * 0.5

    def commutator(self, other):
        """Commutator product: 0.5 * (A*B - B*A)."""
        return (self * other - other * self) * 0.5

    def inv(self, eps=1e-8):
        """
        Hestenes-Shirokov multivector inverse.
        For psi * ~psi = S + P*I: psi^-1 = ~psi * (S - P*I) / (S^2 + P^2).
        """
        rev = ~self
        prod = self * rev
        S = prod.data[:, 0:1]
        P = prod.data[:, 15:16]
        denom = S**2 + P**2 + eps
        
        # Construct (S - P*I) / (S^2 + P^2)
        conj_sp = torch.zeros_like(self.data)
        conj_sp[:, 0:1] = S / denom
        conj_sp[:, 15:16] = -P / denom
        return rev * STATensor(conj_sp, device=self.device, dtype=self.dtype)

    # ==========================================
    # EXPONENTIATION, VECTOR DERIVATIVE, AND GRASSMANNIAN PROJECTION
    # ==========================================

    def exp(self, order=8, squarings=4, normalize_rotor=True):
        """
        Tensor exponentiation via scaling and squaring:
        1. Divides input by 2^squarings.
        2. Evaluates Taylor series of degree 'order'.
        3. Squares the result 'squarings' times.
        4. When normalize_rotor=True, normalizes the output by its spinor norm.
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
            norm_sq = (res * (~res)).data[:, 0:1]
            valid = torch.abs(norm_sq) > 1e-12
            safe_norm = torch.clamp(torch.abs(norm_sq), min=1e-30)
            inv_norm = torch.where(valid, torch.rsqrt(safe_norm), torch.ones_like(norm_sq))
            res = STATensor(res.data * inv_norm, device=self.device, dtype=self.dtype)

        return res

    @staticmethod
    def nabla(F, coords):
        """
        Dirac-Hestenes vector derivative operator: nabla F = g^mu d_mu F.
        Computes derivatives with respect to 'coords' [Batch, 4] via torch.autograd.grad.
        """
        assert coords.requires_grad, "The 'coords' tensor [Batch, 4] must have requires_grad=True."
        jacobiano = torch.zeros((*F.data.shape, 4), device=F.device, dtype=F.dtype)
        
        for k in range(16):
            if F.data[:, k].requires_grad:
                grad_k = torch.autograd.grad(
                    outputs=F.data[:, k].sum(), inputs=coords,
                    create_graph=True, retain_graph=True, allow_unused=True
                )[0]
                if grad_k is not None:
                    jacobiano[:, k, :] = grad_k

        # Reciprocal basis vectors: g^0 = g_0, g^i = -g_i
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
        Grassmannian and spinor manifold projector in Cl(1,3).
        Modes: 'hopf_fiber', 'spinor_manifold', and 'plucker_blade'.
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
            raise ValueError(f"Mode '{mode}' is not recognized.")

    def __repr__(self):
        return f"STATensor(batch_size={self.data.shape[0]}, dtype={self.dtype}, device={self.device})"
