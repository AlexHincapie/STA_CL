# STA_CL: Spacetime Algebra $\mathcal{Cl}(1,3)$ Library in PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c.svg)](https://pytorch.org/)

`STA_CL` is a Python library built on PyTorch that implements David Hestenes' Spacetime Algebra (STA) over the Clifford algebra $\mathcal{Cl}(1,3)$ with metric signature $(+, -, -, -)$. The package executes geometric algebra operations across batched tensors on GPUs (`float32` and `float64`) using a 16-dimensional binary Grassmann-Cayley structure tensor.

---

## Core Architecture and Methods

1. **Bitwise Cayley Structure Tensor (`_build_cayley_tensor`):** Encodes the 16 basis blades of $\mathcal{Cl}(1,3)$ ($\{1, \gamma_\mu, \gamma_\mu \wedge \gamma_\nu, I\gamma_\mu, I\}$) using bitwise `XOR` (`i ^ j`) for target index calculation, bitwise `AND` (`i & j`) for metric contraction, and bit-swap counts for anticommutation signs.
2. **Device-Cached Masks (`_ensure_cache`):** Precomputes and stores sign and grade projection arrays in GPU memory for reversion (`~A`), grade involution (`grade_involution`), Clifford conjugation (`clifford_conjugate`), and subspace filtering (`grade_project`, `even_part`, `odd_part`).
3. **Subspace Constructors and Contraction Operators:** Provides constructors (`escalar`, `vector`, `bivector`, `trivector`, `pseudoescalar`), the Hestenes-Shirokov multivector inverse (`inv`), a scaling-and-squaring exponential with $\text{Spin}^+(1,3)$ norm projection (`exp`), and an inner contraction (`dot`) that evaluates the grade-0 anticommutator for vector pairs and the grade-1 commutator $\frac{1}{2}(Fv - vF)$ for bivector-vector interactions.
4. **Grassmannian Projector (`project_grassmannian`):** Maps multivector wavefields onto projective and spinorial manifolds through three modes:
   - `hopf_fiber`: Bilinear observable mapping $\psi K \tilde{\psi}$ folded onto the positive hemisphere $S^n_+$.
   - `spinor_manifold`: Normalization of even subalgebra elements $\psi \in \mathcal{Cl}^+(1,3)$ using the invariant $\rho = \sqrt{\langle \psi\tilde{\psi}\rangle_0^2 + \langle \psi\tilde{\psi}\rangle_4^2}$.
   - `plucker_blade`: Subspace projection onto $S^{15}_+$ for general and null electromagnetic fields ($E^2 - B^2 = 0$).
5. **Lie-Group Rotor Integrator (`step_lorentz_rotor`):** Integrates the Lorentz equation $\dot{v} = \frac{q}{m}(F \cdot v)$ via sandwich transformations $v(t+\Delta t) = R v(t) \tilde{R}$ with $R = \exp(\frac{q}{2m}F\Delta t)$ and midpoint position updates.
6. **Autograd Vector Derivative (`nabla`):** Evaluates the Dirac-Hestenes derivative operator $\nabla = \gamma^\mu \partial_\mu$ using `torch.autograd.grad` and reciprocal basis vectors $\{\gamma^0, -\gamma^1, -\gamma^2, -\gamma^3\}$.
7. **Binary WebGL Serialization (`STABridge`):** Exports batched spatial coordinates into `.bin` (`Float32Array`) binary files for rendering in Three.js and WebGL.

---

## Installation

```bash
pip install git+[https://github.com/AlexHincapie/STA_CL.git](https://github.com/AlexHincapie/STA_CL.git)
