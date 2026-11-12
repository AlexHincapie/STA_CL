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
pip install git+https://github.com/AlexHincapie/STA_CL.git
```

---

## Usage Example

```python
import torch
from sta_cl.sta_tensor import STATensor
from sta_cl.sta_integrators import GeometricEulerCromer
from sta_cl.sta_bridge import STABridge

# 1. Initialize 100,000 particles on GPU
N = 100_000
pos = STATensor.vector(t=torch.zeros(N), x=torch.randn(N), y=torch.randn(N), z=torch.zeros(N))
vel = STATensor.vector(t=torch.ones(N), x=torch.ones(N), y=torch.zeros(N), z=torch.zeros(N))

# 2. Define Faraday bivector field F = E + I*B
F_field = STATensor.bivector(ex=0.0, ey=0.0, ez=0.0, bx=2.0, by=0.0, bz=0.0)

# 3. Integrate trajectory with Lie-group exponential rotors
integrator = GeometricEulerCromer(dt=0.01)
pos, vel = integrator.step_lorentz_rotor(pos, vel, F_field, q_m_ratio=-1.0)

# 4. Project spinor state onto the Grassmannian Hopf fiber
rotor_state = (F_field * 0.5).exp()
projected_wave = rotor_state.project_grassmannian(mode='hopf_fiber', reference_idx=1)

# 5. Export coordinates to binary file
STABridge.serialize_to_file(pos, "frame_plasma_0000.bin")
```

---

## Repository Structure

* `sta_cl/sta_tensor.py`: 16D $\mathcal{Cl}(1,3)$ tensor class, Cayley table generator, involutions, `dot`, `wedge`, `inv`, `exp`, `nabla`, and `project_grassmannian`.
* `sta_cl/sta_integrators.py`: `GeometricEulerCromer` class containing `step_lorentz`, `step_lorentz_rotor`, and `step_geodesic`.
* `sta_cl/sta_bridge.py`: `STABridge` binary serializer for `.bin` coordinate streams.
* `paper.md` & `paper.bib`: Manuscript and BibTeX database documenting the software architecture.

---

## Theoretical Foundations and Software Genealogy

The methods in `STA_CL` build upon theoretical formulations of Spacetime Algebra and two decades of geometric algebra software packages:

### Spacetime Algebra and Geometric Calculus Foundations
1. **Hestenes, D.** (1966). *Space-Time Algebra*. Gordon and Breach (Reissued by Birkhäuser, 2015). DOI: [10.1007/978-3-319-18413-5](https://doi.org/10.1007/978-3-319-18413-5) — Formulation of $\mathcal{Cl}(1,3)$, the Faraday bivector $F = \mathbf{E} + I\mathbf{B}$, and $\nabla F = J$.
2. **Hestenes, D.** (1967). Real Spinor Fields. *Journal of Mathematical Physics*, 8(4), 798–808. DOI: [10.1063/1.1705279](https://doi.org/10.1063/1.1705279) — Formulation of Dirac spinors in the even subalgebra $\mathcal{Cl}^+(1,3)$ and bilinear observables (`spinor_manifold` and `hopf_fiber` modes).
3. **Hestenes, D., & Sobczyk, G.** (1984). *Clifford Algebra to Geometric Calculus: A Unified Language for Mathematics and Physics*. D. Reidel Publishing / Springer. DOI: [10.1007/978-94-009-6292-7](https://doi.org/10.1007/978-94-009-6292-7) — Definition of the vector derivative operator $\nabla$ (`STATensor.nabla`) and subspace projections.
4. **Lasenby, A., Doran, C., & Gull, S.** (1998). Gravity, gauge theories and geometric algebra. *Philosophical Transactions of the Royal Society of London. Series A*, 356(1737), 487–582. DOI: [10.1098/rsta.1998.0178](https://doi.org/10.1098/rsta.1998.0178) — Rotor connection equations implemented in `step_geodesic`.
5. **Hestenes, D.** (2003). Spacetime Physics with Geometric Algebra. *American Journal of Physics*, 71(7), 691–714. DOI: [10.1119/1.1571836](https://doi.org/10.1119/1.1571836) — Relativistic kinematics via rotor sandwich transformations $R v \tilde{R}$.
6. **Doran, C., & Lasenby, A.** (2003). *Geometric Algebra for Physicists*. Cambridge University Press. DOI: [10.1017/CBO9780511807497](https://doi.org/10.1017/CBO9780511807497) — Computational rules for bivector exponentials (`exp`) and Lie-group rotor integration (`step_lorentz_rotor`).

### Computational Geometric Algebra Precursors
7. **Leopardi, P. C.** (2005). A generalized FFT for Clifford algebras (**GluCat**). *Bulletin of the Belgian Mathematical Society - Simon Stevin*, 11(5), 663–688. DOI: [10.36045/bbms/1110205626](https://doi.org/10.36045/bbms/1110205626) — Integer bitset indexing of basis blades implemented in `_build_cayley_tensor`.
8. **Fontijne, D., Bouma, T., & Dorst, L.** (2007). **Gaigen 2**: A Geometric Algebra Implementation Generator. *Generative Programming and Component Engineering*, ACM, 141–150. DOI: [10.1145/1289971.1289996](https://doi.org/10.1145/1289996) — Grade-specific constructors and runtime active coordinate tracking (`active_mask`).
9. **Perwass, C.** (2009). *Geometric Algebra with Applications in Engineering* (**CLU / CLUCalc**). Springer. DOI: [10.1007/978-3-540-89068-3](https://doi.org/10.1007/978-3-540-89068-3) — Coupling between projective manifolds (`project_grassmannian`) and graphics vertex buffers (`STABridge`).
10. **Hildenbrand, D.** (2013). *Foundations of Geometric Algebra Computing* (**Gaalop**). Springer. DOI: [10.1007/978-3-642-31794-1](https://doi.org/10.1007/978-3-642-31794-1) — Preallocation of sign masks in GPU memory (`_ensure_cache`).
11. **Colapinto, P.** (2015). *Articulating Space: Geometric Algebra for Parametric Design* (**Versor / vsr**). PhD Thesis, University of California, Santa Barbara — Bit-population counting (`bin(i).count('1')`) for static grade mask generation.
12. **Bromborsky, A., & The GAlgebra Developers.** (2020). **GAlgebra**: Symbolic Geometric Algebra/Calculus package for SymPy. Zenodo. DOI: [10.5281/zenodo.3828397](https://doi.org/10.5281/zenodo.3828397) — Formulation of the reciprocal frame vector derivative adapted to `torch.autograd` in `STATensor.nabla`.
13. **Hadfield, H., Kern, R., Arsenovic, A., Wieser, E., & The PyGAE Team.** (2020). **clifford**: Numerical Geometric Algebra for Python. Zenodo. DOI: [10.5281/zenodo.1453978](https://doi.org/10.5281/zenodo.1453978) — 16-component tabular layout for $\mathcal{Cl}(1,3)$ multivector arrays in Python.
14. **Reed, M.** (2020). Differential geometric algebra with Leibniz and Grassmann (**Grassmann.jl**). *Proceedings of the JuliaCon Conferences*. Repository: [github.com/chakravala/Grassmann.jl](https://github.com/chakravala/Grassmann.jl) — Graded multivector contractions and Plücker blade decomposition.
15. **Shirokov, D. S.** (2021). On computing the determinant, other characteristic polynomial coefficients, and inverse in Clifford algebras of arbitrary dimension. *Computational and Applied Mathematics*, 40(173). DOI: [10.1007/s40314-021-01536-0](https://doi.org/10.1007/s40314-021-01536-0) — Algebraic multivector inversion via adjugate grades implemented in `STATensor.inv`.
16. **Roelfs, M., & De Keninck, S.** (2023). Graded Symmetry Groups: Plane and Simple (**kingdon**). *Advances in Applied Clifford Algebras*, 33(30). DOI: [10.1007/s00006-023-01269-9](https://doi.org/10.1007/s00006-023-01269-9) — PyTorch tensor integration and bivector-vector grade-1 contractions in `STATensor.dot`.
17. **Kahlow, R.** (2023). **TFGA and TorchGA**: Geometric Algebra in TensorFlow and PyTorch. GitHub Repository: [github.com/RobinKahlow/tfga](https://github.com/RobinKahlow/tfga) — Batched Einstein summation (`torch.einsum('bi, bj, ijk -> bk', ...)`) over 3D Cayley structure tensors on GPUs.

---

## Author and License

* **Author:** Dane Alexander Hincapié Arango
* **Affiliations:**
  - Grupo de Investigación Innovación en Matemáticas y Nuevas Tecnologías para la Educación (GNOMON), Instituto Tecnológico Metropolitano (ITM), Medellín, Colombia.
  - Museo de Ciencias Naturales de La Salle, Instituto Tecnológico Metropolitano (ITM), Medellín, Colombia.
* **License:** Released under the [MIT License](LICENSE).
