---
title: 'STA_CL: A PyTorch Library for Spacetime Algebra Cl(1,3) and Grassmannian Wave Projections on GPUs'
tags:
  - Python
  - PyTorch
  - Spacetime Algebra
  - Clifford Algebra
  - Relativistic Physics
  - Grassmannian
authors:
  - name: Dane Alexander Hincapié Arango
    affiliation: "1, 2"
affiliations:
 - name: Grupo de Investigación Innovación en Matemáticas y Nuevas Tecnologías para la Educación (GNOMON), Instituto Tecnológico Metropolitano (ITM), Medellín, Colombia
   index: 1
 - name: Museo de Ciencias Naturales de La Salle, Instituto Tecnológico Metropolitano (ITM), Medellín, Colombia
   index: 2
date: 12 November 2026
bibliography: paper.bib
---

# Summary

`STA_CL` is a Python library built on PyTorch that implements Spacetime Algebra $\mathcal{Cl}(1,3)$ with metric signature $(+, -, -, -)$. The package executes geometric algebra products across batched tensors on GPUs, integrates relativistic equations of motion via Lie-group rotors, evaluates the vector derivative operator $\nabla$ through automatic differentiation, projects spinor fields onto Grassmannian manifolds, and exports particle coordinates to binary files for three-dimensional rendering.

# Statement of Need and Computational Genealogy

Spacetime Algebra was formulated by David Hestenes in 1966 to express electromagnetic fields, Dirac spinors, and relativistic kinematics within the real Clifford algebra $\mathcal{Cl}(1,3)$ [@Hestenes1966; @Hestenes1967]. Hestenes and Sobczyk later developed the differential calculus of multivector fields through the vector derivative operator $\nabla$ [@Hestenes1984]. Lasenby, Doran, and Gull applied this algebraic structure to gauge theories of gravity [@Lasenby1998], and Hestenes synthesized rotor dynamics in Minkowski space [@Hestenes2003].

In 2003, Chris Doran and Anthony Lasenby systematized the computational rules for multivector products, bivector exponentials, and Lorentz transformations [@Doran2003]. In 2005, Paul C. Leopardi introduced the architecture of `GluCat`, establishing a correspondence between basis blades of $\mathcal{Cl}(p,q)$ and integer bitmasks where each bit encodes the presence of a generating vector [@Leopardi2005].

The architecture of `STA_CL` synthesizes specific mechanisms developed across two decades of geometric algebra software into three modules (`sta_tensor.py`, `sta_integrators.py`, and `sta_bridge.py`):

1. **Bitwise Cayley Tensor Construction (`_build_cayley_tensor`):** Building upon Leopardi's bitset representation in `GluCat` [@Leopardi2005] and Pablo Colapinto's bit-counting grade indexing in `Versor` [@Colapinto2015], `sta_tensor.py` generates the $16 \times 16 \times 16$ structure tensor using bitwise `i ^ j` operations to determine target indices in $\{0,\dots,15\}$, `i & j` overlaps to apply the metric signature $(+, -, -, -)$, and bit-swap counts to assign anticommutation signs.
2. **Device-Cached Masks (`_ensure_cache`) and Subspace Constructors:** Following the grade specialization introduced by Daniel Fontijne, Tim Bouma, and Leo Dorst in `Gaigen 2` [@Fontijne2007] and the precomputation of sign arrays for GPU registers developed by Dietmar Hildenbrand in `Gaalop` [@Hildenbrand2013], `sta_tensor.py` allocates one-dimensional sign and projection masks in GPU memory for reversion (`__invert__`), grade involution (`grade_involution`), Clifford conjugation (`clifford_conjugate`), and grade filtering (`grade_project`, `even_part`, `odd_part`). Subspace constructors (`escalar`, `vector`, `bivector`, `trivector`, `pseudoescalar`) and the `active_mask` property track populated indices across batches.
3. **Batched Tensor Contraction (`__mul__`):** Drawing from the 16-component array layout in Python implemented by Hugo Hadfield, Robert Kern, and Alex Arsenovic in `clifford` [@Hadfield2020], and from the Einstein summation over Cayley tensors in PyTorch and TensorFlow formulated by Robin Kahlow in `TFGA` and `TorchGA` [@Kahlow2023], `sta_tensor.py` computes the geometric product across batches via `torch.einsum('bi, bj, ijk -> bk', self.data, other.data, self.cache['cayley'])` in `float32` and `float64`.
4. **Unified Inner Contraction (`dot`), Inversion (`inv`), and Exponentiation (`exp`):** Informed by the graded operators of Michael Reed in `Grassmann.jl` [@Reed2020] and Martin Roelfs and Steven De Keninck in `kingdon` [@Roelfs2023], the `dot` method evaluates the grade-0 anticommutator for vector pairs and the grade-1 commutator $\frac{1}{2}(Fv - vF)$ for bivector-vector Lorentz contractions. The `inv` method implements the Clifford algebra inversion formula studied by Hestenes and Dmitry Shirokov [@Hestenes1967; @Shirokov2021] using the scalar and pseudoscalar components of $\psi\tilde{\psi}$. The `exp` method combines a scaling-and-squaring Taylor expansion with a norm projection onto the group $\text{Spin}^+(1,3)$ [@Doran2003].
5. **Autograd Vector Derivative (`nabla`):** Adapting the symbolic formulation of Alan Bromborsky in `galgebra` [@Bromborsky2020] based on the operator of Hestenes and Sobczyk [@Hestenes1984], `STATensor.nabla` couples the reciprocal basis vectors $\{\gamma^0, -\gamma^1, -\gamma^2, -\gamma^3\}$ with `torch.autograd.grad` to compute four-dimensional divergences and curls on GPU tensors.
6. **Lie-Group Rotor Integrators (`sta_integrators.py`):** The `GeometricEulerCromer` class provides the finite-difference `step_lorentz` method alongside `step_lorentz_rotor`, which evolves four-velocities via the sandwich transformation $v(t+\Delta t) = R v(t) \tilde{R}$ with $R = \exp(\frac{q}{2m}F\Delta t)$ formulated by Hestenes, Doran, and Lasenby [@Hestenes2003; @Doran2003], paired with midpoint position updates.
7. **Grassmannian Projection (`project_grassmannian`) and Binary Serialization (`sta_bridge.py`):** Following the integration between projective manifolds and rendering buffers developed by Christian Perwass in `CLUCalc` [@Perwass2009], `project_grassmannian` maps multivector wavefields onto the even spinor subalgebra (`spinor_manifold`), the positive hemisphere of the bilinear Hopf fibration $\psi K \tilde{\psi}$ (`hopf_fiber`), and normalized subspace blades on the light cone (`plucker_blade`). The `STABridge.serialize_to_file` method writes spatial coordinates into binary `.bin` streams for direct ingestion into WebGL and Three.js `Float32Array` buffers.

# References
