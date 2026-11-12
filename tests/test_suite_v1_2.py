import torch
from sta_cl.sta_tensor import STATensor
from sta_cl.sta_integrators import GeometricEulerCromer


def test_grassmannian_projector():
    """
    Test 1: Validation of Grassmannian and spinor manifold projections in Cl(1,3).
    Evaluates 'hopf_fiber', 'spinor_manifold', and 'plucker_blade' modes.
    """
    print("TEST 1: GRASSMANNIAN PROJECTOR IN CL(1,3)\n" + "=" * 68)

    # 1A. Hopf fibration and antipodal identification (+psi and -psi)
    B_gen = STATensor.bivector(ex=0.5, ey=-0.2, ez=0.1, bx=1.2, by=-0.8, bz=0.4)
    psi_wave = B_gen.exp()
    psi_antipodal = psi_wave * (-1.0)

    proj_pos = psi_wave.project_grassmannian(mode='hopf_fiber', reference_idx=1)
    proj_inv = psi_antipodal.project_grassmannian(mode='hopf_fiber', reference_idx=1)

    diff_hopf = torch.max(torch.abs(proj_pos.data - proj_inv.data)).item()
    g0_comp = proj_pos.data[0, 1].item()
    print(f"1A. [Hopf Fiber] Projected g0 component: {g0_comp:.4f} (Target >= 0.0)")
    print(f"    Difference between (+psi) and (-psi) projections: {diff_hopf:.8f} (Target: 0.0)")
    assert g0_comp >= 0.0
    assert diff_hopf < 1e-6

    # 1B. Dirac-Hestenes spinor manifold (Yvon-Takabayashi invariant)
    raw_spinor = torch.zeros((1, 16))
    raw_spinor[0, 0] = 3.0   # Grade 0 (Scalar)
    raw_spinor[0, 15] = 4.0  # Grade 4 (Pseudoscalar I)
    raw_spinor[0, 1] = 99.0  # Grade 1 component to be filtered by even_part()

    psi_input = STATensor(raw_spinor)
    psi_projected = psi_input.project_grassmannian(mode='spinor_manifold')

    norm_check = psi_projected * (~psi_projected)
    rho_final = torch.sqrt(norm_check.data[0, 0]**2 + norm_check.data[0, 15]**2).item()
    grade_1_residual = psi_projected.data[0, 1].item()

    print(f"1B. [Spinor Manifold] Invariant rho: {rho_final:.6f} (Target: 1.0)")
    print(f"    Grade 1 component after projection: {grade_1_residual:.6f} (Target: 0.0)")
    assert abs(rho_final - 1.0) < 1e-5
    assert abs(grade_1_residual) < 1e-7

    # 1C. Plucker blade projection on null field (E^2 - B^2 = 0)
    null_wave = STATensor.bivector(ex=5.0, ey=0.0, ez=0.0, bx=0.0, by=5.0, bz=0.0)
    plucker_blade = null_wave.project_grassmannian(mode='plucker_blade', target_grade=2)

    has_nan = torch.isnan(plucker_blade.data).any().item()
    sphere_norm = torch.norm(plucker_blade.data).item()

    print(f"1C. [Plucker Blade] NaN presence: {has_nan} (Target: False)")
    print(f"    Norm on S^15+: {sphere_norm:.6f} (Target: 1.0)")
    print(f"    Projected components (Ex, By): ({plucker_blade.data[0, 3].item():.4f}, {plucker_blade.data[0, 12].item():.4f})\n")
    assert not has_nan
    assert abs(sphere_norm - 1.0) < 1e-5


def test_contraction_and_inverse():
    """
    Test 2: Validation of inner contraction (dot) and Hestenes-Shirokov inverse (inv).
    """
    print("TEST 2: INNER CONTRACTION (DOT) AND MULTIVECTOR INVERSE (INV)\n" + "=" * 68)

    # 2A. Vector-vector contraction (Grade 0 output)
    v_vec = STATensor.vector(t=5.0, x=3.0, y=0.0, z=0.0)
    metric_val = v_vec.dot(v_vec)
    g0_val = metric_val.data[0, 0].item()
    other_grades_sum = metric_val.data[0, 1:].abs().sum().item()

    print(f"2A. [Vector . Vector] Minkowski norm (Target: 16.0):")
    print(f"    Grade 0: {g0_val:.4f} | Sum of remaining indices: {other_grades_sum:.6f}")
    assert abs(g0_val - 16.0) < 1e-5
    assert other_grades_sum < 1e-6

    # 2B. Bivector-vector Lorentz contraction (Grade 1 output)
    vel_x = STATensor.vector(t=1.0, x=2.0, y=0.0, z=0.0)
    F_mag = STATensor.bivector(ex=0.0, ey=0.0, ez=0.0, bx=3.0, by=0.0, bz=0.0)

    lorentz_force = F_mag.dot(vel_x)
    ax = lorentz_force.data[0, 2].item()
    ay = lorentz_force.data[0, 4].item()
    az = lorentz_force.data[0, 8].item()

    print(f"2B. [Bivector . Vector] Contraction F.v:")
    print(f"    Components (ax, ay, az): ({ax:.4f}, {ay:.4f}, {az:.4f}) (Target ay: 6.0)")
    assert abs(ay - 6.0) < 1e-5

    # 2C. Hestenes-Shirokov inverse and division operator
    psi = (STATensor.escalar(2.0)
           + STATensor.bivector(0.5, -1.0, 0.0, 1.5, 0.0, -0.5)
           + STATensor.pseudoescalar(3.0))

    psi_inv = psi.inv()
    identity_prod = psi * psi_inv
    direct_div = psi / psi

    err_identity = torch.max(torch.abs(identity_prod.data - STATensor.escalar(1.0).data)).item()
    err_division = torch.max(torch.abs(direct_div.data - STATensor.escalar(1.0).data)).item()

    print(f"2C. [Multivector Inverse] psi * psi.inv():")
    print(f"    Scalar output (Target: 1.0): {identity_prod.data[0, 0].item():.6f}")
    print(f"    Maximum deviation across 16 components: {err_identity:.8f}")
    print(f"    Division operator (psi / psi) deviation: {err_division:.8f}\n")
    assert err_identity < 1e-6
    assert err_division < 1e-6


def test_exp_involutions_and_nabla():
    """
    Test 3: Validation of scaling-and-squaring exponentiation, involutions, and nabla operator.
    """
    print("TEST 3: EXPONENTIATION, INVOLUTIONS, AND NABLA OPERATOR\n" + "=" * 68)

    # 3A. Lorentz boost exponentiation (phi = 6.0) in float32 and float64
    phi = 6.0
    B_f32 = STATensor.bivector(ex=-phi / 2.0, ey=0.0, ez=0.0, bx=0.0, by=0.0, bz=0.0, dtype=torch.float32)
    R_f32 = B_f32.exp(normalize_rotor=True)
    unit_f32 = (R_f32 * (~R_f32)).data[0, 0].item()

    B_f64 = STATensor.bivector(ex=-phi / 2.0, ey=0.0, ez=0.0, bx=0.0, by=0.0, bz=0.0, dtype=torch.float64)
    R_f64 = B_f64.exp(normalize_rotor=True)
    u0_f64 = STATensor.vector(t=1.0, x=0.0, y=0.0, z=0.0, dtype=torch.float64)
    u_boost_f64 = R_f64 * u0_f64 * (~R_f64)
    inv_f64 = u_boost_f64.dot(u_boost_f64).data[0, 0].item()

    print(f"3A. [Scaling & Squaring] Boost (phi = {phi}):")
    print(f"    Rotor norm R*~R in float32 (Target: 1.0):   {unit_f32:.7f}")
    print(f"    Invariant u'.u' in float64 (Target: 1.0):   {inv_f64:.10f}")
    assert abs(unit_f32 - 1.0) < 1e-5
    assert abs(inv_f64 - 1.0) < 1e-9

    # 3B. Pseudoscalar square and grade involution signs
    I_pseudo = STATensor.pseudoescalar(1.0)
    I_sq = (I_pseudo * I_pseudo).data[0, 0].item()

    M_all = (STATensor.escalar(1.0)
             + STATensor.vector(1.0, 0, 0, 0)
             + STATensor.bivector(1.0, 0, 0, 0, 0, 0)
             + STATensor.trivector(1.0, 0, 0, 0)
             + STATensor.pseudoescalar(1.0))

    M_inv = M_all.grade_involution()
    inv_signs = [M_inv.data[0, idx].item() for idx in [0, 1, 3, 7, 15]]

    print(f"3B. [Clifford Involutions]:")
    print(f"    Pseudoscalar square I^2 (Target: -1.0): {I_sq:.4f}")
    print(f"    Signs by grade [0,1,2,3,4] after grade_involution: {inv_signs}")
    assert abs(I_sq + 1.0) < 1e-6
    assert inv_signs == [1.0, -1.0, 1.0, -1.0, 1.0]

    # 3C. Vector derivative operator (nabla) via autograd
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    coords = torch.tensor([[2.0, 3.0, -1.0, 5.0]], dtype=torch.float32, requires_grad=True, device=dev)
    X_field = STATensor.vector(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], device=dev)
    div_4d = STATensor.nabla(X_field, coords)

    div_scalar = div_4d.data[0, 0].item()
    curl_bivector = div_4d.grade_project(2).data.abs().sum().item()

    print(f"3C. [Vector Derivative Nabla]:")
    print(f"    4D divergence in Grade 0 (Target: 4.0): {div_scalar:.4f}")
    print(f"    4D curl in Grade 2 (Target: 0.0):       {curl_bivector:.6f}\n")
    assert abs(div_scalar - 4.0) < 1e-5
    assert curl_bivector < 1e-6


def test_rotor_integrator():
    """
    Test 4: Validation of Lie-group exponential rotor integrator (step_lorentz_rotor).
    """
    print("TEST 4: LIE-GROUP ROTOR INTEGRATOR (STEP_LORENTZ_ROTOR)\n" + "=" * 68)

    pos = STATensor.vector(t=0.0, x=0.0, y=0.0, z=0.0)
    vel = STATensor.vector(t=1.0, x=1.0, y=0.0, z=0.0)
    F_field = STATensor.bivector(ex=0.0, ey=0.0, ez=0.0, bx=2.0, by=0.0, bz=0.0)

    integrator = GeometricEulerCromer(dt=0.001)
    q_m = 1.0

    for _ in range(1000):
        pos, vel = integrator.step_lorentz_rotor(pos, vel, F_field, q_m)

    vx_num = vel.data[0, 2].item()
    vy_num = vel.data[0, 4].item()
    x_num = pos.data[0, 2].item()
    y_num = pos.data[0, 4].item()
    kinetic_norm = vx_num**2 + vy_num**2

    vx_analytical = torch.cos(torch.tensor(2.0)).item()
    vy_analytical = torch.sin(torch.tensor(2.0)).item()
    x_analytical = (torch.sin(torch.tensor(2.0)) / 2.0).item()
    y_analytical = ((1.0 - torch.cos(torch.tensor(2.0))) / 2.0).item()

    print(f"4A. [Trajectory after 1000 steps, dt=0.001]:")
    print(f"    Numerical velocity  (vx, vy): ({vx_num:.6f}, {vy_num:+.6f})")
    print(f"    Analytical velocity (vx, vy): ({vx_analytical:.6f}, {vy_analytical:+.6f})")
    print(f"    Numerical position  (x,  y ): ({x_num:.6f}, {y_num:.6f})")
    print(f"    Analytical position (x,  y ): ({x_analytical:.6f}, {y_analytical:.6f})")
    print(f"    Spatial velocity squared (vx^2 + vy^2) (Target: 1.0): {kinetic_norm:.6f}\n")

    assert abs(vx_num - vx_analytical) < 1e-3
    assert abs(vy_num - vy_analytical) < 1e-3
    assert abs(x_num - x_analytical) < 1e-3
    assert abs(y_num - y_analytical) < 1e-3
    assert abs(kinetic_norm - 1.0) < 1e-4


if __name__ == "__main__":
    test_grassmannian_projector()
    test_contraction_and_inverse()
    test_exp_involutions_and_nabla()
    test_rotor_integrator()
    print("ALL 4 VALIDATION SUITES PASSED.")
