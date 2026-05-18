"""CUDA-accelerated dynamics for EDAP model v4.0. CPU clamping fixed."""

import numpy as np
from numba import njit, cuda
import math

@njit
def dynamics_cpu_single(
    T,
    K,
    C,
    alpha_T,
    beta_T,
    gamma_T,
    T_min,
    alpha_K_base,
    beta_K,
    K_max_coeff,
    K_half,
    T_automation,
    alpha_C,
    gamma_C,
    eta,
    K0,
    delta,
    T_singularity,
    epsilon_tech,
    K_shadow_threshold,
    lambda_shadow,
    mu_shadow,
    S,
    historical_T_peak,
    recovery_potential,
    in_reset_zone,
    T_reset_threshold,
    sigma_T,
    sigma_K,
    delta_T_reset,
    delta_K_reset,
    delta_C_reset,
    alpha_recovery_base,
    epsilon_recovery,
    rng_state,
):
    """CPU dynamics for a single state point. Used by solve_ivp."""
    # Clamp input
    T = min(max(T, 0.01), 0.99)
    K = min(max(K, 0.01), 0.99)
    C = min(max(C, 0.01), 0.99)

    Kc = K0 - delta * T + epsilon_tech * max(T - T_singularity, 0.0) ** 2
    if Kc < 0.01:
        Kc = 0.01

    if T < T_automation:
        Km = K_max_coeff * max(T, 0.0) / (max(T, 0.0) + K_half)
    else:
        tr = (T - T_automation) / (1.0 - T_automation)
        base = K_max_coeff * T / (T + K_half)
        Km = base + (1.0 - base) * tr
    if Km < 0.01:
        Km = 0.01

    T_eff = T - T_min
    if T_eff < 0.001:
        T_eff = 0.001

    dT = (
        alpha_T * T_eff * (1.0 - T) * C * (1.0 - K / Kc)
        - beta_T * (T - T_min)
        - gamma_T * K * T
        + eta * S * (1.0 - T / max(historical_T_peak, 0.01))
    )

    alpha_K_eff = alpha_K_base * (1.0 - C)

    lam = lambda_shadow * max(0.0, 1.0 - T / T_automation)
    shadow_drain = lam * max(K - K_shadow_threshold, 0.0) * K

    dK = (
        alpha_K_eff * K * (1.0 - T - C) * (1.0 - K / Km)
        - beta_K * (T * C) * K
        - shadow_drain
    )

    # Clamp dK to prevent K from going negative when K is near floor
    if K <= 0.015 and dK < 0.0:
        dK = 0.0

    mu = mu_shadow * max(0.0, 1.0 - T / T_automation)
    shadow_growth = mu * max(K - K_shadow_threshold, 0.0) * (1.0 - C)

    dC = (
        alpha_C * (T - K) * C * (1.0 - C)
        + eta * S * (1.0 - C)
        + shadow_growth
        - gamma_C * K * C
    )

    # Clamp dC to prevent C from exceeding 1.0 or going negative
    if C >= 0.985 and dC > 0.0:
        dC = 0.0
    if C <= 0.015 and dC < 0.0:
        dC = 0.0

    # Recovery
    d_rp = alpha_recovery_base * (1.0 - recovery_potential) * max(C - K, 0.0)
    recovery_potential += d_rp * 0.1
    if recovery_potential > 1.0:
        recovery_potential = 1.0
    if recovery_potential < 0.01:
        recovery_potential = 0.01

    # Cycle trigger
    sigmoid_T = 1.0 / (1.0 + math.exp(-(T_reset_threshold - T) / sigma_T))
    sigmoid_K = 1.0 / (1.0 + math.exp(-(K - 0.85 * Km) / sigma_K))
    p_reset = sigmoid_T * sigmoid_K

    rng_state = (rng_state * 1103515245 + 12345) % 2147483648
    rand_val = rng_state / 2147483648.0

    if (
        not in_reset_zone
        and p_reset > 0.5
        and rand_val < p_reset
        and recovery_potential > 0.05
    ):
        in_reset_zone = True

    if in_reset_zone:
        dT = delta_T_reset * recovery_potential
        dK = -delta_K_reset * recovery_potential
        dC = delta_C_reset * recovery_potential
        in_reset_zone = False

    return np.array([dT, dK, dC]), recovery_potential, in_reset_zone, rng_state


# ============================================================
# GPU VERSION — already has clamping via _clamp_gpu
# ============================================================


@cuda.jit(device=True)
def _clamp_gpu(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@cuda.jit
def simulate_on_gpu_kernel(
    T0_arr,
    K0_arr,
    C0_arr,
    T_out,
    K_out,
    C_out,
    alpha_T,
    beta_T,
    gamma_T,
    T_min,
    alpha_K_base,
    beta_K,
    K_max_coeff,
    K_half,
    T_automation,
    alpha_C,
    gamma_C,
    eta,
    K0,
    delta,
    T_singularity,
    epsilon_tech,
    K_shadow_threshold,
    lambda_shadow,
    mu_shadow,
    S_val,
    historical_T_peak,
    T_reset_threshold,
    sigma_T,
    sigma_K,
    delta_T_reset,
    delta_K_reset,
    delta_C_reset,
    alpha_recovery_base,
    epsilon_recovery,
    seed,
    n_simulations,
    n_steps,
    dt,
):
    sim_idx = cuda.grid(1)
    if sim_idx >= n_simulations:
        return

    T = T0_arr[sim_idx]
    K = K0_arr[sim_idx]
    C = C0_arr[sim_idx]
    rng_state = seed + sim_idx
    recovery_potential = 1.0

    for step in range(n_steps):
        T = _clamp_gpu(T, 0.01, 0.99)
        K = _clamp_gpu(K, 0.01, 0.99)
        C = _clamp_gpu(C, 0.01, 0.99)

        Kc = K0 - delta * T + epsilon_tech * max(T - T_singularity, 0.0) ** 2
        if Kc < 0.01:
            Kc = 0.01

        if T < T_automation:
            Km = K_max_coeff * max(T, 0.0) / (max(T, 0.0) + K_half)
        else:
            tr = (T - T_automation) / (1.0 - T_automation)
            base = K_max_coeff * T / (T + K_half)
            Km = base + (1.0 - base) * tr
        if Km < 0.01:
            Km = 0.01

        T_eff = T - T_min
        if T_eff < 0.001:
            T_eff = 0.001

        dT = (
            alpha_T * T_eff * (1.0 - T) * C * (1.0 - K / Kc)
            - beta_T * (T - T_min)
            - gamma_T * K * T
            + eta * S_val * (1.0 - T / max(historical_T_peak, 0.01))
        )

        alpha_K_eff = alpha_K_base * (1.0 - C)

        lam = lambda_shadow * max(0.0, 1.0 - T / T_automation)
        shadow_drain = lam * max(K - K_shadow_threshold, 0.0) * K

        dK = (
            alpha_K_eff * K * (1.0 - T - C) * (1.0 - K / Km)
            - beta_K * (T * C) * K
            - shadow_drain
        )

        if K <= 0.015 and dK < 0.0:
            dK = 0.0

        mu = mu_shadow * max(0.0, 1.0 - T / T_automation)
        shadow_growth = mu * max(K - K_shadow_threshold, 0.0) * (1.0 - C)

        dC = (
            alpha_C * (T - K) * C * (1.0 - C)
            + eta * S_val * (1.0 - C)
            + shadow_growth
            - gamma_C * K * C
        )

        if C >= 0.985 and dC > 0.0:
            dC = 0.0
        if C <= 0.015 and dC < 0.0:
            dC = 0.0

        d_rp = alpha_recovery_base * (1.0 - recovery_potential) * max(C - K, 0.0)
        recovery_potential += d_rp * 0.05
        if recovery_potential > 1.0:
            recovery_potential = 1.0
        if recovery_potential < 0.01:
            recovery_potential = 0.01

        sigmoid_T = 1.0 / (1.0 + math.exp(-(T_reset_threshold - T) / sigma_T))
        sigmoid_K = 1.0 / (1.0 + math.exp(-(K - 0.85 * Km) / sigma_K))
        p_reset = sigmoid_T * sigmoid_K

        rng_state = (rng_state * 1103515245 + 12345) % 2147483648
        rand_val = rng_state / 2147483648.0

        if p_reset > 0.5 and rand_val < p_reset and recovery_potential > 0.05:
            dT = delta_T_reset * recovery_potential
            dK = -delta_K_reset * recovery_potential
            dC = delta_C_reset * recovery_potential
            recovery_potential *= 0.7

        T += dT * dt
        K += dK * dt
        C += dC * dt

        if step % 10 == 0:
            idx = sim_idx * (n_steps // 10) + step // 10
            T_out[idx] = T
            K_out[idx] = K
            C_out[idx] = C


def simulate_gpu_batch(model, T0, K0, C0, n_simulations, t_span, n_steps, seed=42):
    dt = t_span / n_steps
    store_every = 10
    n_stored = n_steps // store_every

    rng = np.random.RandomState(seed)
    T0_arr = np.full(n_simulations, T0, dtype=np.float64) + rng.normal(
        0, 0.001, n_simulations
    )
    K0_arr = np.full(n_simulations, K0, dtype=np.float64) + rng.normal(
        0, 0.001, n_simulations
    )
    C0_arr = np.full(n_simulations, C0, dtype=np.float64) + rng.normal(
        0, 0.001, n_simulations
    )
    T0_arr = np.clip(T0_arr, 0.01, 0.99)
    K0_arr = np.clip(K0_arr, 0.01, 0.99)
    C0_arr = np.clip(C0_arr, 0.01, 0.99)

    T_out = np.zeros(n_simulations * n_stored, dtype=np.float64)
    K_out = np.zeros(n_simulations * n_stored, dtype=np.float64)
    C_out = np.zeros(n_simulations * n_stored, dtype=np.float64)

    d_T0 = cuda.to_device(T0_arr)
    d_K0 = cuda.to_device(K0_arr)
    d_C0 = cuda.to_device(C0_arr)
    d_Tout = cuda.to_device(T_out)
    d_Kout = cuda.to_device(K_out)
    d_Cout = cuda.to_device(C_out)

    threads_per_block = 128
    blocks = (n_simulations + threads_per_block - 1) // threads_per_block

    simulate_on_gpu_kernel[blocks, threads_per_block](
        d_T0,
        d_K0,
        d_C0,
        d_Tout,
        d_Kout,
        d_Cout,
        model.alpha_T,
        model.beta_T,
        model.gamma_T,
        model.T_min,
        model.alpha_K_base,
        model.beta_K,
        model.K_max_coeff,
        model.K_half,
        model.T_automation,
        model.alpha_C,
        model.gamma_C,
        model.eta,
        model.K0,
        model.delta,
        model.T_singularity,
        model.epsilon_tech,
        model.K_shadow_threshold,
        model.lambda_shadow,
        model.mu_shadow,
        0.05,
        model.cycles.historical_T_peak,
        model.cycles.T_reset_threshold,
        model.cycles.sigma_T,
        model.cycles.sigma_K,
        model.cycles.delta_T_reset,
        model.cycles.delta_K_reset,
        model.cycles.delta_C_reset,
        model.cycles.alpha_recovery_base,
        model.cycles.epsilon_recovery,
        seed,
        n_simulations,
        n_steps,
        dt,
    )

    T_out = d_Tout.copy_to_host().reshape(n_simulations, n_stored)
    K_out = d_Kout.copy_to_host().reshape(n_simulations, n_stored)
    C_out = d_Cout.copy_to_host().reshape(n_simulations, n_stored)

    return T_out, K_out, C_out


# ============================================================
# EXPORTS
# ============================================================

dynamics_numba = dynamics_cpu_single
HAS_NUMBA = True
HAS_CUDA = cuda.is_available()

if HAS_CUDA:
    print("CUDA GPU available — use simulate_gpu_batch() for Monte Carlo")
else:
    print("CUDA not available — using CPU dynamics")
