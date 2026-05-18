"""EDAP Model v3.1 main class."""

import numpy as np
from scipy.integrate import solve_ivp
from edap_model.dynamics import dynamics_numba, HAS_NUMBA
from edap_model.cycles import CycleManager


class EDAPModel:
    def __init__(self, params=None):
        p = params or {}
        # Dynamics
        self.alpha_T = p.get('alpha_T', 0.52)
        self.beta_T = p.get('beta_T', 0.12)
        self.gamma_T = p.get('gamma_T', 0.20)
        self.T_min = p.get('T_min', 0.08)
        self.alpha_K_base = p.get('alpha_K_base', 0.67)
        self.beta_K = p.get('beta_K', 0.18)
        self.K_max_coeff = p.get('K_max_coeff', 0.95)
        self.K_half = p.get('K_half', 0.05)
        self.T_automation = p.get('T_automation', 0.85)
        self.alpha_C = p.get('alpha_C', 0.31)
        self.gamma_C = p.get('gamma_C', 0.20)
        self.eta = p.get('eta', 0.40)
        self.K0 = p.get('K0', 0.55)
        self.delta = p.get('delta', 0.25)
        self.sigma = p.get('sigma', 0.12)
        self.T_singularity = p.get('T_singularity', 0.65)
        self.epsilon_tech = p.get('epsilon_tech', 0.40)
        self.K_shadow_threshold = p.get('K_shadow_threshold', 0.50)
        self.lambda_shadow = p.get('lambda_shadow', 0.10)
        self.mu_shadow = p.get('mu_shadow', 0.15)

        # Cycle manager
        self.cycles = CycleManager(p)

        # Shock state
        self.shock_history = {float(k): v for k, v in p.get('shocks', {}).items()}
        self.rng = np.random.RandomState(42)
        self._S_current = 0.05
        self._S_time = 0.0

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)
        self._S_current = 0.05
        self._S_time = 0.0
        self.cycles.reset()
        self.cycles.set_seed(seed)

    def K_critical(self, T):
        T = np.array(T)
        return self.K0 - self.delta * T + self.epsilon_tech * np.maximum(T - self.T_singularity, 0.0)**2

    def external_shock(self, t):
        if t in self.shock_history:
            return self.shock_history[t]
        dt = t - self._S_time
        if dt > 0:
            dW = self.rng.normal(0, np.sqrt(dt))
            self._S_current = max(0, self._S_current + 0.3 * (0.05 - self._S_current) * dt + self.sigma * dW)
            self._S_time = t
        return self._S_current

    def _param_tuple(self, S):
        return (self.alpha_T, self.beta_T, self.gamma_T, self.T_min,
                self.alpha_K_base, self.beta_K,
                self.K_max_coeff, self.K_half, self.T_automation,
                self.alpha_C, self.gamma_C, self.eta,
                self.K0, self.delta, self.T_singularity, self.epsilon_tech,
                self.K_shadow_threshold, self.lambda_shadow, self.mu_shadow,
                S,
                *self.cycles.tuple_for_numba())

    def dynamics(self, t, y):
        S = self.external_shock(t)
        if HAS_NUMBA:
            result, rp, in_rz, rng_s = dynamics_numba(y[0], y[1], y[2], *self._param_tuple(S))
        else:
            result, rp, in_rz, rng_s = dynamics_numba.py_func(y[0], y[1], y[2], *self._param_tuple(S))
        self.cycles.update_from_numba(rp, in_rz, rng_s)
        return result

    def simulate(self, T0, K0, C0, t_span=100, n_points=2000, method='RK45'):
        self._S_current = 0.05
        self._S_time = 0.0
        self.cycles.reset()
        t_eval = np.linspace(0, t_span, n_points)
        sol = solve_ivp(self.dynamics, [0, t_span], [T0, K0, C0],
                        t_eval=t_eval, method=method, rtol=1e-8, atol=1e-10)
        T, K, C = sol.y
        S_hist = np.array([self.external_shock(t) for t in t_eval])
        Kc_hist = self.K_critical(T)
        return {'t': t_eval, 'T': T, 'K': K, 'C': C, 'S': S_hist, 'K_crit': Kc_hist,
                'success': sol.success, 'final_recovery_potential': self.cycles.recovery_potential}