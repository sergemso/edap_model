"""Cycle management for EDAP model v3.1."""

class CycleManager:
    def __init__(self, params=None):
        p = params or {}
        self.recovery_potential_0 = p.get('recovery_potential_0', 1.0)
        self.decay_factor = p.get('decay_factor', 0.70)
        self.historical_T_peak = p.get('historical_T_peak', 0.50)
        self.T_reset_threshold = p.get('T_reset_threshold', 0.06)
        self.sigma_T = p.get('sigma_T', 0.01)
        self.sigma_K = p.get('sigma_K', 0.05)
        self.delta_T_reset = p.get('delta_T_reset', 0.10)
        self.delta_K_reset = p.get('delta_K_reset', 0.10)
        self.delta_C_reset = p.get('delta_C_reset', 0.10)
        self.alpha_recovery_base = p.get('alpha_recovery_base', 0.05)
        self.epsilon_recovery = p.get('epsilon_recovery', 0.01)

        self.recovery_potential = self.recovery_potential_0
        self.in_reset_zone = False
        self.rng_state = 42

    def reset(self):
        self.recovery_potential = self.recovery_potential_0
        self.in_reset_zone = False
        self.rng_state = 42

    def set_seed(self, seed):
        self.rng_state = seed

    def apply_decay(self):
        self.recovery_potential = round(self.recovery_potential * self.decay_factor, 10)
        if self.recovery_potential < 0.01:
            self.recovery_potential = 0.01

    def tuple_for_numba(self):
        return (self.historical_T_peak, self.recovery_potential, self.in_reset_zone,
                self.T_reset_threshold, self.sigma_T, self.sigma_K,
                self.delta_T_reset, self.delta_K_reset, self.delta_C_reset,
                self.alpha_recovery_base, self.epsilon_recovery,
                self.rng_state)

    def update_from_numba(self, recovery_potential, in_reset_zone, rng_state):
        was_in_reset = self.in_reset_zone
        self.recovery_potential = recovery_potential
        self.in_reset_zone = in_reset_zone
        self.rng_state = rng_state
        if was_in_reset and not in_reset_zone:
            self.apply_decay()