"""Test suite for EDAP Model v3.1."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from edap_model.model import EDAPModel
from edap_model.cycles import CycleManager
from edap_model.normalizer import Normalizer
from edap_model.utils import load_json


def make_params(**overrides):
    p = {
        'alpha_T': 0.52, 'beta_T': 0.12, 'gamma_T': 0.20, 'T_min': 0.08,
        'alpha_K_base': 0.67, 'beta_K': 0.18,
        'K_max_coeff': 0.95, 'K_half': 0.05, 'T_automation': 0.85,
        'alpha_C': 0.31, 'gamma_C': 0.20, 'eta': 0.40,
        'K0': 0.55, 'delta': 0.25, 'sigma': 0.12,
        'T_singularity': 0.65, 'epsilon_tech': 0.40,
        'K_shadow_threshold': 0.50, 'lambda_shadow': 0.10, 'mu_shadow': 0.15,
        'historical_T_peak': 0.50, 'decay_factor': 0.70,
        'recovery_potential_0': 1.0, 'T_reset_threshold': 0.06,
        'sigma_T': 0.01, 'sigma_K': 0.05,
        'delta_T_reset': 0.10, 'delta_K_reset': 0.10, 'delta_C_reset': 0.10,
        'alpha_recovery_base': 0.05, 'epsilon_recovery': 0.01,
    }
    p.update(overrides)
    return p


class TestModelInstantiation:
    def test_default_construction(self):
        m = EDAPModel()
        assert m.alpha_T == 0.52
        assert m.alpha_K_base == 0.67
        assert m.eta == 0.40

    def test_custom_params(self):
        m = EDAPModel(make_params(alpha_K_base=0.30))
        assert m.alpha_K_base == 0.30

    def test_cycle_manager_created(self):
        m = EDAPModel()
        assert m.cycles is not None
        assert m.cycles.recovery_potential == 1.0


class TestCycleManager:
    def test_initial_state(self):
        cm = CycleManager(make_params())
        assert cm.recovery_potential == 1.0
        assert not cm.in_reset_zone

    def test_reset(self):
        cm = CycleManager(make_params())
        cm.recovery_potential = 0.3
        cm.in_reset_zone = True
        cm.reset()
        assert cm.recovery_potential == 1.0
        assert not cm.in_reset_zone

    def test_apply_decay(self):
        cm = CycleManager(make_params(decay_factor=0.70, recovery_potential_0=1.0))
        cm.apply_decay()
        assert cm.recovery_potential == 0.70
        cm.apply_decay()
        assert cm.recovery_potential == pytest.approx(0.49, rel=1e-9)

    def test_decay_floor(self):
        cm = CycleManager(make_params(decay_factor=0.70, recovery_potential_0=0.02))
        cm.apply_decay()
        assert cm.recovery_potential == pytest.approx(0.014, rel=1e-9)

    def test_recovery_replenishment(self):
        cm = CycleManager(make_params(recovery_potential_0=0.5, alpha_recovery_base=0.10))
        cm.recovery_potential = 0.5
        # Simulate good times: C > K
        # In real dynamics this would be called from dynamics_numba
        # Here we just test the parameter is set
        assert cm.alpha_recovery_base == 0.10


class TestKCritical:
    def test_linear_at_low_T(self):
        m = EDAPModel(make_params(T_singularity=0.65, epsilon_tech=0.40))
        assert m.K_critical(0.0) == 0.55
        assert abs(m.K_critical(0.30) - (0.55 - 0.25 * 0.30)) < 1e-9

    def test_bend_above_singularity(self):
        m = EDAPModel(make_params(T_singularity=0.65, epsilon_tech=0.40))
        assert m.K_critical(0.75) > (0.55 - 0.25 * 0.75)


class TestAlphaKInhibition:
    def test_high_C_inhibits_K_growth(self):
        m = EDAPModel(make_params())
        _, dK, _ = m.dynamics(0, [0.5, 0.1, 0.8])
        assert dK <= 0.05

    def test_low_C_allows_K_growth(self):
        m = EDAPModel(make_params())
        _, dK, _ = m.dynamics(0, [0.2, 0.4, 0.1])
        assert dK > 0


class TestTechnologyDynamics:
    def test_T_grows_when_conditions_good(self):
        m = EDAPModel(make_params(alpha_K_base=0.30))
        dT, _, _ = m.dynamics(0, [0.3, 0.05, 0.7])
        assert dT > 0

    def test_T_declines_when_K_above_K_crit(self):
        m = EDAPModel(make_params())
        Kc = m.K_critical(0.3)
        dT, _, _ = m.dynamics(0, [0.3, Kc + 0.1, 0.3])
        assert dT < 0

    def test_eta_boosts_T_recovery(self):
        m_eta = EDAPModel(make_params(eta=0.40, historical_T_peak=0.80))
        m_no = EDAPModel(make_params(eta=0.0, historical_T_peak=0.80))
        dT_eta, _, _ = m_eta.dynamics(0, [0.2, 0.3, 0.3])
        dT_no, _, _ = m_no.dynamics(0, [0.2, 0.3, 0.3])
        assert dT_eta > dT_no

    def test_T_stays_in_bounds(self):
        m = EDAPModel()
        m.set_seed(42)
        r = m.simulate(0.5, 0.5, 0.5, t_span=50, n_points=500)
        assert r['T'].max() <= 1.0 and r['T'].min() >= 0.0


class TestShadowCooperation:
    def test_shadow_drain_at_high_K(self):
        m = EDAPModel(make_params(K_shadow_threshold=0.50, lambda_shadow=0.10))
        _, dK1, _ = m.dynamics(0, [0.2, 0.3, 0.2])
        _, dK2, _ = m.dynamics(0, [0.2, 0.7, 0.2])
        assert dK2 < dK1


class TestRecoveryDynamics:
    def test_recovery_potential_in_results(self):
        m = EDAPModel()
        m.set_seed(42)
        r = m.simulate(0.3, 0.4, 0.3, t_span=10, n_points=100)
        assert 'final_recovery_potential' in r

    def test_recovery_potential_bounded(self):
        m = EDAPModel()
        m.set_seed(42)
        r = m.simulate(0.3, 0.4, 0.3, t_span=10, n_points=100)
        assert 0.0 <= r['final_recovery_potential'] <= 1.0


class TestSimulation:
    def test_simulate_returns_correct_structure(self):
        m = EDAPModel()
        m.set_seed(42)
        r = m.simulate(0.3, 0.4, 0.3, t_span=10, n_points=100)
        assert 't' in r and 'T' in r and 'K' in r and 'C' in r
        assert len(r['t']) == 100

    def test_same_seed_reproducible(self):
        m1 = EDAPModel()
        m1.set_seed(42)
        m2 = EDAPModel()
        m2.set_seed(42)
        r1 = m1.simulate(0.3, 0.4, 0.3, t_span=10, n_points=50)
        r2 = m2.simulate(0.3, 0.4, 0.3, t_span=10, n_points=50)
        assert np.allclose(r1['T'], r2['T'], rtol=1e-10)

    def test_C_never_negative(self):
        m = EDAPModel()
        m.set_seed(42)
        r = m.simulate(0.1, 0.8, 0.05, t_span=50, n_points=500)
        assert r['C'].min() >= 0.0


class TestAttractors:
    def test_feudal_trap_exists(self):
        m = EDAPModel(make_params(alpha_K_base=0.67, recovery_potential_0=0.01))
        m.set_seed(42)
        r = m.simulate(0.15, 0.70, 0.10, t_span=80, n_points=500)
        final_T = r['T'][-50:].mean()
        assert final_T < 0.25

    def test_sustainable_with_low_alpha_K(self):
        m = EDAPModel(make_params(alpha_K_base=0.08))
        m.set_seed(42)
        r = m.simulate(0.40, 0.05, 0.70, t_span=80, n_points=500)
        final_T = r['T'][-50:].mean()
        assert final_T > 0.35

    def test_bistability(self):
        m_bad = EDAPModel(make_params(alpha_K_base=0.25, recovery_potential_0=0.01))
        m_bad.set_seed(42)
        r_bad = m_bad.simulate(0.10, 0.70, 0.05, t_span=80, n_points=500)
        m_good = EDAPModel(make_params(alpha_K_base=0.25))
        m_good.set_seed(42)
        r_good = m_good.simulate(0.50, 0.05, 0.80, t_span=80, n_points=500)
        assert abs(r_bad['T'][-50:].mean() - r_good['T'][-50:].mean()) > 0.10


if __name__ == "__main__":
    pytest.main([__file__, '-v'])