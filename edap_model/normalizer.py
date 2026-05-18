"""Proxy normalization for EDAP model."""

import numpy as np
from edap_model.utils import load_json


class Normalizer:
    def __init__(self, params_path):
        self.params = load_json(params_path)

    def _normalize_component(self, raw_dict, component_config):
        proxies = component_config['proxies']
        value = 0.0
        total_weight = 0.0
        for proxy_name, config in proxies.items():
            if proxy_name in raw_dict:
                raw_val = raw_dict[proxy_name]
                norm_val = (raw_val - config['min_global']) / (config['max_global'] - config['min_global'])
                norm_val = np.clip(norm_val, 0.0, 1.0)
                value += config['weight'] * norm_val
                total_weight += config['weight']
        return value / total_weight if total_weight > 0 else 0.0

    def normalize(self, raw_point):
        T = self._normalize_component(raw_point.get('T_proxies', {}), self.params['T_normalization'])
        K = self._normalize_component(raw_point.get('K_proxies', {}), self.params['K_normalization'])
        C = self._normalize_component(raw_point.get('C_proxies', {}), self.params['C_normalization'])
        return T, K, C

    def normalize_civilization(self, civ_data):
        normalized = []
        for point in civ_data.get('data_points', []):
            T, K, C = self.normalize(point)
            normalized.append({'year': point['year'], 'label': point.get('label', ''), 'T': T, 'K': K, 'C': C})
        for point in civ_data.get('projection_points', []):
            T, K, C = self.normalize(point)
            normalized.append({'year': point['year'], 'label': point.get('label', ''), 'T': T, 'K': K, 'C': C, 'is_projection': True})
        return normalized