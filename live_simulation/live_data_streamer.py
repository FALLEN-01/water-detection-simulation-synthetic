"""
Live Synthetic Data Generator
================================
Generates realistic water flow data in real-time using India priors.
Supports fast-forward simulation — speed_multiplier advances the
simulation clock by N minutes per tick, compressing a full day
into seconds for rapid testing.
"""

import random
import numpy as np
from datetime import datetime, timedelta
import json
import os


class LiveSyntheticDataGenerator:
    """Generates realistic synthetic water flow data with fast-forward support"""

    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.priors = self._load_priors()

        building = self.config.get('building_config', {})
        self.num_apartments   = building.get('num_apartments', 50)
        self.avg_household    = building.get('avg_household_size', 3)
        self.base_occupancy   = building.get('base_occupancy', 0.85)

        self.max_flow         = self.config.get('max_flow_rate', 15.0)
        self.leak_probability = self.config.get('leak_injection_probability', 0.02)
        self.speed            = int(self.config.get('speed_multiplier', 1))

        # Leak state
        self.leak_active        = False
        self.leak_start_time    = None
        self.leak_duration      = 0
        self.leak_severity      = 0.0
        self.last_leak_end_time = None
        self.leak_cooldown_min  = 5      # minimum gap between leaks

        # Simulation clock
        self.simulation_time       = datetime.now()
        # Monotonic flow_duration counter — always increments by 60/tick
        # regardless of speed, so LSTM sees same scale as training data
        self._flow_dur_counter     = 0.0

        print(f"LiveSyntheticDataGenerator ready:")
        print(f"  Building: {self.num_apartments} apts, "
              f"{self.avg_household} ppl/apt")
        print(f"  Speed: {self.speed}x | "
              f"Leak prob: {self.leak_probability*100:.1f}%")
        print(f"  Priors: {len(self.priors)} appliances loaded")

    def _load_priors(self):
        priors = {}
        d = self.config.get('priors_directory', '../priors_india')
        if os.path.exists(d):
            for fname in os.listdir(d):
                if fname.endswith('.json'):
                    with open(os.path.join(d, fname)) as f:
                        priors[fname.replace('.json', '')] = json.load(f)
        return priors

    # ------------------------------------------------------------------
    # Flow calculation using priors
    # ------------------------------------------------------------------
    def _time_multiplier(self, hour):
        """Usage intensity multiplier from India patterns"""
        if   6  <= hour < 9:   return random.uniform(2.5, 4.0)
        elif 19 <= hour < 22:  return random.uniform(2.0, 3.5)
        elif 12 <= hour < 14:  return random.uniform(1.4, 2.0)
        elif hour >= 23 or hour < 5: return random.uniform(0.05, 0.25)
        else:                        return random.uniform(0.7, 1.4)

    def _occupancy(self, hour, is_weekend):
        base = self.base_occupancy
        wk = is_weekend
        if wk:
            mult = random.uniform(1.1, 1.3) if 8 <= hour < 22 else random.uniform(0.88, 1.05)
        else:
            if   6  <= hour < 9:  mult = random.uniform(1.15, 1.35)
            elif 9  <= hour < 17: mult = random.uniform(0.50, 0.72)
            elif 17 <= hour < 22: mult = random.uniform(1.05, 1.25)
            else:                  mult = random.uniform(0.88, 1.02)
        return min(0.98, max(0.40, base * mult))

    def _calc_base_flow(self, hour, is_weekend):
        """Calculate aggregate building flow from appliance priors"""
        time_mult = self._time_multiplier(hour)
        occ       = self._occupancy(hour, is_weekend)
        total_people = self.num_apartments * self.avg_household * occ

        agg_flow = 0.0
        for name, prior in self.priors.items():
            # Expected events at this minute across the building
            hour_prob    = prior['timing']['start_hour']['p'][hour]
            lambda_day   = prior['activation']['events_per_day']['lambda']
            exp_events   = hour_prob * lambda_day * total_people / 60.0

            # Appliance flow from prior lognormal
            fp = prior['flow']['mean_flow']
            if fp['type'] == 'lognormal' and fp['shape'] > 0:
                flow_ml_s = fp['scale'] * np.exp(np.random.normal(0, fp['shape']))
            else:
                flow_ml_s = fp['scale']
            flow_lpm = max(0.01, flow_ml_s * 0.06)  # ml/s → L/min

            # Duration fraction
            dp = prior['duration']
            if dp['type'] == 'lognormal' and dp['shape'] > 0:
                dur_s = dp['scale'] * np.exp(np.random.normal(0, dp['shape']))
            else:
                dur_s = dp.get('value', 60.0)
            time_frac = min(1.0, max(0.01, dur_s / 60.0))

            agg_flow += exp_events * flow_lpm * time_frac * time_mult

        # Baseline (common areas, always-on taps)
        agg_flow += random.uniform(0.05, 0.20) * (total_people / 150.0)
        agg_flow += random.gauss(0, 0.06)

        return float(np.clip(agg_flow, 0.02, self.max_flow * 0.85))

    # ------------------------------------------------------------------
    # Leak management
    # ------------------------------------------------------------------
    def _maybe_inject_leak(self):
        if self.leak_active:
            return
        if self.last_leak_end_time is not None:
            elapsed = (self.simulation_time - self.last_leak_end_time).total_seconds() / 60
            if elapsed < self.leak_cooldown_min:
                return
        # Scale probability by speed so leaks appear at similar intervals
        # regardless of time compression (leak probability per sim-minute)
        prob_per_tick = (self.leak_probability / 60.0) * self.speed
        if random.random() < prob_per_tick:
            self.leak_active     = True
            self.leak_start_time = self.simulation_time
            self.leak_duration   = random.randint(3, 15)         # sim-minutes
            self.leak_severity   = random.uniform(0.20, 0.70)
            print(f"  [LEAK] @ {self.simulation_time.strftime('%H:%M')} "
                  f"dur={self.leak_duration}min sev={self.leak_severity*100:.0f}%")

    def _update_leak(self):
        if not self.leak_active:
            return
        elapsed = (self.simulation_time - self.leak_start_time).total_seconds() / 60
        if elapsed >= self.leak_duration:
            self.leak_active        = False
            self.last_leak_end_time = self.simulation_time
            print(f"  [END]  Leak ended @ {self.simulation_time.strftime('%H:%M')}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_speed(self, multiplier: int):
        """Change simulation speed (1 = real-time, N = N min/tick)"""
        self.speed = max(1, min(int(multiplier),
                                self.config.get('fast_forward_max', 120)))
        print(f"  Speed set to {self.speed}x")

    def generate_sample(self) -> dict:
        """Generate one sensor data sample and advance simulation clock"""
        self._update_leak()
        self._maybe_inject_leak()

        hour       = self.simulation_time.hour
        dow        = self.simulation_time.weekday()
        is_weekend = int(dow >= 5)
        month      = self.simulation_time.month

        flow = self._calc_base_flow(hour, is_weekend)

        if self.leak_active:
            flow = min(self.max_flow, flow * (1.0 + self.leak_severity))

        flow           = float(np.clip(flow + random.gauss(0, 0.05), 0.0, self.max_flow))
        flow_normalized = float(np.clip(flow / self.max_flow, 0.0, 1.0))
        turbidity = float(np.clip(
            0.35 + (flow / self.max_flow) * 1.0 + (random.uniform(0.4, 1.5) if self.leak_active else 0)
            + random.gauss(0, 0.08),
            0.05, 4.0
        ))
        # Monotonic counter: always +60 per sample (matches training)
        self._flow_dur_counter = (self._flow_dur_counter + 60.0) % 86400.0
        flow_duration = self._flow_dur_counter

        sample = {
            'timestamp':       self.simulation_time.isoformat(),
            'sim_time':        self.simulation_time.strftime('%H:%M %d/%m'),
            'flow_rate':       round(flow, 3),
            'flow_normalized': round(flow_normalized, 3),
            'turbidity':       round(turbidity, 3),
            'flow_duration':   flow_duration,
            'hour':            hour,
            'day_of_week':     dow,
            'month':           month,
            'is_weekend':      is_weekend,
            'label':           int(self.leak_active),
            'leak_active':     self.leak_active,
            'speed':           self.speed,
        }

        # Advance sim clock by speed_multiplier minutes
        self.simulation_time += timedelta(minutes=self.speed)
        return sample

    def reset(self):
        self.leak_active           = False
        self.leak_start_time       = None
        self.last_leak_end_time    = None
        self.simulation_time       = datetime.now()
        self._flow_dur_counter     = 0.0
        print("LiveSyntheticDataGenerator reset")
