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

        self.priors = self._load_appliances()

        building = self.config.get('building_config', {})
        self.num_apartments   = building.get('num_apartments', 50)
        self.avg_household    = building.get('avg_household_size', 3)
        self.base_occupancy   = building.get('base_occupancy', 0.85)

        self.max_flow         = self.config.get('max_flow_rate', 15.0)
        self.speed            = int(self.config.get('speed_multiplier', 1))

        # Leak state
        self.leak_active        = False
        self.leak_start_time    = None
        self.leak_duration      = 0        # minutes
        self.leak_intensity     = 0.0      # L/min
        self.leak_mode          = 'instant' # 'instant' or 'ramp'
        self.leak_ramp_minutes  = 5        # ramp duration
        self.last_leak_end_time = None
        self.leak_cooldown_min  = 5        # minimum gap between leaks

        # Simulation clock
        self.simulation_time       = datetime.now()
        # Monotonic flow_duration counter — always increments by 60/tick
        # regardless of speed, so LSTM sees same scale as training data
        self._flow_dur_counter     = 0.0

        print(f"LiveSyntheticDataGenerator ready:")
        print(f"  Building: {self.num_apartments} apts, "
              f"{self.avg_household} ppl/apt")
        print(f"  Speed: {self.speed}x")
        print(f"  Priors: {len(self.priors)} appliances loaded")

    def _load_appliances(self):
        """Load appliances from all_appliances.json"""
        appliances = {}
        app_file = os.path.join('..', 'household_simulator', 'all_appliances.json')
        if os.path.exists(app_file):
            try:
                with open(app_file, 'r') as f:
                    all_apps = json.load(f)
                # Convert all_appliances.json format to priors format
                for app_name, app_data in all_apps.items():
                    appliances[app_name] = {
                        'timing': {
                            'start_hour': {
                                'p': [app_data.get('start_hour_probability', 0.5)] * 24
                            }
                        },
                        'activation': {
                            'events_per_day': {
                                'lambda': app_data.get('events_per_day', 2.0)
                            }
                        },
                        'flow': {
                            'mean_flow': {
                                'type': 'lognormal',
                                'scale': app_data.get('flow_rate_lpm', 5.0),
                                'shape': app_data.get('flow_shape', 0.3)
                            }
                        },
                        'duration': {
                            'type': 'lognormal',
                            'scale': app_data.get('duration_seconds', 60.0),
                            'shape': app_data.get('duration_shape', 0.2)
                        }
                    }
                print(f"Loaded {len(appliances)} appliances from all_appliances.json")
                return appliances
            except Exception as e:
                print(f"Error loading all_appliances.json: {e}")
                return {}
        else:
            print(f"all_appliances.json not found at {app_file}")
            return {}

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
    def inject_leak(self, intensity: float = None, duration: int = None,
                    mode: str = 'instant', ramp_minutes: int = 5):
        """Manually inject a leak into the simulation

        Args:
            intensity: Leak intensity in L/min (0.1 - 2.0)
            duration: Leak duration in minutes (5 - 180)
            mode: 'instant' or 'ramp'
            ramp_minutes: Ramp-up duration in minutes (1 - 30)
        """
        if self.leak_active:
            print(f"  [INFO] Leak already active")
            return
        if self.last_leak_end_time is not None:
            elapsed = (self.simulation_time - self.last_leak_end_time).total_seconds() / 60
            if elapsed < self.leak_cooldown_min:
                print(f"  [INFO] Leak cooldown active ({self.leak_cooldown_min}min required)")
                return
        self.leak_active     = True
        self.leak_start_time = self.simulation_time
        self.leak_duration   = duration if duration is not None else 60
        self.leak_intensity  = intensity if intensity is not None else 0.5
        self.leak_mode       = mode
        self.leak_ramp_minutes = ramp_minutes
        print(f"  [LEAK] MANUALLY INJECTED @ {self.simulation_time.strftime('%H:%M')} "
              f"intensity={self.leak_intensity}L/min dur={self.leak_duration}min mode={mode}")

    def get_leak_remaining(self) -> int:
        """Get remaining leak time in minutes"""
        if not self.leak_active:
            return 0
        elapsed = (self.simulation_time - self.leak_start_time).total_seconds() / 60
        remaining = self.leak_duration - elapsed
        return max(0, int(np.ceil(remaining)))

    def stop_leak(self):
        """Manually stop the current leak"""
        if not self.leak_active:
            print(f"  [INFO] No active leak to stop")
            return
        self.leak_active        = False
        self.last_leak_end_time = self.simulation_time
        print(f"  [STOP] Leak manually stopped @ {self.simulation_time.strftime('%H:%M')}")

    def _update_leak(self):
        if not self.leak_active:
            return
        elapsed = (self.simulation_time - self.leak_start_time).total_seconds() / 60
        if elapsed >= self.leak_duration:
            self.leak_active        = False
            self.last_leak_end_time = self.simulation_time
            print(f"  [END]  Leak ended @ {self.simulation_time.strftime('%H:%M')}")

    def _get_leak_factor(self) -> float:
        """Get leak factor for flow calculation (0.0 to 1.0+)"""
        if not self.leak_active:
            return 0.0

        if self.leak_mode == 'instant':
            return self.leak_intensity
        elif self.leak_mode == 'ramp':
            elapsed = (self.simulation_time - self.leak_start_time).total_seconds() / 60
            ramp_progress = min(1.0, elapsed / self.leak_ramp_minutes)
            return self.leak_intensity * ramp_progress
        return 0.0

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

        hour       = self.simulation_time.hour
        dow        = self.simulation_time.weekday()
        is_weekend = int(dow >= 5)
        month      = self.simulation_time.month

        flow = self._calc_base_flow(hour, is_weekend)

        leak_factor = self._get_leak_factor()
        if leak_factor > 0:
            flow = min(self.max_flow, flow + leak_factor)

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
            'leak_mode':       self.leak_mode,
            'leak_intensity':  round(self.leak_intensity, 1),
            'leak_remaining':  self.get_leak_remaining(),
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
