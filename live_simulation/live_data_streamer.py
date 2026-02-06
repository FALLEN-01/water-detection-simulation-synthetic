"""
Live Synthetic Data Generator
Generates fresh synthetic water flow data in real-time using India-based patterns
Similar to emulate_data.py but for live streaming
"""

import random
import numpy as np
from datetime import datetime, timedelta
import json
import os


class LiveSyntheticDataGenerator:
    """Generates realistic synthetic water flow data in real-time"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the live data generator"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load India-based priors
        self.priors = self._load_india_priors()
        
        # Building configuration
        building_config = self.config.get('building_config', {})
        self.num_apartments = building_config.get('num_apartments', 50)
        self.avg_household_size = building_config.get('avg_household_size', 3)
        self.base_occupancy = building_config.get('base_occupancy', 0.85)
        
        # Flow parameters
        self.max_flow = self.config.get('max_flow_rate', 15.0)
        self.leak_probability = self.config.get('leak_injection_probability', 0.02)
        
        # Leak state
        self.leak_active = False
        self.leak_start_time = None
        self.leak_duration = 0
        self.leak_severity = 0
        
        # Simulation time (starts at current time)
        self.simulation_time = datetime.now()
        
        print(f"LiveSyntheticDataGenerator initialized:")
        print(f"  Building: {self.num_apartments} apartments")
        print(f"  Leak probability: {self.leak_probability*100:.1f}%")
        print(f"  Loaded {len(self.priors)} appliance patterns")
    
    def _load_india_priors(self):
        """Load India-based water usage patterns"""
        priors = {}
        priors_dir = self.config.get('priors_directory', '../priors_india')
        
        if os.path.exists(priors_dir):
            for filename in os.listdir(priors_dir):
                if filename.endswith('.json'):
                    appliance_name = filename.replace('.json', '')
                    with open(os.path.join(priors_dir, filename), 'r') as f:
                        priors[appliance_name] = json.load(f)
        
        return priors
    
    def _get_time_of_day_multiplier(self, hour):
        """Get usage multiplier based on time of day (India patterns)"""
        # Morning peak: 6-9 AM
        if 6 <= hour < 9:
            return random.uniform(2.5, 4.0)
        # Evening peak: 7-10 PM
        elif 19 <= hour < 22:
            return random.uniform(2.0, 3.5)
        # Afternoon: 12-2 PM
        elif 12 <= hour < 14:
            return random.uniform(1.5, 2.0)
        # Night: 11 PM - 5 AM
        elif hour >= 23 or hour < 5:
            return random.uniform(0.1, 0.3)
        # Other times
        else:
            return random.uniform(0.8, 1.5)
    
    def _get_occupancy_factor(self, hour, is_weekend):
        """Get occupancy factor based on time and day"""
        base = self.base_occupancy
        
        if is_weekend:
            # Higher occupancy on weekends
            if 8 <= hour < 22:
                return base * random.uniform(1.1, 1.3)
            else:
                return base * random.uniform(0.9, 1.1)
        else:
            # Weekday patterns
            if 6 <= hour < 9:  # Morning rush
                return base * random.uniform(1.2, 1.4)
            elif 9 <= hour < 17:  # Work hours (lower occupancy)
                return base * random.uniform(0.6, 0.8)
            elif 17 <= hour < 22:  # Evening
                return base * random.uniform(1.1, 1.3)
            else:  # Night
                return base * random.uniform(0.9, 1.0)
    
    def _calculate_base_flow(self, hour, is_weekend):
        """Calculate base flow rate for the building"""
        # Time of day multiplier
        time_multiplier = self._get_time_of_day_multiplier(hour)
        
        # Occupancy factor
        occupancy = self._get_occupancy_factor(hour, is_weekend)
        
        # Base flow per apartment (L/min)
        base_per_apartment = 0.15  # Conservative baseline
        
        # Calculate total base flow
        total_people = self.num_apartments * self.avg_household_size * occupancy
        base_flow = (total_people / 10) * base_per_apartment * time_multiplier
        
        # Add randomness
        base_flow *= random.uniform(0.8, 1.2)
        
        return max(0.5, min(base_flow, self.max_flow * 0.7))
    
    def _inject_leak(self):
        """Randomly inject a leak event"""
        if not self.leak_active and random.random() < self.leak_probability:
            self.leak_active = True
            self.leak_start_time = self.simulation_time
            # Leak duration: 5-30 minutes
            self.leak_duration = random.randint(5, 30)
            # Leak severity: 20-80% increase in flow
            self.leak_severity = random.uniform(0.2, 0.8)
            print(f"💧 Leak injected at {self.simulation_time.strftime('%H:%M:%S')} "
                  f"(duration: {self.leak_duration}min, severity: {self.leak_severity*100:.0f}%)")
    
    def _update_leak_state(self):
        """Update leak state based on duration"""
        if self.leak_active:
            elapsed = (self.simulation_time - self.leak_start_time).total_seconds() / 60
            if elapsed >= self.leak_duration:
                self.leak_active = False
                print(f"✓ Leak ended at {self.simulation_time.strftime('%H:%M:%S')}")
    
    def generate_sample(self) -> dict:
        """Generate a single real-time sample"""
        # Update leak state
        self._update_leak_state()
        
        # Randomly inject new leaks
        self._inject_leak()
        
        # Extract time features
        hour = self.simulation_time.hour
        day_of_week = self.simulation_time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Calculate base flow
        flow_rate = self._calculate_base_flow(hour, is_weekend)
        
        # Apply leak if active
        if self.leak_active:
            flow_rate *= (1 + self.leak_severity)
        
        # Add measurement noise
        flow_rate += random.gauss(0, 0.1)
        flow_rate = max(0.0, min(flow_rate, self.max_flow))
        
        # Calculate turbidity (correlates with flow)
        turbidity = 0.5 + (flow_rate / self.max_flow) * 1.5
        turbidity += random.gauss(0, 0.05)
        turbidity = max(0.3, min(turbidity, 3.0))
        
        # Calculate flow_normalized
        flow_normalized = flow_rate / self.max_flow
        
        # Calculate flow_duration (simulated)
        flow_duration = random.uniform(0.5, 5.0)  # minutes
        
        # Extract month
        month = self.simulation_time.month
        
        # Create sample with all required fields
        sample = {
            'timestamp': self.simulation_time.isoformat(),
            'flow_rate': round(flow_rate, 3),
            'flow_normalized': round(flow_normalized, 3),
            'turbidity': round(turbidity, 3),
            'flow_duration': round(flow_duration, 3),
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'label': 1 if self.leak_active else 0,
            'leak_active': self.leak_active
        }
        
        # Advance simulation time by 1 minute (or 1 second for faster demo)
        self.simulation_time += timedelta(seconds=1)
        
        return sample
    
    def reset(self):
        """Reset the generator"""
        print("Resetting live data generator...")
        self.leak_active = False
        self.leak_start_time = None
        self.simulation_time = datetime.now()


if __name__ == '__main__':
    # Test the generator
    print("Testing LiveSyntheticDataGenerator...")
    print("=" * 60)
    
    generator = LiveSyntheticDataGenerator()
    
    print("\nGenerating 20 test samples:")
    for i in range(20):
        sample = generator.generate_sample()
        leak_status = "🚨 LEAK" if sample['leak_active'] else "✅ NORMAL"
        print(f"Sample {i+1}: {sample['timestamp'][-8:]} | "
              f"Flow={sample['flow_rate']:.2f} L/min, "
              f"Turb={sample['turbidity']:.2f} NTU, "
              f"Hour={sample['hour']:02d}, "
              f"Status={leak_status}")
        
        import time
        time.sleep(0.1)
