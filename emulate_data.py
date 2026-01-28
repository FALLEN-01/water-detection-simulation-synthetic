import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os

MAX_FLOW = 15.0
SAMPLES_PER_DAY = 1440
DAYS = 180  # 6 months
TOTAL_SAMPLES = SAMPLES_PER_DAY * DAYS

print(f"Generating 6 months of water flow data...")
print(f"Frequency: 1 sample/minute, Total: {TOTAL_SAMPLES:,}")

# Load India-based water usage priors
def load_india_priors():
    """Load realistic water usage patterns from India-based data"""
    priors = {}
    priors_dir = 'priors_india'
    
    if os.path.exists(priors_dir):
        for filename in os.listdir(priors_dir):
            if filename.endswith('.json'):
                appliance_name = filename.replace('.json', '')
                with open(os.path.join(priors_dir, filename), 'r') as f:
                    priors[appliance_name] = json.load(f)
        print(f"Loaded {len(priors)} India-based appliance patterns")
    else:
        print("Warning: priors_india folder not found, using default patterns")
    
    return priors

INDIA_PRIORS = load_india_priors()

def validate_and_optimize_usage(df, building_size=50):
    """Validate and optimize water usage to ensure realistic values"""
    print("\n--- Water Usage Validation ---")
    
    # Expected daily usage per person in India: 135-150 liters (Bureau of Indian Standards)
    # For 50 apartments with avg 3 people each = 150 people
    avg_persons = building_size * 3
    
    # Calculate daily totals
    df['date'] = df['timestamp'].dt.date
    # flow_rate is L/min, sampled every 1 minute, so sum directly gives total liters
    daily_usage = df.groupby('date')['flow_rate'].sum()
    
    avg_daily_total = daily_usage.mean()
    avg_per_person_per_day = avg_daily_total / avg_persons
    
    print(f"Building size: {building_size} apartments (~{avg_persons} people)")
    print(f"Average daily total: {avg_daily_total:.1f} liters")
    print(f"Average per person: {avg_per_person_per_day:.1f} liters/day")
    print(f"Expected range: 135-150 liters/person/day (BIS standard)")
    
    # Check peak flow rates
    peak_flow = df['flow_rate'].quantile(0.99)
    print(f"Peak flow (99th percentile): {peak_flow:.2f} L/min")
    print(f"Max allowed: {MAX_FLOW} L/min")
    
    # Hourly distribution check
    hourly_avg = df.groupby('hour')['flow_rate'].mean()
    morning_peak = hourly_avg.loc[6:8].mean()
    evening_peak = hourly_avg.loc[19:21].mean()
    night_avg = hourly_avg.loc[0:5].mean()
    
    print(f"Morning peak (6-8 AM): {morning_peak:.2f} L/min")
    print(f"Evening peak (7-9 PM): {evening_peak:.2f} L/min")
    print(f"Night average (0-5 AM): {night_avg:.2f} L/min")
    print(f"Peak/Night ratio: {morning_peak/night_avg:.1f}x")
    
    # Validation warnings
    if avg_per_person_per_day < 100:
        print("⚠️ WARNING: Usage below realistic minimum")
    elif avg_per_person_per_day > 200:
        print("⚠️ WARNING: Usage above realistic maximum")
    else:
        print("✓ Usage within realistic range")
    
    if morning_peak / night_avg < 3:
        print("⚠️ WARNING: Peak/night ratio too low (expected 4-10x)")
    elif morning_peak / night_avg > 15:
        print("⚠️ WARNING: Peak/night ratio too high")
    else:
        print("✓ Peak patterns realistic")
    
    print("--- Validation Complete ---\n")
    
    return df

def get_appliance_category_peaks(hour, day_of_week):
    """
    Get randomized usage multipliers by appliance category for peak hours.
    Returns dict of appliance-specific intensity factors.
    """
    # Base multipliers by appliance category based on typical Indian usage
    categories = {
        'morning_personal': ['shower', 'toilet', 'washbasin', 'bidet'],  # Morning hygiene
        'kitchen': ['kitchenfaucet', 'dishwasher30'],  # Meal prep times
        'laundry': ['washingmachine'],  # Flexible timing
        'continuous': ['toilet', 'washbasin']  # Used throughout day
    }
    
    multipliers = {}
    
    # Morning peak (6-9 AM): Personal hygiene dominates
    if 6 <= hour < 9:
        for app in categories['morning_personal']:
            multipliers[app] = random.uniform(1.8, 2.4)  # High morning usage
        for app in categories['kitchen']:
            multipliers[app] = random.uniform(1.2, 1.6)  # Breakfast prep
        for app in categories['laundry']:
            multipliers[app] = random.uniform(1.4, 1.8) if day_of_week >= 5 else random.uniform(0.9, 1.2)
    
    # Lunch time (12-2 PM): Kitchen activity
    elif 12 <= hour < 14:
        for app in categories['kitchen']:
            multipliers[app] = random.uniform(1.5, 2.0)
        for app in categories['morning_personal']:
            multipliers[app] = random.uniform(0.6, 0.9)
        for app in categories['laundry']:
            multipliers[app] = random.uniform(1.0, 1.4)
    
    # Evening peak (6-10 PM): Mixed usage
    elif 18 <= hour < 22:
        for app in categories['morning_personal']:
            multipliers[app] = random.uniform(1.4, 2.0)  # Evening showers
        for app in categories['kitchen']:
            multipliers[app] = random.uniform(1.8, 2.5)  # Dinner prep & cleanup
        for app in categories['laundry']:
            multipliers[app] = random.uniform(0.8, 1.2)
    
    # Night time (10 PM - 6 AM): Minimal usage
    elif hour >= 22 or hour < 6:
        for cat_apps in categories.values():
            for app in cat_apps:
                multipliers[app] = random.uniform(0.2, 0.5)
    
    # Regular hours (9 AM - 6 PM): Moderate usage
    else:
        for cat_apps in categories.values():
            for app in cat_apps:
                multipliers[app] = random.uniform(0.7, 1.1)
    
    # Flatten and ensure all appliances have a multiplier
    final_multipliers = {}
    for app_list in categories.values():
        for app in app_list:
            if app not in final_multipliers:
                final_multipliers[app] = multipliers.get(app, 1.0)
    
    return final_multipliers

def get_building_usage_profile(hour, day_of_week, total_people, appliance_availability):
    """
    Get expected aggregate building flow based on India usage patterns.
    Returns expected flow rate in L/min for the entire building.
    """
    # Get appliance-specific peak hour multipliers with randomization
    peak_multipliers = get_appliance_category_peaks(hour, day_of_week)
    
    aggregate_hourly_weight = 0.0
    total_weight = 0.0
    
    # Weight by frequency of use (how many times per day each appliance is used)
    for appliance_name, prior in INDIA_PRIORS.items():
        hour_prob = prior['timing']['start_hour']['p'][hour]
        events_per_day = prior['activation']['events_per_day']['lambda']
        
        # Apply appliance availability (not all apartments have all appliances)
        availability = appliance_availability.get(appliance_name, 0.5)
        
        # Apply peak hour multiplier with randomization
        peak_multiplier = peak_multipliers.get(appliance_name, 1.0)
        
        # Add random variation to individual appliance contribution (±15%)
        random_variation = random.uniform(0.85, 1.15)
        
        # Combine all factors
        effective_weight = (hour_prob * events_per_day * availability * 
                           peak_multiplier * random_variation)
        
        aggregate_hourly_weight += effective_weight
        total_weight += events_per_day * availability
    
    # Normalize to get aggregate usage probability (0-1 scale)
    if total_weight > 0:
        usage_intensity = aggregate_hourly_weight / total_weight
    else:
        usage_intensity = 0.05
    
    # Scale usage intensity to realistic building-level concurrency
    # At any given minute, expect 5-15% of people to be using water (varies by hour)
    # Peak hours will have higher multipliers already applied
    base_activity_rate = usage_intensity * 5.0  # Scale factor to match realistic usage
    
    # Weekend adjustment (people home more, different patterns)
    if day_of_week >= 5:
        base_activity_rate *= random.uniform(1.15, 1.35)  # Randomized weekend boost
    
    # Calculate expected flow for the building
    # Average flow per active person: ~2.5 L/min (mix of all appliances)
    expected_active_people = total_people * base_activity_rate
    expected_flow = expected_active_people * 2.5
    
    return expected_flow

def generate_training_data():
    """Generate training data using India-based aggregate flow patterns"""
    print("\nGenerating training data (India-based aggregate building flow)...")
    
    data = []
    start_date = datetime(2025, 1, 25)
    num_apartments = 50
    
    # Randomize appliance availability across building with realistic distributions
    # Each apartment has different appliances, but we model aggregate availability
    appliance_availability = {
        'toilet': random.uniform(0.98, 1.0),           # ~100% (essential)
        'washbasin': random.uniform(0.98, 1.0),        # ~100% (essential)
        'kitchenfaucet': random.uniform(0.98, 1.0),    # ~100% (essential)
        'shower': random.uniform(0.90, 0.98),          # ~95% (most have)
        'washingmachine': random.uniform(0.65, 0.75),  # ~70% (common)
        'bidet': random.uniform(0.25, 0.35),           # ~30% (less common)
        'dishwasher30': random.uniform(0.12, 0.18)     # ~15% (rare in India)
    }
    
    # Add daily variation - not every day is the same
    # Some days people use more water, some less (guests, events, etc.)
    daily_usage_pattern = np.random.normal(1.0, 0.15, DAYS)  # Daily multiplier
    daily_usage_pattern = np.clip(daily_usage_pattern, 0.7, 1.3)
    
    print(f"  Appliance availability randomized:")
    for app, avail in appliance_availability.items():
        print(f"    {app}: {avail*100:.1f}%")
    
    for i in range(TOTAL_SAMPLES):
        timestamp = start_date + timedelta(minutes=i)
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()
        month = timestamp.month
        day_index = i // 1440  # Which day we're on
        
        # Occupancy varies by season
        if month in [4, 5, 6]:  # Summer
            base_occupancy = random.uniform(0.70, 0.85)
        elif month in [12, 1]:  # Winter holidays
            base_occupancy = random.uniform(0.65, 0.80)
        else:
            base_occupancy = random.uniform(0.80, 0.95)
        
        daily_variation = random.uniform(-0.10, 0.05)
        occupancy_rate = max(0.55, min(0.98, base_occupancy + daily_variation))
        
        # Household size variation (2-4 people per apartment in India)
        avg_household_size = random.uniform(2.5, 3.5)
        total_people = int(num_apartments * occupancy_rate * avg_household_size)
        
        # Get expected aggregate flow based on India patterns with randomization
        expected_flow = get_building_usage_profile(hour, day_of_week, total_people, 
                                                   appliance_availability)
        
        # Apply daily usage pattern (some days are busier than others)
        expected_flow *= daily_usage_pattern[day_index]
        
        # Add stochastic variation (real-world randomness in usage timing)
        # Use gamma distribution for realistic positive-skewed variation
        flow_variation = np.random.gamma(shape=2.0, scale=0.5)  # Mean=1.0, realistic spread
        actual_flow = expected_flow * flow_variation
        
        # Baseline building flow (common areas, minor background)
        baseline_flow = random.uniform(0.1, 0.25) * (total_people / 100)
        
        # Total measured flow at building meter
        total_flow = baseline_flow + actual_flow
        
        # Add sensor measurement noise (not usage variation)
        measurement_noise = random.gauss(0, 0.1)
        total_flow += measurement_noise
        
        # Turbidity (correlates with flow - higher flow can stir sediment)
        base_turbidity = 0.5 + (total_flow / MAX_FLOW) * 0.8
        turbidity = base_turbidity + random.gauss(0, 0.15)
        
        # Ensure valid ranges
        total_flow = max(0.1, min(MAX_FLOW, total_flow))
        turbidity = max(0.1, min(3.0, turbidity))
        
        # Flow duration (cumulative seconds of flow in current day)
        flow_duration = (i % 1440) * 60
        
        data.append({
            'timestamp': timestamp,
            'flow_rate': total_flow,
            'flow_normalized': total_flow / MAX_FLOW,
            'turbidity': turbidity,
            'flow_duration': flow_duration,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': int(day_of_week >= 5),
            'label': 0
        })
        
        if (i + 1) % 50000 == 0:
            print(f"  Generated {i+1:,} samples...")
    
    print(f"  Building: {num_apartments} apartments, ~{int(num_apartments * 0.85 * 3)} people avg")
    return pd.DataFrame(data)

def generate_test_data(train_df):
    """Generate test data with simulated leaks following realistic patterns"""
    print("\nGenerating test data (with leaks based on usage patterns)...")
    
    test_df = train_df.copy()
    
    # Pattern-based leak injection
    num_leak_events = 8  # Reduced for 6 months
    leak_locations = []
    
    # Pattern 1: Stress-induced leaks during peak usage (40% of leaks)
    # High flow periods stress pipes, causing gradual failures
    peak_hours = [7, 8, 19, 20, 21]  # Morning and evening peaks
    peak_candidates = test_df[(test_df['hour'].isin(peak_hours)) & 
                              (test_df['flow_rate'] > test_df['flow_rate'].quantile(0.75))].index.tolist()
    stress_leaks = random.sample(peak_candidates, min(3, len(peak_candidates)))
    leak_locations.extend(stress_leaks)
    
    # Pattern 2: Seasonal/temperature-related leaks (30% of leaks)
    # Cold months cause pipe contraction/expansion
    winter_candidates = test_df[test_df['month'].isin([1, 2, 12])].index.tolist()
    if winter_candidates:
        seasonal_leaks = random.sample(winter_candidates, min(2, len(winter_candidates)))
        leak_locations.extend(seasonal_leaks)
    
    # Pattern 3: Night-time leaks (30% of leaks)
    # Easier to detect due to low baseline usage, often unnoticed
    night_candidates = test_df[(test_df['hour'] >= 0) & (test_df['hour'] < 6)].index.tolist()
    night_leaks = random.sample(night_candidates, min(3, len(night_candidates)))
    leak_locations.extend(night_leaks)
    
    print(f"Leak patterns: {len(stress_leaks)} stress-induced, {len(seasonal_leaks) if winter_candidates else 0} seasonal, {len(night_leaks)} night-time")
    
    for leak_idx, start_idx in enumerate(leak_locations):
        # Leak duration varies by type
        if start_idx in stress_leaks:
            # Stress leaks: gradual, longer duration (1-3 days)
            leak_duration = random.randint(1440, 4320)
            leak_severity = random.uniform(0.20, 0.40)
        elif start_idx in night_leaks:
            # Night leaks: often go unnoticed, very long (2-5 days)
            leak_duration = random.randint(2880, 7200)
            leak_severity = random.uniform(0.15, 0.30)
        else:
            # Seasonal leaks: sudden, medium duration (4 hours to 2 days)
            leak_duration = random.randint(240, 2880)
            leak_severity = random.uniform(0.25, 0.45)
        
        end_idx = min(start_idx + leak_duration, len(test_df))
        
        for i in range(start_idx, end_idx):
            if test_df.iloc[i]['flow_rate'] < 0.1:
                # Leak during no-usage period
                test_df.at[i, 'flow_rate'] = leak_severity * MAX_FLOW
                test_df.at[i, 'flow_normalized'] = leak_severity
                test_df.at[i, 'flow_duration'] = i - start_idx + 1
            else:
                # Leak during usage period
                test_df.at[i, 'flow_rate'] += leak_severity * MAX_FLOW * 0.3
                test_df.at[i, 'flow_normalized'] = test_df.at[i, 'flow_rate'] / MAX_FLOW
            
            # Leak effects on sensors
            # Turbidity increases (auxiliary sensor - if available)
            test_df.at[i, 'turbidity'] += random.uniform(0.5, 2.0)
            test_df.at[i, 'label'] = 1
    
    leak_count = (test_df['label'] == 1).sum()
    print(f"Leak samples: {leak_count:,} ({leak_count/len(test_df)*100:.2f}%)")
    
    return test_df

def visualize_data(train_df, test_df):
    """Create visualizations of the generated data"""
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Sample data for visualization (first 7 days)
    sample_days = 7
    sample_size = SAMPLES_PER_DAY * sample_days
    train_sample = train_df.iloc[:sample_size]
    test_sample = test_df.iloc[:sample_size]
    
    # Plot 1: Training Flow Rate
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(train_sample['timestamp'], train_sample['flow_rate'], linewidth=0.5, alpha=0.7)
    ax1.set_title('Training: Normal Flow Rate (First 7 Days)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Flow Rate (L/min)')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Plot 2: Test Flow Rate with Leaks
    ax2 = plt.subplot(3, 2, 2)
    normal = test_sample[test_sample['label'] == 0]
    leak = test_sample[test_sample['label'] == 1]
    ax2.plot(normal['timestamp'], normal['flow_rate'], linewidth=0.5, alpha=0.7, label='Normal', color='blue')
    ax2.plot(leak['timestamp'], leak['flow_rate'], linewidth=0.8, alpha=0.9, label='Leak', color='red')
    ax2.set_title('Testing: Flow Rate with Leaks (First 7 Days)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Flow Rate (L/min)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Plot 3: Hourly Average Flow
    ax3 = plt.subplot(3, 2, 3)
    hourly_avg = train_sample.groupby('hour')['flow_rate'].mean()
    ax3.bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7)
    ax3.set_title('Average Flow Rate by Hour', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Avg Flow Rate (L/min)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Flow Rate Distribution
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(test_sample[test_sample['label']==0]['flow_rate'], bins=50, 
             alpha=0.6, label='Normal', color='blue', density=True)
    ax4.hist(test_sample[test_sample['label']==1]['flow_rate'], bins=30, 
             alpha=0.7, label='Leak', color='red', density=True)
    ax4.set_title('Flow Rate Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Flow Rate (L/min)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Turbidity Distribution (Auxiliary Sensor)
    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(train_sample['turbidity'], bins=50, alpha=0.7, label='Training', color='green')
    ax5.hist(test_sample[test_sample['label']==1]['turbidity'], bins=30, alpha=0.7, label='Test (Leak)', color='red')
    ax5.set_title('Turbidity Distribution (Auxiliary)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Turbidity (NTU) - Optional Sensor')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Monthly Statistics
    ax6 = plt.subplot(3, 2, 6)
    train_df['month'] = train_df['timestamp'].dt.month
    test_df['month'] = test_df['timestamp'].dt.month
    monthly_flow = train_df.groupby('month')['flow_rate'].mean()
    monthly_leaks = test_df.groupby('month')['label'].sum()
    
    ax6.bar(monthly_flow.index, monthly_flow.values, color='steelblue', alpha=0.7)
    ax6.set_title('Monthly Average Flow & Leak Count', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Month')
    ax6.set_ylabel('Avg Flow Rate (L/min)', color='steelblue')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(monthly_leaks.index, monthly_leaks.values, color='red', marker='o', linewidth=2, label='Leaks')
    ax6_twin.set_ylabel('Leak Sample Count', color='red')
    ax6.tick_params(axis='y', labelcolor='steelblue')
    ax6_twin.tick_params(axis='y', labelcolor='red')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('water_flow_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: water_flow_analysis.png")
    
    # Timeline visualization
    fig2, (ax7, ax8) = plt.subplots(2, 1, figsize=(16, 8))
    
    # Downsample for visualization (every 60 minutes)
    downsample = 60
    train_down = train_df.iloc[::downsample]
    test_down = test_df.iloc[::downsample]
    
    ax7.plot(train_down['timestamp'], train_down['flow_rate'], linewidth=0.3, alpha=0.6)
    ax7.set_title('6-Month Timeline - Training Data (Normal)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Flow Rate (L/min)')
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    normal_down = test_down[test_down['label'] == 0]
    leak_down = test_down[test_down['label'] == 1]
    ax8.plot(normal_down['timestamp'], normal_down['flow_rate'], linewidth=0.3, alpha=0.6, color='blue')
    ax8.scatter(leak_down['timestamp'], leak_down['flow_rate'], s=1, alpha=0.8, color='red', label='Leaks')
    ax8.set_title('6-Month Timeline - Test Data (with Leaks)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Flow Rate (L/min)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    plt.savefig('water_flow_timeline.png', dpi=150, bbox_inches='tight')
    print("  Saved: water_flow_timeline.png")
    
    # 24-hour patterns for all 6 months (one day per month)
    fig3 = plt.figure(figsize=(15, 10))
    
    # Show one representative day from each month
    for month_num in range(1, 7):
        ax = plt.subplot(2, 3, month_num)
        
        # Get middle day of each month
        month_data = train_df[train_df['month'] == month_num]
        mid_day = len(month_data) // 2
        day_start = mid_day - (mid_day % SAMPLES_PER_DAY)
        day_end = day_start + SAMPLES_PER_DAY
        day_data = month_data.iloc[day_start:day_end]
        
        if len(day_data) > 0:
            # Create time axis in hours (0-24)
            minutes = range(len(day_data))
            hours = [m / 60 for m in minutes]
            
            ax.plot(hours, day_data['flow_rate'].values, linewidth=0.6, alpha=0.8, color='steelblue')
            month_name = day_data.iloc[0]['timestamp'].strftime('%B %Y')
            day_name = day_data.iloc[0]['timestamp'].strftime('%A, %b %d')
            ax.set_title(f'{month_name}\n{day_name}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Hour of Day', fontsize=9)
            ax.set_ylabel('Flow Rate (L/min)', fontsize=9)
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 3))
            ax.grid(True, alpha=0.3)
            
            # Highlight night hours
            ax.axvspan(0, 6, alpha=0.1, color='gray')
    
    plt.suptitle('24-Hour Water Flow Patterns - All 6 Months', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('water_flow_24hour_patterns.png', dpi=150, bbox_inches='tight')
    print("  Saved: water_flow_24hour_patterns.png")
    
    plt.close('all')

# Generate datasets
train_df = generate_training_data()

# Validate and optimize usage patterns
train_df = validate_and_optimize_usage(train_df, building_size=50)

test_df = generate_test_data(train_df)

# Save datasets
print("\nSaving datasets...")
train_df.to_csv('water_train.csv', index=False)
print(f"Saved: water_train.csv ({len(train_df):,} samples)")

test_df.to_csv('water_test.csv', index=False)
print(f"Saved: water_test.csv ({len(test_df):,} samples)")

# Visualize
visualize_data(train_df, test_df)

print(f"\nComplete: {len(train_df):,} samples, 6 months, 1 sample/minute")