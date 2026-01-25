import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MAX_FLOW = 15.0
SAMPLES_PER_DAY = 1440
DAYS = 365
TOTAL_SAMPLES = SAMPLES_PER_DAY * DAYS

print(f"Generating 12 months of water flow data...")
print(f"Frequency: 1 sample/minute, Total: {TOTAL_SAMPLES:,}")

def get_time_based_usage_prob(hour):
    """Get usage probability based on time of day"""
    if 6 <= hour < 9:  # Morning peak
        return 0.6
    elif 12 <= hour < 14:  # Lunch time
        return 0.5
    elif 18 <= hour < 22:  # Evening peak
        return 0.7
    elif 0 <= hour < 6:  # Night time
        return 0.1
    else:
        return 0.3

def generate_training_data():
    """Generate training data simulating apartment building water flow (no leaks)"""
    print("\nGenerating training data (apartment building, normal flow)...")
    
    data = []
    start_date = datetime(2025, 1, 25)
    num_apartments = 50  # Total apartment units
    
    for i in range(TOTAL_SAMPLES):
        timestamp = start_date + timedelta(minutes=i)
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Calculate occupancy rate (varies by season and time)
        # Summer (Jun-Aug) and Winter holidays: more vacancies/seasonal visitors
        if month in [6, 7, 8]:  # Summer - some residents travel, seasonal visitors
            base_occupancy = random.uniform(0.70, 0.85)  # 70-85% occupied
        elif month in [12, 1]:  # Winter holidays - more travel
            base_occupancy = random.uniform(0.65, 0.80)
        else:  # Regular months
            base_occupancy = random.uniform(0.80, 0.95)  # 80-95% occupied
        
        # Add random daily variation (some units vacant, some visitors)
        daily_variation = random.uniform(-0.10, 0.05)
        occupancy_rate = max(0.55, min(0.98, base_occupancy + daily_variation))
        
        # Calculate number of occupied apartments
        occupied_apartments = int(num_apartments * occupancy_rate)
        
        # Baseline flow - common areas, minor leaks in empty units
        baseline_flow = random.uniform(0.3, 1.0) * occupancy_rate
        
        # Time-based multiplier for apartment building
        if 6 <= hour < 9:  # Morning peak (showers, breakfast)
            usage_rate = random.uniform(0.35, 0.65)  # 35-65% of occupied units active
            unit_flow = random.uniform(2.0, 4.0)  # L/min per active unit
        elif 12 <= hour < 14:  # Lunch time
            usage_rate = random.uniform(0.15, 0.30)
            unit_flow = random.uniform(1.5, 3.0)
        elif 18 <= hour < 23:  # Evening peak (cooking, washing, showers)
            usage_rate = random.uniform(0.45, 0.75)
            unit_flow = random.uniform(2.5, 4.5)
        elif 0 <= hour < 6:  # Night time (minimal usage)
            usage_rate = random.uniform(0.05, 0.15)
            unit_flow = random.uniform(0.8, 2.0)
        else:  # Day time (most people at work/school)
            usage_rate = random.uniform(0.10, 0.25)
            unit_flow = random.uniform(1.0, 2.5)
        
        # Weekend adjustment (people home more, different patterns)
        if day_of_week >= 5:
            usage_rate *= 1.4  # More people home on weekends
            baseline_flow *= 1.2
        
        # Calculate active units based on occupancy
        active_units = int(occupied_apartments * usage_rate)
        
        # Calculate total flow
        total_flow = baseline_flow + (active_units * unit_flow / num_apartments)
        
        # Add random variations to simulate real usage
        total_flow *= random.uniform(0.85, 1.15)
        total_flow += random.gauss(0, 0.3)
        
        # Pressure varies inversely with flow (higher flow = lower pressure)
        if total_flow < 3:
            pressure = random.uniform(50, 58)
        elif total_flow < 6:
            pressure = random.uniform(45, 52)
        elif total_flow < 10:
            pressure = random.uniform(40, 48)
        else:
            pressure = random.uniform(35, 45)
        
        # Add pressure noise
        pressure += random.gauss(0, 1.5)
        
        # Other parameters
        turbidity = random.uniform(0.3, 1.5)  # Lower turbidity for normal flow
        temperature = random.uniform(18, 24)  # Celsius
        turbidity += random.gauss(0, 0.1)
        temperature += random.gauss(0, 0.3)
        
        # Ensure valid ranges
        total_flow = max(0.2, min(MAX_FLOW, total_flow))  # Always some flow in building
        pressure = max(30, min(60, pressure))
        turbidity = max(0, turbidity)
        
        # Flow duration tracking
        flow_duration = i % 1440 + 1  # Reset daily
        
        data.append({
            'timestamp': timestamp,
            'flow_rate': total_flow,
            'flow_normalized': total_flow / MAX_FLOW,
            'pressure': pressure,
            'turbidity': turbidity,
            'temperature': temperature,
            'flow_duration': flow_duration,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': int(day_of_week >= 5),
            'label': 0  # Normal
        })
        
        if (i + 1) % 100000 == 0:
            print(f"  Generated {i+1:,} samples...")
    
    return pd.DataFrame(data)

def generate_test_data(train_df):
    """Generate test data with simulated leaks"""
    print("\nGenerating test data (with leaks)...")
    
    test_df = train_df.copy()
    
    # Inject leak events
    num_leak_events = 15
    leak_locations = np.random.choice(len(test_df), num_leak_events, replace=False)
    
    for leak_idx, start_idx in enumerate(leak_locations):
        # Each leak lasts between 2 hours to 3 days
        leak_duration = random.randint(120, 4320)  # minutes
        end_idx = min(start_idx + leak_duration, len(test_df))
        leak_severity = random.uniform(0.15, 0.45)
        
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
            
            # Leak effects on other parameters
            test_df.at[i, 'pressure'] -= random.uniform(5, 15)
            test_df.at[i, 'pressure'] = max(20, test_df.at[i, 'pressure'])
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
    
    # Plot 4: Pressure vs Flow
    ax4 = plt.subplot(3, 2, 4)
    ax4.scatter(test_sample[test_sample['label']==0]['flow_rate'], 
                test_sample[test_sample['label']==0]['pressure'], 
                alpha=0.3, s=1, label='Normal', color='blue')
    ax4.scatter(test_sample[test_sample['label']==1]['flow_rate'], 
                test_sample[test_sample['label']==1]['pressure'], 
                alpha=0.7, s=2, label='Leak', color='red')
    ax4.set_title('Pressure vs Flow Rate', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Flow Rate (L/min)')
    ax4.set_ylabel('Pressure (PSI)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Turbidity Distribution
    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(train_sample['turbidity'], bins=50, alpha=0.7, label='Training', color='green')
    ax5.hist(test_sample[test_sample['label']==1]['turbidity'], bins=30, alpha=0.7, label='Test (Leak)', color='red')
    ax5.set_title('Turbidity Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Turbidity (NTU)')
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
    ax7.set_title('12-Month Timeline - Training Data (Normal)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Flow Rate (L/min)')
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    normal_down = test_down[test_down['label'] == 0]
    leak_down = test_down[test_down['label'] == 1]
    ax8.plot(normal_down['timestamp'], normal_down['flow_rate'], linewidth=0.3, alpha=0.6, color='blue')
    ax8.scatter(leak_down['timestamp'], leak_down['flow_rate'], s=1, alpha=0.8, color='red', label='Leaks')
    ax8.set_title('12-Month Timeline - Test Data (with Leaks)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Flow Rate (L/min)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    plt.savefig('water_flow_timeline.png', dpi=150, bbox_inches='tight')
    print("  Saved: water_flow_timeline.png")
    
    plt.close('all')

# Generate datasets
train_df = generate_training_data()
test_df = generate_test_data(train_df)

# Save datasets
print("\nSaving datasets...")
train_df.to_csv('water_train.csv', index=False)
print(f"Saved: water_train.csv ({len(train_df):,} samples)")

test_df.to_csv('water_test.csv', index=False)
print(f"Saved: water_test.csv ({len(test_df):,} samples)")

# Visualize
visualize_data(train_df, test_df)

print(f"\nComplete: {len(train_df):,} samples, 12 months, 1 sample/minute")