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
    print("\nGenerating training data (normal flow, no leaks)...")
    
    data = []
    start_date = datetime(2025, 1, 25)
    flow_duration = 0
    
    for i in range(TOTAL_SAMPLES):
        timestamp = start_date + timedelta(minutes=i)
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()
        
        # Time-based usage patterns
        usage_prob = get_time_based_usage_prob(hour)
        
        # Weekend adjustment (less usage)
        if day_of_week >= 5:
            usage_prob *= 0.7
        
        # Generate flow based on usage probability
        rand_val = random.random()
        
        if rand_val < usage_prob * 0.4:  # Heavy usage
            flow = random.uniform(0.6, 1.0) * MAX_FLOW
            flow_duration += 1
            pressure = random.uniform(45, 55)  # Normal pressure (PSI)
        elif rand_val < usage_prob * 0.7:  # Medium usage
            flow = random.uniform(0.3, 0.6) * MAX_FLOW
            flow_duration += 1
            pressure = random.uniform(40, 50)
        elif rand_val < usage_prob:  # Light usage
            flow = random.uniform(0.1, 0.3) * MAX_FLOW
            flow_duration += 1
            pressure = random.uniform(35, 45)
        else:  # No usage
            flow = 0.0
            flow_duration = 0
            pressure = random.uniform(50, 60)  # Higher pressure when not in use
        
        # Add natural variations
        turbidity = random.uniform(0.5, 2.0)  # NTU (Nephelometric Turbidity Units)
        temperature = random.uniform(15, 25)  # Celsius
        
        # Add noise
        flow += random.gauss(0, 0.05)
        turbidity += random.gauss(0, 0.05)
        pressure += random.gauss(0, 1.0)
        temperature += random.gauss(0, 0.5)
        
        # Ensure non-negative values
        flow = max(0, flow)
        turbidity = max(0, turbidity)
        pressure = max(0, pressure)
        
        data.append({
            'timestamp': timestamp,
            'flow_rate': flow,
            'flow_normalized': flow / MAX_FLOW,
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

def generate_phase2_data(phase1_df):
    """Phase test_data(train_df):
    print("\nGenerating test data (with leaks)...")
    
    test_df = train_df.copy()
    
    num_leak_events = 15
    leak_locations = np.random.choice(len(test
    for leak_idx, start_idx in enumerate(leak_locations):
        # Each leak lasts between 2 hours to 3 days
        leak_duration = random.randint(120, 4320)  # minutes
        leak_duration = random.randint(120, 4320)
        end_idx = min(start_idx + leak_duration, len(test_df))
        leak_severity = random.uniform(0.15, 0.45)
        
        for i in range(start_idx, end_idx):
            if test_df.iloc[i]['flow_rate'] < 0.1:
                test_df.at[i, 'flow_rate'] = leak_severity * MAX_FLOW
                test_df.at[i, 'flow_normalized'] = leak_severity
                test_df.at[i, 'flow_duration'] = i - start_idx + 1
            else:
                test_df.at[i, 'flow_rate'] += leak_severity * MAX_FLOW * 0.3
                test_df.at[i, 'flow_normalized'] = test_df.at[i, 'flow_rate'] / MAX_FLOW
            
            test_df.at[i, 'pressure'] -= random.uniform(5, 15)
            test_df.at[i, 'pressure'] = max(20, test_df.at[i, 'pressure'])
            test_df.at[i, 'turbidity'] += random.uniform(0.5, 2.0)
            test_df.at[i, 'label'] = 1
    
    leak_count = (test_df['label'] == 1).sum()
    print(f"Leak samples: {leak_count:,} ({leak_count/len(test_df)*100:.2f}%)")
    
    return testa(df, phase2_df):
    """Create visualizations of the generated data"""
    print("\n--- Generating Visualizations ---")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Sample data for visualization (first 7 days)
    sample_days = 7
    sample_size = SAMPLES_PER_DAY * sample_days
    df_sample = df.iloc[:sample_size]
    phase2_sample = phase2_df.iloc[:sample_size]
    
    # Plot 1: Flow train_df, test_df):
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    sample_days = 7
    sample_size = SAMPLES_PER_DAY * sample_days
    train_sample = train_df.iloc[:sample_size]
    test_sample = test_df.iloc[:sample_size]
    
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(train_sample['timestamp'], train_sample['flow_rate'], linewidth=0.5, alpha=0.7)
    ax1.set_title('Training: Normal Flow Rate (First 7 Days)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Flow Rate (L/min)')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    ax2 = plt.subplot(3, 2, 2)
    normal = test_sample[test_sample['label'] == 0]
    leak = test_sample[test_sample['label'] == 1]
    ax2.plot(normal['timestamp'], normal['flow_rate'], linewidth=0.5, alpha=0.7, label='Normal', color='blue')
    ax2.plot(leak['timestamp'], leak['flow_rate'], linewidth=0.8, alpha=0.9, label='Leak', color='red')
    ax2.set_title('Testingf Day')
    ax3.set_ylabel('Avg Flow Rate (L/min)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Pressure vs Flow (Phase 2)
    ax4 = plt.subplot(3, 2, 4)
    ax4.scatter(phase2_sample[phase2_sample['label']==0]['flow_rate'], 
                phase2_sample[phase2_sample['label']==0]['pressure'], 
                alpha=0.3, s=1, label='Normal', color='blue')
    ax4.scatter(phase2_sample[phase2_sample['label']==1]['flow_rate'], 
                phase2_sample[phase2_sample['label']==1]['pressure'], 
    ax3 = plt.subplot(3, 2, 3)
    hourly_avg = train_sample.groupby('hour')['flow_rate'].mean()
    ax3.bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7)
    ax3.set_title('Average Flow Rate by Hour', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Avg Flow Rate (L/min)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = plt.subplot(3, 2, 4)
    ax4.scatter(test_sample[test_sample['label']==0]['flow_rate'], 
                test_sample[test_sample['label']==0]['pressure'], 
                alpha=0.3, s=1, label='Normal', color='blue')
    ax4.scatter(test_sample[test_sample['label']==1]['flow_rate'], 
                test_sample[testy')
    
    # Plot 6: Monthly Statistics
    ax6 = plt.subplot(3, 2, 6)
    df['month'] = df['timestamp'].dt.month
    phase2_df['month'] = phase2_df['timestamp'].dt.month
    monthly_flow = df.groupby('month')['flow_rate'].mean()
    monthly_leaks = phase2_df.groupby('month')['label'].sum()
    
    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(train_sample['turbidity'], bins=50, alpha=0.7, label='Training', color='green')
    ax5.hist(test_sample[test_sample['label']==1]['turbidity'], bins=30, alpha=0.7, label='Test
    ax6.set_xlabel('Month')
    ax6.set_ylabel('Avg Flow Rate (L/min)', color='steelblue')
    ax6_twin.set_ylabel('Leak Sample Count', color='red')
    ax6.tick_params(axis='y', labelcolor='steelblue')
    ax6_twin.tick_params(axis='y', labelcolor='red')
    ax6.grid(True, alpha=0.3, axis='y')
    
    ax6 = plt.subplot(3, 2, 6)
    train_df['month'] = train_df['timestamp'].dt.month
    test_df['month'] = test_df['timestamp'].dt.month
    monthly_flow = train_df.groupby('month')['flow_rate'].mean()
    monthly_leaks = test.subplots(2, 1, figsize=(16, 8))
    
    # Downsample for visualization (every 60 minutes)
    downsample = 60
    df_down = df.iloc[::downsample]
    phase2_Saved: water_flow_analysis.png")
    
    fig2, (ax7, ax8) = plt.subplots(2, 1, figsize=(16, 8))
    
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
    ax8.set_title('12-Month Timeline - Test Datatight')
    print("  Saved visualization: water_flow_timeline.png")
    
    plt.close('all')

# Generate datasets
phase1_df = generate_phase1_data()
phase2_df = generate_phase2_data(phase1_df)

# Save datasets
print("\n--- Saving Datasets ---")
phase1_df.tSaved: water_flow_timeline.png")
    plt.close('all')

train_df = generate_training_data()
test_df = generate_test_data(train_df)

print("\nSaving datasets...")
train_df.to_csv('water_train.csv', index=False)
print(f"Saved: water_train.csv ({len(train_df):,} samples)")

test_df.to_csv('water_test.csv', index=False)
print(f"Saved: water_test.csv ({len(test_df):,} samples)")

visualize_data(train_df, test_df)

print(f"\nComplete: {len(train_df):,} samples, 12 months, 1 sample/minute"