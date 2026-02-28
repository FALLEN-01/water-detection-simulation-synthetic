"""
Water Leak Detection — Synthetic Data Generator
================================================
Generates training and test data from India-based appliance priors.

Strategy for generalization:
  - TRAINING: N_BUILDINGS different building configs, each with different
    appliance availability, household sizes, apartment counts.
    Covers diverse seasonal patterns across the full year.
  - TESTING:  2 completely unseen building configs (different seeds),
    covering only Q3 (summer-to-monsoon transition) with leaks injected.

This teaches the model "what normal looks like across all buildings"
so it generalizes to any new apartment deployment.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os

# =====================================================
# Configuration
# =====================================================
MAX_FLOW = 15.0          # L/min ceiling (building main meter)
SAMPLES_PER_DAY = 1440   # 1 sample per minute

# Training: 5 different buildings × 60 days = 300 days of diverse data
N_TRAIN_BUILDINGS = 5
TRAIN_DAYS_PER_BUILDING = 60    # 2 months per building

# Test: 2 unseen buildings × 45 days each
N_TEST_BUILDINGS = 2
TEST_DAYS_PER_BUILDING = 45

# Building configurations — each building has unique characteristics
# These MUST NOT overlap with seeds used for test buildings (seeds 200+)
TRAIN_BUILDING_SEEDS = [10, 20, 30, 40, 50]
# Test buildings use different seeds AND different start dates
TEST_BUILDING_SEEDS  = [201, 202]

print("=" * 60)
print("Water Leak Detection — Data Generator")
print(f"Training: {N_TRAIN_BUILDINGS} buildings × {TRAIN_DAYS_PER_BUILDING} days")
print(f"Testing:  {N_TEST_BUILDINGS} buildings × {TEST_DAYS_PER_BUILDING} days (with leaks)")
print("=" * 60)

# =====================================================
# Load Priors
# =====================================================
def load_india_priors():
    priors = {}
    priors_dir = 'priors_india'
    if not os.path.exists(priors_dir):
        raise FileNotFoundError(f"priors_india/ not found — cannot generate data")
    for fname in os.listdir(priors_dir):
        if fname.endswith('.json'):
            name = fname.replace('.json', '')
            with open(os.path.join(priors_dir, fname)) as f:
                priors[name] = json.load(f)
    print(f"Loaded {len(priors)} appliances: {sorted(priors.keys())}")
    return priors

PRIORS = load_india_priors()

# =====================================================
# Per-appliance flow sampler using actual prior distributions
# =====================================================
def sample_appliance_flow_lpm(prior):
    """
    Sample a single-event mean flow rate in L/min from the appliance prior.
    Priors store flow in ml/s, convert to L/min (* 0.06).
    Uses lognormal distribution parameters from the prior.
    """
    flow_prior = prior['flow']['mean_flow']
    if flow_prior['type'] == 'lognormal':
        # np.random.lognormal takes (mean, sigma) of underlying normal
        # prior stores 'scale' = median of lognormal, 'shape' = sigma
        median_ml_s = flow_prior['scale']
        sigma = flow_prior['shape'] if flow_prior['shape'] > 0 else 0.01
        # Draw from lognormal: median * exp(N(0, sigma))
        flow_ml_s = median_ml_s * np.exp(np.random.normal(0, sigma))
    else:
        flow_ml_s = flow_prior['scale']
    return max(0.01, flow_ml_s * 0.06)  # ml/s → L/min

def sample_appliance_duration_s(prior):
    """Sample event duration in seconds from prior"""
    dur = prior['duration']
    if dur['type'] == 'lognormal':
        median_s = dur['scale']
        sigma = dur['shape'] if dur['shape'] > 0 else 0.01
        return max(10.0, median_s * np.exp(np.random.normal(0, sigma)))
    elif dur['type'] == 'fixed':
        return dur['value']
    return 60.0

def events_per_minute(prior, hour, availability):
    """
    Expected number of active events at this minute for this appliance.
    = P(start at this hour) × events_per_day × availability / 60
    """
    hour_prob = prior['timing']['start_hour']['p'][hour]
    lambda_day = prior['activation']['events_per_day']['lambda']
    return hour_prob * lambda_day * availability / 60.0

# =====================================================
# Building profile generator
# =====================================================
def make_building_profile(seed):
    """
    Generate a unique apartment building profile.
    Returns dict with appliance availability, count, household size.
    """
    rng = np.random.RandomState(seed)
    r = rng.uniform

    n_apartments = int(rng.randint(30, 100))     # 30–100 apartments
    avg_household = float(r(2.2, 4.2))           # 2.2–4.2 people per apt

    availability = {
        'toilet':         float(r(0.97, 1.00)),  # near-universal
        'washbasin':      float(r(0.96, 1.00)),
        'kitchenfaucet':  float(r(0.95, 1.00)),
        'shower':         float(r(0.75, 0.98)),  # most have
        'washingmachine': float(r(0.40, 0.82)),  # varies a lot
        'bidet':          float(r(0.05, 0.45)),  # less common
        'dishwasher30':   float(r(0.02, 0.25)),  # rare
    }

    # Cooking behaviour — affects kitchenfaucet peak timing
    cooking_peak_hours = {
        'early_cook':   [6, 7, 11, 18],    # typical middle-class
        'late_cook':    [8, 13, 20],        # working families
        'extended_cook':[7, 12, 19, 21],   # joint family
    }
    style = rng.choice(list(cooking_peak_hours.keys()))

    return {
        'seed':            seed,
        'n_apartments':    n_apartments,
        'avg_household':   avg_household,
        'availability':    availability,
        'cooking_style':   style,
        'cooking_peaks':   cooking_peak_hours[style],
    }

# =====================================================
# Occupancy model (time + season + building)
# =====================================================
def get_occupancy(hour, dow, month, profile, rng):
    """Fraction of apartments occupied at a given minute"""
    # Season baseline
    if month in [4, 5, 6]:       base = rng.uniform(0.65, 0.80)  # summer (holidays/heat)
    elif month in [12, 1]:       base = rng.uniform(0.60, 0.78)  # winter (travel season)
    elif month in [7, 8, 9]:     base = rng.uniform(0.78, 0.92)  # monsoon (stay-in)
    else:                        base = rng.uniform(0.75, 0.90)  # other

    # Time-of-day modifier
    if dow < 5:  # weekday
        if   6 <= hour < 9:    mod = rng.uniform(1.1, 1.3)   # morning rush
        elif 9 <= hour < 17:   mod = rng.uniform(0.5, 0.75)  # most at work
        elif 17 <= hour < 22:  mod = rng.uniform(1.05, 1.2)  # evening return
        else:                   mod = rng.uniform(0.88, 1.0)  # night
    else:  # weekend
        if   8 <= hour < 22:   mod = rng.uniform(1.1, 1.3)
        else:                   mod = rng.uniform(0.90, 1.05)

    return min(0.98, max(0.40, base * mod))

# =====================================================
# Core per-minute flow calculator
# =====================================================
def calc_flow_at_minute(hour, dow, month, profile, rng):
    """
    Compute the aggregate building flow rate (L/min) at one minute.
    Uses actual prior flow distributions, appliance concurrency, and occupancy.
    """
    occ = get_occupancy(hour, dow, month, profile, rng)
    total_people = profile['n_apartments'] * occ * profile['avg_household']

    # Cooking time boost for kitchen appliances
    in_cooking_peak = hour in profile['cooking_peaks']

    agg_flow = 0.0

    for name, prior in PRIORS.items():
        avail = profile['availability'].get(name, 0.0)
        if avail < 0.01:
            continue

        # Expected concurrent events at this minute for the whole building
        # = events_per_minute × n_apartments × availability × occupancy
        exp_events = events_per_minute(prior, hour, avail) * total_people

        # Kitchen faucet boost during cooking peaks
        if name in ('kitchenfaucet', 'dishwasher30') and in_cooking_peak:
            exp_events *= rng.uniform(1.5, 2.2)

        if exp_events < 1e-4:
            continue

        # Sample actual flow from prior distribution
        flow_per_event = sample_appliance_flow_lpm(prior)
        duration_s = sample_appliance_duration_s(prior)

        # Fraction of a minute this event occupies (duration/60, capped at 1)
        time_fraction = min(1.0, duration_s / 60.0)

        # Contribution = concurrent_events × flow × time_fraction
        agg_flow += exp_events * flow_per_event * time_fraction

    # Baseline: common areas, leaky taps, misc (0.1–0.3% of max flow)
    agg_flow += rng.uniform(0.05, 0.25) * (total_people / 150.0)

    # Sensor noise
    agg_flow += rng.normal(0, 0.06)

    return float(np.clip(agg_flow, 0.02, MAX_FLOW))

# =====================================================
# Data generator for one building, one period
# =====================================================
def generate_building_data(profile, start_date, num_days, label_leaks=False,
                           daily_seed_offset=0):
    """Generate `num_days` × 1440 samples for one building from priors."""
    rng = np.random.RandomState(profile['seed'] + daily_seed_offset)
    # Per-day usage multiplier (some days heavier, some lighter)
    day_mult = np.clip(rng.normal(1.0, 0.11, num_days), 0.72, 1.28)

    data = []
    total = num_days * SAMPLES_PER_DAY

    for i in range(total):
        ts  = start_date + timedelta(minutes=i)
        h   = ts.hour
        dow = ts.weekday()
        mon = ts.month
        day_idx = i // SAMPLES_PER_DAY

        flow = calc_flow_at_minute(h, dow, mon, profile, rng)
        flow *= day_mult[day_idx]

        # Turbidity — correlated with flow rate, slight own noise
        turbidity = 0.35 + (flow / MAX_FLOW) * 1.0 + rng.normal(0, 0.10)
        turbidity = float(np.clip(turbidity, 0.05, 4.0))

        flow_normalized = float(np.clip(flow / MAX_FLOW, 0.0, 1.0))
        flow_duration   = (i % SAMPLES_PER_DAY) * 60  # seconds into current day

        data.append({
            'timestamp':       ts,
            'flow_rate':       round(flow, 4),
            'flow_normalized': round(flow_normalized, 4),
            'turbidity':       round(turbidity, 4),
            'flow_duration':   flow_duration,
            'hour':            h,
            'day_of_week':     dow,
            'month':           mon,
            'is_weekend':      int(dow >= 5),
            'label':           0,
        })

        if (i + 1) % 50000 == 0:
            pct = (i + 1) / total * 100
            print(f"    {i+1:,}/{total:,} ({pct:.0f}%)...")

    df = pd.DataFrame(data)

    if label_leaks:
        df = inject_leaks(df, rng)

    return df

# =====================================================
# Leak injection — only for test data
# =====================================================
def inject_leaks(df, rng):
    """
    Inject physically plausible leaks into a dataframe.
    Three types:
      - Stress leaks: during high-flow peaks (pipe pressure)
      - Night leaks:  continuous anomalous flow in quiet hours
      - Seasonal:     winter pipe failures (contraction/expansion)
    """
    leak_locs = []

    # Stress leaks during peaks
    peak_h = [7, 8, 19, 20, 21]
    peak_cands = df[(df['hour'].isin(peak_h)) &
                    (df['flow_rate'] > df['flow_rate'].quantile(0.78))].index.tolist()
    if len(peak_cands) >= 3:
        leak_locs.extend(rng.choice(peak_cands, 3, replace=False).tolist())

    # Night leaks
    night_cands = df[df['hour'].isin([0, 1, 2, 3, 4])].index.tolist()
    if len(night_cands) >= 3:
        leak_locs.extend(rng.choice(night_cands, 3, replace=False).tolist())

    # Seasonal leak (winter months)
    winter_cands = df[df['month'].isin([1, 2, 12])].index.tolist()
    if len(winter_cands) >= 2:
        leak_locs.extend(rng.choice(winter_cands, 2, replace=False).tolist())

    print(f"    Injecting {len(leak_locs)} leak events...")

    for start_idx in leak_locs:
        row = df.iloc[start_idx]
        h_val = row['hour']
        mon_val = row['month']

        if h_val in peak_h:
            duration = int(rng.randint(720, 3600))    # 12h–2.5 days
            severity = float(rng.uniform(0.18, 0.38))
        elif h_val < 5:
            duration = int(rng.randint(2160, 6480))   # 1.5–4.5 days
            severity = float(rng.uniform(0.10, 0.25))
        else:
            duration = int(rng.randint(180, 2160))    # 3h–1.5 days
            severity = float(rng.uniform(0.22, 0.42))

        end_idx = min(start_idx + duration, len(df))

        for k in range(start_idx, end_idx):
            base = df.at[k, 'flow_rate']
            if base < 0.15:
                # Pure leak during silent period — most detectable
                new_flow = severity * MAX_FLOW
            else:
                # Additive leak during usage
                new_flow = min(MAX_FLOW, base + severity * MAX_FLOW * 0.28)
            df.at[k, 'flow_rate']       = round(new_flow, 4)
            df.at[k, 'flow_normalized'] = round(new_flow / MAX_FLOW, 4)
            df.at[k, 'flow_duration']   = int((k - start_idx + 1) * 60)
            df.at[k, 'turbidity']       = round(
                min(4.0, df.at[k, 'turbidity'] + float(rng.uniform(0.35, 1.6))), 4
            )
            df.at[k, 'label'] = 1

    leak_n = (df['label'] == 1).sum()
    pct = leak_n / len(df) * 100
    print(f"    Leak samples: {leak_n:,} ({pct:.2f}%)")
    return df

# =====================================================
# Validation
# =====================================================
def validate(df, name):
    print(f"\n--- {name} Validation ---")
    df2 = df.copy()
    df2['_date'] = df2['timestamp'].dt.date
    avg_daily = df2.groupby('_date')['flow_rate'].sum().mean()
    # estimate people from first row building size (n_apartments×avg_household×occ)
    avg_per_person = avg_daily / 150   # approximate 150 people
    peak = df2['flow_rate'].quantile(0.99)
    hourly = df2.groupby('hour')['flow_rate'].mean()
    morning = hourly.loc[6:8].mean()
    night = hourly.loc[0:5].mean()
    ratio = morning / night if night > 0 else 0
    print(f"  Avg daily flow: {avg_daily:.0f} L | Est. per person: {avg_per_person:.1f} L/day")
    print(f"  Peak (p99): {peak:.2f} L/min | Morning/Night ratio: {ratio:.1f}x")
    ok = "✓ OK" if 80 < avg_per_person < 250 else "⚠ CHECK"
    print(f"  Realism check: {ok}")

# =====================================================
# Visualizations
# =====================================================
def visualize(train_df, test_df):
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    s7 = SAMPLES_PER_DAY * 7
    tr = train_df.iloc[:s7]
    te = test_df.iloc[:s7]

    # Plot 1: Training flow (first week)
    axes[0, 0].plot(tr['timestamp'], tr['flow_rate'], lw=0.5, alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Training: Flow Rate (first 7 days, building 1)', fontweight='bold')
    axes[0, 0].set_ylabel('L/min'); axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Test flow with leaks
    normal = te[te['label'] == 0]; leaks = te[te['label'] == 1]
    axes[0, 1].plot(normal['timestamp'], normal['flow_rate'], lw=0.5, alpha=0.7, color='blue', label='Normal')
    if len(leaks): axes[0, 1].plot(leaks['timestamp'], leaks['flow_rate'], lw=0.8, color='red', label='Leak')
    axes[0, 1].set_title('Test: Flow with Leaks (first 7 days)', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Avg hourly (across all training buildings)
    hourly = train_df.groupby('hour')['flow_rate'].mean()
    axes[1, 0].bar(hourly.index, hourly.values, color='steelblue', alpha=0.7)
    axes[1, 0].set_title('Avg Flow by Hour (all training buildings)', fontweight='bold')
    axes[1, 0].set_xlabel('Hour'); axes[1, 0].set_ylabel('L/min'); axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Flow distribution
    axes[1, 1].hist(test_df[test_df['label']==0]['flow_rate'], bins=80, alpha=0.6, label='Normal', color='blue', density=True)
    axes[1, 1].hist(test_df[test_df['label']==1]['flow_rate'], bins=50, alpha=0.7, label='Leak', color='red', density=True)
    axes[1, 1].set_title('Test Flow Distribution (Normal vs Leak)', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Turbidity
    axes[2, 0].hist(train_df.sample(min(50000, len(train_df)))['turbidity'], bins=60, alpha=0.7, label='Train', color='green')
    axes[2, 0].hist(test_df[test_df['label']==1]['turbidity'], bins=40, alpha=0.7, label='Test Leak', color='red')
    axes[2, 0].set_title('Turbidity Distribution', fontweight='bold')
    axes[2, 0].legend(); axes[2, 0].grid(True, alpha=0.3, axis='y')

    # Plot 6: Monthly avg (seasonality)
    monthly = train_df.groupby('month')['flow_rate'].mean()
    axes[2, 1].bar(monthly.index, monthly.values, color='steelblue', alpha=0.7)
    axes[2, 1].set_title('Monthly Avg Flow (seasonal pattern)', fontweight='bold')
    axes[2, 1].set_xlabel('Month'); axes[2, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('water_flow_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: water_flow_analysis.png")
    plt.close()

    # Timeline
    fig2, (ax7, ax8) = plt.subplots(2, 1, figsize=(16, 8))
    ds = 60
    ax7.plot(train_df.iloc[::ds]['timestamp'], train_df.iloc[::ds]['flow_rate'], lw=0.3, alpha=0.6, color='steelblue')
    ax7.set_title(f'Full Training Timeline ({N_TRAIN_BUILDINGS} buildings, {N_TRAIN_BUILDINGS*TRAIN_DAYS_PER_BUILDING} days)', fontweight='bold')
    ax7.set_ylabel('L/min'); ax7.grid(True, alpha=0.3)

    nd = test_df[test_df['label']==0].iloc[::ds]
    ld = test_df[test_df['label']==1].iloc[::ds]
    ax8.plot(nd['timestamp'], nd['flow_rate'], lw=0.3, alpha=0.6, color='blue', label='Normal')
    if len(ld): ax8.scatter(ld['timestamp'], ld['flow_rate'], s=1.5, alpha=0.8, color='red', label='Leak')
    ax8.set_title(f'Test Timeline ({N_TEST_BUILDINGS} unseen buildings, {N_TEST_BUILDINGS*TEST_DAYS_PER_BUILDING} days, with leaks)', fontweight='bold')
    ax8.set_ylabel('L/min'); ax8.legend(); ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('water_flow_timeline.png', dpi=150, bbox_inches='tight')
    print("  Saved: water_flow_timeline.png")
    plt.close()

    # Seasonal 24-hour patterns (one building, different months)
    fig3 = plt.figure(figsize=(15, 10))
    months_present = sorted(train_df['month'].unique())[:6]
    for idx, m in enumerate(months_present, 1):
        ax = plt.subplot(2, 3, idx)
        mdata = train_df[train_df['month'] == m]
        mid = len(mdata) // 2
        s0 = mid - (mid % SAMPLES_PER_DAY)
        day = mdata.iloc[s0:s0 + SAMPLES_PER_DAY]
        if len(day) > 10:
            ax.plot([i/60 for i in range(len(day))], day['flow_rate'].values, lw=0.7, color='steelblue')
            ax.set_title(day.iloc[0]['timestamp'].strftime('%B %Y'), fontweight='bold', fontsize=10)
            ax.set_xlabel('Hour'); ax.set_ylabel('L/min')
            ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
            ax.axvspan(0, 6, alpha=0.07, color='gray')
            ax.grid(True, alpha=0.3)
    plt.suptitle('24-Hour Patterns by Month (Seasonal Variation)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('water_flow_24hour_patterns.png', dpi=150, bbox_inches='tight')
    print("  Saved: water_flow_24hour_patterns.png")
    plt.close('all')

# =====================================================
# MAIN — Generate all data
# =====================================================

# --- TRAINING DATA ---
# 5 different buildings, spread across different months (covering full year)
print("\n=== Generating TRAINING data ===")
print(f"Strategy: {N_TRAIN_BUILDINGS} buildings × {TRAIN_DAYS_PER_BUILDING} days")
print("(Different appliance mixes, household sizes, cooking styles)\n")

TRAIN_STARTS = [
    datetime(2025, 1, 1),   # Jan–Feb  (winter)
    datetime(2025, 3, 1),   # Mar–Apr  (spring/summer)
    datetime(2025, 5, 1),   # May–Jun  (peak summer)
    datetime(2025, 7, 1),   # Jul–Aug  (monsoon)
    datetime(2025, 9, 1),   # Sep–Oct  (post-monsoon)
]

train_frames = []
for i, (seed, start) in enumerate(zip(TRAIN_BUILDING_SEEDS, TRAIN_STARTS)):
    profile = make_building_profile(seed)
    print(f"Building {i+1}: seed={seed} | {profile['n_apartments']} apts | "
          f"{profile['avg_household']:.1f} ppl/apt | "
          f"cooking={profile['cooking_style']} | "
          f"start={start.strftime('%b %Y')}")
    for k, v in profile['availability'].items():
        print(f"  {k}: {v*100:.0f}%", end='  ')
    print()
    df = generate_building_data(
        profile, start,
        num_days=TRAIN_DAYS_PER_BUILDING,
        label_leaks=False,
        daily_seed_offset=seed * 100
    )
    train_frames.append(df)
    print(f"  -> {len(df):,} samples\n")

train_df = pd.concat(train_frames, ignore_index=True)
print(f"Total training samples: {len(train_df):,}")
validate(train_df, "Training (all buildings combined)")

# --- TEST DATA ---
# 2 completely unseen buildings with leaks injected
print("\n=== Generating TEST data (unseen buildings + leaks) ===")
print(f"Strategy: {N_TEST_BUILDINGS} NEW buildings × {TEST_DAYS_PER_BUILDING} days\n")

TEST_STARTS = [
    datetime(2025, 11, 1),  # Nov–Dec (winter chill — seasonal leak risk)
    datetime(2025, 4, 15),  # Apr–May (summer heat — stress leak risk)
]

test_frames = []
for i, (seed, start) in enumerate(zip(TEST_BUILDING_SEEDS, TEST_STARTS)):
    profile = make_building_profile(seed)
    print(f"Test Building {i+1}: seed={seed} | {profile['n_apartments']} apts | "
          f"{profile['avg_household']:.1f} ppl/apt | "
          f"cooking={profile['cooking_style']} | "
          f"start={start.strftime('%b %Y')}")
    df = generate_building_data(
        profile, start,
        num_days=TEST_DAYS_PER_BUILDING,
        label_leaks=True,
        daily_seed_offset=seed * 77
    )
    test_frames.append(df)
    print(f"  -> {len(df):,} samples\n")

test_df = pd.concat(test_frames, ignore_index=True)
print(f"Total test samples: {len(test_df):,}")
validate(test_df, "Test (unseen buildings)")

# --- SAVE ---
print("\nSaving datasets...")
os.makedirs("models", exist_ok=True)
train_df.to_csv('water_train.csv', index=False)
print(f"Saved: water_train.csv — {len(train_df):,} samples, no leaks")
test_df.to_csv('water_test.csv', index=False)
print(f"Saved: water_test.csv  — {len(test_df):,} samples, "
      f"{(test_df['label']==1).sum():,} leak samples "
      f"({(test_df['label']==1).sum()/len(test_df)*100:.2f}%)")

# Visualize
visualize(train_df, test_df)

print("\nData generation complete!")
print(f"  Train: {N_TRAIN_BUILDINGS} buildings, {len(train_df):,} samples")
print(f"  Test:  {N_TEST_BUILDINGS} buildings, {len(test_df):,} samples (with leaks)")