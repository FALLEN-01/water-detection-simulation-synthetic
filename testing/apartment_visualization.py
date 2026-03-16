"""
Generate comprehensive leak detection performance visualization for apartment building.
Creates a 2x2 subplot figure showing:
1. Confusion Matrix
2. Detection Performance (Recall, Precision, F1, Specificity)
3. Detection Delay (Min, Avg, Max)
4. Alert Counts (True Positives, False Positives)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json


def load_apartment_results(results_dir):
    """Load apartment test results from CSV."""
    results_csv = Path(results_dir) / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    df = pd.read_csv(results_csv)
    return df


def calculate_metrics(results_df):
    """Calculate detection metrics from results dataframe."""
    # Detection rate
    detected = sum(results_df["detection_time"].notna())
    detection_rate = detected / len(results_df) if len(results_df) > 0 else 0.0
    
    # False alarm rate
    false_alarms = sum(results_df["false_alarm"])
    false_alarm_rate = false_alarms / len(results_df) if len(results_df) > 0 else 0.0
    
    # Average detection delay (only for successful detections, convert to minutes)
    delays = results_df[results_df["delay"].notna()]["delay"].values
    avg_delay_minutes = np.mean(delays) if len(delays) > 0 else 0.0
    min_delay_minutes = np.min(delays) if len(delays) > 0 else 0.0
    max_delay_minutes = np.max(delays) if len(delays) > 0 else 0.0
    
    # Build confusion matrix components
    tp = sum((results_df["detection_time"].notna()) & (~results_df["false_alarm"]))
    fp = sum(results_df["false_alarm"])
    fn = sum(results_df["missed_detection"])
    tn = len(results_df) - tp - fp - fn
    
    # Calculate standard metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(results_df) if len(results_df) > 0 else 0.0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "avg_delay_minutes": avg_delay_minutes,
        "min_delay_minutes": min_delay_minutes,
        "max_delay_minutes": max_delay_minutes,
    }


def create_comprehensive_visualization(metrics, output_path):
    """
    Create a comprehensive 2x2 subplot figure showing all key metrics.
    Matches the style of the reference image provided.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Confusion Matrix (top-left)
    # ─────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array([
        [metrics["tp"], metrics["fp"]],
        [metrics["fn"], metrics["tn"]]
    ])
    
    # Create heatmap manually without seaborn
    im = ax1.imshow(cm, cmap="YlOrRd", aspect='auto')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Leak", "Normal"])
    ax1.set_yticklabels(["Leak", "Normal"])
    ax1.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax1.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{int(cm[i, j])}',
                          ha="center", va="center", color="black",
                          fontsize=14, fontweight="bold")
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Detection Performance Metrics (top-right)
    # ─────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_labels = ["Recall", "Precision", "F1", "Specificity"]
    metrics_values = [
        metrics["recall"],
        metrics["precision"],
        metrics["f1_score"],
        metrics["specificity"]
    ]
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax2.bar(metrics_labels, metrics_values, color=colors, width=0.6, edgecolor="black", linewidth=1.5)
    ax2.set_ylim([0, 1.05])
    ax2.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax2.set_title("Detection Performance", fontsize=13, fontweight="bold", pad=15)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )
    
    # ─────────────────────────────────────────────────────────────────
    # 3. Detection Delay (bottom-left)
    # ─────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    delay_labels = ["Min", "Avg", "Max"]
    delay_values = [
        metrics["min_delay_minutes"],
        metrics["avg_delay_minutes"],
        metrics["max_delay_minutes"]
    ]
    
    bars = ax3.bar(delay_labels, delay_values, color="#1f77b4", width=0.5, edgecolor="black", linewidth=1.5)
    ax3.set_ylabel("Minutes", fontsize=12, fontweight="bold")
    ax3.set_title("Detection Delay (minutes)", fontsize=13, fontweight="bold", pad=15)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add value labels on bars
    for bar, value in zip(bars, delay_values):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(delay_values) * 0.02,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Alert Counts (bottom-right)
    # ─────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    alert_labels = ["True Positives", "False Positives"]
    alert_values = [metrics["tp"], metrics["fp"]]
    
    colors_alert = ["#2ca02c", "#d62728"]
    bars = ax4.bar(alert_labels, alert_values, color=colors_alert, width=0.5, edgecolor="black", linewidth=1.5)
    ax4.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax4.set_title("Alert Counts", fontsize=13, fontweight="bold", pad=15)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add value labels on bars
    for bar, value in zip(bars, alert_values):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(alert_values) * 0.02,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )
    
    # Add overall title
    fig.suptitle(
        "Apartment Building Leak Detection Performance",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


def save_metrics_summary(metrics, output_path):
    """Save detailed metrics summary as text file."""
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("APARTMENT BUILDING LEAK DETECTION TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DETECTION METRICS:\n")
        f.write(f"  Detection Rate:        {metrics['detection_rate']:.2%}\n")
        f.write(f"  False Alarm Rate:      {metrics['false_alarm_rate']:.2%}\n")
        f.write(f"  Avg Detection Delay:   {metrics['avg_delay_minutes']:.2f} minutes\n")
        f.write(f"  Min Detection Delay:   {metrics['min_delay_minutes']:.2f} minutes\n")
        f.write(f"  Max Detection Delay:   {metrics['max_delay_minutes']:.2f} minutes\n\n")
        
        f.write("ACCURACY METRICS:\n")
        f.write(f"  Accuracy:              {metrics['accuracy']:.2%}\n")
        f.write(f"  Precision:             {metrics['precision']:.2%}\n")
        f.write(f"  Recall (Sensitivity):  {metrics['recall']:.2%}\n")
        f.write(f"  Specificity:           {metrics['specificity']:.2%}\n")
        f.write(f"  F1-Score:              {metrics['f1_score']:.2f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  True Positives:        {metrics['tp']}\n")
        f.write(f"  False Positives:       {metrics['fp']}\n")
        f.write(f"  False Negatives:       {metrics['fn']}\n")
        f.write(f"  True Negatives:        {metrics['tn']}\n")
    
    print(f"✓ Metrics summary saved to {output_path}")


def main():
    """Main execution function."""
    results_dir = Path(__file__).parent / "results" / "apartment"
    
    # Load and validate results
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    try:
        results_df = load_apartment_results(results_dir)
        print(f"✓ Loaded {len(results_df)} test results")
        
        # Calculate metrics
        metrics = calculate_metrics(results_df)
        print(f"✓ Calculated metrics")
        
        # Create visualizations
        output_figure = results_dir / "apartment_performance_dashboard.png"
        create_comprehensive_visualization(metrics, output_figure)
        
        # Save metrics summary
        output_summary = results_dir / "apartment_metrics_detailed.txt"
        save_metrics_summary(metrics, output_summary)
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("APARTMENT BUILDING LEAK DETECTION PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"\nDetection Performance:")
        print(f"  Recall (TP Detection Rate):  {metrics['recall']:.1%}")
        print(f"  Precision (False Positive):  {metrics['precision']:.1%}")
        print(f"  F1-Score:                    {metrics['f1_score']:.2f}")
        print(f"  Specificity:                 {metrics['specificity']:.1%}")
        print(f"\nDetection Timing:")
        print(f"  Average Delay:               {metrics['avg_delay_minutes']:.1f} minutes")
        print(f"  Min/Max Delay:               {metrics['min_delay_minutes']:.1f} / {metrics['max_delay_minutes']:.1f} minutes")
        print(f"\nAlert Statistics:")
        print(f"  True Positives:              {metrics['tp']}")
        print(f"  False Positives:             {metrics['fp']}")
        print(f"  False Negatives:             {metrics['fn']}")
        print(f"  True Negatives:              {metrics['tn']}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
