"""
Metrics calculation and visualization module for testing framework.
Generates confusion matrices, accuracy dashboards, and individual run plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def calculate_metrics(results):
    """
    Calculate detection rate, false alarm rate, average delay, and accuracy metrics.
    
    Args:
        results: List of dicts with keys: leak_time, detection_time, false_alarm, missed_detection
    
    Returns:
        Dict with all computed metrics
    """
    
    # Detection rate
    detected = sum(1 for r in results if r["detection_time"] is not None)
    detection_rate = detected / len(results) if len(results) > 0 else 0.0
    
    # False alarm rate
    false_alarms = sum(1 for r in results if r["false_alarm"])
    false_alarm_rate = false_alarms / len(results) if len(results) > 0 else 0.0
    
    # Average detection delay (only for successful detections)
    delays = [r["delay"] for r in results if r["delay"] is not None]
    avg_delay = sum(delays) / len(delays) if len(delays) > 0 else 0.0
    
    # Build confusion matrix components
    tp = sum(1 for r in results if r["detection_time"] is not None and not r["false_alarm"])
    fp = sum(1 for r in results if r["false_alarm"])
    fn = sum(1 for r in results if r["missed_detection"])
    tn = len(results) - tp - fp - fn
    
    # Calculate standard metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0.0
    
    return {
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "avg_delay_seconds": avg_delay,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def plot_confusion_matrix(metrics, output_path):
    """
    Plot and save confusion matrix heatmap.
    """
    cm = np.array([
        [metrics["tp"], metrics["fp"]],
        [metrics["fn"], metrics["tn"]]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Detected", "Not Detected"],
        yticklabels=["Actual Leak", "No Leak"],
        ax=ax,
        cbar_kws={"label": "Count"}
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix - Leak Detection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_dashboard(metrics, output_path):
    """
    Plot bar chart with accuracy metrics: precision, recall, F1, detection rate.
    """
    metrics_data = {
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-Score": metrics["f1_score"],
        "Detection Rate": metrics["detection_rate"],
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics_data.keys(), metrics_data.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    ax.set_ylim([0, 1.0])
    ax.set_ylabel("Score")
    ax.set_title("Detection Accuracy Metrics")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_individual_run(run_data, output_path):
    """
    Plot individual run trace: flow signal + CUSUM + leak/detection markers.
    
    Args:
        run_data: Dict with keys: 'flow_history', 'cusum_history', 'leak_time', 'detection_time'
        output_path: File path to save plot
    """
    flow = np.array(run_data["flow_history"])
    cusum = np.array(run_data["cusum_history"])
    leak_time = run_data["leak_time"]
    detection_time = run_data["detection_time"]
    
    # Normalize CUSUM for visibility on same plot
    cusum_norm = cusum / (np.max(cusum) + 1e-6) if np.max(cusum) > 0 else cusum
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot flow
    ax1.plot(flow, label="Flow (L/min)", color="blue", linewidth=1.5, alpha=0.7)
    if leak_time is not None:
        ax1.axvline(leak_time, color="red", linestyle="--", linewidth=2, label=f"Leak Injected (min {leak_time})")
    if detection_time is not None:
        ax1.axvline(detection_time, color="green", linestyle="--", linewidth=2, label=f"Detection (min {detection_time})")
    ax1.set_ylabel("Flow (L/min)")
    ax1.set_title("Water Flow Signal")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    
    # Plot CUSUM
    ax2.plot(cusum_norm, label="CUSUM (normalized)", color="orange", linewidth=1.5, alpha=0.7)
    if leak_time is not None:
        ax2.axvline(leak_time, color="red", linestyle="--", linewidth=2)
    if detection_time is not None:
        ax2.axvline(detection_time, color="green", linestyle="--", linewidth=2)
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("CUSUM (normalized)")
    ax2.set_title("CUSUM Anomaly Score")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_results_csv(results, output_path):
    """
    Save experiment results to CSV.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def save_metrics_summary(metrics, output_path):
    """
    Save metrics summary as human-readable text file.
    """
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("LEAK DETECTION TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DETECTION METRICS:\n")
        f.write(f"  Detection Rate:        {metrics['detection_rate']:.2%}\n")
        f.write(f"  False Alarm Rate:      {metrics['false_alarm_rate']:.2%}\n")
        f.write(f"  Avg Detection Delay:   {metrics['avg_delay_seconds']:.2f} seconds\n\n")
        
        f.write("ACCURACY METRICS:\n")
        f.write(f"  Accuracy:              {metrics['accuracy']:.2%}\n")
        f.write(f"  Precision:             {metrics['precision']:.2%}\n")
        f.write(f"  Recall (Sensitivity):  {metrics['recall']:.2%}\n")
        f.write(f"  F1-Score:              {metrics['f1_score']:.2f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  True Positives:        {metrics['tp']}\n")
        f.write(f"  False Positives:       {metrics['fp']}\n")
        f.write(f"  False Negatives:       {metrics['fn']}\n")
        f.write(f"  True Negatives:        {metrics['tn']}\n")
    
    print(f"Summary saved to {output_path}")
