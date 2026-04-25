"""
Generate comprehensive visualization plots from sprint results.
Creates all charts from task artifacts and saves to outputs/plots/sprint/
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Paths
PROJECT_ROOT = Path(__file__).parent.resolve()
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots" / "sprint"
ARTIFACTS_DIR = PROJECT_ROOT / "outputs" / "artifacts"

# Ensure plots directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    'baseline': '#1f77b4',
    'attention': '#ff7f0e',
    'ensemble': '#2ca02c',
    'aggressive': '#d62728',
    'original': '#9467bd'
}

def load_data():
    """Load all sprint artifacts."""
    print("Loading sprint artifacts...")
    
    # Task A: Blend sweep
    task_a = pd.read_csv(REPORTS_DIR / "task_a_blend_sweep.csv")
    
    # Task D: Loss tuning
    task_d = pd.read_csv(REPORTS_DIR / "task_d_loss_tuning.csv")
    
    # Task C: Ensemble weights
    task_c = pd.read_csv(REPORTS_DIR / "task_c_ensemble_weights.csv")
    
    # Final report
    with open(REPORTS_DIR / "FINAL_SPRINT_REPORT.json", "r") as f:
        final_report = json.load(f)
    
    print("✓ All artifacts loaded successfully")
    return task_a, task_d, task_c, final_report

def plot_blend_sweep(task_a):
    """Plot Task A: Blend sweep results with multiple metrics."""
    print("\n📊 Generating Task A: Blend Sweep Visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task A: Blend Sweep Analysis\n(Baseline 0.0 → Attention 1.0)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Peak MAE (primary metric)
    axes[0, 0].plot(task_a['blend'], task_a['peak_mae'], 'o-', color=COLORS['baseline'], linewidth=2, markersize=6)
    axes[0, 0].axhline(y=25.0, color='green', linestyle='--', label='Target (≤25)', linewidth=1.5)
    axes[0, 0].set_xlabel('Blend Ratio (0=Baseline, 1=Attention)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Peak MAE', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Peak MAE vs Blend Ratio', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([20, 85])
    
    # Plot 2: Day MAE
    axes[0, 1].plot(task_a['blend'], task_a['day_mae'], 's-', color=COLORS['attention'], linewidth=2, markersize=6)
    axes[0, 1].axhline(y=35.0, color='orange', linestyle='--', label='Target (≤35)', linewidth=1.5)
    axes[0, 1].set_xlabel('Blend Ratio', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Day MAE', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Day MAE vs Blend Ratio', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([35, 140])
    
    # Plot 3: RMSE
    axes[1, 0].plot(task_a['blend'], task_a['rmse'], '^-', color=COLORS['ensemble'], linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Blend Ratio', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('RMSE', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('RMSE vs Blend Ratio', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: R² Score
    axes[1, 1].plot(task_a['blend'], task_a['r2'], 'd-', color=COLORS['original'], linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Blend Ratio', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('R² Score vs Blend Ratio', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0.87, 0.975])
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "sprint_task_a_blend_sweep.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def plot_loss_tuning(task_d):
    """Plot Task D: Loss function variant comparison."""
    print("📊 Generating Task D: Loss Tuning Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Task D: Loss Function Variant Comparison\n(Peak-weighted Huber vs Aggressive Variants)', 
                 fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(task_d))
    width = 0.2
    
    # Left plot: All metrics for each variant
    ax1 = axes[0]
    ax1.bar(x_pos - 1.5*width, task_d['rmse'], width, label='RMSE', color='#1f77b4')
    ax1.bar(x_pos - 0.5*width, task_d['mae'], width, label='MAE', color='#ff7f0e')
    ax1.bar(x_pos + 0.5*width, task_d['day_mae'], width, label='Day MAE', color='#2ca02c')
    ax1.bar(x_pos + 1.5*width, task_d['peak_mae'], width, label='Peak MAE', color='#d62728')
    
    ax1.set_xlabel('Loss Variant', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax1.set_title('All Metrics by Loss Variant', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(task_d['variant'], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Peak MAE focus with targets
    ax2 = axes[1]
    bars = ax2.bar(task_d['variant'], task_d['peak_mae'], color=['#9467bd', '#d62728', '#d62728'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=25.0, color='green', linestyle='--', linewidth=2, label='Target (≤25)')
    ax2.set_xlabel('Loss Variant', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Peak MAE', fontsize=11, fontweight='bold')
    ax2.set_title('Peak MAE by Loss Variant (with Target)', fontsize=12)
    ax2.set_ylim([20, 26])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, task_d['peak_mae'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "sprint_task_d_loss_tuning.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def plot_ensemble_weights(task_c):
    """Plot Task C: Ensemble weight optimization."""
    print("📊 Generating Task C: Ensemble Weight Optimization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Task C: Ensemble Weight Optimization\n(Baseline vs Attention Mix)', 
                 fontsize=16, fontweight='bold')
    
    # Left plot: Baseline and Attention weights
    ax1 = axes[0]
    baseline_weights = task_c['w_baseline'].values
    attention_weights = task_c['w_attention'].values
    weight_labels = [f"{b:.0%}" for b in baseline_weights]
    
    x_pos = np.arange(len(task_c))
    width = 0.35
    
    ax1.bar(x_pos - width/2, baseline_weights, width, label='Baseline Weight', color=COLORS['baseline'], alpha=0.8)
    ax1.bar(x_pos + width/2, attention_weights, width, label='Attention Weight', color=COLORS['attention'], alpha=0.8)
    ax1.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Weight', fontsize=11, fontweight='bold')
    ax1.set_title('Baseline vs Attention Weights', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"Config {i+1}" for i in range(len(task_c))], rotation=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.0])
    
    # Right plot: Peak MAE by configuration
    ax2 = axes[1]
    peak_mae_values = task_c['peak_mae'].values
    colors = [COLORS['baseline'] if b > 0.5 else COLORS['attention'] for b in baseline_weights]
    bars = ax2.bar(x_pos, peak_mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=25.0, color='green', linestyle='--', linewidth=2, label='Target (≤25)')
    ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Peak MAE', fontsize=11, fontweight='bold')
    ax2.set_title('Ensemble Peak MAE by Configuration', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{int(b*100)}%B/{int(a*100)}%A" for b, a in zip(baseline_weights, attention_weights)], rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, peak_mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "sprint_task_c_ensemble_weights.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def plot_final_metrics(final_report):
    """Plot Task E: Final metrics comparison across all models."""
    print("📊 Generating Task E: Final Metrics Comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task E: Final Model Comparison\n(All Sprint Results)', 
                 fontsize=16, fontweight='bold')
    
    results = final_report['all_results']
    model_names = list(results.keys())
    
    # Extract metrics
    rmse_vals = [results[m]['RMSE'] for m in model_names]
    mae_vals = [results[m]['MAE'] for m in model_names]
    day_mae_vals = [results[m]['Day_MAE'] for m in model_names]
    peak_mae_vals = [results[m]['Peak_MAE'] for m in model_names]
    
    x_pos = np.arange(len(model_names))
    
    # Plot 1: RMSE
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, rmse_vals, color=[COLORS['baseline'], COLORS['baseline'], COLORS['aggressive'], 
                                              COLORS['attention'], COLORS['ensemble']], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax1.set_title('RMSE Comparison', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: MAE
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, mae_vals, color=[COLORS['baseline'], COLORS['baseline'], COLORS['aggressive'], 
                                             COLORS['attention'], COLORS['ensemble']], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('MAE', fontsize=11, fontweight='bold')
    ax2.set_title('MAE Comparison', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Day MAE with target
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, day_mae_vals, color=[COLORS['baseline'], COLORS['baseline'], COLORS['aggressive'], 
                                                 COLORS['attention'], COLORS['ensemble']], alpha=0.7, edgecolor='black')
    ax3.axhline(y=35.0, color='orange', linestyle='--', linewidth=2, label='Target (≤35)')
    ax3.set_ylabel('Day MAE', fontsize=11, fontweight='bold')
    ax3.set_title('Day MAE Comparison (Target: ≤35)', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    for bar, val in zip(bars3, day_mae_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Peak MAE with target (PRIMARY METRIC)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, peak_mae_vals, color=[COLORS['baseline'], COLORS['baseline'], COLORS['aggressive'], 
                                                  COLORS['attention'], COLORS['ensemble']], alpha=0.7, edgecolor='black', linewidth=2)
    ax4.axhline(y=25.0, color='green', linestyle='--', linewidth=2, label='Target (≤25)')
    ax4.set_ylabel('Peak MAE', fontsize=11, fontweight='bold')
    ax4.set_title('Peak MAE Comparison (Target: ≤25)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    # Highlight winning model
    winning_model_full = final_report['winning_model']
    if winning_model_full in model_names:
        winner_idx = model_names.index(winning_model_full)
        bars4[winner_idx].set_linewidth(3)
        bars4[winner_idx].set_edgecolor('gold')
    
    for bar, val in zip(bars4, peak_mae_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "sprint_task_e_final_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def plot_target_achievement(final_report):
    """Plot target achievement visualization."""
    print("📊 Generating Target Achievement Chart...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    results = final_report['all_results']
    targets = final_report['targets']
    targets_met = final_report['targets_met']
    
    model_names = list(results.keys())
    day_mae_vals = [results[m]['Day_MAE'] for m in model_names]
    peak_mae_vals = [results[m]['Peak_MAE'] for m in model_names]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x_pos - width/2, peak_mae_vals, width, label='Peak MAE (Current)', 
                   color=COLORS['baseline'], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, day_mae_vals, width, label='Day MAE (Current)', 
                   color=COLORS['attention'], alpha=0.7, edgecolor='black')
    
    # Add target lines
    ax.axhline(y=targets['peak_mae'], color='green', linestyle='--', linewidth=2.5, 
               label=f"Peak MAE Target ({targets['peak_mae']})", alpha=0.8)
    ax.axhline(y=targets['day_mae'], color='orange', linestyle='--', linewidth=2.5, 
               label=f"Day MAE Target ({targets['day_mae']})", alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE Value', fontsize=12, fontweight='bold')
    ax.set_title('Target Achievement Analysis\nGreen=Peak Met, Orange=Day Target', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 140])
    
    # Add value labels
    for bar, val in zip(bars1, peak_mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, day_mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add status text
    status_text = f"Peak MAE Target: {'✓ MET' if targets_met['peak_mae'] else '✗ NOT MET'}\n"
    status_text += f"Day MAE Target: {'✓ MET' if targets_met['day_mae'] else '✗ NOT MET'}"
    ax.text(0.98, 0.97, status_text, transform=ax.transAxes, fontsize=11, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontweight='bold')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "sprint_target_achievement.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def create_summary_report(final_report, task_a, task_d):
    """Create a summary statistics visualization."""
    print("📊 Generating Summary Statistics...")
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('GHI Model Improvement Sprint - Summary Report', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Winning model info (text box)
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')
    
    winning = final_report['winning_model']
    metrics = final_report['final_metrics']
    targets = final_report['targets']
    
    summary_text = f"""
WINNING MODEL: {winning}

Final Metrics:
  • RMSE: {metrics['rmse']:.2f}
  • MAE: {metrics['mae']:.2f}
  • Peak MAE: {metrics['peak_mae']:.2f} (Target: ≤{targets['peak_mae']}) {'✓ MET' if metrics['peak_mae'] <= targets['peak_mae'] else '✗ NOT MET'}
  • Day MAE: {metrics['day_mae']:.2f} (Target: ≤{targets['day_mae']}) {'✓ MET' if metrics['day_mae'] <= targets['day_mae'] else '✗ NOT MET'}
    """
    
    ax_text.text(0.05, 0.5, summary_text, transform=ax_text.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Key findings
    ax_findings = fig.add_subplot(gs[1, :])
    ax_findings.axis('off')
    
    findings_text = f"""
KEY FINDINGS:

Task A (Blend Sweep):  Best blend ratio = 0.0 (Pure Baseline), Peak MAE = 25.00
                       Attention alone (blend=1.0) degrades to Peak MAE = 82.54
                       
Task D (Loss Tuning):  aggressive_peak_2x achieves best Peak MAE = 21.47 (Improvement: 14.1% vs baseline)
                       aggressive_peak_3x = 21.49, original_huber = 22.38
                       
Task C (Ensemble):     80% Baseline / 20% Attention blend achieves Peak MAE = 23.20
                       Marginal improvement over pure baseline due to attention weakness
                       
Task B (Attention):    LSTM with attention mechanism achieved lower performance (Peak MAE = 43.14)
                       Limited benefit for solar GHI with current data volume
    """
    
    ax_findings.text(0.05, 0.5, findings_text, transform=ax_findings.transAxes, fontsize=10,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Blend sweep trend
    ax_blend = fig.add_subplot(gs[2, 0])
    ax_blend.plot(task_a['blend'], task_a['peak_mae'], 'o-', color=COLORS['baseline'], linewidth=2, markersize=5)
    ax_blend.axhline(y=25.0, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax_blend.set_xlabel('Baseline ← Blend Ratio → Attention', fontsize=10, fontweight='bold')
    ax_blend.set_ylabel('Peak MAE', fontsize=10, fontweight='bold')
    ax_blend.set_title('Blend Sweep Trend', fontsize=11, fontweight='bold')
    ax_blend.grid(True, alpha=0.3)
    
    # Loss variant comparison
    ax_loss = fig.add_subplot(gs[2, 1])
    variants = task_d['variant'].values
    peak_maes = task_d['peak_mae'].values
    colors_loss = [COLORS['original'], COLORS['aggressive'], COLORS['aggressive']]
    ax_loss.barh(variants, peak_maes, color=colors_loss, alpha=0.7, edgecolor='black')
    ax_loss.axvline(x=25.0, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax_loss.set_xlabel('Peak MAE', fontsize=10, fontweight='bold')
    ax_loss.set_title('Loss Variant Comparison', fontsize=11, fontweight='bold')
    ax_loss.grid(True, alpha=0.3, axis='x')
    for i, (v, pm) in enumerate(zip(variants, peak_maes)):
        ax_loss.text(pm + 0.2, i, f'{pm:.2f}', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "sprint_summary_report.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()

def main():
    """Generate all plots."""
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE SPRINT VISUALIZATION PLOTS")
    print("="*70)
    
    # Load all data
    task_a, task_d, task_c, final_report = load_data()
    
    # Generate all plots
    plot_blend_sweep(task_a)
    plot_loss_tuning(task_d)
    plot_ensemble_weights(task_c)
    plot_final_metrics(final_report)
    plot_target_achievement(final_report)
    create_summary_report(final_report, task_a, task_d)
    
    print("\n" + "="*70)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Plots saved to: {PLOTS_DIR}")
    print("\n📊 Generated files:")
    for plot_file in sorted(PLOTS_DIR.glob("sprint_*.png")):
        print(f"   • {plot_file.name}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
