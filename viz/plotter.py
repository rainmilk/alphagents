"""
Visualization Module

This module provides plotting and visualization tools for the AAAI 2027 paper,
including performance charts, factor analysis plots, and ablation study figures.

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """
    Visualizer for factor analysis and portfolio performance.
    
    Generates plots for:
    1. Portfolio performance (cumulative returns, drawdown)
    2. Factor analysis (IC distribution, correlation heatmap)
    3. Ablation studies (component contribution)
    4. Baseline comparisons (performance bar chart)
    """
    
    def __init__(self, save_dir: str = "paper/figures"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Visualizer initialized. Figures will be saved to {save_dir}")
    
    def plot_cumulative_returns(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Cumulative Returns",
        save_name: str = "cumulative_returns.png",
    ):
        """
        Plot cumulative returns.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns (optional)
            title: Plot title
            save_name: File name to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label='Portfolio', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                   label='Benchmark', linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {self.save_dir}/{save_name}")
        
        plt.show()
    
    def plot_drawdown(
        self,
        returns: pd.Series,
        title: str = "Drawdown Chart",
        save_name: str = "drawdown.png",
    ):
        """
        Plot drawdown chart.
        
        Args:
            returns: Series of returns
            title: Plot title
            save_name: File name to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, linewidth=2, color='darkred')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {self.save_dir}/{save_name}")
        
        plt.show()
    
    def plot_ic_distribution(
        self,
        ic_values: List[float],
        title: str = "IC Distribution",
        save_name: str = "ic_distribution.png",
    ):
        """
        Plot IC distribution.
        
        Args:
            ic_values: List of IC values
            title: Plot title
            save_name: File name to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(ic_values, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(ic_values), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean IC = {np.mean(ic_values):.4f}')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Information Coefficient (IC)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {self.save_dir}/{save_name}")
        
        plt.show()
    
    def plot_factor_correlation_heatmap(
        self,
        factor_values: pd.DataFrame,
        title: str = "Factor Correlation Heatmap",
        save_name: str = "factor_correlation.png",
    ):
        """
        Plot factor correlation heatmap.
        
        Args:
            factor_values: DataFrame of factor values
            title: Plot title
            save_name: File name to save
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = factor_values.corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {self.save_dir}/{save_name}")
        
        plt.show()
    
    def plot_ablation_studies(
        self,
        ablation_results: Dict,
        metric: str = 'sharpe_ratio',
        title: str = "Ablation Studies",
        save_name: str = "ablation.png",
    ):
        """
        Plot ablation study results.
        
        Args:
            ablation_results: Dict of ablation results
            metric: Metric to plot
            title: Plot title
            save_name: File name to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract metric values
        names = list(ablation_results.keys())
        values = [ablation_results[name].get(metric, 0) for name in names]
        
        # Plot bar chart
        bars = ax.bar(names, values, color='steelblue', edgecolor='black')
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Component Removed', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {self.save_dir}/{save_name}")
        
        plt.show()
    
    def plot_baseline_comparison(
        self,
        baseline_results: Dict,
        metrics: List[str] = ['sharpe_ratio', 'max_drawdown', 'information_ratio'],
        title: str = "Baseline Comparison",
        save_name: str = "baseline_comparison.png",
    ):
        """
        Plot baseline comparison results.
        
        Args:
            baseline_results: Dict of baseline results
            metrics: List of metrics to plot
            title: Plot title
            save_name: File name to save
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Extract metric values
            names = list(baseline_results.keys())
            values = [baseline_results[name].get(metric, 0) for name in names]
            
            # Plot bar chart
            bars = ax.bar(names, values, color='steelblue', edgecolor='black')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {self.save_dir}/{save_name}")
        
        plt.show()
    
    def plot_performance_summary(
        self,
        metrics: Dict,
        title: str = "Performance Summary",
        save_name: str = "performance_summary.png",
    ):
        """
        Plot performance summary (radar chart).
        
        Args:
            metrics: Dict of performance metrics
            title: Plot title
            save_name: File name to save
        """
        # Select metrics for radar chart
        radar_metrics = ['sharpe_ratio', 'calmar_ratio', 'information_ratio', 'win_rate']
        values = [metrics.get(m, 0) for m in radar_metrics]
        
        # Normalize values to 0-1
        values = np.array(values)
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        # Plot radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        values = values.tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {self.save_dir}/{save_name}")
        
        plt.show()
    
    def plot_factor_importance(
        self,
        factor_weights: Dict[str, float],
        title: str = "Factor Importance (Weights)",
        save_name: str = "factor_importance.png",
    ):
        """
        Plot factor importance (weights).
        
        Args:
            factor_weights: Dict of factor weights
            title: Plot title
            save_name: File name to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by weight
        sorted_factors = sorted(factor_weights.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_factors]
        weights = [x[1] for x in sorted_factors]
        
        # Plot horizontal bar chart
        bars = ax.barh(names, weights, color='steelblue', edgecolor='black')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{weight:.4f}', ha='left', va='center', fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Weight', fontsize=12)
        ax.set_ylabel('Factor', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {self.save_dir}/{save_name}")
        
        plt.show()


def generate_all_figures():
    """
    Generate all figures for the AAAI 2027 paper.
    """
    visualizer = Visualizer()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    
    # Figure 1: Cumulative returns
    returns = pd.Series(np.random.randn(n_days) * 0.01 + 0.0005, index=dates)
    visualizer.plot_cumulative_returns(returns)
    
    # Figure 2: Drawdown
    visualizer.plot_drawdown(returns)
    
    # Figure 3: IC distribution
    ic_values = np.random.randn(100) * 0.02 + 0.03
    visualizer.plot_ic_distribution(ic_values)
    
    # Figure 4: Ablation studies
    ablation_results = {
        'without_debate': {'sharpe_ratio': 1.25},
        'without_evolution': {'sharpe_ratio': 1.10},
        'without_memory': {'sharpe_ratio': 1.18},
        'without_fusion': {'sharpe_ratio': 1.05},
        'without_all': {'sharpe_ratio': 0.85},
    }
    visualizer.plot_ablation_studies(ablation_results)
    
    # Figure 5: Baseline comparison
    baseline_results = {
        'equal_weight': {'sharpe_ratio': 0.85, 'max_drawdown': -0.22, 'information_ratio': 0.45},
        'ic_weighted': {'sharpe_ratio': 1.10, 'max_drawdown': -0.18, 'information_ratio': 0.72},
        'alphagrail': {'sharpe_ratio': 1.25, 'max_drawdown': -0.16, 'information_ratio': 0.88},
        'gpt_factor': {'sharpe_ratio': 1.05, 'max_drawdown': -0.20, 'information_ratio': 0.65},
        'ours': {'sharpe_ratio': 1.42, 'max_drawdown': -0.13, 'information_ratio': 1.15},
    }
    visualizer.plot_baseline_comparison(baseline_results)
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    # Generate all figures
    generate_all_figures()
