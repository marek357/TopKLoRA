#!/usr/bin/env python3
"""
Focused analysis of autointerp results for evaluating monosemanticity
of sparse LoRA latents.

This script provides targeted insights into the interpretability and
monosemantic properties of the sparse LoRA adapter latents.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re
from typing import Dict, List, Tuple
import argparse


def load_latent_data(scores_dir: str, explanations_dir: str) -> pd.DataFrame:
    """Load and combine scores and explanations into a single DataFrame."""
    scores_path = Path(scores_dir)
    explanations_path = Path(explanations_dir)
    
    data = []
    
    for config_dir in scores_path.iterdir():
        if not config_dir.is_dir():
            continue
            
        config = config_dir.name
        scores_config_path = config_dir / "default_detection"
        explanations_config_path = explanations_path / config / "default"
        
        if not (scores_config_path.exists() and explanations_config_path.exists()):
            continue
        
        print(f"Processing configuration: {config}")
        
        for score_file in scores_config_path.glob("*.json"):
            try:
                # Parse filename
                filename = score_file.stem
                match = re.match(r'.*_latent(\d+)', filename)
                if not match:
                    continue
                latent_id = int(match.group(1))
                
                # Load explanation
                explanation_file = explanations_config_path / f"{filename}.json"
                explanation = ""
                if explanation_file.exists():
                    with open(explanation_file, 'r') as f:
                        exp_data = json.load(f)
                        explanation = exp_data.get('explanation', '')
                
                # Load detection scores
                with open(score_file, 'r') as f:
                    detection_data = json.load(f)
                
                # Calculate metrics
                metrics = calculate_metrics(detection_data)
                
                data.append({
                    'config': config,
                    'latent_id': latent_id,
                    'explanation': explanation,
                    **metrics
                })
                
            except Exception as e:
                print(f"Error processing {score_file}: {e}")
                continue
    
    return pd.DataFrame(data)


def calculate_metrics(detection_data: List[Dict]) -> Dict:
    """Calculate interpretability metrics from detection data."""
    if not detection_data:
        return {}
    
    total = len(detection_data)
    activating = sum(1 for ex in detection_data if ex.get('activating', False))
    
    # Confusion matrix
    tp = sum(1 for ex in detection_data 
            if ex.get('activating', False) and ex.get('prediction', False))
    fp = sum(1 for ex in detection_data 
            if not ex.get('activating', False) and ex.get('prediction', False))
    tn = sum(1 for ex in detection_data 
            if not ex.get('activating', False) and not ex.get('prediction', False))
    fn = sum(1 for ex in detection_data 
            if ex.get('activating', False) and not ex.get('prediction', False))
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # Activation statistics
    activations = []
    for ex in detection_data:
        if 'activations' in ex and ex['activations']:
            activations.extend([a for a in ex['activations'] if isinstance(a, (int, float)) and a > 0])
    
    activation_strength = np.mean(activations) if activations else 0
    activation_consistency = np.std(activations) if len(activations) > 1 else 0
    
    return {
        'total_examples': total,
        'activation_rate': activating / total if total > 0 else 0,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'activation_strength': activation_strength,
        'activation_consistency': activation_consistency,
        'num_activations': len(activations)
    }


def analyze_monosemanticity(df: pd.DataFrame) -> Dict:
    """Analyze monosemanticity properties of the latents."""
    print("\n" + "="*80)
    print("MONOSEMANTICITY ANALYSIS")
    print("="*80)
    
    # 1. High-quality interpretable latents
    high_quality = df[df['f1_score'] > 0.7]
    print(f"\n1. HIGH-QUALITY INTERPRETABLE LATENTS ({len(high_quality)} total)")
    print("-" * 50)
    
    if len(high_quality) > 0:
        print("Top interpretable latents:")
        for _, row in high_quality.nlargest(5, 'f1_score').iterrows():
            print(f"  Config {row['config']}, Latent {row['latent_id']}: F1={row['f1_score']:.3f}")
            print(f"    Explanation: {row['explanation']}")
            print()
        
        # Analyze explanation patterns in high-quality latents
        explanations = ' '.join(high_quality['explanation'].tolist())
        words = re.findall(r'\b\w+\b', explanations.lower())
        word_counts = Counter(words)
        
        # Filter common words
        stopwords = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        filtered_words = {w: c for w, c in word_counts.items() if w not in stopwords and len(w) > 2}
        top_concepts = dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10])
        
        print("Most common concepts in high-quality latents:")
        for word, count in top_concepts.items():
            print(f"  {word}: {count}")
        print()
    
    # 2. Configuration comparison for monosemanticity
    print("2. MONOSEMANTICITY BY CONFIGURATION")
    print("-" * 50)
    
    config_stats = df.groupby('config').agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': 'mean',
        'recall': 'mean',
        'activation_rate': 'mean',
        'activation_strength': 'mean'
    }).round(3)
    
    print(config_stats)
    print()
    
    # Count high-quality latents per config
    quality_by_config = df.groupby('config').apply(
        lambda x: pd.Series({
            'high_quality': len(x[x['f1_score'] > 0.7]),
            'medium_quality': len(x[(x['f1_score'] >= 0.3) & (x['f1_score'] <= 0.7)]),
            'poor_quality': len(x[x['f1_score'] < 0.3]),
            'total': len(x)
        })
    )
    
    print("Quality distribution by configuration:")
    print(quality_by_config)
    print()
    
    # 3. Sparsity vs Interpretability
    print("3. SPARSITY vs INTERPRETABILITY ANALYSIS")
    print("-" * 50)
    
    # Categorize by sparsity
    df['sparsity_category'] = pd.cut(
        df['activation_rate'],
        bins=[0, 0.1, 0.3, 1.0],
        labels=['Very Sparse', 'Moderate', 'Dense']
    )
    
    sparsity_interp = df.groupby('sparsity_category')['f1_score'].agg(['mean', 'count']).round(3)
    print("Interpretability by sparsity level:")
    print(sparsity_interp)
    print()
    
    # 4. Identify potential polysemantic latents
    print("4. POTENTIAL POLYSEMANTIC LATENTS")
    print("-" * 50)
    
    # Latents with low precision but high recall might be polysemantic
    polysemantic_candidates = df[
        (df['precision'] < 0.5) & 
        (df['recall'] > 0.6) & 
        (df['f1_score'] > 0.3)
    ]
    
    print(f"Found {len(polysemantic_candidates)} potential polysemantic latents:")
    if len(polysemantic_candidates) > 0:
        for _, row in polysemantic_candidates.nlargest(5, 'recall').iterrows():
            print(f"  Config {row['config']}, Latent {row['latent_id']}: "
                  f"Precision={row['precision']:.3f}, Recall={row['recall']:.3f}")
            print(f"    Explanation: {row['explanation']}")
            print()
    
    # 5. Dead/inactive latents
    print("5. INACTIVE/DEAD LATENTS")
    print("-" * 50)
    
    inactive = df[df['activation_rate'] < 0.01]
    print(f"Found {len(inactive)} potentially inactive latents (activation rate < 1%)")
    
    very_poor = df[df['f1_score'] < 0.1]
    print(f"Found {len(very_poor)} latents with very poor interpretability (F1 < 0.1)")
    print()
    
    # 6. Configuration efficiency
    print("6. CONFIGURATION EFFICIENCY ANALYSIS")
    print("-" * 50)
    
    efficiency_stats = df.groupby('config').apply(
        lambda x: pd.Series({
            'interpretable_ratio': len(x[x['f1_score'] > 0.5]) / len(x),
            'mean_f1': x['f1_score'].mean(),
            'mean_sparsity': x['activation_rate'].mean(),
            'efficiency_score': len(x[x['f1_score'] > 0.5]) / len(x) * x['f1_score'].mean()
        })
    ).round(3)
    
    print("Configuration efficiency (higher is better):")
    print(efficiency_stats.sort_values('efficiency_score', ascending=False))
    print()
    
    return {
        'high_quality_count': len(high_quality),
        'polysemantic_count': len(polysemantic_candidates),
        'inactive_count': len(inactive),
        'config_efficiency': efficiency_stats.to_dict('index'),
        'quality_by_config': quality_by_config.to_dict('index')
    }


def generate_monosemantic_plots(df: pd.DataFrame, output_dir: str = "monosemantic_analysis"):
    """Generate plots focused on monosemanticity analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Interpretability distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # F1 distribution by config
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        axes[0, 0].hist(config_data['f1_score'], alpha=0.6, label=config, bins=20)
    axes[0, 0].set_title('F1 Score Distribution by Configuration')
    axes[0, 0].set_xlabel('F1 Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].axvline(0.7, color='red', linestyle='--', label='High Quality Threshold')
    
    # Precision vs Recall scatter
    colors = {'1024_8': 'blue', '512_2': 'orange', '512_4': 'green'}
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        axes[0, 1].scatter(config_data['recall'], config_data['precision'], 
                          alpha=0.6, label=config, c=colors.get(config, 'gray'))
    axes[0, 1].set_title('Precision vs Recall by Configuration')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Sparsity vs Interpretability
    axes[1, 0].scatter(df['activation_rate'], df['f1_score'], alpha=0.6)
    axes[1, 0].set_title('Sparsity vs Interpretability')
    axes[1, 0].set_xlabel('Activation Rate')
    axes[1, 0].set_ylabel('F1 Score')
    
    # Quality distribution by config
    quality_data = []
    for config in df['config'].unique():
        config_df = df[df['config'] == config]
        quality_data.append({
            'Config': config,
            'High Quality': len(config_df[config_df['f1_score'] > 0.7]),
            'Medium Quality': len(config_df[(config_df['f1_score'] >= 0.3) & (config_df['f1_score'] <= 0.7)]),
            'Poor Quality': len(config_df[config_df['f1_score'] < 0.3])
        })
    
    quality_df = pd.DataFrame(quality_data)
    quality_df.set_index('Config')[['High Quality', 'Medium Quality', 'Poor Quality']].plot(
        kind='bar', stacked=True, ax=axes[1, 1])
    axes[1, 1].set_title('Quality Distribution by Configuration')
    axes[1, 1].set_ylabel('Number of Latents')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / "monosemantic_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze monosemanticity of sparse LoRA latents')
    parser.add_argument('--scores_dir', default='scores', help='Directory containing score JSON files')
    parser.add_argument('--explanations_dir', default='explanations', help='Directory containing explanation JSON files')
    parser.add_argument('--output_dir', default='monosemantic_analysis', help='Output directory for analysis')
    
    args = parser.parse_args()
    
    print("Loading autointerp data...")
    df = load_latent_data(args.scores_dir, args.explanations_dir)
    
    if df.empty:
        print("No data found! Check your directories.")
        return
    
    print(f"Loaded data for {len(df)} latents across {df['config'].nunique()} configurations")
    
    # Perform monosemanticity analysis
    analysis_results = analyze_monosemanticity(df)
    
    # Generate plots
    generate_monosemantic_plots(df, args.output_dir)
    
    # Save detailed results
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save analysis results
    with open(output_path / "monosemantic_analysis_results.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save detailed data
    df.to_csv(output_path / "monosemantic_data.csv", index=False)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    
    # Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVING MONOSEMANTICITY")
    print("="*80)
    
    best_config = max(analysis_results['config_efficiency'].keys(), 
                     key=lambda x: analysis_results['config_efficiency'][x]['efficiency_score'])
    
    print(f"1. Best performing configuration: {best_config}")
    print(f"   - Efficiency score: {analysis_results['config_efficiency'][best_config]['efficiency_score']:.3f}")
    print(f"   - Interpretable ratio: {analysis_results['config_efficiency'][best_config]['interpretable_ratio']:.3f}")
    
    print(f"\n2. Overall interpretability:")
    print(f"   - {analysis_results['high_quality_count']} highly interpretable latents ({analysis_results['high_quality_count']/len(df)*100:.1f}%)")
    print(f"   - {analysis_results['polysemantic_count']} potentially polysemantic latents")
    print(f"   - {analysis_results['inactive_count']} inactive/dead latents")
    
    if analysis_results['high_quality_count'] / len(df) < 0.2:
        print(f"\n3. RECOMMENDATION: Low interpretability rate suggests:")
        print(f"   - Consider increasing sparsity (lower k)")
        print(f"   - Experiment with different r/k ratios")
        print(f"   - Add explicit monosemanticity loss terms")
        print(f"   - Use contrastive learning for latent separation")


if __name__ == "__main__":
    main()
