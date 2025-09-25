# plotting_functions.py
from preambles import *
# ─────────────────────────────── Visualisation of Results ──────────────────────────────

ALL_METHODS = [
    "BAS_full_mcq",
    "iBAS_all",
    "iBAS_wrong_only",
]

# Plot a single run’s performance
def plot_bar(results, title="Model Performance", save_path=None, show=False):
    keys = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 4))
    bars = plt.bar(keys, values)
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, max(values) + 0.10)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha='center', va='bottom')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Bar plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

# Print benchmark results across methods, layers, strengths, on the training set
def plot_results(
    results,
    raw_acc,
    methods=ALL_METHODS,
    save_dir="tables",
    benchmark_name="benchmark_name", 
    model_name="model_name"
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    layers = sorted(set(layer for (layer, _) in results.keys()))
    strengths = sorted(set(s for (_, s) in results.keys()))

    # Collect table entries
    records = []

    for method in methods:
        print(f"\n📊 {method} Results:")

        header = "Layer".ljust(8) + "".join(f"{s:^20}" for s in strengths)
        print(header)
        print("-" * len(header))
        
        for layer in layers:
            row = f"{layer:<8}"  # Left-align the layer number
            for s in strengths:
                acc = results[(layer, s)][method]
                delta = acc - raw_acc
                row += f"{acc:.2%} (Δ {delta:+.3f})".center(20)
                records.append({
                    "method": method,
                    "layer": layer,
                    "steer_strength": s,
                    "accuracy": acc,
                    "delta": delta
                })
            print(row)

    # Save as CSV
    csv_filename = f"train_{benchmark_name}.csv"
    csv_path = save_path / csv_filename
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\n📄 Saved grid results to {csv_path}")


# Heatmaps for steering strength × layer grid

def plot_delta_heatmaps(
    results,
    raw_acc,
    save_dir="overleaf/figures",
    benchmark_name="benchmark_name",
    model_name="model_name",
    show=False,
    methods=ALL_METHODS
):
    layers = sorted(set(layer for (layer, _) in results))
    strengths = sorted(set(strength for (_, strength) in results))
    strength_labels = [str(s) for s in strengths]


    benchmark_dir = os.path.join(save_dir, benchmark_name)
    os.makedirs(benchmark_dir, exist_ok=True)  # will create intermediate dirs if missing

    for method in methods:
        # Compute Δ accuracy matrix
        delta_matrix = np.array([
            [results[(layer, strength)][method] - raw_acc for strength in strengths]
            for layer in layers
        ])

        # Plot
        plt.figure(figsize=(8, 6))
        im = plt.imshow(delta_matrix, cmap="coolwarm", aspect="auto", origin="lower")
        plt.colorbar(im, label="Δ Accuracy vs. Unsteered")
        plt.xticks(ticks=np.arange(len(strengths)), labels=strength_labels)
        plt.yticks(ticks=np.arange(len(layers)), labels=layers)
        plt.xlabel("Steering Strength")
        plt.ylabel("Layer")
        plt.title(f"{method} Δ Accuracy Heatmap on {benchmark_name}")

        # Annotate cells
        for i in range(len(layers)):
            for j in range(len(strengths)):
                val = delta_matrix[i, j]
                plt.text(j, i, f"{val:+.2f}", ha="center", va="center", color="black")

        plt.tight_layout()

        # Save
        filename = f"{method.lower()}_delta_heatmap.png"
        save_path = os.path.join(benchmark_dir, filename)
        plt.savefig(save_path)
        print(f"\n Saved heatmap to {save_path}")
        if show:
            plt.show()
        else:
            plt.close()


# ────────────────────────────── multiple benchmarks ──────────────────────────────
# code for visualising results across many benchmarks

def plot_cross_benchmark_comparison(all_test_results, all_best_configs, save_dir, model_name="model_name"):
    """
    Generate comprehensive cross-benchmark comparison plots.
    
    Args:
        all_test_results: Dict[benchmark_name] -> Dict[method_name] -> accuracy
        all_best_configs: Dict[benchmark_name] -> Dict[method_name] -> config_dict
    """
    # Create output directory
    output_dir = Path(save_dir) / "Cross_Benchmark_Comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    method_names = list(next(iter(all_test_results.values())).keys())
    benchmark_names = list(all_test_results.keys())
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Heatmap of accuracy across benchmarks and methods
    plot_accuracy_heatmap(all_test_results, benchmark_names, method_names, output_dir)
    
    # 2. Bar plot comparison across benchmarks
    plot_grouped_bar_comparison(all_test_results, benchmark_names, method_names, output_dir)
    
    # 3. Performance improvement over baseline
    plot_improvement_heatmap(all_test_results, benchmark_names, method_names, output_dir)
    
    # 4. Best hyperparameter analysis
    plot_hyperparameter_analysis(all_best_configs, benchmark_names, method_names, output_dir)
    
    # 5. Method ranking across benchmarks
    plot_method_rankings(all_test_results, benchmark_names, method_names, output_dir)
    
    print(f"✅ Cross-benchmark comparison plots saved to {output_dir}")

def plot_accuracy_heatmap(all_test_results, benchmark_names, method_names, output_dir):
    """Plot heatmap of raw accuracies across benchmarks and methods."""
    # Create accuracy matrix
    acc_matrix = np.zeros((len(benchmark_names), len(method_names)))
    
    for i, benchmark in enumerate(benchmark_names):
        for j, method in enumerate(method_names):
            acc_matrix[i, j] = all_test_results[benchmark][method]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(benchmark_names)))
    ax.set_xticklabels([name.replace('_', '\n') for name in method_names], rotation=45, ha='right')
    ax.set_yticklabels(benchmark_names)
    
    # Add text annotations
    for i in range(len(benchmark_names)):
        for j in range(len(method_names)):
            text = ax.text(j, i, f'{acc_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=9)
    
    # Add colorbar and labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test Accuracy', rotation=270, labelpad=20)
    
    plt.title('Test Accuracy Across Benchmarks and Methods', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Steering Methods', fontsize=12)
    plt.ylabel('Benchmarks', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_grouped_bar_comparison(all_test_results, benchmark_names, method_names, output_dir):
    """Plot grouped bar chart comparing methods across benchmarks."""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(benchmark_names))
    width = 0.8 / len(method_names)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
    
    for i, method in enumerate(method_names):
        accuracies = [all_test_results[benchmark][method] for benchmark in benchmark_names]
        offset = (i - len(method_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=method.replace('_', ' '), 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Benchmarks', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Method Performance Comparison Across Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_names, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max([max(all_test_results[b].values()) for b in benchmark_names]) + 0.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grouped_bar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_improvement_heatmap(all_test_results, benchmark_names, method_names, output_dir):
    """Plot heatmap of improvement over unsteered baseline."""
    # Create improvement matrix
    improvement_matrix = np.zeros((len(benchmark_names), len(method_names)))
    
    for i, benchmark in enumerate(benchmark_names):
        unsteered_acc = all_test_results[benchmark]["Unsteered"]
        for j, method in enumerate(method_names):
            if method != "Unsteered":
                improvement = all_test_results[benchmark][method] - unsteered_acc
                improvement_matrix[i, j] = improvement
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use diverging colormap centered at 0
    vmax = max(abs(improvement_matrix.min()), abs(improvement_matrix.max()))
    im = ax.imshow(improvement_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Set ticks and labels
    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(benchmark_names)))
    ax.set_xticklabels([name.replace('_', '\n') for name in method_names], rotation=45, ha='right')
    ax.set_yticklabels(benchmark_names)
    
    # Add text annotations
    for i in range(len(benchmark_names)):
        for j in range(len(method_names)):
            if method_names[j] != "Unsteered":
                color = "white" if abs(improvement_matrix[i, j]) > vmax * 0.5 else "black"
                text = ax.text(j, i, f'{improvement_matrix[i, j]:+.3f}',
                             ha="center", va="center", color=color, fontsize=9)
    
    # Add colorbar and labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy Improvement over Unsteered', rotation=270, labelpad=20)
    
    plt.title('Accuracy Improvement Over Unsteered Baseline', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Steering Methods', fontsize=12)
    plt.ylabel('Benchmarks', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_hyperparameter_analysis(all_best_configs, benchmark_names, method_names, output_dir):
    """Plot analysis of best hyperparameters across benchmarks."""
    # Extract layer and strength information
    layers_data = []
    strengths_data = []
    
    for benchmark in benchmark_names:
        for method in method_names:
            if method != "Unsteered" and method in all_best_configs[benchmark]:
                config = all_best_configs[benchmark][method]
                layers_data.append({
                    'benchmark': benchmark,
                    'method': method,
                    'layer': config['layer']
                })
                strengths_data.append({
                    'benchmark': benchmark,
                    'method': method,
                    'strength': config['strength']
                })
    
    # Create subplots for layer and strength analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Layer analysis
    layers_df = pd.DataFrame(layers_data)
    if not layers_df.empty:
        layer_pivot = layers_df.pivot(index='benchmark', columns='method', values='layer')
        sns.heatmap(layer_pivot, annot=True, fmt='g', cmap='viridis', ax=ax1, cbar_kws={'label': 'Best Layer'})
        ax1.set_title('Best Layers Across Benchmarks and Methods', fontweight='bold')
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Benchmarks')
    
    # Strength analysis
    strengths_df = pd.DataFrame(strengths_data)
    if not strengths_df.empty:
        strength_pivot = strengths_df.pivot(index='benchmark', columns='method', values='strength')
        sns.heatmap(strength_pivot, annot=True, fmt='g', cmap='plasma', ax=ax2, cbar_kws={'label': 'Best Strength'})
        ax2.set_title('Best Steering Strengths Across Benchmarks and Methods', fontweight='bold')
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Benchmarks')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_method_rankings(all_test_results, benchmark_names, method_names, output_dir):
    """Plot ranking of methods across benchmarks."""
    # Calculate rankings for each benchmark
    rankings = {}
    
    for benchmark in benchmark_names:
        # Sort methods by accuracy (descending)
        sorted_methods = sorted(all_test_results[benchmark].items(), 
                               key=lambda x: x[1], reverse=True)
        rankings[benchmark] = {method: rank+1 for rank, (method, _) in enumerate(sorted_methods)}
    
    # Create ranking matrix
    rank_matrix = np.zeros((len(benchmark_names), len(method_names)))
    
    for i, benchmark in enumerate(benchmark_names):
        for j, method in enumerate(method_names):
            rank_matrix[i, j] = rankings[benchmark][method]
    
    # Create heatmap (lower rank = better, so reverse colormap)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', 
                   vmin=1, vmax=len(method_names))
    
    # Set ticks and labels
    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(benchmark_names)))
    ax.set_xticklabels([name.replace('_', '\n') for name in method_names], rotation=45, ha='right')
    ax.set_yticklabels(benchmark_names)
    
    # Add text annotations
    for i in range(len(benchmark_names)):
        for j in range(len(method_names)):
            rank = int(rank_matrix[i, j])
            color = "white" if rank > len(method_names) * 0.6 else "black"
            text = ax.text(j, i, f'{rank}', ha="center", va="center", 
                         color=color, fontsize=10, fontweight='bold')
    
    # Add colorbar and labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Method Ranking (1=Best)', rotation=270, labelpad=20)
    
    plt.title('Method Rankings Across Benchmarks', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Steering Methods', fontsize=12)
    plt.ylabel('Benchmarks', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'method_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a summary ranking plot
    plot_average_rankings(rankings, method_names, output_dir)

def plot_average_rankings(rankings, method_names, output_dir):
    """Plot average rankings across all benchmarks."""
    # Calculate average rankings
    avg_rankings = {}
    for method in method_names:
        ranks = [rankings[benchmark][method] for benchmark in rankings.keys()]
        avg_rankings[method] = np.mean(ranks)
    
    # Sort by average ranking
    sorted_methods = sorted(avg_rankings.items(), key=lambda x: x[1])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods, avg_ranks = zip(*sorted_methods)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(methods)))
    
    bars = ax.barh(range(len(methods)), avg_ranks, color=colors)
    
    # Add value labels
    for i, (bar, rank) in enumerate(zip(bars, avg_ranks)):
        ax.text(rank + 0.05, i, f'{rank:.2f}', va='center', fontweight='bold')
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace('_', ' ') for m in methods])
    ax.set_xlabel('Average Ranking (Lower is Better)', fontsize=12)
    ax.set_title('Average Method Rankings Across All Benchmarks', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(avg_ranks) + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'average_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
            
