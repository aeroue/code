# run_experiments.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import time
import os
import copy
import warnings

# Import your simulation code AS A MODULE
import spacecraft_re_tom as sim

# Suppress specific warnings if they become annoying (e.g., tight_layout)
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

# --------------------------------------------------------------------------
# Experiment Configuration
# --------------------------------------------------------------------------
# List of robot counts to test
NUM_AGENTS_TO_TEST = [2, 3, 4, 5]
NUM_TRIALS = 50  # Number of simulations per agent count per algorithm (like the example plot)
SAVE_INDIVIDUAL_PLOTS = False # Keep False unless debugging specific runs

# Base Simulation Parameters (num_agents will be overridden in the loop)
base_sim_params = {
    'env_size': 20.0,
    # 'num_agents': 3, # Removed, will be set in the loop
    'num_goals': 3,   # Keep num_goals fixed, or vary it in another outer loop? Let's keep fixed for now.
    'min_goal_distance': 20.0 / 4,
    'convergence_distance': 1.5,
    'max_iterations': 150,
    'dt': 0.2,
    # 'agent_types': ['A', 'A', 'A'], # Will be adapted in the loop
    'observation_error_std': 2.0, # From paper figure, likely used specific noise
    'velocity_options': [1.0], # Simpler actions might match paper? Or keep [0.5, 1.0, 1.5]? Let's try 1.0
    'heading_options': [-np.pi/6, 0, np.pi/6], # Reduced options? [-np.pi/8, 0, np.pi/8] is also fine
}

# Algorithms to Compare (Map to Paper Figure Labels)
# Match the order and names for consistent plotting
algorithms_to_test = [
    {
        'name': 'Zero-Order', # Assuming Greedy maps to Zero-Order
        'params': {'algorithm': 'greedy', 'use_epistemic_reasoning': False}
    },
    {
        'name': 'First-Order', # AIF without EP
        'params': {'algorithm': 'active_inference', 'use_epistemic_reasoning': False}
    },
    {
        'name': 'Higher-Order', # AIF with EP
        'params': {'algorithm': 'active_inference', 'use_epistemic_reasoning': True}
    }
]

# --- Academic Plotting Style Setup ---
try:
    # Attempt to use LaTeX for rendering if available
    # --- CHANGE THIS LINE ---
    # plt.rcParams.update({
    #     "text.usetex": True, # Comment out or set to False
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"], # Or Times New Roman, etc.
    #     # ... other settings ...
    # })
    # --- Use this instead ---
    plt.rcParams.update({
        "text.usetex": False, # Explicitly disable LaTeX
        "font.family": "sans-serif", # Use a common sans-serif font
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })
    # --- End Change ---
    print("Using default Matplotlib fonts (LaTeX not used/found).")
    use_latex = False # Keep this variable consistent
except Exception as e:
    print(f"Error setting plotting parameters: {e}. Using default Matplotlib fonts.")
    # Fallback to default style
    plt.rcParams.update(plt.rcParamsDefault)
    use_latex = False

# Consistent colors inspired by the example figure
# Define colors specifically for the algorithms in the desired order
algo_colors = {
    'Zero-Order': '#66c2a5', # Teal/Greenish
    'First-Order': '#fc8d62', # Orange
    'Higher-Order': '#8da0cb' # Blue/Purple
}
algo_names_ordered = [a['name'] for a in algorithms_to_test] # Ensure order

# --- Saving Setup ---
timestamp = time.strftime("%Y%m%d-%H%M%S")
experiment_label = f"N{min(NUM_AGENTS_TO_TEST)}-{max(NUM_AGENTS_TO_TEST)}_G{base_sim_params['num_goals']}_T{NUM_TRIALS}"
base_save_dir = f"experiment_{experiment_label}_{timestamp}"
comparative_plot_dir = os.path.join(base_save_dir, "comparative_plots")
results_dir = os.path.join(base_save_dir, "results_data")
os.makedirs(comparative_plot_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
print(f"Saving results and plots to: {base_save_dir}")

# --------------------------------------------------------------------------
# Experiment Execution Loop
# --------------------------------------------------------------------------
all_results_list = []
total_sims = len(NUM_AGENTS_TO_TEST) * NUM_TRIALS * len(algorithms_to_test)
sim_count = 0

for n_agents in NUM_AGENTS_TO_TEST:
    print(f"\n===== Testing with {n_agents} Agents =====")
    current_agent_types = ['A', 'B'] * (n_agents // 2) + ['A'] * (n_agents % 2) # Assuming homogeneous 'A' type for now

    for trial in range(NUM_TRIALS):
        sim_start_time_trial = time.time()
        print(f"\n--- Starting Trial {trial + 1}/{NUM_TRIALS} for {n_agents} Agents ---")

        # Generate SAME initial conditions for all algorithms in this trial
        current_goals = sim.generate_spread_out_goals(
            base_sim_params['num_goals'],
            base_sim_params['env_size'],
            base_sim_params['min_goal_distance']
        )
        if len(current_goals) < base_sim_params['num_goals']:
            print(f"Warning: Trial {trial+1} (N={n_agents}) failed to generate enough goals. Skipping.")
            sim_count += len(algorithms_to_test) # Increment count even if skipped
            continue

        current_initial_positions = []
        min_start_dist_goal = base_sim_params['env_size'] / 5
        placement_successful = True
        for i in range(n_agents):
            attempts = 0
            pos_found = False
            while attempts < 100:
                pos = np.random.rand(2) * base_sim_params['env_size']
                heading = np.random.rand() * 2 * np.pi - np.pi
                # Check distance to goals AND other placed agents
                valid_goal_dist = all(np.linalg.norm(pos - g[:2]) >= min_start_dist_goal for g in current_goals)
                valid_agent_dist = all(np.linalg.norm(pos - p[:2]) >= 1.0 for p in current_initial_positions) if current_initial_positions else True

                if valid_goal_dist and valid_agent_dist:
                     current_initial_positions.append(np.concatenate([pos, [heading]]))
                     pos_found = True
                     break
                attempts += 1
            if not pos_found:
                print(f"Warning: Trial {trial+1} (N={n_agents}) failed to place agent {i}. Skipping trial.")
                placement_successful = False
                break # Break inner agent placement loop
        if not placement_successful:
            sim_count += len(algorithms_to_test) # Increment count even if skipped
            continue # Skip to next trial

        current_initial_positions = np.array(current_initial_positions)

        # Generate rendezvous configs for this trial setup
        rendezvous_configs = sim.identify_rendezvous_configs(
            base_sim_params['num_goals'], n_agents
        )

        for algo_config in algorithms_to_test:
            sim_count += 1
            algo_name = algo_config['name']
            print(f"  ({sim_count}/{total_sims}) Running Algorithm: {algo_name} ... ", end='', flush=True)
            run_start_time = time.time()

            # Create params for this specific run
            exp_params = copy.deepcopy(base_sim_params)
            exp_params.update(algo_config['params']) # Add algo-specific params
            exp_params['num_agents'] = n_agents
            exp_params['agent_types'] = current_agent_types
            exp_params['goals'] = current_goals.tolist()
            exp_params['initial_positions'] = current_initial_positions.tolist()
            exp_params['rendezvous_configs'] = rendezvous_configs

            # --- Run the Simulation ---
            sim_results = sim.run_simulation(exp_params)
            # --- End Simulation ---
            run_end_time = time.time()
            print(f"Converged: {sim_results.get('converged', False)}, Iterations: {sim_results.get('final_iteration', 'N/A')}, Time: {run_end_time - run_start_time:.2f}s")


            # Store key results
            result_summary = {
                'num_agents': n_agents,
                'trial': trial + 1,
                'algorithm': algo_name,
                'converged': sim_results.get('converged', False),
                'iterations_taken': sim_results.get('final_iteration', exp_params['max_iterations']),
                'final_target_goal': sim_results.get('final_target_goal', -1),
                'use_ep': exp_params.get('use_epistemic_reasoning', None)
            }
            all_results_list.append(result_summary)

            # Optional: Save individual plots
            if SAVE_INDIVIDUAL_PLOTS and sim_results.get('log'):
                # ... (optional saving code - keep short for brevity) ...
                pass

        print(f"--- Trial {trial + 1} (N={n_agents}) completed in {time.time() - sim_start_time_trial:.2f}s ---")


print("\n===== Experiment Finished =====")

# --------------------------------------------------------------------------
# Results Aggregation and Comparative Plotting
# --------------------------------------------------------------------------
if not all_results_list:
    print("No results were collected.")
else:
    results_df = pd.DataFrame(all_results_list)
    results_csv_path = os.path.join(results_dir, "raw_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nRaw results saved to: {results_csv_path}")

    # --- Aggregate Results ---
    print("\n--- Aggregate Convergence Rate (%) ---")
    agg_conv_rate = results_df.groupby(['num_agents', 'algorithm'])['converged'].mean().unstack() * 100.0
    # Reorder columns to match algorithms_to_test
    agg_conv_rate = agg_conv_rate[algo_names_ordered]
    print(agg_conv_rate.to_string(float_format="%.1f%%"))
    agg_csv_path = os.path.join(results_dir, "aggregate_convergence_rate.csv")
    agg_conv_rate.to_csv(agg_csv_path, float_format="%.1f")
    print(f"Aggregate convergence rates saved to: {agg_csv_path}")

    # --- Plotting Function ---
    def plot_convergence_comparison(agg_df, save_path):
        """Plots the grouped bar chart for convergence rates."""
        # Use the ordered algorithm names and colors
        algorithms = algo_names_ordered
        colors = [algo_colors[algo] for algo in algorithms]
        n_groups = len(agg_df.index) # Number of agent counts
        n_bars = len(algorithms)    # Number of algorithms
        bar_width = 0.8 / n_bars    # Width of each bar
        index = np.arange(n_groups) # Base positions for groups

        fig, ax = plt.subplots(figsize=(8, 5)) # Adjust size as needed

        for i, algo in enumerate(algorithms):
            # Calculate position for each bar in the group
            pos = index + (i - (n_bars - 1) / 2) * bar_width
            rates = agg_df[algo].values
            bars = ax.bar(pos, rates, bar_width, label=algo, color=colors[i])
            ax.bar_label(bars, fmt='%.0f\\%%' if use_latex else '%.0f%%', padding=3, fontsize=9) # Add labels

        ax.set_xlabel('Number of Robots')
        ax.set_ylabel('\\% of Scenarios Converged' if use_latex else '% of Scenarios Converged')
        ax.set_title(f'Multi-Robot Reasoning-Based Rendezvous (N={NUM_TRIALS} trials per point)')
        ax.set_xticks(index)
        ax.set_xticklabels(agg_df.index)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, max(105, agg_df.max().max() * 1.1)) # Adjust y-limit slightly above max bar
        ax.legend(title='Algorithm') # Add title to legend if desired
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False) # Remove top spine
        ax.spines['right'].set_visible(False) # Remove right spine

        try:
            fig.tight_layout()
        except ValueError:
            print("Warning: tight_layout failed. Plot might have overlapping elements.")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparative convergence plot saved to: {save_path}")
        # plt.show() # Moved outside

    # --- Generate Plot ---
    plot_path = os.path.join(comparative_plot_dir, "multi_robot_convergence_comparison.png")
    plot_convergence_comparison(agg_conv_rate, plot_path)
    plt.show() # Show the final plot