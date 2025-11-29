#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_quadtree_vs_simulation_log(all_nodes, simulation_log, nodes_visited_dict, params, depth_level):
    """
    Evaluate how many ground-truth grid points (from simulation_log) fall inside
    in_range vs out_range quadtree nodes, up to a given depth level.
    Also reports how many grid points remain unclassified (not covered by any
    in_range or out_range node), and how many nodes were visited and computed.

    Parameters
    ----------
    all_nodes : list[QuadNode]
        All nodes from the quadtree. Each node must have: 
        x_min, x_max, y_min, y_max, depth, is_leaf, status.
    simulation_log : pd.DataFrame
        DataFrame containing at least:
            - param_x, param_y : grid search coordinates
            - status : True (in-range) or False (out-of-range)
    params : dict
        Contains 'params_to_change' with two parameters being varied.
    depth_level : int
        Only consider quadtree nodes up to this depth.

    Returns
    -------
    results_df : pd.DataFrame
        Per-node statistics.
    summary : pd.Series
        Aggregated counts including number of unclassified grid points,
        nodes visited, and nodes computed.
    """


    # Determine which parameter names correspond to X and Y
    param_x, param_y = list(params['params_to_change'].keys())

    # Keep only grid search entries
    simulation_log = simulation_log[simulation_log['algorithm'] == 'grid_search'].copy()

    # Extract numpy arrays for faster computation
    grid_x = simulation_log[param_x].to_numpy()
    grid_y = simulation_log[param_y].to_numpy()
    grid_status = simulation_log['status'].astype(bool).to_numpy()
    total_points = len(simulation_log)

    counted_indices = set()
    results = []

    # Initialize counters for visited and computed nodes
    nodes_visited = 0
    nodes_computed = 0

    for node in all_nodes:
        # Count the node as visited if it is within the depth level
        if node.depth <= depth_level:
            nodes_visited += 1

        # Only process leaf nodes up to target depth
        if not node.is_leaf or node.depth > depth_level:
            continue

        # Count the node as computed if it is a leaf node at the depth level
        nodes_computed += 1

        if node.status not in ("in_range", "out_range"):
            # Skip mixed or undefined nodes when classifying points
            continue

        x_min, x_max, y_min, y_max = node.x_min, node.x_max, node.y_min, node.y_max

        # Identify points inside this node
        inside_mask = (
            (grid_x >= x_min) &
            (grid_x <= x_max) &
            (grid_y >= y_min) &
            (grid_y <= y_max)
        )
        inside_indices = np.where(inside_mask)[0]

        # Remove duplicates (points already counted)
        new_indices = [i for i in inside_indices if i not in counted_indices]
        counted_indices.update(new_indices)
        new_indices = np.array(new_indices)

        # Count how many are True vs False in the ground truth
        if len(new_indices) > 0:
            in_true = np.sum(grid_status[new_indices])
            out_true = len(new_indices) - in_true
        else:
            in_true = out_true = 0

        results.append({
            "depth": node.depth,
            "status": node.status,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "n_points_total": len(new_indices),
            "n_points_true_inrange": in_true,
            "n_points_true_outrange": out_true,
        })

    results_df = pd.DataFrame(results)

    # ---- Aggregate summary ----
    # handle case of empty results_df
    if results_df.empty:
        results_df = pd.DataFrame(columns=[
            "depth", "status", "x_min", "x_max", "y_min", "y_max",
            "n_points_total", "n_points_true_inrange", "n_points_true_outrange"
        ])
    
    total_in_nodes = np.sum(results_df["status"] == "in_range")
    total_out_nodes = np.sum(results_df["status"] == "out_range")

    points_in_inrange_nodes = int(results_df.loc[results_df["status"] == "in_range", "n_points_total"].sum())
    points_in_outrange_nodes = int(results_df.loc[results_df["status"] == "out_range", "n_points_total"].sum())

    unique_points_classified = len(counted_indices)
    unclassified_points = total_points - unique_points_classified

    # Get the unclassified points
    unclassified_indices = [i for i in range(total_points) if i not in counted_indices]
    unclassified_points_df = simulation_log.iloc[unclassified_indices][[param_x, param_y]]

    print(nodes_visited_dict)
    summary = {
        "depth_level": depth_level,
        "total_nodes": len(results_df),
        "in_range_nodes": total_in_nodes,
        "out_range_nodes": total_out_nodes,
        "points_in_inrange_nodes": points_in_inrange_nodes,
        "points_in_outrange_nodes": points_in_outrange_nodes,
        "unique_points_classified": unique_points_classified,
        "unclassified_points": unclassified_points,
        "fraction_inrange_nodes_points": (
            results_df.loc[results_df["status"] == "in_range", "n_points_true_inrange"].sum() / total_points
            if total_points > 0 else 0
        ),
        "fraction_outrange_nodes_points": (
            results_df.loc[results_df["status"] == "out_range", "n_points_true_outrange"].sum() / total_points
            if total_points > 0 else 0
        ),
        "fraction_unclassified_points": (
            unclassified_points / total_points if total_points > 0 else 0
        ),
        # "nodes_visited": nodes_visited,  # Total nodes visited at this depth
        # "nodes_computed": nodes_computed,  # Total nodes computed at this depth
        "nodes_visited": nodes_visited_dict[depth_level-1]['all_nodes'],  # Total nodes visited at this depth
        "nodes_computed": nodes_visited_dict[depth_level-1]['sampled_points'],  # Total nodes computed at this depth
        # 'all_nodes': nodes_visited_dict[depth_level-1]['all_nodes'],
        # 'sampled_points': nodes_visited_dict[depth_level-1]['all_nodes'],
        
    }

    # display(unclassified_points_df)

    return results_df, pd.Series(summary)



import pandas as pd
import numpy as np

def compute_quadtree_metrics(results_df, summary):
    """
    Compute precision, recall, and accuracy for quadtree classification
    against the ground truth (simulation_log), based on output from
    `evaluate_quadtree_vs_simulation_log`.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output DataFrame from evaluate_quadtree_vs_simulation_log, containing:
          - status (in_range / out_range)
          - n_points_true_inrange
          - n_points_true_outrange
    summary : pd.Series
        Summary output including total_points or unclassified counts.

    Returns
    -------
    pd.Series
        Metrics including precision, recall, accuracy, and unclassified fraction.
    """

    # --- Count confusion matrix elements ---
    TP = results_df.loc[results_df["status"] == "in_range", "n_points_true_inrange"].sum()
    FP = results_df.loc[results_df["status"] == "in_range", "n_points_true_outrange"].sum()
    TN = results_df.loc[results_df["status"] == "out_range", "n_points_true_outrange"].sum()
    FN = results_df.loc[results_df["status"] == "out_range", "n_points_true_inrange"].sum()

    total_classified = TP + FP + TN + FN
    total_points = summary.get("unique_points_classified", total_classified) + summary.get("unclassified_points", 0)

    # --- Metrics ---
    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    accuracy = (TP + TN) / total_classified if total_classified > 0 else np.nan
    mcc = (
        (TP * TN - FP * FN) /
        np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        if (TP + FP) > 0 and (TP + FN) > 0 and (TN + FP) > 0 and (TN + FN) > 0 else np.nan
    )

    unclassified_fraction = (
        summary.get("unclassified_points", 0) / total_points if total_points > 0 else np.nan
    )

    return pd.Series({
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "precision_inrange": precision,
        "recall_inrange": recall,
        "accuracy": accuracy,
        "mcc": mcc,
        "unclassified_fraction": unclassified_fraction,
        "total_points": int(total_points),
        "classified_points": int(total_classified),
        "unclassified_points": int(summary.get("unclassified_points", 0)),
    })



def plot_quadtree_metrics_over_depth(quadtree_nodes, simulation_log, params, nodes_visited_dict, max_depth, metric="accuracy"):
    """
    Plots quadtree metrics (accuracy, precision, or recall) and the fraction of classified points
    over the depth levels of the quadtree.

    Args:
        all_nodes (list[QuadNode]): All nodes from the quadtree.
        simulation_log (pd.DataFrame): DataFrame containing ground-truth grid points.
        params (dict): Contains 'params_to_change' with two parameters being varied.
        max_depth (int): Maximum depth level to evaluate.
        metric (str): The metric to plot on the left y-axis. Options: "accuracy", "precision", "recall".

    Returns:
        None
    """
    # Initialize lists to store metrics for each depth level
    depths = []
    metric_values = []
    fractions_classified = []

    # Map the metric argument to the correct key in the metrics object
    metric_key_map = {
        "accuracy": "accuracy",
        "mcc": "mcc",
        "precision": "precision_inrange",
        "recall": "recall_inrange"
    }
    if metric not in metric_key_map:
        raise ValueError("Invalid metric. Choose from 'accuracy', 'precision', or 'recall'.")

    metric_key = metric_key_map[metric]

    # Iterate over depth levels
    for depth in range(1, max_depth + 1):
        # Evaluate the quadtree at the current depth level
        results_df, summary = evaluate_quadtree_vs_simulation_log(quadtree_nodes, simulation_log, nodes_visited_dict, params, depth)

        # Compute metrics for the current depth level
        metrics = compute_quadtree_metrics(results_df, summary)

        # Append depth and metrics
        depths.append(depth)
        metric_values.append(metrics[metric_key])  # Use the correct key for the selected metric
        fractions_classified.append(1 - metrics["unclassified_fraction"])  # Fraction classified

    # Plot the metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the selected metric on the left y-axis
    ax1.set_xlabel("Tree Depth Level")
    ax1.set_ylabel(metric.capitalize(), color="tab:blue")
    ax1.plot(depths, metric_values, label=metric.capitalize(), color="tab:blue", marker="o")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Plot the fraction of classified points on the right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction of Classified Points", color="tab:orange")
    ax2.plot(depths, fractions_classified, label="Fraction Classified", color="tab:orange", marker="x")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Add a title and legend
    fig.suptitle(f"Quadtree Metrics vs Depth (Metric: {metric.capitalize()})", fontsize=14)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Show the plot
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(os.path.join(params['experiment_output_dir'], f'quadtree_{metric}_vs_depth.png'))
    plt.show()



    

def generate_and_plot_quadtree_metrics(quadtree_nodes, simulation_log, params, nodes_visited_dict, max_depth):
    """
    Generate quadtree metrics for each depth level and plot the required metrics.

    Args:
        quadtree_nodes (list[QuadNode]): All nodes from the quadtree.
        simulation_log (pd.DataFrame): DataFrame containing grid search points.
        params (dict): Contains 'params_to_change' with two parameters being varied.
        max_depth (int): Maximum depth level to evaluate.
        nodes_visited (dict): Dictionary containing the number of nodes visited and evaluated at each depth.

    Returns:
        None
    """
    # Initialize lists to store metrics
    quadtree_metrics = []

    # Generate metrics for each depth level
    for depth in range(1, max_depth + 1):
        # depth += 1
        results_df, summary = evaluate_quadtree_vs_simulation_log(quadtree_nodes, simulation_log, nodes_visited_dict, params, depth)
        metrics = compute_quadtree_metrics(results_df, summary)
        quadtree_metrics.append(metrics)

    # Extract depth levels from nodes_visited
    # depth_levels = list(nodes_visited.keys())

    depth_levels = [d for d in range(1, max_depth + 1)]

    # Extract metrics for plotting
    grid_search_points = len(simulation_log[simulation_log['algorithm'] == 'grid_search'])  # Total number of grid search points (horizontal line)
    classified_points = [
        metrics["classified_points"] for metrics in quadtree_metrics
    ]  # Absolute number of classified points
    all_nodes_visited = [nodes_visited_dict[depth-1]["all_nodes"] for depth in depth_levels]  # All nodes visited
    sampled_points =    [nodes_visited_dict[depth-1]["sampled_points"] for depth in depth_levels]  # Evaluated points

    # Plot the metrics
    plt.figure(figsize=(10, 6))

    # Horizontal line for grid search points
    plt.axhline(y=grid_search_points, color="tab:blue", linestyle="--", label="Grid Search Points (Total)")

    # Absolute number of classified points
    plt.plot(
        depth_levels,
        classified_points,
        label="Classified Points",
        color="tab:orange",
        marker="o",
    )

    # # All nodes visited
    # plt.plot(
    #     depth_levels,
    #     all_nodes_visited,
    #     label="All Nodes Visited",
    #     color="tab:green",
    #     marker="s",
    # )

    # Evaluated points
    plt.plot(
        depth_levels,
        sampled_points,
        label="Evaluated Points",
        color="tab:red",
        marker="x",
    )

    # Add labels, legend, and title
    plt.xlabel("Tree Depth Level", fontsize=12)
    plt.ylabel("Number of Parameter Settings", fontsize=12)
    plt.title("Quadtree Evaluation Metrics vs Tree Depth Level", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(os.path.join(params['experiment_output_dir'], 'quadtree_evaluation_metrics.png'))

    # Show the plot
    plt.show()




# def compute_and_save_quadtree_metrics(quadtree_nodes, simulation_log, params, max_depth, output_file_name='quadtree_metrics_over_depth.csv'):
#     """
#     Computes quadtree metrics across all depth levels and saves the results to a CSV file.

#     Args:
#         quadtree_nodes (list): List of quadtree nodes.
#         simulation_log (pd.DataFrame): The simulation log DataFrame.
#         params (dict): Parameters for the quadtree evaluation.
#         max_depth (int): Maximum depth of the quadtree to evaluate.
#         output_csv_path (str): Path to save the resulting CSV file.

#     Returns:
#         pd.DataFrame: DataFrame containing all quadtree metrics across depth levels.
#     """
#     all_metrics = []

#     for depth in range(1, max_depth + 1):
#         # Evaluate the quadtree at the current depth
#         results_df, summary = evaluate_quadtree_vs_simulation_log(quadtree_nodes, simulation_log, params, depth)
        
#         # Compute the metrics for the current depth
#         metrics = compute_quadtree_metrics(results_df, summary)
        
#         # Add the depth level to the metrics
#         metrics['depth'] = depth
        
#         # Append the metrics to the list
#         all_metrics.append(metrics)

#     # Combine all metrics into a single DataFrame
#     all_metrics_df = pd.DataFrame(all_metrics)

#     output_csv_path = os.path.join(params['experiment_output_dir'], output_file_name)

#     # Save the DataFrame to a CSV file
#     all_metrics_df.to_csv(output_csv_path, index=False)

#     print(f"Quadtree metrics saved to {output_csv_path}")
#     return all_metrics_df




def compute_and_save_quadtree_metrics(quadtree_nodes, simulation_log, params, nodes_visited_dict, max_depth, algorithm, output_file_name='quadtree_metrics_over_depth.csv'):
    """
    Computes quadtree metrics across all depth levels, including the number of evaluated points
    and the classified fraction, and saves the results to a CSV file.

    Args:
        quadtree_nodes (list): List of quadtree nodes.
        simulation_log (pd.DataFrame): The simulation log DataFrame.
        params (dict): Parameters for the quadtree evaluation.
        max_depth (int): Maximum depth of the quadtree to evaluate.
        output_csv_path (str): Path to save the resulting CSV file.

    Returns:
        pd.DataFrame: DataFrame containing all quadtree metrics across depth levels.
    """
    all_metrics = []

    for depth in range(1, max_depth + 1):
        # Evaluate the quadtree at the current depth
        if algorithm == 'quadtree':
            results_df, summary = evaluate_quadtree_vs_simulation_log(quadtree_nodes, simulation_log, nodes_visited_dict, params, depth)
        if algorithm == 'hyperquadtree':
            results_df, summary = evaluate_hyperquadtree_vs_simulation_log(quadtree_nodes, simulation_log, nodes_visited_dict, params, depth)
        
        # Compute the metrics for the current depth
        metrics = compute_quadtree_metrics(results_df, summary)
        
        # Add the depth level to the metrics
        metrics['depth'] = depth

        # # Add the number of evaluated points at this depth (from the summary)
        # metrics['num_evaluated_points'] = summary.get('num_evaluated_points', 0)

        # Add the total number of nodes at this depth (from the summary)
        metrics['nodes_computed'] = summary.get('nodes_computed', 0)

        # Compute the classified fraction (fraction of nodes classified as "in_range" or "out_range")
        # num_classified = summary.get('num_in_range', 0) + summary.get('num_out_range', 0)
        # total_nodes = summary.get('total_nodes', 1)  # Avoid division by zero
        # classified_fraction = num_classified / total_nodes
        # metrics['classified_fraction'] = classified_fraction
        metrics['classified_fraction'] = 1-metrics['unclassified_fraction']
        # print(metrics['unclassified_fraction'])

        # Append the metrics to the list
        all_metrics.append(metrics)

    # Combine all metrics into a single DataFrame
    all_metrics_df = pd.DataFrame(all_metrics)

    output_csv_path = os.path.join(params['experiment_output_dir'], output_file_name)

    # Save the DataFrame to a CSV file
    all_metrics_df.to_csv(output_csv_path, index=False)

    print(f"Quadtree metrics saved to {output_csv_path}")
    return all_metrics_df








###############################################################
###############################################################



def evaluate_hyperquadtree_vs_simulation_log(all_nodes, simulation_log, nodes_visited_dict, params, depth_level):
    """
    Evaluate how many ground-truth grid points (from simulation_log) fall inside
    in_range vs out_range hyperquadtree nodes, up to a given depth level.

    Parameters
    ----------
    all_nodes : list[HyperQuadNode]
        All nodes from the hyperquadtree. Each node must have:
        bounds, depth, is_leaf, status.
    simulation_log : pd.DataFrame
        DataFrame containing at least:
            - param_1, param_2, ..., param_n : grid search coordinates
            - status : True (in-range) or False (out-of-range)
    nodes_visited_dict : dict
        Dictionary tracking nodes visited and sampled points at each depth.
    params : dict
        Contains 'params_to_change' with n parameters being varied.
    depth_level : int
        Only consider hyperquadtree nodes up to this depth.

    Returns
    -------
    results_df : pd.DataFrame
        Per-node statistics.
    summary : pd.Series
        Aggregated counts including number of unclassified grid points,
        nodes visited, and nodes computed.
    """
    # Extract parameter names
    param_names = list(params['params_to_change'].keys())
    n_dims = len(param_names)

    # Keep only grid search entries
    simulation_log = simulation_log[simulation_log['algorithm'] == 'grid_search'].copy()

    # Extract numpy arrays for faster computation
    grid_coords = [simulation_log[param].to_numpy() for param in param_names]
    grid_status = simulation_log['status'].astype(bool).to_numpy()
    total_points = len(simulation_log)

    counted_indices = set()
    results = []

    # Initialize counters for visited and computed nodes
    nodes_visited = 0
    nodes_computed = 0

    for node in all_nodes:
        # Count the node as visited if it is within the depth level
        if node.depth <= depth_level:
            nodes_visited += 1

        # Only process leaf nodes up to target depth
        if not node.is_leaf or node.depth > depth_level:
            continue

        # Count the node as computed if it is a leaf node at the depth level
        nodes_computed += 1

        if node.status not in ("in_range", "out_range"):
            # Skip mixed or undefined nodes when classifying points
            continue

        # Identify points inside this node
        inside_mask = np.ones(total_points, dtype=bool)
        for dim, (dim_min, dim_max) in enumerate(node.bounds):
            inside_mask &= (grid_coords[dim] >= dim_min) & (grid_coords[dim] <= dim_max)

        inside_indices = np.where(inside_mask)[0]

        # Remove duplicates (points already counted)
        new_indices = [i for i in inside_indices if i not in counted_indices]
        counted_indices.update(new_indices)
        new_indices = np.array(new_indices)

        # Count how many are True vs False in the ground truth
        if len(new_indices) > 0:
            in_true = np.sum(grid_status[new_indices])
            out_true = len(new_indices) - in_true
        else:
            in_true = out_true = 0

        results.append({
            "depth": node.depth,
            "status": node.status,
            **{f"dim_{dim}_min": node.bounds[dim][0] for dim in range(n_dims)},
            **{f"dim_{dim}_max": node.bounds[dim][1] for dim in range(n_dims)},
            "n_points_total": len(new_indices),
            "n_points_true_inrange": in_true,
            "n_points_true_outrange": out_true,
        })

    results_df = pd.DataFrame(results)

    # ---- Aggregate summary ----
    if results_df.empty:
        results_df = pd.DataFrame(columns=[
            "depth", "status", *[f"dim_{dim}_min" for dim in range(n_dims)],
            *[f"dim_{dim}_max" for dim in range(n_dims)],
            "n_points_total", "n_points_true_inrange", "n_points_true_outrange"
        ])

    total_in_nodes = np.sum(results_df["status"] == "in_range")
    total_out_nodes = np.sum(results_df["status"] == "out_range")

    points_in_inrange_nodes = int(results_df.loc[results_df["status"] == "in_range", "n_points_total"].sum())
    points_in_outrange_nodes = int(results_df.loc[results_df["status"] == "out_range", "n_points_total"].sum())

    unique_points_classified = len(counted_indices)
    unclassified_points = total_points - unique_points_classified

    summary = {
        "depth_level": depth_level,
        "total_nodes": len(results_df),
        "in_range_nodes": total_in_nodes,
        "out_range_nodes": total_out_nodes,
        "points_in_inrange_nodes": points_in_inrange_nodes,
        "points_in_outrange_nodes": points_in_outrange_nodes,
        "unique_points_classified": unique_points_classified,
        "unclassified_points": unclassified_points,
        "fraction_inrange_nodes_points": (
            results_df.loc[results_df["status"] == "in_range", "n_points_true_inrange"].sum() / total_points
            if total_points > 0 else 0
        ),
        "fraction_outrange_nodes_points": (
            results_df.loc[results_df["status"] == "out_range", "n_points_true_outrange"].sum() / total_points
            if total_points > 0 else 0
        ),
        "fraction_unclassified_points": (
            unclassified_points / total_points if total_points > 0 else 0
        ),
        "nodes_visited": nodes_visited_dict[depth_level - 1]['all_nodes'],
        "nodes_computed": nodes_visited_dict[depth_level - 1]['sampled_points'],
    }

    return results_df, pd.Series(summary)














# -------------------------------------


# def calculate_normalized_space(simulation_log, params):
#     """
#     Calculate the normalized space between in_range, out_range, and mixed cubes.

#     Args:
#         gs_df (pd.DataFrame): DataFrame containing grid search points with columns:
#                               ['arriaval_distr_mean', 'resource_count_unified_resource_profile',
#                                'branching_probability_node_51629->node_75a93', 'status'].

#     Returns:
#         dict: A dictionary with the normalized space for each category:
#               {'in_range': float, 'out_range': float, 'mixed': float}.
#     """
#     # Ensure the DataFrame is sorted by dimensions for proper cube formation
#     gs_df = simulation_log[simulation_log['algorithm'] == 'grid_search'].copy()
#     gs_df = gs_df.sort_values(by=list(params['params_to_change'].keys())).reset_index(drop=True)

#     # Extract unique values for each dimension
#     dim_0_vals = gs_df['arriaval_distr_mean'].unique()
#     dim_1_vals = gs_df['resource_count_unified_resource_profile'].unique()
#     dim_2_vals = gs_df['branching_probability_node_51629->node_75a93'].unique()

#     # Initialize total volumes for each category
#     total_volumes = {'in_range': 0.0, 'out_range': 0.0, 'mixed': 0.0}

#     # Iterate through all possible cubes
#     for i in range(len(dim_0_vals) - 1):
#         for j in range(len(dim_1_vals) - 1):
#             for k in range(len(dim_2_vals) - 1):
#                 # Define the corner points of the cube
#                 cube_points = gs_df[
#                     (gs_df['arriaval_distr_mean'].between(dim_0_vals[i], dim_0_vals[i + 1], inclusive='both')) &
#                     (gs_df['resource_count_unified_resource_profile'].between(dim_1_vals[j], dim_1_vals[j + 1], inclusive='both')) &
#                     (gs_df['branching_probability_node_51629->node_75a93'].between(dim_2_vals[k], dim_2_vals[k + 1], inclusive='both'))
#                 ]

#                 # Skip if the cube is not fully defined (less than 8 corner points)
#                 if len(cube_points) < 8:
#                     continue

#                 # Check the status of all corner points
#                 statuses = cube_points['status'].values

#                 # Classify the cube
#                 if np.all(statuses):  # All points are in_range
#                     category = 'in_range'
#                 elif not np.any(statuses):  # All points are out_range
#                     category = 'out_range'
#                 else:  # Mixed points
#                     category = 'mixed'

#                 # Calculate the volume of the cube
#                 volume = (
#                     (dim_0_vals[i + 1] - dim_0_vals[i]) *
#                     (dim_1_vals[j + 1] - dim_1_vals[j]) *
#                     (dim_2_vals[k + 1] - dim_2_vals[k])
#                 )

#                 # Add the volume to the corresponding category
#                 total_volumes[category] += volume

#     # Normalize the volumes
#     total_volume = sum(total_volumes.values())
#     if total_volume > 0:
#         for category in total_volumes:
#             total_volumes[category] /= total_volume

#     return total_volumes

def calculate_normalized_space(simulation_log, params):
    """
    Calculate the normalized space between in_range, out_range, and mixed cubes.

    Args:
        simulation_log (pd.DataFrame): DataFrame containing grid search points with columns
                                       corresponding to params['params_to_change'] and 'status'.
        params (dict): Dictionary containing 'params_to_change', which specifies the parameters
                       being varied.

    Returns:
        dict: A dictionary with the normalized space for each category:
              {'in_range': float, 'out_range': float, 'mixed': float}.
    """
    # Ensure the DataFrame is sorted by dimensions for proper cube formation
    param_keys = list(params['params_to_change'].keys())
    gs_df = simulation_log[simulation_log['algorithm'] == 'grid_search'].copy()
    gs_df = gs_df.sort_values(by=param_keys).reset_index(drop=True)

    # Extract unique values for each dimension
    unique_values = {key: gs_df[key].unique() for key in param_keys}

    # Initialize total volumes for each category
    total_volumes = {'in_range': 0.0, 'out_range': 0.0, 'mixed': 0.0}

    # Recursive function to iterate through all dimensions
    def iterate_cubes(dim_idx, current_ranges):
        if dim_idx == len(param_keys):  # Base case: all dimensions processed
            # Define the corner points of the cube
            cube_points = gs_df
            for dim, (start, end) in enumerate(current_ranges):
                key = param_keys[dim]
                cube_points = cube_points[
                    cube_points[key].between(start, end, inclusive='both')
                ]

            # Skip if the cube is not fully defined (less than 2^n corner points)
            if len(cube_points) < 2 ** len(param_keys):
                return

            # Check the status of all corner points
            statuses = cube_points['status'].values

            # Classify the cube
            if np.all(statuses):  # All points are in_range
                category = 'in_range'
            elif not np.any(statuses):  # All points are out_range
                category = 'out_range'
            else:  # Mixed points
                category = 'mixed'

            # Calculate the volume of the cube
            volume = 1.0
            for dim, (start, end) in enumerate(current_ranges):
                volume *= (end - start)

            # Add the volume to the corresponding category
            total_volumes[category] += volume
        else:
            # Recursive case: iterate through the current dimension
            key = param_keys[dim_idx]
            values = unique_values[key]
            for i in range(len(values) - 1):
                iterate_cubes(dim_idx + 1, current_ranges + [(values[i], values[i + 1])])

    # Start the recursive iteration
    iterate_cubes(0, [])

    # Normalize the volumes
    total_volume = sum(total_volumes.values())
    if total_volume > 0:
        for category in total_volumes:
            total_volumes[category] /= total_volume

    return total_volumes