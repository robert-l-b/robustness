#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

def plot_param_relationships(results_df, algorithms, params):
    """
    Plots the relationship between two parameters for multiple algorithms, highlighting points in and out of the target range.

    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
        params_to_change (list): A list of two parameter names to plot on the x and y axes.
        algorithms (list): A list of algorithm names to differentiate results.
        params (dict): A dictionary containing simulation parameters, including "target_range".
        output_figures (str): Path to save the output figure.

    Returns:
        None
    """

    # filter the algorithms list for those present in the results_df
    algorithms = [alg for alg in algorithms if alg in results_df['algorithm'].unique()]

    target_ppi = params['viz']['2d_params']['target_ppi']
    # target_range = params['target_range'][target_ppi]

    # if len(params_to_change) != 2:
    #     raise ValueError("params_to_change must contain exactly two parameters.")

    # x_param, y_param = params_to_change
    x_param, y_param = params['viz']['2d_params']['x_param'], params['viz']['2d_params']['y_param']

    # Define colors for remaining algorithms
    colors = ['blue', 'purple', 'black']
    color_map = {alg: colors[i % len(colors)] for i, alg in enumerate(algorithms) if alg not in ["grid_search", "orig_run"]}

    plt.figure(figsize=(10, 8))

    # Plot "orig_run" first if it exists
    if "orig_run" in algorithms:
        orig_run_results = results_df[results_df['algorithm'] == "orig_run"]
        plt.scatter(
            orig_run_results[x_param],
            orig_run_results[y_param],
            color='darkblue',
            s=100,  # Larger scatter points
            label="orig_run",
            alpha=0.9
        )

    # Plot "grid_search" next
    if "grid_search" in algorithms:
        grid_search_results = results_df[results_df['algorithm'] == "grid_search"]

        # Separate points in and out of the target range
        # in_range = grid_search_results[
        #     (grid_search_results['target_ppi_val'] >= target_range[0]) &
        #     (grid_search_results['target_ppi_val'] <= target_range[1])
        # ]
        # out_of_range = grid_search_results[
        #     (grid_search_results['target_ppi_val'] < target_range[0]) |
        #     (grid_search_results['target_ppi_val'] > target_range[1])
        # ]
        in_range     = grid_search_results[grid_search_results['status'] == 'in']
        out_of_range = grid_search_results[grid_search_results['status'] == 'out']

        # Plot in-range points in green
        plt.scatter(
            in_range[x_param],
            in_range[y_param],
            color='green',
            label="grid_search (In Range)",
            alpha=0.7
        )

        # Plot out-of-range points in red
        plt.scatter(
            out_of_range[x_param],
            out_of_range[y_param],
            color='red',
            label="grid_search (Out of Range)",
            alpha=0.7
        )

    # Plot remaining algorithms
    for algorithm in algorithms:
        if algorithm not in ["grid_search", "orig_run"]:
            algo_results = results_df[results_df['algorithm'] == algorithm]

            # Separate points in and out of the target range
            # in_range = algo_results[
            #     (algo_results['target_ppi_val'] >= target_range[0]) &
            #     (algo_results['target_ppi_val'] <= target_range[1])
            # ]
            # out_of_range = algo_results[
            #     (algo_results['target_ppi_val'] < target_range[0]) |
            #     (algo_results['target_ppi_val'] > target_range[1])
            # ]
            in_range     = algo_results[algo_results['status'] == 'in']
            out_of_range = algo_results[algo_results['status'] == 'out']

            # Plot in-range points
            plt.scatter(
                in_range[x_param],
                in_range[y_param],
                color=color_map[algorithm],
                # label=f"{algorithm} (In Range)",
                alpha=0.7,
                marker = 'x',
                s=80
            )

            # Plot out-of-range points
            plt.scatter(
                out_of_range[x_param],
                out_of_range[y_param],
                color=color_map[algorithm],
                edgecolor='black',
                label=f"{algorithm}",
                alpha=0.7,
                marker = 'x',
                s=80
            )

    # Add labels, legend, and title
    plt.xlabel(x_param.replace("_", " ").capitalize())
    plt.ylabel(y_param.replace("_", " ").capitalize())
    plt.title("Parameter Relationships Across Algorithms")
    plt.legend()
    plt.grid(True)

    

    # Save the plot
    fig_name = f'Scatter_{x_param}_vs_{y_param}_{algorithms}'
    for extension in params['viz']['figure_extensions']:
        plt.savefig(os.path.join(params['viz']['output_figures_path'], fig_name + extension))

    plt.show()






def plot_3d_results_with_target_range(results_df, params, algorithm, x_col, z_col, y_col,
                                      target_range, y_max_value='max',
                                      use_log_scale=False, draw_plane=False):
    """
    Plots a 3D graph with x, z as the independent variables and y as the dependent variable,
    including target range as planes or color coding. Optionally, draw a plane connecting the scattered dots.
    """

    # --- Filter dataframe ---
    results_df = results_df[results_df['algorithm'] == algorithm]
    # if x_col not in results_df.columns or z_col not in results_df.columns or y_col not in results_df.columns:
    #     print(f"Missing required columns: {x_col}, {z_col}, or {y_col}")
    #     return

    filtered_df = results_df.dropna(subset=[x_col, z_col])
    x = filtered_df[x_col].values
    z = filtered_df[z_col].values

    y = np.array([d[y_col] for d in  filtered_df['target_ppi_means']])
    lower_bound, upper_bound = target_range

    # --- Figure setup ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Scatter or surface ---
    if draw_plane:
        cmap = plt.cm.get_cmap('RdYlGn')
        ax.plot_trisurf(x, z, y, cmap=cmap, alpha=0.7, shade=True, antialiased=True)
    else:
        colors = np.where((y >= lower_bound) & (y <= upper_bound), 'g', 'r')
        ax.scatter(x, z, y, c=colors, alpha=0.8)

    # --- Target range planes ---
    X, Z = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(z.min(), z.max(), 10))
    if lower_bound > 0:
        ax.plot_surface(X, Z, np.full_like(X, lower_bound), color='blue', alpha=0.25)
    ax.plot_surface(X, Z, np.full_like(X, upper_bound), color='orange', alpha=0.25)

    # --- Labels and title ---
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(z_col.replace("_", " ").title(), fontsize=12)
    ax.set_zlabel(y_col.replace("_", " ").title(), fontsize=12)
    
    title = f"3D Plot of {y_col} vs {x_col} and {z_col} (Target Range)"
    # ax.set_title(title, fontsize=14)
    
    ax.view_init(elev=30, azim=45)

    # --- Handle Y (Z-axis) scale ---
    y_min, y_max = y.min(), y.max()


    if use_log_scale:
        y_min = max(y_min, 1e-6)
        y_max = float(y_max)
        ax.set_zscale('log')

        # --- Determine a "nice" top tick (always >= y_max) ---
        exponent = int(np.floor(np.log10(y_max)))
        fraction = y_max / 10**exponent
        if fraction <= 1.5:
            nice_top_tick = 2 * 10**exponent
        elif fraction <= 3:
            nice_top_tick = 3 * 10**exponent
        elif fraction <= 7:
            nice_top_tick = 5 * 10**exponent
        else:
            nice_top_tick = 10 * 10**exponent
        if nice_top_tick < y_max:
            nice_top_tick = 10**(exponent + 1)

        # --- Generate candidate ticks (log spacing) ---
        min_exp = 0
        max_exp = int(np.floor(np.log10(nice_top_tick)))
        candidate_ticks = [10 ** e for e in range(min_exp, max_exp + 1)]

        # --- Apply gap factor to reduce clutter ---
        gap_factor = 10
        major_ticks = []
        last_tick = None
        for t in candidate_ticks:
            if last_tick is None or t / last_tick >= gap_factor:
                major_ticks.append(t)
                last_tick = t

        # --- Ensure top tick is included ---
        if nice_top_tick not in major_ticks:
            major_ticks.append(nice_top_tick)

        # --- Replace first half + 1 with zero ---
        n = len(major_ticks)
        num_to_replace = n // 2 + 2
        for i in range(num_to_replace):
            major_ticks[i] = 0.0

        # --- Deduplicate and ensure single zero tick ---
        major_ticks = sorted(list(set(major_ticks)))
        if major_ticks.count(0.0) > 1:
            major_ticks = [0.0] + [t for t in major_ticks if t != 0.0]

        # --- Apply ticks and limits ---
        ax.set_zticks(major_ticks)
        ax.set_zlim(bottom=0.0, top=nice_top_tick)

        # --- Label formatting ---
        def pretty_label(val):
            if val == 0:
                return "0"
            elif val >= 1000:
                return f"{int(val):,}"
            elif val >= 1:
                return f"{val:g}"
            else:
                return f"{val:.3g}"

        ax.zaxis.set_major_formatter(FuncFormatter(lambda v, _: pretty_label(v)))

        # --- Gridlines and aesthetics ---
        ax.zaxis.set_minor_locator(LogLocator(base=10, subs=[]))
        ax.grid(which='minor', visible=False)
        ax.grid(which='major', linestyle='-', linewidth=0.4, alpha=0.3)
        ax.tick_params(axis='z', pad=10)


    else:
        # Linear scale
        ax.set_zlim(0, y_max_value if isinstance(y_max_value, (int, float)) else y_max)

        # Adaptive, rounded ticks
        locator = MaxNLocator(nbins=6, prune=None)
        ax.zaxis.set_major_locator(locator)

        # Clean formatting
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.zaxis.set_major_formatter(formatter)

    # # --- Add target range annotation ---
    # ax.text2D(0.05, 0.95, f"Target Range: {lower_bound}–{upper_bound}",
    #           transform=ax.transAxes, fontsize=10)
    
    fig.tight_layout()
    
    # Save the figure
    fig_name = f'3D_{algorithm}_yMax:{y_max_value}_{x_col}_{z_col}_vs_{y_col}'
    for extension in params['viz']['figure_extensions']:
        plt.savefig(os.path.join(params['viz']['output_figures_path'], fig_name + extension))


    plt.show()






def plot_3d_in_out(results_df, params, algorithm, use_connected_surface=False):
    """
    Plots a 3D graph with three parameters on the axes, visualizing in-range and out-of-range points.
    Optionally, plots out-of-range points as a connected red surface.

    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
        params (dict): A dictionary containing simulation parameters, including "viz".
        algorithm (str): The algorithm to filter the results for.

    Returns:
        None
    """
    # Extract 3D plotting parameters
    plot_params = params['viz']['3d_InOut_params']
    x_param = plot_params['x_param']
    y_param = plot_params['y_param']
    z_param = plot_params['z_param']

    # Extract parameter bounds
    x_bounds = params['params_to_change'][x_param]['values']
    y_bounds = params['params_to_change'][y_param]['values']
    z_bounds = params['params_to_change'][z_param]['values']

    # Filter the DataFrame for the given algorithm
    filtered_df = results_df[results_df['algorithm'] == algorithm]

    # Separate in-range and out-of-range points
    in_range = filtered_df[filtered_df['status'] == 'in']
    out_of_range = filtered_df[filtered_df['status'] == 'out']

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Parameter Space for {algorithm}")

    # Plot in-range points as green scatter points
    ax.scatter(in_range[x_param], in_range[y_param], in_range[z_param],
               c='green', label="In Range", alpha=0.8)

    # if use_connected_surface:
    #     # Plot out-of-range points as a connected red surface
    #     if len(out_of_range) > 0:
    #         x_out = out_of_range[x_param].values
    #         y_out = out_of_range[y_param].values
    #         z_out = out_of_range[z_param].values

    #         # Create a triangulated surface
    #         ax.plot_trisurf(x_out, y_out, z_out, color='red', alpha=0.7, label="Out of Range (Surface)")

    if use_connected_surface:
        # Plot out-of-range points as a connected red surface
        if len(out_of_range) >= 3:  # Ensure there are at least 3 points
            x_out = out_of_range[x_param].values
            y_out = out_of_range[y_param].values
            z_out = out_of_range[z_param].values

            # Check for duplicate points
            unique_points = np.unique(np.column_stack((x_out, y_out, z_out)), axis=0)
            if len(unique_points) >= 3:  # Ensure there are at least 3 unique points
                x_out, y_out, z_out = unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]
                ax.plot_trisurf(x_out, y_out, z_out, color='red', alpha=0.7, label="Out of Range (Surface)")
            else:
                print("Not enough unique points to create a surface.")
        else:
            print("Not enough points to create a surface.")


    else:
        # Plot out-of-range points as red scatter points
        ax.scatter(out_of_range[x_param], out_of_range[y_param], out_of_range[z_param],
                   c='red', label="Out of Range", alpha=0.8)

    # Set axis labels and limits
    ax.set_xlabel(f"{x_param} (Range: {x_bounds[0]} to {x_bounds[1]})")
    ax.set_ylabel(f"{y_param} (Range: {y_bounds[0]} to {y_bounds[1]})")
    ax.set_zlabel(f"{z_param} (Range: {z_bounds[0]} to {z_bounds[1]})")
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_zlim(z_bounds)

    # Add legend and grid
    ax.legend()
    ax.grid(True)

    # Save the plot
    fig_name = f"3D_Param_Space_{algorithm}_{'surface' if use_connected_surface else 'scatter'}"
    for extension in params['viz']['figure_extensions']:
        plt.savefig(os.path.join(params['viz']['output_figures_path'], fig_name + extension))

    # Show the plot
    plt.show()