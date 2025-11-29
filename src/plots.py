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



def get_label(param_name, label_dict):
    for key in label_dict.keys():
        if param_name.startswith(key):
            label = label_dict[key]
            return label
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def plot_gs_hqt_overlay_with_areas(results_df, params, hqt_df):
    """
    Plots the grid search results as the first layer (red and green dots), overlays other algorithms,
    and plots the areas from hqt_df based on their coordinates.

    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
        params (dict): A dictionary containing simulation parameters, including "viz".
        hqt_df (pd.DataFrame): The DataFrame containing hyperquadtree data.

    Returns:
        None
    """
    # Extract parameters for the plot
    x_param = params['viz']['2d_params']['x_param']
    y_param = params['viz']['2d_params']['y_param']
    algorithms = results_df['algorithm'].unique()

    # Extract plot variables
    plot_vars = params['viz']['plot_vars']
    label_dict = plot_vars['label_dict']
    x_label = get_label(x_param, label_dict)
    y_label = get_label(y_param, label_dict)
    gr_s = 25

    plt.figure(figsize=plot_vars['figure_size'])



    import matplotlib.ticker as ticker

    # Set the y-axis to print every 2nd tick
    plt.yticks(fontsize=plot_vars['tick_size'])
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))  # Show every 2nd tick on the y-axis

    # Format the x-axis from seconds to minutes
    def seconds_to_minutes(x, pos):
        return f"{x / 60:.1f}"  # Convert seconds to minutes and format to 1 decimal place

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_minutes))
    plt.xticks(fontsize=plot_vars['tick_size'])

    # Set the x-axis to display meaningful ticks (e.g., every 5 minutes)
    def seconds_to_minutes(x, pos):
        return f"{x / 60:.0f}"  # Convert seconds to minutes and format as an integer

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(900))  # Set ticks every 300 seconds (5 minutes)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_minutes))  # Format ticks as minutes

    # Set the font size for the x-axis ticks
    plt.xticks(fontsize=plot_vars['tick_size'])


    # Plot grid search results first
    if "grid_search" in algorithms:
        grid_search_results = results_df[results_df['algorithm'] == "grid_search"]

        # Separate points in and out of the target range
        in_range = grid_search_results[grid_search_results['status'] == 'in']
        out_of_range = grid_search_results[grid_search_results['status'] == 'out']

        # Plot in-range points in green
        plt.scatter(
            in_range[x_param],
            in_range[y_param],
            color='green',
            label="Grid Search (In Range)",
            alpha=plot_vars['transparancy'],
            s=gr_s
        )

        # Plot out-of-range points in red
        plt.scatter(
            out_of_range[x_param],
            out_of_range[y_param],
            color='red',
            label="Grid Search (Out of Range)",
            alpha=plot_vars['transparancy'],
            s = gr_s
        )

    # Overlay other algorithms
    for algorithm in algorithms:
        if algorithm != "grid_search":
            algo_results = results_df[results_df['algorithm'] == algorithm]

            # Separate points in and out of the target range
            in_range = algo_results[algo_results['status'] == 'in']
            out_of_range = algo_results[algo_results['status'] == 'out']

            # Plot in-range points
            plt.scatter(
                in_range[x_param],
                in_range[y_param],
                color=plot_vars['tree_node_colour'],
                alpha=plot_vars['transparancy'],
                label=f"{algorithm} (In Range)",
                s=gr_s+20
            )

            # Plot out-of-range points
            plt.scatter(
                out_of_range[x_param],
                out_of_range[y_param],
                color=plot_vars['tree_node_colour'],
                alpha=plot_vars['transparancy'],
                label=f"{algorithm} (Out of Range)",
                s=gr_s+20
            )

    # Plot areas from hqt_df
    for _, row in hqt_df.iterrows():
        if row['is_leaf']:  # Only plot areas for rows where is_leaf is True
            # Determine the color based on the status
            if row['status'] == 'mixed':
                color = 'white'
            elif row['status'] == 'in_range':
                color = 'green'
            elif row['status'] == 'out_range':
                color = 'red'
            else:
                continue  # Skip rows with unknown status

            # Create a rectangle for the area
            rect = patches.Rectangle(
                (row['dim_0_min'], row['dim_1_min']),  # Bottom-left corner
                row['dim_0_max'] - row['dim_0_min'],  # Width
                row['dim_1_max'] - row['dim_1_min'],  # Height
                linewidth=0,
                edgecolor=None,
                facecolor=color,
                alpha=0.1  # Transparency factor
            )
            plt.gca().add_patch(rect)

    # Add labels, legend, and title
    plt.xlabel(x_label, fontsize=plot_vars['label_size'])
    plt.ylabel(y_label, fontsize=plot_vars['label_size'])
    plt.xticks(fontsize=plot_vars['tick_size'])
    plt.yticks(fontsize=plot_vars['tick_size'])

    title = params['process_name'] + "_GS_HQT_overlay"
    if plot_vars['plot_title']:
        plt.title(title, fontsize=plot_vars['label_size'])

    if plot_vars['plot_legend']:
        plt.legend(
            loc=plot_vars['legend_location'],
            fontsize=plot_vars['legend_size']
        )

    plt.grid(alpha=plot_vars['grid_alpha'])

    plt.tight_layout()

    # Save the plot
    log_name = params.get('process_name', 'log_name')
    fig_name = f'{log_name}_Overlay_{x_param}_vs_{y_param}_with_areas'
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






def plot_3d_in_out(results_df, params, algorithm, use_connected_surface=False, angles=(45, 45), show_points=None):
    """
    Plots a 3D graph with three parameters on the axes, visualizing in-range and out-of-range points.
    Optionally, plots out-of-range points as a connected red surface.

    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
        params (dict): A dictionary containing simulation parameters, including "viz".
        algorithm (str): The algorithm to filter the results for.
        use_connected_surface (bool): Whether to plot out-of-range points as a connected red surface.
        angles (tuple): Tuple specifying the azimuth angles for rotating the plot.
        show_points (str or list): Whether to show only "in" points, only "out" points, or both (["in", "out"]).

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

    # Determine which points to show based on the `show_points` parameter
    show_in = show_points in ["in", ["in", "out"]]
    show_out = show_points in ["out", ["in", "out"]]

    # Set azimuth angles for rotating the plot
    for azim_angle in range(angles[0], angles[1] + 1, 20):
        # Create the 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=azim_angle)
        ax.set_title(f"3D Parameter Space for {algorithm}")

        # Plot in-range points as green scatter points if `show_points` includes "in"
        if show_in:
            ax.scatter(in_range[x_param], in_range[y_param], in_range[z_param],
                       c='green', label="In Range", alpha=0.8)

        # Plot out-of-range points as a connected red surface or scatter points if `show_points` includes "out"
        if show_out:
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

        plt.tight_layout()

        # Save the plot
        fig_name = f"3D_Param_Space_{algorithm}_{'surface' if use_connected_surface else 'scatter'}_{show_points}_azim{azim_angle}"
        for extension in params['viz']['figure_extensions']:
            plt.savefig(os.path.join(params['viz']['output_figures_path'], fig_name + extension))

        # Show the plot
        plt.show()


import plotly.graph_objects as go
from scipy.spatial import Delaunay
import numpy as np

def plot_3d_in_out_interactive(results_df, params, algorithm, use_connected_surface=None, show_points=None):
    """
    Creates an interactive 3D plot with three parameters on the axes, visualizing in-range and out-of-range points.
    Optionally, plots in-range or out-of-range points as a connected surface.

    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
        params (dict): A dictionary containing simulation parameters, including "viz".
        algorithm (str): The algorithm to filter the results for.
        use_connected_surface (str or None): Whether to plot a connected surface for "in" (green), "out" (red), or None.
        show_points (str or None): Whether to show only "in" points, only "out" points, or both (None).

    Returns:
        None
    """
    # Extract 3D plotting parameters
    plot_params = params['viz']['3d_InOut_params']
    x_param = plot_params['x_param']
    y_param = plot_params['y_param']
    z_param = plot_params['z_param']

    # Filter the DataFrame for the given algorithm
    filtered_df = results_df[results_df['algorithm'] == algorithm]

    # Separate in-range and out-of-range points
    in_range = filtered_df[filtered_df['status'] == 'in']
    out_of_range = filtered_df[filtered_df['status'] == 'out']

    # Create the 3D scatter plot
    fig = go.Figure()

    # Add in-range points (green) if show_points is "in" or None
    if show_points in ["in", None]:
        fig.add_trace(go.Scatter3d(
            x=in_range[x_param],
            y=in_range[y_param],
            z=in_range[z_param],
            mode='markers',
            marker=dict(size=5, color='green'),
            name='In Range'
        ))

    # Add out-of-range points (red) if show_points is "out" or None
    if show_points in ["out", None]:
        fig.add_trace(go.Scatter3d(
            x=out_of_range[x_param],
            y=out_of_range[y_param],
            z=out_of_range[z_param],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Out of Range'
        ))

    # Add surface based on use_connected_surface
    if use_connected_surface == "in" and len(in_range) >= 3:
        x_in = in_range[x_param].values
        y_in = in_range[y_param].values
        z_in = in_range[z_param].values

        # Remove duplicate points
        unique_points = np.unique(np.column_stack((x_in, y_in, z_in)), axis=0)
        if len(unique_points) >= 3:
            x_in, y_in, z_in = unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]

            # Perform Delaunay triangulation for the surface
            tri = Delaunay(np.column_stack((x_in, y_in)))
            fig.add_trace(go.Mesh3d(
                x=x_in,
                y=y_in,
                z=z_in,
                i=tri.simplices[:, 0],
                j=tri.simplices[:, 1],
                k=tri.simplices[:, 2],
                color='green',
                opacity=0.5,
                name='In Range (Surface)'
            ))
        else:
            print("Not enough unique points to create an in-range surface.")

    elif use_connected_surface == "out" and len(out_of_range) >= 3:
        x_out = out_of_range[x_param].values
        y_out = out_of_range[y_param].values
        z_out = out_of_range[z_param].values

        # Remove duplicate points
        unique_points = np.unique(np.column_stack((x_out, y_out, z_out)), axis=0)
        if len(unique_points) >= 3:
            x_out, y_out, z_out = unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]

            # Perform Delaunay triangulation for the surface
            tri = Delaunay(np.column_stack((x_out, y_out)))
            fig.add_trace(go.Mesh3d(
                x=x_out,
                y=y_out,
                z=z_out,
                i=tri.simplices[:, 0],
                j=tri.simplices[:, 1],
                k=tri.simplices[:, 2],
                color='red',
                opacity=0.5,
                name='Out of Range (Surface)'
            ))
        else:
            print("Not enough unique points to create an out-of-range surface.")

    # Set axis labels and layout
    fig.update_layout(
        title=f"3D Parameter Space for {algorithm}",
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_param
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the interactive plot
    fig.show()





from scipy.interpolate import griddata
import numpy as np

def plot_surface_with_interpolation(results_df, params, algorithm, use_connected_surface=None):
    # Extract 3D plotting parameters
    plot_params = params['viz']['3d_InOut_params']
    x_param = plot_params['x_param']
    y_param = plot_params['y_param']
    z_param = plot_params['z_param']

    # Filter the DataFrame for the given algorithm
    filtered_df = results_df[results_df['algorithm'] == algorithm]

    # Separate in-range and out-of-range points
    in_range = filtered_df[filtered_df['status'] == 'in']
    out_of_range = filtered_df[filtered_df['status'] == 'out']

    # Choose the data to create the surface
    if use_connected_surface == "in":
        x, y, z = in_range[x_param], in_range[y_param], in_range[z_param]
        color = 'green'
    elif use_connected_surface == "out":
        x, y, z = out_of_range[x_param], out_of_range[y_param], out_of_range[z_param]
        color = 'red'
    else:
        return  # No surface to plot

    # Interpolate the data onto a grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 50),
        np.linspace(y.min(), y.max(), 50)
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

    # Create the 3D surface plot
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale=[[0, color], [1, color]],
        opacity=0.5,
        name=f"{use_connected_surface.capitalize()} Range (Surface)"
    ))

    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=5, color=color),
        name=f"{use_connected_surface.capitalize()} Range (Points)"
    ))

    # Set axis labels and layout
    fig.update_layout(
        title=f"3D Parameter Space for {algorithm}",
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_param
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the interactive plot
    fig.show()



from alphashape import alphashape
import numpy as np
import plotly.graph_objects as go

def plot_surface_with_alpha_shape(results_df, params, algorithm, use_connected_surface=None, alpha=1.0):
    """
    Creates a 3D plot with an alpha shape surface for in-range or out-of-range points.

    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
        params (dict): A dictionary containing simulation parameters, including "viz".
        algorithm (str): The algorithm to filter the results for.
        use_connected_surface (str or None): Whether to plot a connected surface for "in" (green), "out" (red), or None.
        alpha (float): Alpha parameter for the alpha shape.

    Returns:
        None
    """
    # Extract 3D plotting parameters
    plot_params = params['viz']['3d_InOut_params']
    x_param = plot_params['x_param']
    y_param = plot_params['y_param']
    z_param = plot_params['z_param']

    # Filter the DataFrame for the given algorithm
    filtered_df = results_df[results_df['algorithm'] == algorithm]

    # Separate in-range and out-of-range points
    in_range = filtered_df[filtered_df['status'] == 'in']
    out_of_range = filtered_df[filtered_df['status'] == 'out']

    # Choose the data to create the surface
    if use_connected_surface == "in":
        x, y, z = in_range[x_param].values, in_range[y_param].values, in_range[z_param].values
        color = 'green'
    elif use_connected_surface == "out":
        x, y, z = out_of_range[x_param].values, out_of_range[y_param].values, out_of_range[z_param].values
        color = 'red'
    else:
        print("No surface to plot. Exiting.")
        return

    # Check for missing or invalid values
    valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    # Combine into a points array
    points = np.column_stack((x, y, z))

    # Debug: Check the points array
    print(f"Points array shape: {points.shape}")
    print(f"Points array: {points}")

    # Ensure there are enough points
    if len(points) < 4:
        print("Not enough points to create an alpha shape. Skipping surface creation.")
        return

    # Compute the alpha shape
    try:
        alpha_shape = alphashape(points, alpha)
        triangles = np.array(list(alpha_shape.triangles))
    except Exception as e:
        print(f"Error creating alpha shape: {e}")
        return

    # Create the 3D surface plot
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color=color,
        opacity=0.5,
        name=f"{use_connected_surface.capitalize()} Range (Surface)"
    ))

    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=5, color=color),
        name=f"{use_connected_surface.capitalize()} Range (Points)"
    ))

    # Set axis labels and layout
    fig.update_layout(
        title=f"3D Parameter Space for {algorithm}",
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_param
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the interactive plot
    fig.show()



from scipy.spatial import Delaunay
import plotly.graph_objects as go
import numpy as np

def plot_3d_connected_all_points(results_df, params, algorithm, use_connected_surface="in"):
    """
    Creates an interactive 3D plot with a connected surface visualizing the area covered by all points.

    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
        params (dict): A dictionary containing simulation parameters, including "viz".
        algorithm (str): The algorithm to filter the results for.
        use_connected_surface (str): Whether to plot a connected surface for "in" (green) or "out" (red).

    Returns:
        None
    """
    # Extract 3D plotting parameters
    plot_params = params['viz']['3d_InOut_params']
    x_param = plot_params['x_param']
    y_param = plot_params['y_param']
    z_param = plot_params['z_param']

    # Filter the DataFrame for the given algorithm
    filtered_df = results_df[results_df['algorithm'] == algorithm]

    # Separate in-range and out-of-range points
    in_range = filtered_df[filtered_df['status'] == 'in']
    out_of_range = filtered_df[filtered_df['status'] == 'out']

    # Choose the data to create the surface
    if use_connected_surface == "in":
        x, y, z = in_range[x_param].values, in_range[y_param].values, in_range[z_param].values
        color = 'green'
    elif use_connected_surface == "out":
        x, y, z = out_of_range[x_param].values, out_of_range[y_param].values, out_of_range[z_param].values
        color = 'red'
    else:
        print("Invalid value for use_connected_surface. Choose 'in' or 'out'.")
        return

    # Combine the points into a single array
    points = np.column_stack((x, y, z))

    # Ensure there are enough points to create a surface
    if len(points) < 4:
        print("Not enough points to create a surface. At least 4 points are required.")
        return

    # Use Delaunay triangulation to connect all points
    try:
        tri = Delaunay(points)  # Use all 3 dimensions for triangulation
    except Exception as e:
        print(f"Error during Delaunay triangulation: {e}")
        return

    # Create the 3D surface plot
    fig = go.Figure()

    # Add the surface
    fig.add_trace(go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=tri.simplices[:, 0],
        j=tri.simplices[:, 1],
        k=tri.simplices[:, 2],
        color=color,
        opacity=0.5,
        name=f"{use_connected_surface.capitalize()} Range (Surface)"
    ))

    # Set axis labels and layout
    fig.update_layout(
        title=f"3D Connected Area for {algorithm}",
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_param
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the interactive plot
    fig.show()



from scipy.spatial import Delaunay
import plotly.graph_objects as go
import numpy as np

def plot_3d_connected_filtered_points(results_df, params, algorithm, use_connected_surface="in"):
    """
    Creates an interactive 3D plot with a connected surface visualizing the area covered by points,
    ensuring no point from the other range is in between.

    Args:
        results_df (pd.DataFrame): The DataFrame containing simulation results.
        params (dict): A dictionary containing simulation parameters, including "viz".
        algorithm (str): The algorithm to filter the results for.
        use_connected_surface (str): Whether to plot a connected surface for "in" (green) or "out" (red).

    Returns:
        None
    """
    # Extract 3D plotting parameters
    plot_params = params['viz']['3d_InOut_params']
    x_param = plot_params['x_param']
    y_param = plot_params['y_param']
    z_param = plot_params['z_param']

    # Filter the DataFrame for the given algorithm
    filtered_df = results_df[results_df['algorithm'] == algorithm]

    # Separate in-range and out-of-range points
    in_range = filtered_df[filtered_df['status'] == 'in']
    out_of_range = filtered_df[filtered_df['status'] == 'out']

    # Choose the data to create the surface
    if use_connected_surface == "in":
        x, y, z = in_range[x_param].values, in_range[y_param].values, in_range[z_param].values
        other_points = np.column_stack((
            out_of_range[x_param].values,
            out_of_range[y_param].values,
            out_of_range[z_param].values
        ))
        color = 'green'
    elif use_connected_surface == "out":
        x, y, z = out_of_range[x_param].values, out_of_range[y_param].values, out_of_range[z_param].values
        other_points = np.column_stack((
            in_range[x_param].values,
            in_range[y_param].values,
            in_range[z_param].values
        ))
        color = 'red'
    else:
        print("Invalid value for use_connected_surface. Choose 'in' or 'out'.")
        return

    # Combine the points into a single array
    points = np.column_stack((x, y, z))

    # Ensure there are enough points to create a surface
    if len(points) < 4:
        print("Not enough points to create a surface. At least 4 points are required.")
        return

    # Filter connections to ensure no other points are in between
    def is_connection_valid(p1, p2, other_points):
        """
        Check if the connection between p1 and p2 is valid (no other points in between).
        """
        midpoint = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2) / 2
        distances = np.linalg.norm(other_points - midpoint, axis=1)
        return np.all(distances > radius)

    # Perform Delaunay triangulation
    try:
        tri = Delaunay(points)  # Use all 3 dimensions for triangulation
    except Exception as e:
        print(f"Error during Delaunay triangulation: {e}")
        return

    # Extract triangular faces from the tetrahedra
    tetrahedra = tri.simplices
    faces = set()
    for tetra in tetrahedra:
        # Each tetrahedron has 4 triangular faces
        faces.update([
            tuple(sorted([tetra[0], tetra[1], tetra[2]])),
            tuple(sorted([tetra[0], tetra[1], tetra[3]])),
            tuple(sorted([tetra[0], tetra[2], tetra[3]])),
            tuple(sorted([tetra[1], tetra[2], tetra[3]]))
        ])

    # Filter triangles based on the validity of connections
    valid_faces = []
    for face in faces:
        p1, p2, p3 = points[list(face)]
        if (
            is_connection_valid(p1, p2, other_points) and
            is_connection_valid(p2, p3, other_points) and
            is_connection_valid(p3, p1, other_points)
        ):
            valid_faces.append(face)

    valid_faces = np.array(valid_faces)

    # Create the 3D surface plot
    fig = go.Figure()

    # Add the surface
    if len(valid_faces) > 0:
        fig.add_trace(go.Mesh3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            i=valid_faces[:, 0],
            j=valid_faces[:, 1],
            k=valid_faces[:, 2],
            color=color,
            opacity=0.5,
            name=f"{use_connected_surface.capitalize()} Range (Surface)"
        ))
    else:
        print("No valid connections to create a surface.")

    # Set axis labels and layout
    fig.update_layout(
        title=f"3D Filtered Connected Area for {algorithm}",
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_param
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the interactive plot
    fig.show()



    

# def plot_3d_in_out_interactive(results_df, params, algorithm, use_connected_surface=False):
#     """
#     Creates an interactive 3D plot with three parameters on the axes, visualizing in-range and out-of-range points.
#     Optionally, plots out-of-range points as a connected red surface.

#     Args:
#         results_df (pd.DataFrame): The DataFrame containing simulation results.
#         params (dict): A dictionary containing simulation parameters, including "viz".
#         algorithm (str): The algorithm to filter the results for.
#         use_connected_surface (bool): Whether to plot out-of-range points as a connected red surface.

#     Returns:
#         None
#     """

#     markersize = 3

#     # Extract 3D plotting parameters
#     plot_params = params['viz']['3d_InOut_params']
#     x_param = plot_params['x_param']
#     y_param = plot_params['y_param']
#     z_param = plot_params['z_param']

#     # Filter the DataFrame for the given algorithm
#     filtered_df = results_df[results_df['algorithm'] == algorithm]

#     # Separate in-range and out-of-range points
#     in_range = filtered_df[filtered_df['status'] == 'in']
#     out_of_range = filtered_df[filtered_df['status'] == 'out']

#     # Create the 3D scatter plot
#     fig = go.Figure()

#     # Add in-range points (green)
#     fig.add_trace(go.Scatter3d(
#         x=in_range[x_param],
#         y=in_range[y_param],
#         z=in_range[z_param],
#         mode='markers',
#         marker=dict(size=markersize, color='green'),
#         name='In Range'
#     ))

#     if use_connected_surface:
#         # Add out-of-range points as a connected red surface
#         if len(out_of_range) >= 3:  # Ensure there are at least 3 points
#             x_out = out_of_range[x_param].values
#             y_out = out_of_range[y_param].values
#             z_out = out_of_range[z_param].values

#             # Check for duplicate points
#             unique_points = np.unique(np.column_stack((x_out, y_out, z_out)), axis=0)
            
#             # print(f"Unique points for surface: {len(unique_points)}")   
#             # print(unique_points[:5])  
#             # s, e = -5, None
#             # # for unique_point in unique_points[:5]:
#             #     # print('\n','\n',unique_point[, 0], '\n', unique_point[, 1], '\n',unique_point[, 2])
#             # print('\n','\n',unique_points[s:e, 0], '\n', unique_points[s:e, 1], '\n',unique_points[s:e, 2])
#             # raise Exception("Debugging stop")


#             if len(unique_points) >= 3:  # Ensure there are at least 3 unique points
#                 x_out, y_out, z_out = unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]

#                 # Perform Delaunay triangulation for the surface
#                 tri = Delaunay(np.column_stack((x_out, y_out)))
#                 fig.add_trace(go.Mesh3d(
#                     x=x_out,
#                     y=y_out,
#                     z=z_out,
#                     i=tri.simplices[:, 0],
#                     j=tri.simplices[:, 1],
#                     k=tri.simplices[:, 2],
#                     color='red',
#                     opacity=0.5,
#                     name='Out of Range (Surface)'
#                 ))
#             else:
#                 print("Not enough unique points to create a surface.")
#         else:
#             print("Not enough points to create a surface.")
#     else:
#         # Add out-of-range points as red scatter points
#         fig.add_trace(go.Scatter3d(
#             x=out_of_range[x_param],
#             y=out_of_range[y_param],
#             z=out_of_range[z_param],
#             mode='markers',
#             marker=dict(size=markersize, color='red'),
#             name='Out of Range'
#         ))

#     # Set axis labels and layout
#     fig.update_layout(
#         title=f"3D Parameter Space for {algorithm}",
#         scene=dict(
#             xaxis_title=x_param,
#             yaxis_title=y_param,
#             zaxis_title=z_param
#         ),
#         width=1200,  # Set the width of the plot
#         height=800,  # Set the height of the plot
#         margin=dict(l=0, r=0, b=0, t=40)
#     )

#     # Show the interactive plot
#     fig.show()