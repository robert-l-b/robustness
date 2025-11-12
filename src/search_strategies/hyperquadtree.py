import numpy as np

from src.logging import log_simulation
import csv
import json

# ================================
# Hyperquadtree Node
# ================================
class HyperQuadNode:
    def __init__(self, bounds, depth=1):
        """
        bounds: List of tuples [(min1, max1), (min2, max2), ...] for each dimension.
        depth: Current depth of the node in the hyperquadtree.
        """
        self.bounds = bounds  # [(min1, max1), (min2, max2), ...]
        self.depth = depth
        self.children = []
        self.is_leaf = True
        self.status = None  # "in_range", "out_range", or "mixed"
        self.corner_results = []

# ================================
# Adaptive Hyperquadtree Algorithm
# ================================
def adaptive_hyperquadtree(set_sim_params_get_sim_stats, is_in_target_range, params, simulation_log, min_depth=1, max_depth=4):
    """
    Adaptive hyperquadtree algorithm for n-dimensional parameter spaces.

    Args:
        set_sim_params_get_sim_stats: Function that sets parameters and returns simulation stats.
        is_in_target_range: Function that checks if the simulation result is in the target range.
        params: Dictionary containing simulation parameters, including 'params_to_change'.
        max_depth: Maximum depth of the hyperquadtree.

    Returns:
        frontier: List of leaf nodes at the final depth.
        all_nodes: List of all nodes in the hyperquadtree.
        sampled_points: Dictionary of sampled points and their results.
        nodes_visited: Dictionary tracking nodes visited and sampled points at each depth.
    """
    params_to_change = params['params_to_change']
    algorithm = 'hyperquadtree'

    # Extract parameter definitions
    param_names = list(params_to_change.keys())
    n_dims = len(param_names)  # Number of dimensions
    if n_dims < 1:
        raise ValueError("params_to_change must contain at least one parameter.")

    # Initialize root node
    bounds = [(info["values"][0], info["values"][1]) for info in params_to_change.values()]
    root = HyperQuadNode(bounds)
    all_nodes = [root]
    sampled_points = {}

    def all_values_same(lst):
        return all(x == lst[0] for x in lst)

    def cast_value(val, is_disc):
        """Cast value to integer if discrete, otherwise leave continuous."""
        return int(round(val)) if is_disc else val

    def evaluate(node, simulation_log):
        """Evaluate the simulation function at the corners and decide if subdivision is needed."""
        # Generate all corners of the hypercube
        corners = []
        for corner in np.ndindex(*(2,) * n_dims):  # Generate 2^n corners
            corner_coords = [
                cast_value(node.bounds[i][corner[i]], params_to_change[param_names[i]]["type"] == "disc")
                for i in range(n_dims)
            ]
            corners.append(tuple(corner_coords))

        results = []
        results_direction = []
        for corner in corners:
            if params['print_intermediate_results']:
                print(f"\n\n # Evaluating corner at {corner}")

            if corner not in sampled_points:
                # Prepare parameter values for the simulation
                param_values = {param_names[i]: corner[i] for i in range(n_dims)}
                target_ppi_dict = set_sim_params_get_sim_stats(params, param_values)
                sampled_points[corner] = target_ppi_dict

                # Log the simulation results
                simulation_log = log_simulation(
                    simulation_log=simulation_log,
                    algorithm=algorithm,
                    params=params,
                    target_ppi_dict=target_ppi_dict,
                    param_values=param_values,
                )
            else:
                target_ppi_dict = sampled_points[corner]

            if params['print_intermediate_results']:
                print(f"  Target PPI Dict: {target_ppi_dict}")

            # Determine if in target range
            in_range, in_out_mixed = is_in_target_range(target_ppi_dict, params, above_below=True)
            results.append(in_range)
            results_direction.append(in_out_mixed)

            if params['print_intermediate_results']:
                print(f"  In Target Range: {in_range}, Direction: {in_out_mixed}")

        # Check if region is "mixed" (contains both in-range and out-of-range corners)
        in_range = any(results)
        out_range = any(not r for r in results)
        direction_range = all_values_same(results_direction)

        if params['print_intermediate_results']:
            print(f"Node at depth {node.depth} - In Range: {in_range}, Out Range: {out_range}, Direction Range Same: {direction_range}")

        # Classify node
        node.corner_results = [(corners[i], results[i]) for i in range(len(corners))]

        if all(results):
            node.status = "in_range"
        elif not any(results):
            node.status = "out_range"
        else:
            node.status = "mixed"

        if params['print_intermediate_results']:
            print(f"Node at depth {node.depth} classified as {node.status}")

        # Subdivide if necessary
        if (node.depth<min_depth) or ((in_range and out_range or out_range and not direction_range) and node.depth < max_depth):
        # if ((in_range and out_range or out_range and not direction_range) and node.depth < max_depth):

            # if not (node.depth<min_depth):
            node.is_leaf = False
            midpoints = [
                cast_value((node.bounds[i][0] + node.bounds[i][1]) / 2, params_to_change[param_names[i]]["type"] == "disc")
                for i in range(n_dims)
            ]

            # Generate child nodes
            children = []
            for corner in np.ndindex(*(2,) * n_dims):  # Generate 2^n child nodes
                child_bounds = [
                    (node.bounds[i][0], midpoints[i]) if corner[i] == 0 else (midpoints[i], node.bounds[i][1])
                    for i in range(n_dims)
                ]
                children.append(HyperQuadNode(child_bounds, node.depth + 1))
            node.children = children
            return children, simulation_log
        return [], simulation_log

    # Iterative refinement
    frontier = [root]
    previous_sample_count = 0
    nodes_visited = {}
    for iteration in range(max_depth):
        new_frontier = []
        for node in frontier:
            new_children, simulation_log = evaluate(node, simulation_log)
            new_frontier.extend(new_children)

        previous_sample_count = len(sampled_points)
        frontier = new_frontier
        all_nodes.extend(frontier)

        nodes_visited[iteration] = {
            'all_nodes': len(all_nodes),
            'sampled_points': len(sampled_points)
        }

        if params['print_intermediate_results']:
            # plot_hyperquadtree(params, nodes=all_nodes, samples=nodes_visited, iteration=iteration)
            plot_hyperquadtree(params, nodes=all_nodes, nodes_visited=nodes_visited, iteration=iteration)

    print(f'Number of nodes: {len(all_nodes)}, Number of sampled points: {len(sampled_points)}')
    return all_nodes, sampled_points, nodes_visited, simulation_log



def write_hyperquadtree_nodes_to_file(hyperquadtree_nodes, params):
    """
    Write the hyperquadtree nodes to a CSV file.

    Args:
        hyperquadtree_nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
        params (dict): Simulation pipeline parameters containing output file path.

    Returns:
        None
    """
    output_file_path = params['strategies']['hyperquadtree']['paths']['sampled_points']

    # Dynamically define the header based on the number of dimensions
    n_dims = len(hyperquadtree_nodes[0].bounds) if hyperquadtree_nodes else 0
    header = [f"dim_{i}_min" for i in range(n_dims)] + [f"dim_{i}_max" for i in range(n_dims)]
    header += ["depth", "is_leaf", "status"]

    # Open the file for writing
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        
        # Write the header
        writer.writeheader()
        
        # Write each node's properties
        for node in hyperquadtree_nodes:
            row = {}
            for i, (dim_min, dim_max) in enumerate(node.bounds):
                row[f"dim_{i}_min"] = dim_min
                row[f"dim_{i}_max"] = dim_max
            row["depth"] = node.depth
            row["is_leaf"] = node.is_leaf
            row["status"] = node.status
            writer.writerow(row)

    print(f"Hyperquadtree nodes written to {output_file_path}")


def write_nodes_visited_to_json(nodes_visited, params):
    """
    Write the nodes_visited dictionary to a JSON file.

    Args:
        nodes_visited (dict): Dictionary containing nodes visited and sampled points at each depth level.
        output_file (str): Path to the output JSON file.

    Returns:
        None
    """
    output_file = params['strategies']['hyperquadtree']['paths']['nodes_visited']
    with open(output_file, "w") as file:
        json.dump(nodes_visited, file, indent=4)  # Use indent=4 for pretty formatting
    print(f"Nodes visited written to {output_file}")




#########################################################
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

def plot_hyperquadtree(params, nodes, nodes_visited, iteration):
    """
    Plot the hyperquadtree structure, sampled points, and axis cuts for 1D, 2D, and 3D parameter spaces.

    Args:
        params (dict): Simulation parameters, including 'params_to_change' and output directory.
        nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
        nodes_visited (dict): Metadata about nodes visited at each iteration.
        iteration (int): Current iteration number.

    Returns:
        None
    """
    # Extract parameter names and bounds dynamically
    param_names = list(params['params_to_change'].keys())
    param_bounds = [params['params_to_change'][param]['values'] for param in param_names]
    n_dims = len(param_names)

    if n_dims == 1:
        _plot_1d_hyperquadtree(params, nodes, param_names, param_bounds, iteration)
    elif n_dims == 2:
        _plot_2d_hyperquadtree(params, nodes, param_names, param_bounds, nodes_visited, iteration)
    elif n_dims == 3:
        _plot_3d_hyperquadtree(params, nodes, param_names, param_bounds, nodes_visited, iteration)
    else:
        raise ValueError("Visualization is only supported for 1D, 2D, or 3D parameter spaces.")


def _plot_1d_hyperquadtree(params, nodes, param_names, param_bounds, iteration):
    """
    Plot the 1D hyperquadtree structure.

    Args:
        nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
        param_names (list[str]): Names of the parameters.
        param_bounds (list[tuple]): Bounds for each parameter.
        iteration (int): Current iteration number.

    Returns:
        None
    """
    plt.figure(figsize=(10, 2))
    plt.title(f"Iteration {iteration}")

    # Draw quadtree nodes
    for node in nodes:
        if node.is_leaf:
            continue
        x_min, x_max = node.bounds[0]
        plt.plot([x_min, x_max], [node.depth, node.depth], "b-", lw=2)

    # Format plot
    plt.xlabel(f"{param_names[0]} (Range: {param_bounds[0][0]} to {param_bounds[0][1]})")
    plt.ylabel("Depth")
    plt.xlim(param_bounds[0])
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    # Save and show plot
    _save_plot(params, iteration, "1D")
    plt.show()


def _plot_2d_hyperquadtree(params, nodes, param_names, param_bounds, nodes_visited, iteration):
    """
    Plot the 2D hyperquadtree structure.

    Args:
        nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
        param_names (list[str]): Names of the parameters.
        param_bounds (list[tuple]): Bounds for each parameter.
        nodes_visited (dict): Metadata about nodes visited at each iteration.
        iteration (int): Current iteration number.

    Returns:
        None
    """
    plt.figure(figsize=(8, 8))
    plt.title(f"Iteration {iteration}")

    # Draw quadtree nodes
    for node in nodes:
        if node.is_leaf:
            continue
        x_min, x_max = node.bounds[0]
        y_min, y_max = node.bounds[1]
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2

        # Draw vertical and horizontal cuts
        plt.plot([mid_x, mid_x], [y_min, y_max], "r--", lw=1)
        plt.plot([x_min, x_max], [mid_y, mid_y], "r--", lw=1)

        # Draw the rectangle representing the node
        plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      fill=False, edgecolor='blue', linewidth=1))

    # Draw sampled points from corner results
    for node in nodes:
        for corner, in_range in node.corner_results:
            if in_range:
                plt.scatter(corner[0], corner[1], c='green', s=25, label="In Range")
            else:
                plt.scatter(corner[0], corner[1], c='red', s=25, label="Out of Range")

    # Format plot
    plt.xlabel(f"{param_names[0]} (Range: {param_bounds[0][0]} to {param_bounds[0][1]})")
    plt.ylabel(f"{param_names[1]} (Range: {param_bounds[1][0]} to {param_bounds[1][1]})")
    plt.xlim(param_bounds[0])
    plt.ylim(param_bounds[1])
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    # Save and show plot
    _save_plot(params, iteration, "2D")
    plt.show()




def _plot_3d_hyperquadtree(params, nodes, param_names, param_bounds, nodes_visited, iteration):
    """
    Plot the 3D hyperquadtree structure.

    Args:
        nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
        param_names (list[str]): Names of the parameters.
        param_bounds (list[tuple]): Bounds for each parameter.
        nodes_visited (dict): Metadata about nodes visited at each iteration.
        iteration (int): Current iteration number.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Iteration {iteration}")

    # Draw quadtree nodes
    for node in nodes:
        if node.is_leaf:
            continue
        x_min, x_max = node.bounds[0]
        y_min, y_max = node.bounds[1]
        z_min, z_max = node.bounds[2]
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        mid_z = (z_min + z_max) / 2

        # Draw cuts
        ax.plot([mid_x, mid_x], [y_min, y_max], zs=mid_z, color="r", linestyle="--", linewidth=1)
        ax.plot([x_min, x_max], [mid_y, mid_y], zs=mid_z, color="r", linestyle="--", linewidth=1)

    # Draw sampled points from corner results
    for node in nodes:
        for corner, in_range in node.corner_results:
            color = 'green' if in_range else 'red'
            ax.scatter(corner[0], corner[1], corner[2], c=color, s=25)

    # Format plot
    ax.set_xlabel(f"{param_names[0]} (Range: {param_bounds[0][0]} to {param_bounds[0][1]})")
    ax.set_ylabel(f"{param_names[1]} (Range: {param_bounds[1][0]} to {param_bounds[1][1]})")
    ax.set_zlabel(f"{param_names[2]} (Range: {param_bounds[2][0]} to {param_bounds[2][1]})")
    ax.set_xlim(param_bounds[0])
    ax.set_ylim(param_bounds[1])
    ax.set_zlim(param_bounds[2])
    plt.tight_layout()

    # Save and show plot
    _save_plot(params, iteration, "3D")
    plt.show()

def _save_plot(params, iteration, dimension):
    """
    Save the plot to the output directory.

    Args:
        params (dict): Simulation parameters, including output directory.
        iteration (int): Current iteration number.
        dimension (str): Dimension of the plot (e.g., "1D", "2D", "3D").

    Returns:
        None
    """
    output_dir_path = os.path.join(params['experiment_output_dir'], 'figures')
    os.makedirs(output_dir_path, exist_ok=True)
    output_fig_path = os.path.join(output_dir_path, f'hyperquadtree_{dimension}_iteration_{iteration}.png')
    plt.savefig(output_fig_path)








##########################################################
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def plot_hyperquadtree(params, nodes, samples, iteration):
#     """
#     Plot the hyperquadtree structure, sampled points, and axis cuts for 1D, 2D, and 3D parameter spaces.

#     Args:
#         params (dict): Simulation parameters, including 'params_to_change' and output directory.
#         nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
#         samples (dict): Dictionary of sampled points.
#         iteration (int): Current iteration number.

#     Returns:
#         None
#     """
#     # Extract parameter names and bounds dynamically
#     param_names = list(params['params_to_change'].keys())
#     param_bounds = [params['params_to_change'][param]['values'] for param in param_names]
#     n_dims = len(param_names)

#     if n_dims == 1:
#         _plot_1d_hyperquadtree(params, nodes, samples, param_names, param_bounds, iteration)
#     elif n_dims == 2:
#         _plot_2d_hyperquadtree(params, nodes, samples, param_names, param_bounds, iteration)
#     elif n_dims == 3:
#         _plot_3d_hyperquadtree(params, nodes, samples, param_names, param_bounds, iteration)
#     else:
#         raise ValueError("Visualization is only supported for 1D, 2D, or 3D parameter spaces.")


# def _plot_1d_hyperquadtree(params, nodes, samples, param_names, param_bounds, iteration):
#     """
#     Plot the 1D hyperquadtree structure.

#     Args:
#         params (dict): Simulation parameters, including output directory.
#         nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
#         samples (dict): Dictionary of sampled points.
#         param_names (list[str]): Names of the parameters.
#         param_bounds (list[tuple]): Bounds for each parameter.
#         iteration (int): Current iteration number.

#     Returns:
#         None
#     """
#     plt.figure(figsize=(10, 2))
#     plt.title(f"Iteration {iteration}")

#     # Draw quadtree nodes
#     for node in nodes:
#         x_min, x_max = node.bounds[0]
#         plt.plot([x_min, x_max], [node.depth, node.depth], "b-", lw=2)

#     # Draw sampled points
#     xs = [point[0] for point in samples.keys()]
#     plt.scatter(xs, [0] * len(xs), c='red', s=25, label="Sampled points")

#     # Format plot
#     plt.xlabel(f"{param_names[0]} (Range: {param_bounds[0][0]} to {param_bounds[0][1]})")
#     plt.ylabel("Depth")
#     plt.xlim(param_bounds[0])
#     plt.grid(True, linestyle=':')
#     plt.legend()
#     plt.tight_layout()

#     # Save and show plot
#     _save_plot(params, iteration, "1D")
#     plt.show()


# def _plot_2d_hyperquadtree(params, nodes, samples, param_names, param_bounds, iteration):
#     """
#     Plot the 2D hyperquadtree structure.

#     Args:
#         params (dict): Simulation parameters, including output directory.
#         nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
#         samples (dict): Dictionary of sampled points.
#         param_names (list[str]): Names of the parameters.
#         param_bounds (list[tuple]): Bounds for each parameter.
#         iteration (int): Current iteration number.

#     Returns:
#         None
#     """
#     plt.figure(figsize=(8, 8))
#     plt.title(f"Iteration {iteration}")

#     # Draw quadtree nodes
#     for node in nodes:
#         x_min, x_max = node.bounds[0]
#         y_min, y_max = node.bounds[1]
#         plt.plot([x_min, x_max], [y_min, y_min], "r--", lw=1)  # Horizontal cut
#         plt.plot([x_min, x_max], [y_max, y_max], "r--", lw=1)  # Horizontal cut
#         plt.plot([x_min, x_min], [y_min, y_max], "r--", lw=1)  # Vertical cut
#         plt.plot([x_max, x_max], [y_min, y_max], "r--", lw=1)  # Vertical cut

#     # Draw sampled points
#     xs, ys = zip(*samples.keys())
#     plt.scatter(xs, ys, c='blue', s=25, label="Sampled points")

#     # Format plot
#     plt.xlabel(f"{param_names[0]} (Range: {param_bounds[0][0]} to {param_bounds[0][1]})")
#     plt.ylabel(f"{param_names[1]} (Range: {param_bounds[1][0]} to {param_bounds[1][1]})")
#     plt.xlim(param_bounds[0])
#     plt.ylim(param_bounds[1])
#     plt.grid(True, linestyle=':')
#     plt.legend()
#     plt.tight_layout()

#     # Save and show plot
#     _save_plot(params, iteration, "2D")
#     plt.show()


# def _plot_3d_hyperquadtree(params, nodes, samples, param_names, param_bounds, iteration):
#     """
#     Plot the 3D hyperquadtree structure.

#     Args:
#         params (dict): Simulation parameters, including output directory.
#         nodes (list[HyperQuadNode]): List of hyperquadtree nodes.
#         samples (dict): Dictionary of sampled points.
#         param_names (list[str]): Names of the parameters.
#         param_bounds (list[tuple]): Bounds for each parameter.
#         iteration (int): Current iteration number.

#     Returns:
#         None
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_title(f"Iteration {iteration}")

#     # Draw quadtree nodes
#     for node in nodes:
#         x_min, x_max = node.bounds[0]
#         y_min, y_max = node.bounds[1]
#         z_min, z_max = node.bounds[2]
#         _draw_3d_box(ax, x_min, x_max, y_min, y_max, z_min, z_max)

#     # Draw sampled points
#     xs, ys, zs = zip(*samples.keys())
#     ax.scatter(xs, ys, zs, c='blue', s=25, label="Sampled points")

#     # Format plot
#     ax.set_xlabel(f"{param_names[0]} (Range: {param_bounds[0][0]} to {param_bounds[0][1]})")
#     ax.set_ylabel(f"{param_names[1]} (Range: {param_bounds[1][0]} to {param_bounds[1][1]})")
#     ax.set_zlabel(f"{param_names[2]} (Range: {param_bounds[2][0]} to {param_bounds[2][1]})")
#     ax.set_xlim(param_bounds[0])
#     ax.set_ylim(param_bounds[1])
#     ax.set_zlim(param_bounds[2])
#     ax.legend()
#     plt.tight_layout()

#     # Save and show plot
#     _save_plot(params, iteration, "3D")
#     plt.show()


# def _draw_3d_box(ax, x_min, x_max, y_min, y_max, z_min, z_max):
#     """
#     Draw a 3D bounding box for a hyperquadtree node.

#     Args:
#         ax (Axes3D): Matplotlib 3D axis.
#         x_min, x_max (float): Bounds in the x-dimension.
#         y_min, y_max (float): Bounds in the y-dimension.
#         z_min, z_max (float): Bounds in the z-dimension.

#     Returns:
#         None
#     """
#     # Define the vertices of the box
#     vertices = [
#         [x_min, y_min, z_min], [x_min, y_min, z_max],
#         [x_min, y_max, z_min], [x_min, y_max, z_max],
#         [x_max, y_min, z_min], [x_max, y_min, z_max],
#         [x_max, y_max, z_min], [x_max, y_max, z_max]
#     ]

#     # Define the edges of the box
#     edges = [
#         [0, 1], [0, 2], [0, 4], [1, 3], [1, 5],
#         [2, 3], [2, 6], [3, 7], [4, 5], [4, 6],
#         [5, 7], [6, 7]
#     ]

#     # Draw the edges
#     for edge in edges:
#         ax.plot3D(*zip(*[vertices[edge[0]], vertices[edge[1]]]), color="blue", alpha=0.7)


# def _save_plot(params, iteration, dimension):
#     """
#     Save the plot to the output directory.

#     Args:
#         params (dict): Simulation parameters, including output directory.
#         iteration (int): Current iteration number.
#         dimension (str): Dimension of the plot (e.g., "1D", "2D", "3D").

#     Returns:
#         None
#     """
#     output_dir_path = os.path.join(params['experiment_output_dir'], 'figures')
#     os.makedirs(output_dir_path, exist_ok=True)
#     output_fig_path = os.path.join(output_dir_path, f'hyperquadtree_{dimension}_iteration_{iteration}.png')
#     plt.savefig(output_fig_path)