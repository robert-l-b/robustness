
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from src.simulators.param_manipulation import (
    get_sim_params,
    set_sim_params,
    set_change_param_value,
    get_start_param_settings,
)
from src.sim_execution_and_evalaution import get_simulation_stats
from src.logging import log_simulation  

import numpy as np
import matplotlib.pyplot as plt
import csv
import json


# ================================
# Visualization
# ================================
def plot_quadtree(params, nodes, samples, x_param, y_param, x_info, y_info, iteration):
    """Plot quadtree structure, sampled points, and in-range region."""
    plt.figure(figsize=(7, 6))
    plt.title(f"Iteration {iteration}")

    # Draw quadtree lines
    for node in nodes:
        if node.is_leaf:
            continue
        mid_x = (node.x_min + node.x_max) / 2
        mid_y = (node.y_min + node.y_max) / 2
        plt.plot([mid_x, mid_x], [node.y_min, node.y_max], "r--", lw=1)
        plt.plot([node.x_min, node.x_max], [mid_y, mid_y], "r--", lw=1)

    # Draw sampled points
    xs, ys = zip(*samples.keys())
    plt.scatter(xs, ys, c='blue', s=25, label="Sampled points")

    # Format plot
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.xlim(x_info["values"][0], x_info["values"][1])
    plt.ylim(y_info["values"][0], y_info["values"][1])
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.gca().set_aspect('auto')
    plt.tight_layout()

    output_dir_path = os.path.join(params['experiment_output_dir'], 'figures')
    os.makedirs(output_dir_path, exist_ok=True)
    output_fig_path = os.path.join(output_dir_path, f'hyperquadtree_iteration_{iteration}.png')
    plt.savefig(output_fig_path)
    
    plt.show()






# ================================
# Quadtree Node
# ================================
class QuadNode:
    def __init__(self, x_min, x_max, y_min, y_max, depth=1):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.depth = depth
        self.children = []
        self.is_leaf = True
        self.status = None              # "in_range", "out_range", or "mixed"
        self.corner_results = [] 


# ================================
# Adaptive Quadtree Algorithm
# ================================
def adaptive_quadtree(set_sim_params_get_sim_stats, is_in_target_range, params, simulation_log, min_depth=0, max_depth=4):
    """
    set_sim_params_get_sim_stats(params, param_values): Simulation function that sets parameters and returns stats.
    is_in_target_range(target_ppi_dict, params): Checks if the simulation result is in the target range.
    params_to_change: dict with structure like:
        {
            'param_x': {'type': 'disc' or 'cont', 'values': [min, max]},
            'param_y': {'type': 'disc' or 'cont', 'values': [min, max]},
        }
    params: Dictionary containing simulation parameters.
    simulation_log: DataFrame to log simulation results.
    min_depth: Minimum depth of the quadtree.
    max_depth: Maximum depth of the quadtree.
    """

    params_to_change = params['params_to_change']
    algorithm = 'quadtree'

    # throw error if not exactly two parameters to change
    if len(params_to_change) != 2:
        raise ValueError("params_to_change must contain exactly two parameters.")

    # Extract parameter definitions
    param_names = list(params_to_change.keys())
    x_param, y_param = param_names[0], param_names[1]
    x_info, y_info = params_to_change[x_param], params_to_change[y_param]

    # Range definitions
    x_min, x_max = x_info["values"]
    y_min, y_max = y_info["values"]
    x_is_disc = x_info["type"] == "disc"
    y_is_disc = y_info["type"] == "disc"

    root = QuadNode(x_min, x_max, y_min, y_max)
    all_nodes = [root]
    sampled_points = {}

    def all_values_same(lst):
        return all(x == lst[0] for x in lst)

    def cast_x(val):
        """Cast x to integer if discrete, otherwise leave continuous."""
        return int(round(val)) if x_is_disc else val

    def cast_y(val):
        """Cast y to integer if discrete, otherwise leave continuous."""
        return int(round(val)) if y_is_disc else val

    def evaluate(node, simulation_log):
        """Evaluate the simulation function at the corners and decide if subdivision is needed."""
        corners = [
            (cast_x(node.x_min), cast_y(node.y_min)),
            (cast_x(node.x_max), cast_y(node.y_min)),
            (cast_x(node.x_min), cast_y(node.y_max)),
            (cast_x(node.x_max), cast_y(node.y_max)),
        ]

        results = []
        results_direction = []
        for (x, y) in corners:
            if params['print_intermediate_results']:
                print(f"\n\n # Evaluating corner at ({x}, {y})")

            if (x, y) not in sampled_points:
                # Prepare parameter values for the simulation
                param_values = {x_param: x, y_param: y}
                target_ppi_dict = set_sim_params_get_sim_stats(params, param_values)
                sampled_points[(x, y)] = target_ppi_dict

                # Log the simulation results
                simulation_log = log_simulation(
                    simulation_log=simulation_log,
                    algorithm=algorithm,
                    params=params,
                    target_ppi_dict=target_ppi_dict,
                    param_values=param_values,
                )

            else:
                target_ppi_dict = sampled_points[(x, y)]

            if params['print_intermediate_results']:
                print(f"  Target PPI List: {target_ppi_dict}")
            # Determine if in target range
            # in_range = is_in_target_range(target_ppi_dict, params)
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
        node.corner_results = [(corners[i][0], corners[i][1], results[i]) for i in range(4)]

        if all(results):
            node.status = "in_range"
        elif not any(results):
            node.status = "out_range"
        else:
            node.status = "mixed"

        if params['print_intermediate_results']:
            print(f"Node at depth {node.depth} classified as {node.status}")


        # if in_range and out_range and node.depth < max_depth:
        # if out_range and node.depth < max_depth:
        if (node.depth < min_depth) or ((in_range and out_range or out_range and not direction_range) and node.depth < max_depth):
            print(f" Subdividing node at depth {node.depth}...")
            print((node.depth < min_depth), ((in_range and out_range or out_range and not direction_range) and node.depth < max_depth))        

            node.is_leaf = False
            mid_x = cast_x((node.x_min + node.x_max) / 2)
            mid_y = cast_y((node.y_min + node.y_max) / 2)

            # Avoid zero-width due to rounding (for discrete params)
            if mid_x == node.x_min or mid_x == node.x_max:
                return []

            node.children = [
                QuadNode(node.x_min, mid_x, node.y_min, mid_y, node.depth + 1),
                QuadNode(mid_x, node.x_max, node.y_min, mid_y, node.depth + 1),
                QuadNode(node.x_min, mid_x, mid_y, node.y_max, node.depth + 1),
                QuadNode(mid_x, node.x_max, mid_y, node.y_max, node.depth + 1),
            ]
            return node.children, simulation_log
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

        # # Early stopping
        # if len(new_frontier) == 0 and len(sampled_points) == previous_sample_count:
        #     print(f"Stopping early at iteration {iteration+1}: no change detected.")
        #     break

        previous_sample_count = len(sampled_points)
        frontier = new_frontier
        all_nodes.extend(frontier)
        plot_quadtree(
            params,
            all_nodes,
            sampled_points,
            x_param,
            y_param,
            x_info,
            y_info,
            iteration + 1
        )

        nodes_visited[iteration] = {
            'all_nodes': len(all_nodes),
            'sampled_points': len(sampled_points)
        }

    print(f' Number of nodes: {len(all_nodes)}, Number of sampled points: {len(sampled_points)}')
    return frontier, all_nodes, sampled_points, nodes_visited, simulation_log





def write_quadtree_nodes_to_file(quadtree_nodes, params):
    """
    Write the quadtree nodes to a CSV file.

    Args:
        quadtree_nodes (list[QuadNode]): List of quadtree nodes.
        params (dict): Simulation pipeline parameters containing output file path.
    Returns:
        None
    """
    output_file_path = params['strategies']['quadtree']['paths']['sampled_points']

    # Define the header for the CSV file
    header = ["x_min", "x_max", "y_min", "y_max", "depth", "is_leaf", "status"]

    # Open the file for writing
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        
        # Write the header
        writer.writeheader()
        
        # Write each node's properties
        for node in quadtree_nodes:
            writer.writerow({
                "x_min": node.x_min,
                "x_max": node.x_max,
                "y_min": node.y_min,
                "y_max": node.y_max,
                "depth": node.depth,
                "is_leaf": node.is_leaf,
                "status": node.status,
            })

    print(f"Quadtree nodes written to {output_file_path}")

    import json

def write_nodes_visited_to_json(nodes_visited, params):
    """
    Write the nodes_visited dictionary to a JSON file.

    Args:
        nodes_visited (dict): Dictionary containing nodes visited and sampled points at each depth level.
        output_file (str): Path to the output JSON file.

    Returns:
        None
    """
    output_file = params['strategies']['quadtree']['paths']['nodes_visited']
    with open(output_file, "w") as file:
        json.dump(nodes_visited, file, indent=4)  # Use indent=4 for pretty formatting
    print(f"Nodes visited written to {output_file}")



def read_quadtree_nodes_from_file(input_file):
    """
    Read quadtree nodes from a CSV file and return them as a list of QuadNode objects.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        list[QuadNode]: List of QuadNode objects.
    """
    quadtree_nodes = []

    # Open the file for reading
    with open(input_file, mode="r") as file:
        reader = csv.DictReader(file)
        
        # Read each row and create a QuadNode object
        for row in reader:
            quadtree_nodes.append(
                QuadNode(
                    x_min=float(row["x_min"]),
                    x_max=float(row["x_max"]),
                    y_min=float(row["y_min"]),
                    y_max=float(row["y_max"]),
                    depth=int(row["depth"]),
                    is_leaf=row["is_leaf"] == "True",  # Convert string to boolean
                    status=row["status"],
                )
            )

    print(f"Quadtree nodes read from {input_file}")
    return quadtree_nodes


def read_nodes_visited_from_json(input_file):
    """
    Read the nodes_visited data from a JSON file and return it as a dictionary.

    Args:
        input_file (str): Path to the input JSON file.

    Returns:
        dict: Dictionary containing nodes visited and sampled points at each depth level.
    """
    with open(input_file, "r") as file:
        nodes_visited = json.load(file)
    print(f"Nodes visited read from {input_file}")
    return nodes_visited