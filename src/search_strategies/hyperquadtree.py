


# ================================
# Hyperquadtree Node
# ================================
class HyperQuadNode:
    def __init__(self, bounds, depth=0):
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
def adaptive_hyperquadtree(set_sim_params_get_sim_stats, is_in_target_range, params, max_depth=4):
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

    def evaluate(node):
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
                log_simulation(
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
        if (in_range and out_range or out_range and not direction_range) and node.depth < max_depth:
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
            return children
        return []

    # Iterative refinement
    frontier = [root]
    previous_sample_count = 0
    nodes_visited = {}
    for iteration in range(max_depth):
        new_frontier = []
        for node in frontier:
            new_children = evaluate(node)
            new_frontier.extend(new_children)

        previous_sample_count = len(sampled_points)
        frontier = new_frontier
        all_nodes.extend(frontier)

        nodes_visited[iteration] = {
            'all_nodes': len(all_nodes),
            'sampled_points': len(sampled_points)
        }

    print(f'Number of nodes: {len(all_nodes)}, Number of sampled points: {len(sampled_points)}')
    return all_nodes, sampled_points, nodes_visited



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