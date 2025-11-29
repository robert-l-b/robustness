#!/usr/bin/env python3

from itertools import product
import numpy as np
from src.simulators.param_manipulation import (
    get_sim_params,
    set_sim_params,
    set_change_param_value,
    get_start_param_settings,
)
from src.sim_execution_and_evalaution import get_simulation_stats
from src.logging import log_simulation   



def generate_values_around(base_value, x, y, as_integer=False, step_size=None):
    """
    Generates a list of x values above and below the base value.

    Args:
        base_value (float): The central value around which the values are generated.
        x (int): The number of values to generate above and below the base value.
        y (float): The percentage variation for the step size (e.g., 0.1 for 10%).
        as_integer (bool): If True, the generated values will use integer steps.
        step_size (int or None): The step size for integer values. If None, it will be calculated as y% of base_value.

    Returns:
        list: A list of values around the base value, converted to native Python types.
    """
    if as_integer:
        # Calculate step size dynamically if not provided
        if step_size is None:
            step_size = max(1, int(base_value * y))  # Ensure step size is at least 1
        
        # Generate integer steps above and below the base value
        values_below = [base_value - i * step_size for i in range(1, x + 1)]
        values_above = [base_value + i * step_size for i in range(1, x + 1)]
    else:
        # Calculate the increment with ±y% variation
        increment = 100 * (1 + y)  # Increment is 100 + y%
        # Generate floating-point values above and below the base value
        values_below = [base_value - i * increment for i in range(1, x + 1)]
        values_above = [base_value + i * increment for i in range(1, x + 1)]
    
    # Combine the values, including the base value, and sort them
    all_values = sorted(values_below + [base_value] + values_above)
    
    # Convert all values to native Python types
    if as_integer:
        all_values = [int(value) for value in all_values]  # Ensure all values are integers
    else:
        all_values = [float(value) for value in all_values]  # Ensure all values are floats
    
    return all_values



def generate_grid_search_ranges(params):    
    """
    Generates grid search ranges for each parameter based on the specified granularity.

    Args:
        params (dict): General parameters including strategy settings.

    Returns:
        dict: A dictionary where keys are parameter names and values are lists of values for grid search.
    """

    params_to_change = params['params_to_change']
    
    # Get the starting parameter settings
    start_param_settings = get_start_param_settings(params_to_change, params)
    grid_search_input = params['strategies']['grid_search'].get('input_mode', 'max_granularity')  # Default to 'original_input'
    # Define the granularity (maximum number of values for each parameter)

    # Initialize the grid search ranges
    grid_search_ranges = {}

    if grid_search_input == 'original_input':

        granularity = params['strategies']['grid_search']['granularity']
        # Generate ranges based on params_to_change
        for param, details in params_to_change.items():
            if details['type'] == 'cont':  # Continuous parameter
                # Generate evenly spaced values within the range based on granularity
                grid_search_ranges[param] = np.linspace(details['values'][0], details['values'][1], num=granularity).tolist()
            elif details['type'] == 'disc':  # Discrete parameter
                # Generate integer values within the range
                full_range = list(range(details['values'][0], details['values'][1] + 1))
                if len(full_range) > granularity:
                    # If the range has more values than granularity, sample evenly spaced values
                    grid_search_ranges[param] = np.linspace(details['values'][0], details['values'][1], num=granularity, dtype=int).tolist()
                else:
                    # Use all available discrete values if they are fewer than granularity
                    grid_search_ranges[param] = full_range

    elif grid_search_input == 'max_granularity':

        # Generate ranges based on min_step_size
        for param, details in params_to_change.items():
            min_step_size = details.get('min_step_size', None)
            if min_step_size is None:
                raise ValueError(f"Parameter '{param}' is missing 'min_step_size' for max_granularity mode.")

            if details['type'] == 'cont':  # Continuous parameter
                # Generate a grid with steps of min_step_size within the bounds
                start, end = details['values']
                grid_search_ranges[param] = [round(x, 10) for x in frange(start, end, min_step_size)]
            elif details['type'] == 'disc':  # Discrete parameter
                # Generate a grid with discrete steps of min_step_size
                start, end = details['values']
                grid_search_ranges[param] = list(range(start, end + 1, min_step_size))
            elif details['type'] == 'cat':  # Categorical parameter
                # Use the list of possible values directly
                grid_search_ranges[param] = details['values']
            else:
                raise ValueError(f"Unknown parameter type '{details['type']}' for parameter '{param}'.")

    elif grid_search_input == 'custom_input':
        
        # Generate ranges for non-resource parameters
        for param, value in start_param_settings.items():
            if param.startswith('arriaval_distr_mean'):  # Non-resource parameter
                grid_search_ranges[param] = generate_values_around(value, 5, 1)

        # Generate ranges for resource parameters
        for param, value in start_param_settings.items():
            if param.startswith('resource_count_'):  # Resource parameter
                grid_search_ranges[param] = generate_values_around(value, 6, 0.5, as_integer=True, step_size=2)

    else:
        raise ValueError('Unknown grid_search_input type')

    # Generate all possible combinations of parameter values
    all_combinations = list(product(*grid_search_ranges.values()))

    # print(grid_search_ranges)
    print('\nGrid search ranges generated:')
    for param, values in grid_search_ranges.items():
        print(f'  {param}: {values}')   
    print(f'Total combinations to test: {len(all_combinations)}')

    return grid_search_ranges, all_combinations


def frange(start, stop, step):
    """
    Generate a range of floating-point numbers.

    Args:
        start (float): Start of the range.
        stop (float): End of the range.
        step (float): Step size.

    Yields:
        float: Next number in the range.
    """
    while start < stop or abs(start - stop) < 1e-10:  # Handle floating-point precision issues
        yield start
        start += step




def run_grid_search(params, simulation_log):
    """
    Executes a grid search over the defined parameter ranges.

    Args:
        params (dict): General parameters including strategy settings.

    Returns:
        None
    """

    # Generate grid search ranges
    grid_search_ranges, all_combinations = generate_grid_search_ranges(params)
    # print_statements = False

    algorithm = 'grid_search'

    # Main loop for grid search
    for index, combination in enumerate(all_combinations):

        # For each 10 % done print a progrees update
        if params['print_intermediate_results']:
            percent_done = (all_combinations.index(combination) + 1) / len(all_combinations) * 100
            if percent_done % 10 == 0:
                print(f'Grid-search progress: {percent_done:.0f}% ({index + 1}/{len(all_combinations)} combinations tested)')

        # Map the combination to the corresponding parameters
        param_values = dict(zip(grid_search_ranges.keys(), combination))
        
        # Get the simulation parameters
        sim_params = get_sim_params(params['json_path_temp'])
        
        # Apply the parameter values to the simulation parameters
        for change_param, new_value in param_values.items():
            sim_params = set_change_param_value(change_param, new_value, sim_params, params_to_change=params.get('params_to_change', None))
        
        # Save the updated simulation parameters
        set_sim_params(params['json_path_temp'], sim_params)
        
        # Call simulation and get target PPI values
        target_ppi_dict = get_simulation_stats(params)
        
        # Log the simulation results
        simulation_log = log_simulation(
            simulation_log=simulation_log,
            algorithm=algorithm,
            params=params,
            target_ppi_dict=target_ppi_dict,
            param_values=param_values,
        )

    return simulation_log