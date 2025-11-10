#!/usr/bin/env python3
# Diagonal Search Strategy for Parameter Boundary Identification
# 
# 
#  




# def check_stopping_criteria(current_value, new_value, ppi_value, ppi_min, ppi_max, strictness, param_max_range, last_tried_value, adaptive_step, step_size):
def check_stopping_criteria(params, current_value, new_value, target_ppi_list, strictness, param_max_range, last_tried_value, adaptive_step, step_size):
    """
    Checks if the stopping criteria for the boundary search are met.

    Args:
        current_value (float): The current parameter value.
        new_value (float): The new parameter value.
        ppi_value (float): The current PPI value.
        ppi_min (float): The minimum acceptable PPI value.
        ppi_max (float): The maximum acceptable PPI value.
        strictness (float): The strictness parameter for stopping near the target range borders.
        param_max_range (tuple): The predefined range for the parameter.
        last_tried_value (float): The last tried parameter value.
        adaptive_step (bool): Whether adaptive step size is enabled.
        step_size (float): The current step size.

    Returns:
        tuple: (bool, str, float) - A tuple containing:
            - Whether the search should stop (True/False).
            - The reason for stopping (str).
            - The updated step size (float).
    """

    ppi_min = params['target_range'][0]
    ppi_max = params['target_range'][1]

    # TODO: Figure out how to apply confidence
    # Check if the new value is within the strictness range of the target range
    if ppi_min * (1 - strictness) <= np.mean(target_ppi_list) <= ppi_min * (1 + strictness) or \
       ppi_max * (1 - strictness) <= np.mean(target_ppi_list) <= ppi_max * (1 + strictness):
        return True, "Within strictness range", step_size

    # Check if no further progress will be made
    if current_value == new_value:
        return True, "No further progress (current_value == new_value)", step_size

    # Check if the new value is outside the predefined parameter range
    if new_value < param_max_range[0] or new_value > param_max_range[1]:
        return True, "Outside predefined parameter range", step_size

    # Check if the new value is outside the target range
    if is_in_target_range(target_ppi_list, params, strictness=strictness) == False:
        # Adapt the step size if adaptive_step is enabled
        if adaptive_step:
            step_size *= 0.5  # Reduce the step size for finer adjustments
        if last_tried_value == new_value:
            return True, "Last tried value is the same as new value", step_size

    return False, None, step_size


def find_boundary_for_each_param(params, simulation_log, step_size_initial, step_max, params_to_change, adaptive_step=True, strictness=0.001):
    """
    Iteratively adjusts each parameter to find the range of values that produce simulation outputs within the target range.

    Args:
        params (dict): Simulation pipeline parameters.
        step_size_initial (float): The initial step size for parameter adjustment (in percentage).
        step_max (int): The maximum number of steps to take.
        params_to_change (dict): Dictionary of parameters to vary with their types (e.g., {"param1": "disc"}).

    Returns:
        dict: A dictionary containing the range of values for each parameter.
    """
    decimals = 2
    boundaries = {}
    aD = params['ppi_range_factor']  # Acceptable deviation (e.g., 0.1 for ±10%).
    t_ppi = params['target_ppi']  # Target PPI (e.g., "cycleTime")
    algorithm = "single_param_boundary_search"

    # Get the initial value of the target PPI 
    ppi_start_val = params['orig_target_ppi_val']

    # Calculate the acceptable range for the target PPI
    ppi_min = ppi_start_val * (1 - aD)
    ppi_max = ppi_start_val * (1 + aD)

    # Iterate over each parameter to find its range
    for param, param_dict in params_to_change.items():
        param_type = param_dict['type']
        param_max_range = param_dict['values']
        boundaries[param] = {"min": None, "max": None}  # Initialize the range for this parameter
        boundary_vals = []

        # Test decreasing direction (-1) and increasing direction (+1)
        for direction in [-1, 1]:
            sim_params = get_sim_params(params['json_path'])  # Simulation model parameters
            current_candidate = get_start_param_settings(params_to_change, params)
            current_value = get_change_param_values(params_to_change, param, sim_params)
            search_stopped = False
            last_tried_value = None
            step_size = step_size_initial

            for step in range(step_max):
                # Adjust the parameter value
                new_value = current_value + (direction * current_value * step_size)
                if param_type == "disc":
                    new_value = int(new_value)
                if new_value < 1:
                    new_value = 1
                current_candidate[param] = new_value

                # Update the simulation parameters
                sim_params = set_change_param_value(param, new_value, sim_params)
                set_sim_params(params['json_path_temp'], sim_params)

                # Get the target PPI value for the new parameter value
                # ppi_value = get_simulation_stats(params)
                target_ppi_list = get_simulation_stats(params)
                
                # Log the simulation results
                simulation_log = log_simulation(
                    algorithm=algorithm,
                    params=params,
                    target_ppi_list=target_ppi_list,
                    param_values=current_candidate,
                )

                # Check stopping criteria
                stop, reason, step_size = check_stopping_criteria(
                    params, current_value, new_value, target_ppi_list, 
                    strictness, param_max_range, last_tried_value, adaptive_step, step_size
                )

                if stop:
                    boundary_vals.append(new_value)
                    print(f"Stopping search for {param} in direction {direction}: {reason}")
                    break

                # Update the current value for the next step
                current_value = new_value
                last_tried_value = new_value

            # If the search did not stop, add the last value
            if not search_stopped:
                boundary_vals.append(new_value)

        boundaries[param]["min"] = min(boundary_vals)
        boundaries[param]["max"] = max(boundary_vals)

    return boundaries



if 'diagonal_search' in params['execute_strategy']:

    create_temp_json(input_path=params['json_path'], output_path=None)

    stepsize_initial = params['strategies']['diagonal_search']['stepsize_initial']
    step_max = params['strategies']['diagonal_search']['step_max']
    adaptive_step = params['strategies']['diagonal_search']['adaptive_step']
    strictness = params['strategies']['diagonal_search']['strictness'] 

    # boundaries = find_boundary_for_each_param(params, step_size_initial=0.1, step_max=15, params_to_change=params_to_change)
    boundaries = find_boundary_for_each_param(
        params, 
        step_size_initial=stepsize_initial, 
        step_max=step_max, 
        params_to_change=params_to_change, 
        adaptive_step=adaptive_step, 
        strictness=strictness
        )
    
    # Save the simulation log to a CSV file for later analysis
    simulation_log.to_csv(params['simulation_log_path'], index=False)

    boundaries


######################################################
##### Pseudo Code for Diagonal Search Strategy ######
######################################################


# Pseudo code

# Algorithm: find_boundary_for_each_param
# Input:
#     params — simulation parameters
#     step_size_init — initial fractional step size
#     step_max — maximum number of iterations per direction
#     params_to_change — parameters to vary with their types
#     adaptive_step — enable step size adaptation (boolean)
#     strictness — acceptable deviation tolerance

# Output:
#     boundaries — dictionary containing [min, max] range for each parameter

# aD ← params["ppi_range_factor"]
# target_ppi ← params["target_ppi"]
# ppi_start ← get_simulation_stats(params)
# ppi_min ← ppi_start × (1 − aD)
# ppi_max ← ppi_start × (1 + aD)
# boundaries ← ∅


# for each (param, type) in params_to_change do
#     boundary_vals ← ∅
    
#     for direction in {−1, +1} do
#         sim_params ← get_sim_params(params["json_path"])
#         current_value ← get_change_param_values(params_to_change, param, sim_params)
#         step_size ← step_size_init
#         last_value ← null

#         for step from 1 to step_max do
#             new_value ← max(1, current_value + direction × current_value × step_size)
#             if type = disc then new_value ← ⌊new_value⌋
            
#             sim_params ← set_change_param_value(param, new_value, sim_params)
#             set_sim_params(params["json_path_temp"], sim_params)
#             ppi_val ← get_simulation_stats(params)

#             if |ppi_val − ppi_min| / ppi_min ≤ strictness 
#                or |ppi_val − ppi_max| / ppi_max ≤ strictness then
#                 boundary_vals ← boundary_vals ∪ {new_value}
#                 break

#             if new_value = current_value then
#                 boundary_vals ← boundary_vals ∪ {new_value}
#                 break

#             if ppi_val ∉ [ppi_min, ppi_max] then
#                 if adaptive_step then step_size ← step_size / 2
#                 if last_value = new_value then
#                     boundary_vals ← boundary_vals ∪ {new_value}
#                     break
#             else
#                 current_value ← new_value

#             last_value ← new_value
#         end for
#     end for

#     boundaries[param]["min"] ← min(boundary_vals)
#     boundaries[param]["max"] ← max(boundary_vals)
# end for

# return boundaries


