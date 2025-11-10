
def obective_function(current_candidate, new_candidate, ppi_start_val, aD):
    """
    Objective function to evaluate how close the candidate's target PPI value is to the acceptable range.

    Args:
        candidate (dict): A dictionary of parameter values representing a candidate solution.
        current_ppi_val (float): The current PPI value of the candidate.
        ppi_start_val (float): The initial value of the target PPI.
        aD (float): Acceptable deviation (e.g., 0.1 for ±10%).
    Returns:
        float: The distance to the acceptable range.
    """

    return abs(abs(new_candidate['ppi_value'] - current_candidate['ppi_value']) - ppi_start_val * aD)
    # adapt the objective function to consider the conficence as well
    # return abs(abs(new_candidate['ppi_value'] - current_candidate['ppi_value']) - ppi_start_val * aD) * (1 + (1 - new_candidate['confidence']
    



def get_neighbouring_candidates(current_candidate, params_to_change, step_size, candidate_strategy="all"):
    """
    Generates neighboring scenarios by adjusting the specified parameters in both directions.

    Args:
        current_candidate (dict): The current scenario represented as a dictionary of parameter values.
        params_to_change (dict): Dictionary of parameters to vary with their types and ranges 
                                 (e.g., {"param1": {"type": "disc", "values": [min, max]}}).
        step_size (float): The step size for adjusting parameter values.
        candidate_strategy (str): Strategy for generating candidates. 
                                  "all" - Vary all parameters in both directions.
                                  "random" - Randomly choose one parameter to vary.

    Returns:
        list: A list of neighboring candidates of parameter values, each represented as a dictionary.
    """
    # Initialize the list of neighboring scenarios
    candidates = []

    if candidate_strategy == "all_directions":
        # Iterate over each parameter to change
        for param, param_info in params_to_change.items():
            param_type = param_info['type']
            param_min, param_max = param_info['values']
            param_value = current_candidate[param]

            # Adjust the parameter in both directions
            for direction in [-1, 1]:
                # Create a new candidate by copying the current candidate
                candidate = current_candidate.copy()

                # Update the parameter value in the new candidate
                new_value = param_value + (direction * param_value * step_size)

                # Ensure the new value is within bounds and handle discrete values
                if param_type == 'disc':
                    new_value = int(new_value)
                new_value = max(param_min, min(new_value, param_max))  # Clamp to [min, max]

                candidate[param] = new_value

                # Add the new candidate to the list of neighboring scenarios
                candidates.append(candidate)

    elif candidate_strategy == "random_one":
        # Randomly select one parameter to vary
        param = random.choice(list(params_to_change.keys()))
        param_info = params_to_change[param]
        param_type = param_info['type']
        param_min, param_max = param_info['values']
        param_value = current_candidate[param]

        # Generate a random variation for the selected parameter
        variation = random.uniform(-step_size, step_size)
        new_value = param_value + (param_value * variation)

        # Ensure the new value is within bounds and handle discrete values
        if param_type == 'disc':
            new_value = int(new_value)
        new_value = max(param_min, min(new_value, param_max))  # Clamp to [min, max]

        # Create a new candidate with the randomly varied parameter
        candidate = current_candidate.copy()
        candidate[param] = new_value

        # Add the single candidate to the list
        candidates.append(candidate)

    # Return the list of neighboring scenarios
    return candidates



# def get_neighbouring_scenarios(current_sc, params_to_change, step_size):
#     """
#     Generates neighboring scenarios by adjusting the specified parameters in both directions.

#     Args:
#         current_sc (dict): The current scenario represented as a dictionary of parameter values.
#         params_to_change (list): List of parameters to vary.
#         step_size (float): The step size for adjusting parameter values.

#     Returns:
#         list: A list of neighboring scenarios, each represented as a dictionary.
#     """
#     # Initialize the list of neighboring scenarios
#     scenarios = []

#     # Iterate over each parameter to change
#     for param, param_type in params_to_change.items():
#         # Get the current value of the parameter from the current scenario
#         # param_value = current_sc[param]
#         param_value = get_change_param_values(params_to_change, change_param=param, sim_params=current_sc)

#         # Adjust the parameter in both directions
#         for direction in [-1, 1]:
#             # Create a new scenario by copying the current scenario
#             new_sc = current_sc.copy()

#             # Update the parameter value in the new scenario
#             # new_sc[param] = param_value + (direction * step_size)   #### To be adapted
#             new_value = param_value + (direction * step_size)

#             if param_type == 'disc':
#                 new_value = int(new_value)

#             new_sc = set_change_param_value(change_param, new_value, current_sc)   


#             # Add the new scenario to the list of neighboring scenarios
#             scenarios.append(new_sc)

#     # Return the list of neighboring scenarios
#     return scenarios



def hill_descent(params, step_size_initial, step_max, params_to_change, candidate_strategy='random_one', walk_reps_max=5):
    """
    Finds the boundaries for acceptable simulation parameter values based on the target PPI.

    Args:
        params (dict): Simulation pipeline parameters.
        step_size_initial (float): The initial step size for parameter adjustment (in percentage).
        step_max (int): The maximum number of steps to take.
        params_to_change (list): List of parameters to vary.

    Returns:
        dict: A dictionary containing the boundaries for each parameter.
    """

    print_intermediate_results = True
    algorithm = f'hill_descent_{candidate_strategy}'
    aD = params['ppi_range_factor']                  # (float): Acceptable deviation (e.g., 0.1 for ±10%).
    sim_params = get_sim_params(params['json_path']) # sim_params (SimulationModel): The simulation model to evaluate.
    t_ppi = params['target_ppi']                     # t_ppi (str): The target PPI (e.g., "cycleTime")
    
    # Get the initial value of the target PPI 
    ppi_start_val = params['orig_target_ppi_val']
    # start_params = get_start_param_settings(params_to_change, params)

    # # Calculate the acceptable range for the target PPI
    ppi_min = params['target_range'][0]
    ppi_max = params['target_range'][1]

    step_size = step_size_initial

    # Run multiple walk repetitions to avoid local minima
    for walk_rep in range(walk_reps_max):

        # Initialize candidates for parameter adjustment
        current_candidate = get_start_param_settings(params_to_change, params)
        current_candidate['ppi_value'] = ppi_start_val
        current_candidate['distance'] = ppi_start_val * aD

        if print_intermediate_results:
            print(f"\n\n=== Hill Descent Walk Rep: {walk_rep+1}/{walk_reps_max} ===")
            print(f" Starting Candidate: {current_candidate}, PPI Value: {ppi_start_val}, Target Range: ({ppi_min}, {ppi_max})")

        # Placeholder for the main loop (to be implemented)
        for s_count in range(step_max):
            # Get neighboring candidates
            candidates = get_neighbouring_candidates(current_candidate, params_to_change, step_size, candidate_strategy="random_one")
            candidate_distances = []

            # Evaluate each candidate
            for candidate in candidates:
                
                # Update simulation parameters with the candidate values
                for change_param, new_value in candidate.items():
                    sim_params = set_change_param_value(change_param, new_value, sim_params)
                

                # Save the updated simulation parameters
                set_sim_params(params['json_path_temp'], sim_params)

                # Get the target PPI value list for the current candidate
                target_ppi_list = get_simulation_stats(params)
                # TODO: Figure out how to apply confidence
                candidate['ppi_value'] = np.mean(target_ppi_list)
                # print(f" ## Candidate: {candidate}, PPI Value: {ppi_value_new}")

                # Calculate the distance to the acceptable range
                candidate_distance = obective_function(current_candidate, candidate, ppi_start_val, aD)
                candidate['distance'] = candidate_distance
                candidate_distances.append(candidate_distance)
                # print(f" ## Candidate: {candidate}, PPI Value: {ppi_value_new}")

                if print_intermediate_results:
                    print(f" Evaluating Candidate: {candidate}")
                
                # Log the simulation results
                log_simulation(
                    algorithm=algorithm,
                    params=params,
                    target_ppi_list=target_ppi_list,
                    param_values=current_candidate,
                )

            # Check for the best neighbor
            best_index = np.argmin(candidate_distances)

            # if candidates[best_index]['distance'] == 0 or 
            if candidate_distances[best_index] < current_candidate['distance']:
                current_candidate = candidates[best_index]
                
            else: 
                break


    # Return the boundaries (to be implemented)
    return  candidates, current_candidate



################################################################################
############  Testing the hill climbing strategy ############


# # Step size
# step_size = 0.1
# # Example current candidate
# current_candidate = {
#     "arriaval_distr_mean": 1800,
#     "resource_count_UnifiedResourceProfile": 10
# }


# # # Generate candidates using "all" strategy
# # candidates_all = get_neighbouring_candidates(current_candidate, params_to_change, step_size, candidate_strategy="all")
# # print("Candidates (All Strategy):")
# # for candidate in candidates_all:
# #     print(candidate)

# # Generate candidates using "random" strategy
# for _ in range(10):
#     candidates_random = get_neighbouring_candidates(current_candidate, params_to_change, step_size, candidate_strategy="random")
#     print("Candidates (Random Strategy):")
#     for candidate in candidates_random:
#         print(candidate)





################################################################################
############## Rational and Pseudocode for Hill Climbing Strategy ##############
################################################################################

# <!-- Rational:
# - take the target_ppi_val with the simulation parameters at a start
# <!-- - for each of the simulation parameters,  
# take one simulation parameter -->


# Rational:


# find_boarders:
# - Inputs:
#     - sm <- SimulationModel
#     - aD <- AcceptableDeviation
#     - t_ppi // TargetPPI e.g., cycleTime
#     - step_size_initial
#     - step_max 
#     - params_to_change 
#     <!-- - ...max? -->
# - ppi_start_val <- get_simulation_stats(sm)   // Get value of target PPI from simulating the Simulation model
# - ppi_min <- ppi_start_val * (1 - AcceptableDeviation)
# - ppi_max <- ppi_start_val * (1 + AcceptableDeviation)     // TargetRangeMax
# - scenarios <- get_scenarios()
# - current_sc = sm
# - s_count <- 0
# - while s_count < step_ sIsNotEmpty(Candidates):

# '''
# From a starting point 
# '''
# get_neighbouring_scenarios: 
# - inputs
#     - current_sc
#     - params_to_change
#     - step_size
# - scenarios <- []
# - for each param in params_to_change:
#     - // get param_value from current_sc
#     - for direction in [-1, 1]
#         - // initialize a new scenario sc from the current_sc
#         - // one time for each direction adapt param value of sc 
#         - // scenarios.add(sc)
# - return scenarios





################################################################################
###### Helper functions


# def append_simulation_log(input_path, simulation_log, output_path):
#     """
#     Reads a DataFrame from a file, appends it with the simulation_log DataFrame, 
#     and saves the resulting DataFrame to a new file.

#     Args:
#         input_path (str): Path to the input file containing the existing DataFrame.
#         simulation_log (pd.DataFrame): The DataFrame to append to the existing DataFrame.
#         output_path (str): Path to save the resulting DataFrame.

#     Returns:
#         None
#     """
#     # Read the existing DataFrame from the input path
#     try:
#         existing_df = pd.read_csv(input_path)
#     except FileNotFoundError:
#         print(f"File not found at {input_path}. Creating a new DataFrame.")
#         existing_df = pd.DataFrame()

#     # Append the simulation_log to the existing DataFrame
#     combined_df = pd.concat([existing_df, simulation_log], ignore_index=True)

#     # Save the resulting DataFrame to the output path
#     combined_df.to_csv(output_path, index=False)
#     print(f"Combined DataFrame saved to {output_path}.")

# input_path = os.path.join(params['base_path'], 'output', 'simulation_log.csv')
# output_path = os.path.join(params['base_path'], 'output', 'simulation_log_combined.csv')
# append_simulation_log(input_path, simulation_log, output_path)


