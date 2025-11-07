#!/usr/bin/env python3

# Script for simulation execution and evaluation

from src.simulators.param_manipulation import *
from src.log_stats_calculation import *
import numpy as np
from scipy.stats import norm


def is_in_target_range(target_ppi_dict, params, strictness=None, above_below=False):

    """
    Checks whether the target_ppi_list satisfies the in/out criteria for the target range.

    Args:
        target_ppi_list (list): A list of PPI values to evaluate.
        params (dict): A dictionary containing simulation parameters, including:
                       - "target_range": A tuple (min, max) defining the acceptable range.
                       - "confidence": Confidence level for normal distribution checks (e.g., 0.9).
        in_out_criteria (str): The criteria to check:
                               - "mean": Check if the mean of the list is within the target range.
                               - "confidence": Check if the specified confidence interval of the list
                                 lies within the target range.
        strictness (float): The strictness parameter for adjusting the target range.
        above_below (bool): If True, also returns whether the values are above or below the target range.  

    Returns:
        bool: True if the list satisfies the criteria (inside the range), False otherwise.
        str (optional): "above" if the values are above the range, "below" if below the range.
                        Only returned if above_below=True.
    """
    confidence = params['confidence']
    in_out_criteria = params['in_out_criteria']

    in_range_per_ppi = []
    direction_per_ppi = []

    for ppi in target_ppi_dict.keys():
        target_ppi_list = target_ppi_dict[ppi]


    
        target_min, target_max = params['target_range'][ppi]
        
        # Calculate the mean and standard deviation of the list
        mean = np.mean(target_ppi_list)

        if in_out_criteria == "mean":
            # Check if the mean is within the target range
            in_range = target_min <= mean <= target_max
            if not in_range and above_below:
                direction = "below" if mean < target_min else "above"


        elif in_out_criteria == "confidence":
            std_dev = np.std(target_ppi_list)

            if std_dev == 0:
                # If std_dev is 0, the confidence interval collapses to the mean
                lower_bound = upper_bound = mean
            else:
                # Find the bounds of the confidence interval
                # TO_CHECK: cinfidence interval or with that confidence below/above
                lower_bound = norm.ppf((1 - confidence) / 2, loc=mean, scale=std_dev)
                upper_bound = norm.ppf(1 - (1 - confidence) / 2, loc=mean, scale=std_dev)

            # Check if the confidence interval is fully within the target range
            in_range = target_min <= lower_bound and upper_bound <= target_max
        
            if not in_range and above_below:
                if upper_bound < target_min:
                    direction = "below"
                elif lower_bound > target_max:
                    direction = "above"
                else:
                    direction = "mixed"  # Partially in range

        elif strictness is not None:
            # Adjust the target range based on strictness
            strict_min = target_min * (1 - strictness)
            strict_max = target_max * (1 + strictness)

            # Recursively call the function with "confidence" criteria and adjusted range
            adjusted_params = params.copy()
            adjusted_params['target_range'] = (strict_min, strict_max)

            print(f'### is_in_target_range \n Checking strictness criteria with strictness={strictness}:')

            in_range, direction = is_in_target_range(target_ppi_list, adjusted_params, strictness=None, above_below=True)

        else:
            raise ValueError(f"Invalid in_out_criteria: {in_out_criteria}. Use 'mean', 'confidence'.")
        
        in_range_per_ppi.append(in_range)
        if not in_range and above_below:
            direction_per_ppi.append(direction)
        

    in_range = all(in_range_per_ppi)
    if direction_per_ppi:
        # check if all directions are the same
        unique_directions = set(direction_per_ppi)
        if len(unique_directions) == 1:
            direction = unique_directions.pop()
        else:
            direction = "mixed"

    if above_below:
        return in_range, direction if not in_range else "in_range"
    return in_range



# Function to set simulation parameters and get simulation statistics
def set_sim_params_get_sim_stats(params, param_values):
    """
    Sets the simulation parameters based on the provided parameter values 
    and retrieves the simulation statistics.

    Args:
        params (dict): Simulation pipeline parameters.
        param_values (dict): Dictionary of parameter values to set.

    Returns:
        dict: Simulation statistics after setting the parameters.
    """
    sim_params = get_sim_params(params['json_path'])  # Simulation model parameters

    # Update simulation parameters with the provided parameter values
    for change_param, new_value in param_values.items():
        sim_params = set_change_param_value(change_param, new_value, sim_params)

    # Save the updated simulation parameters
    set_sim_params(params['json_path_temp'], sim_params)

    # Get and return the simulation statistics
    sim_stats = get_simulation_stats(params)
    return sim_stats




def get_simulation_stats(params):
    """
    Executes the simulation n times and returns the average value of the target PPI.

    Args:
        params (dict): Simulation parameters, including the number of simulations to run.

    Returns:
        Average value of the target PPI across all simulations or list of values if confidence intervals are needed.
    """
    nr_simulations = params.get('nr_simulations_per_scenario', 1)  

    calculate_stats = params.get('calculate_stats', 'custom')
    
    if calculate_stats not in ['custom', 'simod']:
        raise ValueError("calculate_stats must be either 'custom' or 'simod'")

    # initialize output dictionary
    ppi_dict = {}
    for ppi in params['target_ppis']:
        ppi_dict[ppi] = []

    # Extract cost per hour per profile from JSON configuration if required
    if 'cost' in params['target_ppis'] and calculate_stats == 'custom':
        cost_per_hour = extract_cost_per_hour(params["json_path"])

        

    for i in range(nr_simulations):

        # Run the simulation
        (r, t) = simulation_engine.run_simulation(
            bpmn_path=params['bpmn_path'], 
            json_path=params['json_path_temp'],
            total_cases=params['cases_to_simulate'],
            # stat_out_path=stat_out_path,
            # log_out_path=log_out_path,
            starting_at=params['starting_at'],
            is_event_added_to_log=False,
            fixed_arrival_times=None,
        )


        # Retrieval of PPI values

        # calcualte PPI values from the log files
        if calculate_stats == 'custom':

            custom_ppis = {
                'lead_time': 'avg',
                'cost': 'total'
                }
            
            if ppi not in custom_ppis.keys():
                raise ValueError(f"PPI '{ppi}' is not supported for custom calculation.")
            
            ppi_dict = calculate_custom_stats(params, ppi_dict, cost_per_hour, r, t)

            # for ppi in params['target_ppis']:
            #     # read the log in to calculate the PPI value

            #     colmap = LogColumnNames()
            #     sim_log = trace_list_to_dataframe(t, colmap)
            #     if ppi == 'lead_time':
            #         value = calculate_lead_time(sim_log, colmap, begin_col="enable_time", metric="avg")
            #     elif ppi == 'cost':
            #         value = calculate_cost(sim_log, cost_per_hour, r, metric="total", cost_calculation='active_time')
            #     else:
            #         raise ValueError(f"PPI '{ppi}' is not supported for custom calculation.")

            #     ppi_dict[ppi].append(value) 
         
        elif calculate_stats == 'simod':
            # Retrieve the target PPI value from inbuild simod
            for ppi in params['target_ppis']:
                value = getattr(getattr(r[0], ppi), 'avg')
                ppi_dict[ppi].append(value)

    # if not params['simulation_results_confidence']:
    #     # Calculate and return the average of the PPI values
    #     ppi_values = sum(ppi_values) / len(ppi_values)


    return ppi_dict


