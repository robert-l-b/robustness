#!/usr/bin/env python3

# Script for logging simulation results and parameters

import pandas as pd
import numpy as np
import json
import datetime
from src.simulators.param_manipulation import *

# Logging function
def log_simulation(algorithm, params, param_values, target_ppi_dict):
    """
    Logs the details of a simulation run.

    Args:
        algorithm (str): Name of the algorithm used.
        params (dict): Simulation parameters including target range and target PPI.
        param_values (dict): Dictionary of parameter values used in the simulation.
        target_ppi_dict (dict): Dictionary of target PPI values obtained from the simulation.
    Returns:
        None
    """
    global simulation_log

    target_ppi_mean_dict = {}
    for ppi in params['target_ppis']:
        target_ppi_mean_dict[ppi] = np.mean(target_ppi_dict[ppi])

    # Determine success status
    # status = "in" if params['target_range'][0] <= target_ppi_val <= params['target_range'][1] else "out"
    status = is_in_target_range(target_ppi_dict, params, strictness=None)

    # Create a log entry
    log_entry = {
        "algorithm": algorithm,
        "simulation_id": len(simulation_log),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "target_range": params['target_range'],
        "status": status,
        "target_ppis": params['target_ppis'],
        "target_ppi_means": target_ppi_mean_dict,
        "target_ppi_dict": target_ppi_dict,
        **param_values  # Dynamically add parameter values as columns
    }

    # Append the log entry to the simulation log
    simulation_log = pd.concat([simulation_log, pd.DataFrame([log_entry])], ignore_index=True)


def initialize_simulation_log(params_to_change):
    """
    Initializes the simulation log DataFrame with specified parameter names.

    Args:
        params_to_change (dict): Dictionary of parameters to change.
    Returns:
        pd.DataFrame: An empty DataFrame with the appropriate columns.
    """
    global simulation_log
    param_names = list(params_to_change.keys())
    columns = ['algorithm', 'simulation_id', 'timestamp', 'target_range', 'status', 'target_ppis', 'target_ppi_means', 'target_ppi_dict'] + param_names
    simulation_log = pd.DataFrame(columns=columns)


def save_params(params):
    """
    Saves the simulation parameters to a JSON file specified in params['output_params_path'].

    Args:
        params (dict): Simulation parameters including 'output_params_path'.
    """
    output_params_path = params.get('output_params_path')
    if output_params_path:
        with open(output_params_path, 'w') as outfile:
            json.dump(params, outfile, indent=4, sort_keys=True)
        print(f"Parameters saved to {output_params_path}")
    else:
        print("Error: 'output_params_path' not found in params dictionary.")


# Function to log execution times per search strategy
execution_times_log = {}    
def log_execution_time(strategy_name, execution_time):
    """
    Logs the execution time for a given search strategy.

    Args:
        strategy_name (str): Name of the search strategy.
        execution_time (float): Execution time in seconds.
    """
    global execution_times_log
    if strategy_name not in execution_times_log:
        execution_times_log[strategy_name] = []
    execution_times_log[strategy_name].append(execution_time)


def save_execution_times_log(output_path):
    """
    Saves the execution times log to a JSON file.

    Args:
        output_path (str): Path to the output JSON file.
    """
    global execution_times_log
    with open(output_path, 'w') as outfile:
        json.dump(execution_times_log, outfile, indent=4, sort_keys=True)
    print(f"Execution times log saved to {output_path}")    