#!/usr/bin/env python3

# Script for logging simulation results and parameters

import os
import shutil
import pandas as pd
import numpy as np
import json
import datetime
# from src.simulators.param_manipulation import *
from src.sim_execution_and_evalaution import is_in_target_range
# from src.sim_execution_and_evalaution import *

# Logging function
def log_simulation(simulation_log, algorithm, params, param_values, target_ppi_dict):
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
    
    return simulation_log


def initialize_simulation_log(params_to_change):
    """
    Initializes the simulation log DataFrame with specified parameter names.

    Args:
        params_to_change (dict): Dictionary of parameters to change.
    Returns:
        pd.DataFrame: An empty DataFrame with the appropriate columns.
    """
    # global simulation_log
    param_names = list(params_to_change.keys())
    columns = ['algorithm', 'simulation_id', 'timestamp', 'target_range', 'status', 'target_ppis', 'target_ppi_means', 'target_ppi_dict'] + param_names
    return pd.DataFrame(columns=columns)




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


def save_simulation_log(simulation_log, params):
    """
    Saves the simulation log DataFrame to a CSV file.

    Args:
        simulation_log (pd.DataFrame): DataFrame containing the simulation log.
        params (dict): Simulation parameters including 'simulation_log_path'.
    """
    # Save the simulation log to a CSV file for later analysis
    simulation_log.to_csv(params['simulation_log_path'], index=False)

    print(f"Simulation log saved to {params['simulation_log_path']}")


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





def set_up_experiment_output_dir(params):
    """
    Sets up the experiment output directory based on the current timestamp.

    Args:
        params (dict): Dictionary containing configuration parameters, including 'base_path'.

    Returns:
        params (dict): Updated params dictionary with 'experiment_output_dir', 'output_params_path', and 'simulation_log_path' keys.
    """
    

    # Define the output directory path
    output_dir = os.path.join(params['base_path'], 'output')

    # Generate a timestamp-based directory name
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nr_of_parameters = len(params['params_to_change'])
    nr_of_ppis = len(params['target_ppis'])
    n = params['nr_simulations_per_scenario']
    c = params.get('confidence', 'NA')
    beta = params.get('beta', 'NA')
    dir_name = f'{current_timestamp}_{nr_of_parameters}D_{nr_of_ppis}PPIs_{n}n_{c}a_{beta}b'
    
    experiment_output_dir = os.path.join(output_dir, dir_name)

    # Create the directory if it does not exist
    os.makedirs(experiment_output_dir, exist_ok=True)
    params['experiment_output_dir'] = experiment_output_dir

    output_params_path = os.path.join(params['experiment_output_dir'], 'params.json')    
    params['output_params_path'] = output_params_path 
    simulation_log_path = os.path.join(params['experiment_output_dir'], 'simulation_log.csv')
    params['simulation_log_path'] = simulation_log_path

    # Copy the file that is under params[json_path'] to params['experiment_output_dir'] 
    shutil.copy(params['json_path'], experiment_output_dir)

    print(f"Output directory ensured at: {experiment_output_dir}")

    return params


def create_results_dataframe():
    """
    Creates an empty DataFrame with the columns:
    experiment, algorithm, evals, time, acc, mcc.

    Returns:
        pd.DataFrame: An empty DataFrame with the specified columns.
    """
    columns = ['experiment', 'algorithm', 'evals', 'time', 'n', 'n_total', 'acc', 'mcc']
    results_df = pd.DataFrame(columns=columns)
    return results_df

def save_results_dataframe(results_df, params):
    """
    Saves the results DataFrame to a CSV file.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results.
        params (dict): Dictionary containing configuration parameters, including 'output_params_path'.
    """
    output_params_path = params.get('experiment_output_dir')
    filename = 'results_df.csv'
    output_path = os.path.join(output_params_path, filename)
    results_df.to_csv(output_path, index=False)
    print(f"Results DataFrame saved to {output_path}")