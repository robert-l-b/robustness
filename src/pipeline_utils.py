#!/usr/bin/env python3

import os
import numpy as np
from src.simulators.param_manipulation import get_sim_params, set_params_to_change
from src.sim_execution_and_evalaution import get_simulation_stats
from src.simulators.param_manipulation import create_temp_json


def prepare_simulation_files(params):
    '''
    Prepare simulation files: BPMN and JSON paths, create temporary JSON.
    '''
    bpmn_path = os.path.join(params['base_path'], params['process_name'] + '.bpmn')
    json_path = os.path.join(params['base_path'], params['process_name'] + '.json')
    params['json_path'] = json_path
    json_path_temp = create_temp_json(input_path=json_path, output_path=None)
    params['bpmn_path'] = bpmn_path
    params['json_path_temp'] = json_path_temp
    return params

def get_params_to_change(params):
    '''
    Determine which parameters to change based on user input and configuration. 
    Args:
        params (dict): Configuration parameters.
        input_parameters (dict): User-specified input parameters.
    Returns:
        params_to_change (dict): Parameters that will be changed during simulation.
    '''

    params_to_change = {}

    update_parameters_list = params['update_parameters_list']
    input_parameters = params['input_parameters']
    sim_params = get_sim_params(params['json_path'])
    params_to_change = set_params_to_change(params_to_change, input_parameters, update_parameters_list, sim_params) 


    if params['print_intermediate_results']:
        print("\nInput parameters:")  
        print(input_parameters)
        print("Parameters to change:")
        print(params_to_change)
        print()

    else:
        params_to_change = input_parameters

    return params_to_change


from src.simulators.param_manipulation import set_change_param_value, get_change_param_values

def set_target_val_and_range(params):
    '''
    Set the target PPI values and their corresponding ranges in the params dictionary.
    
    Args:
        params (dict): Configuration parameters.
    Returns:
        params (dict): Updated configuration parameters with target PPI values and ranges.
    '''

    # Get the initial target PPI value
    target_ppi_dict = get_simulation_stats(params)
    
    ppi_bounds = params['ppi_bounds']

    orig_target_ppi_val_dict = {}
    target_range_dict = {}

    for ppi in params['target_ppis']:
        target_ppi_list = target_ppi_dict[ppi]
        target_ppi_val = np.mean(target_ppi_list)
        orig_target_ppi_val_dict[ppi] = target_ppi_val
        
        # calculate ppi bounds
        target_range = [np.round(target_ppi_val*(1-params['ppi_range_factor'])), np.round(target_ppi_val*(1+params['ppi_range_factor']))]
        # adjust target range based on ppi_bounds
        if ppi_bounds[ppi]=='upper':
            target_range = [0, target_range[1]]
        elif ppi_bounds[ppi]=='lower':
            target_range = [target_range[0], float('inf')]
        elif ppi_bounds[ppi]=='both':
            pass
        else:
            raise ValueError(f"Unknown ppi_bounds value: {ppi_bounds[ppi]}")
        target_range_dict[ppi] = target_range

    params['orig_target_ppi_val_dict'] = orig_target_ppi_val_dict
    params['target_range'] = target_range_dict
    params['target_ppi_dict'] = target_ppi_dict

    return params