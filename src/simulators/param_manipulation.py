#!/usr/bin/env python3

# Getting / setting sim parameters
# SIMOD related parameter manipulation functions

import json
import os
import yaml
from ruamel.yaml import YAML

################################################
# Functions to handle simod configuration files  #
################################################

# create temporal simulation parameters file
def create_temp_json(input_path, output_path=None):

    if output_path == None:
        base, ext = os.path.splitext(input_path)
        if ext != '.json':
            raise ValueError('File extension must be .json')
        output_path = f'{base}_temp{ext}'

    # Step 1: Read the JSON
    with open(input_path, 'r') as infile:
        data = json.load(infile)

    # Step 2: Write a pretty-printed version
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

    return output_path



def read_simod_config_train_path(params):
    """
    Reads the YAML file specified in params['simod_config'] and extracts the 'train_log_path' 
    variable under the 'common' section.

    Args:
        params (dict): A dictionary containing the key 'simod_config' with the path to the YAML file.

    Returns:
        None
    """
    try:
        # Read the YAML file specified in params['simod_config']
        simod_config_path = params.get("simod_config_path")
        if not simod_config_path:
            print("Error: 'simod_config' key is missing in the params dictionary.")
            return
        
        with open(simod_config_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Extract the 'train_log_path' under the 'common' section
        train_log_path = config.get("common", {}).get("train_log_path")
        if train_log_path:
            print(f"Train Log Path: {train_log_path}")
        else:
            print("Error: 'train_log_path' not found under the 'common' section in the YAML file.")
    
    except FileNotFoundError:
        print(f"Error: The file '{simod_config_path}' does not exist.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the YAML file. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def update_simod_config_train_path(params):
    """
    Reads the YAML file specified in params['simod_config'], updates the 'train_log_path' parameter
    under the 'common' section with a new value based on params['process_name'], and writes the
    updated content back to the YAML file while preserving the layout.

    Args:
        params (dict): A dictionary containing:
            - 'simod_config': Path to the YAML file.
            - 'process_name': Name of the process to update the train_log_path with.

    Returns:
        None
    """
    try:
        # Ensure required keys exist in params
        simod_config_path = params.get("simod_config_path")
        process_name = params.get("process_name")
        if not simod_config_path or not process_name:
            print("Error: 'simod_config' or 'process_name' key is missing in the params dictionary.")
            return

        # Load the YAML file while preserving layout
        yaml = YAML()
        yaml.preserve_quotes = True  # Preserve quotes and formatting
        with open(simod_config_path, "r") as file:
            config = yaml.load(file)

        # Find and update the 'train_log_path' under the 'common' section
        train_log_path = config.get("common", {}).get("train_log_path")
        if train_log_path:
            # Replace the data folder and update the path
            updated_path = train_log_path.split("data/")[0] + f"data/{process_name}/{process_name}.csv"
            config["common"]["train_log_path"] = updated_path
            if params.get('print_intermediate_results', True):
                print(f"Updated train_log_path: {updated_path}")
        else:
            print("Error: 'train_log_path' not found under the 'common' section in the YAML file.")
            return

        # Write the updated YAML back to the file
        with open(simod_config_path, "w") as file:
            yaml.dump(config, file)
        if params.get('print_intermediate_results', True):
            print(f"Updated YAML file saved to: {simod_config_path}")

    except FileNotFoundError:
        print(f"Error: The file '{simod_config_path}' does not exist.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the YAML file. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")








################################################
# Functions to get and set simulation parameters #
################################################


def extract_cost_per_hour(json_path):
    """
    Extracts a dictionary mapping resource names to their cost per hour from the given JSON file.

    Args:
        json_path (str): Path to the JSON file containing resource profiles.

    Returns:
        dict: A dictionary where keys are resource names and values are their cost per hour.
    """
    # Load the JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    # Extract resource profiles
    resource_profiles = data.get("resource_profiles", [])
    
    # Create the cost_per_hour dictionary
    cost_per_hour = {}
    for profile in resource_profiles:
        for resource in profile.get("resource_list", []):
            resource_name = resource.get("name")
            cost = resource.get("cost_per_hour")
            if resource_name and cost is not None:
                cost_per_hour[resource_name] = cost

    return cost_per_hour



# get simulation parameters
# currently read json in 
def get_sim_params(json_path_temp):
    with open(json_path_temp, 'r') as infile:
        sim_params = json.load(infile)
    return sim_params


def set_sim_params(json_path_temp, sim_params):
    """
    Write updated simulation parameters back to the temporary JSON file.
    """
    with open(json_path_temp, 'w') as outfile:
        json.dump(sim_params, outfile, indent=4, sort_keys=True)





# change_param = 'arriaval_distr_mean'
# get_change_param_values(params_to_change, change_param, sim_params)

# def set_change_param_value(change_param, new_value, sim_params):
#     """
#     Update the value of a specified parameter in the simulation parameters dictionary.
#     """
#     if change_param == 'arriaval_distr_mean':
#         sim_params['arrival_time_distribution']['distribution_params'][0]['value'] = new_value
#     else:
#         raise ValueError(f"Unsupported parameter for setting: {change_param}")
#     return sim_params


def set_change_param_value(param_name, new_value, sim_params, params_to_change=None):
    """
    Updates a specific parameter in the simulation parameters, including resources.

    Args:
        param_name (str): The name of the parameter to update.
        new_value (float or int): The new value to set for the parameter.
        sim_params (dict): The simulation parameters JSON structure.

    Returns:
        dict: The updated simulation parameters.
    """
    if param_name == "arriaval_distr_mean":
        # Update the arrival time distribution mean
        if "arrival_time_distribution" in sim_params:
            sim_params["arrival_time_distribution"]["distribution_params"][0]["value"] = new_value
    
    elif param_name.startswith("resource_count_"):
        # Update resource-specific parameters
        resource_id = param_name.split("_")[-1]  # Extract the resource ID
        for resource_profile in sim_params.get("resource_profiles", []):
            if resource_profile["id"] == resource_id:
                for resource in resource_profile.get("resource_list", []):
                    resource["amount"] = new_value
                break

    elif param_name.startswith("branching_probability_"):
        # Update branching probabilities in the JSON file
        # Extract the gateway_id and path_ids from the param_name
        gateway_id = param_name.split("_")[-1]  # Extract the gateway_id from the parameter name
        gateway_id = params_to_change[param_name].get('gateway_id', gateway_id)
        path_ids = params_to_change[param_name].get('path_ids', [])

        # Find the gateway in the simulation parameters
        for gateway in sim_params.get("gateway_branching_probabilities", []):
            if gateway["gateway_id"] == gateway_id:
                # Update the probabilities for the specified path_ids
                for probability in gateway["probabilities"]:
                    if probability["path_id"] == path_ids[0]:  # Update the specified path_id
                        probability["value"] = new_value
                    elif probability["path_id"] == path_ids[1]:  # Update the other path_id
                        probability["value"] = 1 - new_value
                break

    return sim_params


def get_resource_profile_ids_and_amounts(resource_model):
    """
    Extracts resource profile IDs and their corresponding amounts from the resource model.

    Args:
        resource_model (dict): The resource model containing resource profiles.

    Returns:
        dict: A dictionary where keys are resource profile IDs and values are their amounts.
    """
    resource_profiles = resource_model.get("resource_profiles", [])
    resource_data = {}

    for profile in resource_profiles:
        profile_id = profile.get("id")
        resource_list = profile.get("resource_list", [])
        
        # Sum up the amounts for all resources in the resource list
        total_amount = sum(resource.get("amount", 0) for resource in resource_list)
        
        # Store the profile ID and total amount
        resource_data[profile_id] = total_amount

    return resource_data





def set_params_to_change(params_to_change, input_parameters, update_parameters_list, sim_params):
    '''
    Adds parameters to params_to_change based on the input_parameters and update_parameters_list.
    Handles both resource parameters and branching probabilities with filtering options.
    Args:
        params_to_change (dict): Existing dictionary of parameters to change.
        input_parameters (dict): Input parameters specifying types and values.
        update_parameters_list (list): List of parameters to update.
        sim_params (dict): Simulation parameters containing resource and branching data.
    Returns:
        dict: Updated params_to_change dictionary.
    '''

    # Initialize params_to_change with all parameters that are not in update_parameters_list 
    params_to_change = {k: v for k, v in input_parameters.items() if k not in update_parameters_list}

    for input_parameter in input_parameters.keys():
        if input_parameter not in update_parameters_list:
            continue
        update_parameter = input_parameter

        if update_parameter == 'resource_count':            
            # Get resource data
            resource_data = get_resource_profile_ids_and_amounts(sim_params)
            
            # Check if 'ignore' exists in the input parameters
            ignore_list = input_parameters.get('resource_count', {}).get('ignore', [])
            ignore_list = [ignore.lower() for ignore in ignore_list]  # Convert ignore list to lowercase
            
            for resource_id, amount in resource_data.items():
                # Convert resource_id to lowercase for case-insensitive comparison
                if any(ignore_str in resource_id.lower() for ignore_str in ignore_list):
                    continue
                
                # Add resource-specific parameter
                param_key = f'resource_count_{resource_id}'
                params_to_change[param_key] = {
                    'type': 'disc',
                    'values': input_parameters['resource_count']['values']
                }

        if update_parameter == 'branching_probability':
            clustered_branching_probs = {}

            # Cluster probabilities by gateway_id
            if "gateway_branching_probabilities" in sim_params:
                for gateway in sim_params["gateway_branching_probabilities"]:
                    gateway_id = gateway.get("gateway_id")
                    if gateway_id is None:
                        continue  # Skip if no gateway_id is found
                    
                    # Initialize the list for this gateway_id if not already present
                    if gateway_id not in clustered_branching_probs:
                        clustered_branching_probs[gateway_id] = []
                    
                    # Add probabilities to the corresponding gateway_id
                    if "probabilities" in gateway:
                        clustered_branching_probs[gateway_id].extend(gateway["probabilities"])
            else:
                raise ValueError("No 'gateway_branching_probabilities' found in sim_params. Choose another process parameter.")

            # Check if 'ignore' or 'use' exists in the input parameters
            use_list = input_parameters.get('branching_probability', {}).get('use', [])
            ignore_list = input_parameters.get('branching_probability', {}).get('ignore', [])

            # Convert all gateway_ids in use_list and ignore_list to lowercase for case-insensitive comparison
            use_list = [gateway_id.lower() for gateway_id in use_list]
            ignore_list = [gateway_id.lower() for gateway_id in ignore_list]

            # Filter clustered_branching_probs based on use_list or ignore_list
            if use_list:
                # Keep only the gateway_ids in use_list
                clustered_branching_probs = {
                    gateway_id: probs
                    for gateway_id, probs in clustered_branching_probs.items()
                    if gateway_id.lower() in use_list
                }
            elif ignore_list:
                # Exclude the gateway_ids in ignore_list
                clustered_branching_probs = {
                    gateway_id: probs
                    for gateway_id, probs in clustered_branching_probs.items()
                    if gateway_id.lower() not in ignore_list
                }

            for gateway in clustered_branching_probs:
                if len(clustered_branching_probs[gateway]) < 2:
                    raise ValueError(f"Less then 2 branching paths detected in gateway {gateway}. Please adjust the 'ignore' and 'use' lists in the input parameters to ensure at least two branching paths are considered for each gateway.")
                elif len(clustered_branching_probs[gateway]) > 2:
                    raise ValueError(f"More then 2 branching paths detected in gateway {gateway}. Current variation strategy only supports 2 branching paths per gateway. Please adjust the 'ignore' and 'use' lists in the input parameters to ensure only two branching paths are considered for each gateway.")
                else:
                    first_path_id = clustered_branching_probs[gateway][0]['path_id']
                    # key = f"branching_probability_{gateway}->{first_path_id}"
                    # Set Key to shorter version
                    key = f"branching_probability_{gateway[:10]}->{first_path_id[:10]}"
                    params_to_change[key] = {
                        'type': input_parameters['branching_probability']['type'],
                        'values':  input_parameters['branching_probability']['values'],
                        'category': 'branching_probability',
                        'gateway_id': gateway,
                        'path_ids': [node['path_id'] for node in clustered_branching_probs[gateway]]
                }
    return params_to_change


# get change_param_values 
def get_change_param_values(params_to_change, change_param, sim_params):
    """ 
    Retrieve the current value of a specified parameter from the simulation parameters. 
    Args:
        params_to_change (list): List of parameters to vary.
        change_param (str): The parameter whose value is to be retrieved.
        sim_params (dict): The simulation parameters JSON structure.
    Returns:
        float or int: The current value of the specified parameter.
    """

    change_param_val = None

    if change_param == 'arriaval_distr_mean':
        change_param_val = sim_params['arrival_time_distribution']['distribution_params'][0]['value']

    # elif change_param == 'resource_count':
    #     change_param_val = sim_params['resource_profiles'][0]['resource_list'][0]['amount']
    elif change_param.startswith("resource_count_"):
        # Split of "resource_count_" from the beginning of the resource_id string
        resource_id = change_param[len("resource_count_"):]
        for resource_profile in sim_params.get("resource_profiles", []):
            if resource_profile["id"] == resource_id:
                for resource in resource_profile.get("resource_list", []):
                    change_param_val = resource["amount"]

    elif change_param.startswith("branching_probability_"):
        # Extract the gateway_id and path_ids from the param_name
        gateway_id = change_param.split("_")[-1]  # Extract the gateway_id from the parameter name
        gateway_id = params_to_change[change_param].get('gateway_id', gateway_id)
        path_ids = params_to_change[change_param].get('path_ids', [])

        # Find the gateway in the simulation parameters
        for gateway in sim_params.get("gateway_branching_probabilities", []):
            if gateway["gateway_id"] == gateway_id:
                # Get the probabilities for the specified path_ids
                for probability in gateway["probabilities"]:
                    if probability["path_id"] == path_ids[0]:  # Get the specified path_id
                        change_param_val = probability["value"]
                break           

    elif change_param not in params_to_change.keys():
        raise ValueError(f"Unsupported parameter for change values: {change_param}")

    return change_param_val 



def get_start_param_settings(params_to_change, params):
    """ Retrieve the starting parameter settings from the simulation parameters.    
    Args:
        params_to_change (list): List of parameters to vary.
        params (dict): General parameters including json_path.
    Returns: 
        dict: A dictionary containing the starting parameter settings.
    """
    sim_params = get_sim_params(params['json_path'])
    start_param_settings = {}
    for param in params_to_change:
        start_param_settings[param] = get_change_param_values(params_to_change, change_param=param, sim_params=sim_params)
        
    return start_param_settings