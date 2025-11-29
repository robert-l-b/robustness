#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

from simod.simod import Simod
from simod.event_log.event_log import EventLog
from simod.settings.simod_settings import SimodSettings

from src.simulators.param_manipulation import update_simod_config_train_path


def copy_simod_files(params):
    """
    Copies the BPMN and JSON files from the 'best_result' directory to the base directory.

    Args:
        params (dict): Dictionary containing configuration parameters, including 'print_intermediate_results'.

    Returns:
        None
    """

    base_path = params['base_path']
    process_name = params['process_name']

    # Define source paths
    bpmn_src = os.path.join(base_path, 'best_result', process_name + '.bpmn')
    json_src = os.path.join(base_path, 'best_result', process_name + '.json')

    # Define destination paths
    bpmn_dest = os.path.join(base_path, process_name + '.bpmn')
    json_dest = os.path.join(base_path, process_name + '.json')

    # Helper function to conditionally print messages
    def conditional_print(message):
        if params.get('print_intermediate_results', False):
            print(message)

    # Copy the BPMN file
    try:
        shutil.copy(bpmn_src, bpmn_dest)
        conditional_print(f"Copied BPMN file from {bpmn_src} to {bpmn_dest}")
    except FileNotFoundError:
        conditional_print(f"Error: BPMN file not found at {bpmn_src}")
    except Exception as e:
        conditional_print(f"Error copying BPMN file: {e}")

    # Copy the JSON file
    try:
        shutil.copy(json_src, json_dest)
        conditional_print(f"Copied JSON file from {json_src} to {json_dest}")
    except FileNotFoundError:
        conditional_print(f"Error: JSON file not found at {json_src}")
    except Exception as e:
        conditional_print(f"Error copying JSON file: {e}")


def discover_BPS_simod(params):
    """ 
    Discover a Business Process Simulation Model using SIMOD.
    Args:
        params (dict): Dictionary containing configuration parameters, including 'base_path'.
    Returns:
        None    
    """
    # Read and update the simod configuration file: set train_log_path
    update_simod_config_train_path(params)

    # Specify the path to the simod directory
    # simod_directory = os.path.join('simulators', 'simod')
    simod_directory = params['simod_directory']

    # output = Path(os.path.join(simod_directory, 'resources', 'output'))
    output = Path(os.path.join(params['base_path']))


    configuration_path = Path(params['simod_config_path'])
    # configuration_path = Path(os.path.join(simod_directory, 'resources', 'config', 'config_one-shot.yml' ))
    settings = SimodSettings.from_path(configuration_path)

    # Read and preprocess event log
    event_log = EventLog.from_path(
        log_ids=settings.common.log_ids,
        train_log_path=settings.common.train_log_path,
        # test_log_path=settings.common.test_log_path,
        preprocessing_settings=settings.preprocessing,
        need_test_partition=settings.common.perform_final_evaluation,
    )

    # Instantiate and run SIMOD
    simod = Simod(settings=settings, event_log=event_log, output_dir=output)
    simod.run()

    # Copy the discovered model to the process data folder
    copy_simod_files(params)