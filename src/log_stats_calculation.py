#!/usr/bin/env python3
import pandas as pd
from src.utils import LogColumnNames


def trace_list_to_dataframe(t, colmap: LogColumnNames = LogColumnNames()):
    rows = []
    for trace in t.trace_list:
        case_id = getattr(trace, "p_case", None)
        for event in trace.event_list:
            rows.append({
                colmap.case_id: case_id,
                colmap.activity: event.task_id,
                colmap.enable_time: event.enabled_datetime,
                colmap.start_time: event.started_datetime,
                colmap.end_time: event.completed_datetime,
                colmap.resource: event.resource_id,
            })

    df = pd.DataFrame(rows)

    # Convert datetime columns
    for attr in ["enable_time", "start_time", "end_time"]:
        df[getattr(colmap, attr)] = pd.to_datetime(df[getattr(colmap, attr)])

    return df


def calculate_metric(values, metric):
    """
    Calculate a specified metric for a list of values.

    Args:
        values (list or pd.Series): List or Series of numerical values.
        metric (str): Metric to calculate ("min", "max", "avg", "total", "median").

    Returns:
        float: The calculated metric value.
    """
    if metric == "min":
        return values.min()
    elif metric == "max":
        return values.max()
    elif metric == "avg":
        return values.mean()
    elif metric == "total":
        return values.sum()
    elif metric == "median":
        return values.median()
    else:
        raise ValueError(f"Invalid metric: {metric}. Choose from 'min', 'max', 'avg', 'total', or 'median'.")


def calculate_lead_time(df, colmap: LogColumnNames = LogColumnNames(), begin_col="enable_time", metric="avg"):
    """
    Calculate lead time and return the specified metric.

    Args:
        df (pd.DataFrame): DataFrame containing event log data.
        colmap (LogColumnNames): Column mapping for the log.
        begin_col (str): Column name for the start time.
        metric (str): Metric to calculate ("min", "max", "avg", "total", "median").

    Returns:
        float: The calculated metric value.
    """
    case_col = getattr(colmap, "case_id")
    begin_col = getattr(colmap, begin_col)
    end_col = getattr(colmap, "end_time")

    # Ensure datetime conversion
    df[begin_col] = pd.to_datetime(df[begin_col])
    df[end_col] = pd.to_datetime(df[end_col])

    # Calculate lead time for each case
    lead_times = df.groupby(case_col).apply(
        lambda g: (g[end_col].max() - g[begin_col].min()).total_seconds()
    )

    # Use the calculate_metric function to compute the desired metric
    return calculate_metric(lead_times, metric)





def calculate_custom_stats(params, ppi_dict, cost_per_hour, r, t):
    """
    Calculates custom statistics from the simulation trace list.
    Args:
        params (dict): Simulation parameters, including the number of simulations to run.
        ppi_dict (dict): Dictionary to store PPI values.
        cost_per_hour (dict): Cost per hour for resources.
        r: Simulation results.
        t: Simulation trace list.
    Returns:
        Updated ppi_dict with calculated PPI values.
    """
    for ppi in params['target_ppis']:
        # read the log in to calculate the PPI value

        colmap = LogColumnNames()
        sim_log = trace_list_to_dataframe(t, colmap)
        if ppi == 'lead_time':
            metric_type = params['ppi_calculation'][ppi]['type']
            value = calculate_lead_time(sim_log, colmap, begin_col="enable_time", metric=metric_type)
        elif ppi == 'cost':
            # metric_type = params['ppi_calculation'][ppi]['type']
            # calculation_method = params['ppi_calculation'][ppi]['method']
            cost_calculation_dict = params['ppi_calculation'][ppi]
            value = calculate_cost(sim_log, cost_per_hour, r, cost_calculation_dict)
        else:
            raise ValueError(f"PPI '{ppi}' is not supported for custom calculation.")

        ppi_dict[ppi].append(value) 
    return ppi_dict



# def calculate_cost(df, cost_per_hour, r, colmap: LogColumnNames = LogColumnNames(), metric="total", cost_calculation="active_time", weight_active_time=0.5):
def calculate_cost(df, cost_per_hour, r, cost_calculation_dict, colmap: LogColumnNames = LogColumnNames()):
    """
    Calculate the cost of the event log based on processing time and resource cost per hour.

    Args:
        df (pd.DataFrame): DataFrame containing event log data.
        cost_per_hour (dict): Dictionary mapping base resource names to cost per hour.
        r: Simulation results containing resource statistics.
        cost_calculation_dict (dict): Dictionary containing 'method' (active_time, full_duration, combined),
        colmap (LogColumnNames): Column mapping for the log.
    Returns:
        float: The calculated cost metric value.
    """
    # Extract parameters from cost_calculation_dict
    metric = cost_calculation_dict.get('type', 'total')
    cost_calculation = cost_calculation_dict.get('method', 'active_time')
    
    # Retrieve column names
    start_col = getattr(colmap, "start_time")
    end_col = getattr(colmap, "end_time")
    resource_col = getattr(colmap, "resource")
    
    # Ensure datetime conversion
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])

    if cost_calculation == "active_time":
        return calculate_active_time_cost(df, cost_per_hour, resource_col, start_col, end_col, metric)

    elif cost_calculation == "full_duration":
        return calculate_full_duration_cost(r, cost_per_hour)

    elif cost_calculation == "combined":
        active_time_cost = calculate_active_time_cost(df, cost_per_hour, resource_col, start_col, end_col, metric)
        full_duration_cost = calculate_full_duration_cost(r, cost_per_hour)
        weight_active_time = cost_calculation_dict.get('weight', 0.5)
        return combine_costs(active_time_cost, full_duration_cost, weight_active_time)

    else:
        raise ValueError(f"Invalid cost_calculation method: {cost_calculation}. Choose from 'active_time', 'full_duration', or 'combined'.")



def calculate_active_time_cost(df, cost_per_hour, resource_col, start_col, end_col, metric):
    """
    Calculate the cost based on active time.

    Args:
        df (pd.DataFrame): DataFrame containing event log data.
        cost_per_hour (dict): Dictionary mapping base resource names to cost per hour.
        resource_col (str): Column name for resource.
        start_col (str): Column name for start time.
        end_col (str): Column name for end time.
        metric (str): Metric to calculate ("min", "max", "avg", "total", "median").

    Returns:
        float: The calculated cost metric value.
    """
    # Extract the base resource name (e.g., "UnifiedResource" from "UnifiedResource_0")
    df["role"] = df[resource_col].str.split("_").str[0]

    # Calculate processing time in hours for each event
    df["processing_time_hours"] = (df[end_col] - df[start_col]).dt.total_seconds() / 3600

    # Map base resource to cost per hour and calculate cost for each event
    df["event_cost"] = df["role"].map(cost_per_hour) * df["processing_time_hours"]

    # Calculate the desired metric
    return calculate_metric(df["event_cost"], metric)


def calculate_full_duration_cost(r, cost_per_hour):
    """
    Calculate the cost based on full duration.

    Args:
        r: Simulation results containing resource statistics.
        cost_per_hour (dict): Dictionary mapping base resource names to cost per hour.

    Returns:
        float: The total cost based on full duration.
    """
    resource_stats = r[2]
    if resource_stats is None:
        raise ValueError("resource_stats must be provided for 'full_duration' cost calculation.")

    # Initialize dictionary to store total cost per profile
    total_cost_per_profile = {profile: 0 for profile in cost_per_hour.keys()}

    # Calculate total cost for each profile
    for resource_id, stats in resource_stats.items():
        # Match the resource_id with the appropriate role in cost_per_hour
        matched_profile = next((profile for profile in cost_per_hour.keys() if resource_id.startswith(profile)), None)
        if matched_profile:
            worked_time_hours = stats.worked_time / 3600  # Convert worked time to hours
            total_cost_per_profile[matched_profile] += worked_time_hours * cost_per_hour[matched_profile]

    # Sum up the total cost across all profiles
    return sum(total_cost_per_profile.values())


def combine_costs(active_time_cost, full_duration_cost, weight_active_time):
    """
    Combine active time cost and full duration cost using a weight factor.

    Args:
        active_time_cost (float): Cost based on active time.
        full_duration_cost (float): Cost based on full duration.
        weight_active_time (float): Weight factor for the active time cost.

    Returns:
        float: The combined cost.
    """
    # return (weight_active_time * active_time_cost) + ((1 - weight_active_time) * full_duration_cost)
    return full_duration_cost + active_time_cost * (1 + weight_active_time)




