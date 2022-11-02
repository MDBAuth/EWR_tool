from typing import List, Dict
from itertools import chain
from collections import defaultdict

import pandas as pd

from . import data_inputs
#--------------------------------------------------------------------------------------------------


def get_frequency(events: list) -> int:
    '''Returns the frequency of years they occur in.
    
    Args:
        events (list): a list of years with events (0 for no event, 1 for event)
    Results:
        int: Frequency of years with events
    
    '''
    if events.count() == 0:
        result = 0
    else:
        result = (int(events.sum())/int(events.count()))*100
    return int(round(result, 0))


def get_ewr_columns(ewr:str, cols:List) -> List:
    """Filter the columns of a particular ewr code in a list of 
    column names.

    Args:
        ewr (str): Ewr code
        cols (List): list of columns to search ewr pattern

    Returns:
        List: List of columns that matches the ewr code
    """
    return [c for c in cols if ewr in c]


def get_columns_attributes(cols: List)-> List:
    """Takes a list of columns with the pattern EwrCode_Attribute
    and relates them returning only the Attribute name.

    Args:
        cols (List): DataFrame columns names as a list

    Returns:
        List: List of the column attribute stripped out of the ewr code
    """
    return [c.split("_")[-1] for c  in cols]

def get_ewrs(pu_df: pd.DataFrame)-> List:
    """Take a DataFrame with the location results and by searching its 
    column return a list with the unique name of ewrs on the results.

    Args:
        pu_df (pd.DataFrame): DataFrame with with location stats results

    Returns:
        List: Returns a list os unique ewrs present in the location results
    """
    cols = pu_df.columns.to_list()
    ewrs_set = set(["_".join(c.split("_")[:-1]) for c  in cols])
    return list(ewrs_set)

def pu_dfs_to_process(detailed_results: Dict)-> List[Dict]:
    """Take the detailed results dictionary of the ewr calculation run,
    either observed or scenario and unpack items into a list of items.
    Each item is a dictionary with the following keys.
                { "scenario" : scenario_name,
                  "gauge" : gauge_id,
                  "pu" : pu_name,
                  "pu_df : DataFrame}

    Args:
        detailed_results (Dict): Dictionary with the following structure
        { "scenario_name/or observed": {"gaugeID": {"pu_name": pu_DateFrame}
        
            }

        } 
        It packs in a dictionary all the gauge ewr calculation for the scenario
        or observed dates run.

    Returns:
        List[Dict]: list of dict with the items to be processed
    """
    items_to_process = []
    for scenario in detailed_results:
        for gauge in detailed_results[scenario]:
            for pu in detailed_results[scenario][gauge]:
                item = {}
                item["scenario"] = scenario
                item["gauge"] = gauge
                item["pu"] = pu
                item["pu_df"] = detailed_results[scenario][gauge][pu]
                items_to_process.append(item)
    return items_to_process


def process_df(scenario:str, gauge:str, pu:str, pu_df: pd.DataFrame)-> pd.DataFrame:
    """Process all the pu_dfs into a tidy format

    Args:
        scenario (str): scenario name metadata
        gauge (str): gauge name metadata
        pu (str): planning unit name metadata
        pu_df (pd.DataFrame): DataFrame to be transformed

    Returns:
        pd.DataFrame: DataFrame with all processed pu_dfs into one.
    """
    ewrs = get_ewrs(pu_df)
    returned_dfs = []
    for ewr in ewrs:
        columns_ewr = get_ewr_columns(ewr, pu_df.columns.to_list())
        ewr_df = pu_df[columns_ewr]
        column_attributes = get_columns_attributes(ewr_df.columns.to_list())
        ewr_df.columns = column_attributes
        ewr_df = ewr_df.reset_index().rename(columns={"index":'Year'})
        ewr_df["ewrCode"] = ewr
        ewr_df["scenario"] = scenario
        ewr_df["gauge"] = gauge
        ewr_df["pu"] = pu
        ewr_df = ewr_df.loc[:,~ewr_df.columns.duplicated()]
        returned_dfs.append(ewr_df)
    return pd.concat(returned_dfs, ignore_index=True)


def process_df_results(results_to_process: List[Dict])-> pd.DataFrame:
    """Manage the processing and concatenating the processed dfs into one
    single dataframe with the results of all ewr calculations.

    Args:
        results_to_process (List[Dict]): List with all items to process.

    Returns:
        pd.DataFrame: Single DataFrame with all the ewr results
    """
    returned_dfs = []
    for item in results_to_process:
        try:
            transformed_df = process_df(**item)
            returned_dfs.append(transformed_df)
        except Exception as e:
            print(f"Could not process due to {e}")
    return pd.concat(returned_dfs, ignore_index=True)

def get_events_to_process(gauge_events: dict)-> List:
    """Take the detailed gauge events results dictionary of the ewr calculation run,
    and unpack items into a list of items.
    Each item is a dictionary with the following keys.
                { "scenario" : scenario_name,
                  "gauge" : gauge_id,
                  "pu" : pu_name,
                  "ewr": ewr_code
                  "ewr_events" : yearly_events_dictionary}

    Args:
        gauge_events (dict): Gauge events captured by the ewr calculations.
        Dictionary with the following structure
        {'observed': {'419001': {'Keepit to Boggabri': {'CF1_a': ({2010: [],
                2011: [],
                2012: [],
                2013: [],
                2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},)}
                                }
                     }
        }
        It packs in a dictionary all the gauge ewr yearly events and threshold flows.

    Returns:
        List: list of dict with the items to be processed
    """
    items_to_process = []
    for scenario in gauge_events:
        for gauge in gauge_events[scenario]:
            for pu in gauge_events[scenario][gauge]:
                for ewr in gauge_events[scenario][gauge][pu]:
                    try:
                        item = {}
                        item["scenario"] = scenario
                        item["gauge"] = gauge
                        item["pu"] = pu
                        item["ewr"] = ewr
                        item["ewr_events"],  = gauge_events[scenario][gauge][pu][ewr]
                        items_to_process.append(item)
                    except Exception as e:
                        print(f"fail to process events for {scenario}-{pu}-{ewr}-{gauge} with error {e}")
                        continue
    return items_to_process


def count_events(yearly_events:dict)-> int:
    """count the events in a collection of years

    Args:
        yearly_events (dict): ewr yearly events dictionary of lists of lists

    Returns:
        int: count of length of all events in the collection of years
    """
    return sum([len(events) for events in yearly_events.values()])


def sum_events(yearly_events:dict)-> int:
    """sum the total event days in a collection of years

    Args:
        yearly_events (dict): ewr yearly events dictionary of lists of lists

    Returns:
        int: count of total days of all events in the collection of years
    """

    flattened_events = [list(chain(*events)) for events in yearly_events.values()]
    return len(list(chain(*flattened_events)))


def process_yearly_events(scenario:str, gauge:str, pu:str, ewr:str, ewr_events: Dict)-> pd.DataFrame:
    """process each item for the gauge and return the statistics in a DataFrame

    Args:
        scenario (str): scenario name metadata
        gauge (str): gauge name metadata
        pu (str): planning unit name metadata
        ewr (str): DataFrame to be transformed
        ewr_events (Dict): Dict with all yearly events list with date and flow/level 

    Returns:
        pd.DataFrame: DataFrame with events statistics
    """
    row_data = defaultdict(list)
    yearly_events = ewr_events
    total_events = count_events(yearly_events)
    total_event_days = sum_events(yearly_events)
    average_event_length = total_event_days/total_events if total_events else 0
    row_data['scenario'].append(scenario)
    row_data['gauge'].append(gauge)
    row_data['pu'].append(pu)
    row_data['ewrCode'].append(ewr)
    row_data['totalEvents'].append(total_events)
    row_data['totalEventDays'].append(total_event_days)
    row_data['averageEventLength'].append(average_event_length)
    
    return pd.DataFrame(row_data)

def process_ewr_events_stats(events_to_process: List[Dict])-> pd.DataFrame:
    """Manage the processing of yearly events and concatenate into a
    single dataframe with the results of all ewr calculations.

    Args:
        events_to_process (List[Dict]): List with all items to process.

    Returns:
        pd.DataFrame: Single DataFrame with all the ewr events stats results
    """
    returned_dfs = []
    for item in events_to_process:
        row_data = process_yearly_events(**item)
        returned_dfs.append(row_data)
    return pd.concat(returned_dfs, ignore_index=True)

def process_all_yearly_events(scenario:str, gauge:str, pu:str, ewr:str, ewr_events: Dict)-> pd.DataFrame():
    """process each item for the gauge and return all events. Each event is a row with a start and end date
    duration and event length

    Args:
        scenario (str): scenario name metadata
        gauge (str): gauge name metadata
        pu (str): planning unit name metadata
        ewr (str): DataFrame to be transformed
        ewr_events (Dict): Dict with all yearly events list with date and flow/level
        

    Returns:
        pd.DataFrame: DataFrame with all events of Pu-ewr-gauge combination
    """
    df_data = defaultdict(list)
    for year in ewr_events:
        for i, ev in enumerate(ewr_events[year]):
            start_date, _ = ev[0]
            end_date, _ = ev[-1]
            df_data["scenario"].append(scenario)
            df_data["gauge"].append(gauge)
            df_data["pu"].append(pu)
            df_data["ewr"].append(ewr)
            df_data["waterYear"].append(year)
            df_data["startDate"].append(start_date )
            df_data["endDate"].append(end_date)
            df_data["eventDuration"].append((end_date - start_date).days + 1)
            df_data["eventLength"].append(len(ev))          
    
    return pd.DataFrame(df_data)

def process_all_events_results(results_to_process: List[Dict])-> pd.DataFrame:
    """Manage the processing of yearly events and concatenate into a
    single dataframe with the results of all ewr calculations.

    Args:
        results_to_process (List[Dict]):List with all items to process.

    Returns:
        pd.DataFrame: Single DataFrame with all the ewr events
    """
    returned_dfs = []
    for item in results_to_process:
        try:
            df = process_all_yearly_events(**item)
            returned_dfs.append(df)
        except Exception as e:
            print(f"could not process due to {e}")
            continue
    return pd.concat(returned_dfs, ignore_index=True)

def fill_empty(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: x if x != '' else '0')
    return df

def join_ewr_parameters(cols_to_add:List, left_table:pd.DataFrame, left_on:List, 
                        selected_columns:List = None, renamed_columns:List = None,
                        parameter_sheet_path:str = None)-> pd.DataFrame:
    
    EWR_table, bad_EWRs = data_inputs.get_EWR_table(parameter_sheet_path)

    EWR_table = fill_empty(EWR_table, ['TargetFrequency','MaxInter-event'])

    columns_right_table = ['Gauge','PlanningUnitName','Code']

    columns_right_table += cols_to_add

    EWR_table = EWR_table[columns_right_table]
    
    output_table = left_table.merge(EWR_table, 
                                                  'left',
                                                  left_on=left_on, 
                                                  right_on=['Gauge','PlanningUnitName','Code'])

    if selected_columns:
        output_table = output_table[selected_columns]

    if renamed_columns:    
        output_table.columns = renamed_columns

    return output_table

def sum_0(series:pd.Series) -> int:
    '''
    Custom agg function for counting occurences of 0's in a series
    
    Args:
        series (pd.Series): pandas series of 0s and 1s

    Results:
        int: sum of 0s
    '''
    return series[series==0].count()

    # return series.value_counts()[0]


def summarise(input_dict:Dict , events:Dict, parameter_sheet_path:str = None)-> pd.DataFrame:
    """orchestrate the processing of the pu_dfs items and the gauge events and join
    in one summary DataFrame and join with EWR parameters for comparison

    Args:
        input_dict (Dict): DataFrame result by yearly with statistics for the ewr calculations.
        events (Dict): Gauge events captured by the ewr calculations.

    Returns:
        pd.DataFrame: Summary statistics for all ewr calculation for the whole period of the run
    """
    to_process = pu_dfs_to_process(input_dict)
    yearly_ewr_results = process_df_results(to_process)
    
    # aggregate by "gauge","pu","ewrCode"
    final_summary_output = (yearly_ewr_results
    .groupby(["scenario","gauge","pu","ewrCode"])
    .agg( EventYears = ("eventYears", sum),
          Frequency = ("eventYears", get_frequency),
          AchievementCount = ("numAchieved", sum),
          AchievementPerYear = ("numAchieved", 'mean'),
          EventCount = ("numEvents",sum),
          EventCountAll = ("numEventsAll",sum),
          EventsPerYear = ("numEvents",'mean'),
          EventsPerYearAll = ("numEventsAll",'mean'),
          ThresholdDays = ("totalEventDays", sum),
          InterEventExceedingCount = ("maxInterEventDaysAchieved", sum_0),
          NoDataDays =  ("missingDays" , sum),
          TotalDays = ("totalPossibleDays" , sum),
          )
    )
    # summarize gauge events
    
    events_to_process = get_events_to_process(events)
    ewr_event_stats = process_ewr_events_stats(events_to_process)
    
    # join summary with gauge events
    
    final_summary_output = final_summary_output.merge(ewr_event_stats, 
                                                      'left',
                                                      left_on=['gauge','pu','ewrCode'], 
                                                      right_on=['gauge','pu',"ewrCode"])
    # Join Ewr parameter to summary

    final_merged = join_ewr_parameters(cols_to_add=['TargetFrequency','MaxInter-event','Multigauge'],
                                left_table=final_summary_output,
                                left_on=['gauge','pu','ewrCode'],
                                selected_columns=["scenario",'gauge',
                                                    'pu', 
                                                    'ewrCode',
                                                    'Multigauge',
                                                    'EventYears',
                                                    'Frequency',
                                                    'TargetFrequency',
                                                    'AchievementCount',
                                                    'AchievementPerYear',
                                                    'EventCount',
                                                    'EventCountAll',
                                                    'EventsPerYear',
                                                    'EventsPerYearAll',
                                                    'averageEventLength',
                                                    'ThresholdDays',
                                                    'InterEventExceedingCount',
                                                    'MaxInter-event',
                                                    'NoDataDays',
                                                    'TotalDays'],
                                renamed_columns=['Scenario','Gauge', 'PlanningUnit', 'EwrCode', 'Multigauge','EventYears', 'Frequency', 'TargetFrequency',
                                    'AchievementCount', 'AchievementPerYear', 'EventCount', 'EventCountAll','EventsPerYear', 'EventsPerYearAll',
                                    'AverageEventLength', 'ThresholdDays', 'InterEventExceedingCount',
                                    'MaxInterEventYears', 'NoDataDays', 'TotalDays'],
                                    parameter_sheet_path=parameter_sheet_path)
    
    return final_merged
