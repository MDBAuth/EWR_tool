from datetime import timedelta
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from . import data_inputs, evaluate_EWRs, summarise_results
from mdba_gauge_getter import gauge_getter as gg

def categorise_gauges(gauges):
    '''Seperate gauges into level, flow, or both'''
    menindee_gauges, weirpool_gauges = data_inputs.get_level_gauges()
    multi_gauges = data_inputs.get_multi_gauges('gauges')
    simultaneous_gauges = data_inputs.get_simultaneous_gauges('gauges')
    
    level_gauges = []
    flow_gauges = []
    # Loop through once to get the special gauges:
    for gauge in gauges:
        if gauge in multi_gauges.keys():
            flow_gauges.append(gauge)
            flow_gauges.append(multi_gauges[gauge])
        if gauge in simultaneous_gauges:
            flow_gauges.append(gauge)
            flow_gauges.append(simultaneous_gauges[gauge])
        if gauge in menindee_gauges:
            level_gauges.append(gauge)
        if gauge in weirpool_gauges.keys(): # need level and flow gauges
            flow_gauges.append(gauge)
            level_gauges.append(weirpool_gauges[gauge])
    # Then loop through again and allocate remaining gauges to the flow category
    for gauge in gauges:
        if ((gauge not in level_gauges) and (gauge not in flow_gauges)):
            # Otherwise, assume its a flow gauge and add
            flow_gauges.append(gauge)
        
        unique_flow_gauges = list(set(flow_gauges))
        unique_level_gauges = list(set(level_gauges))
            
    return unique_flow_gauges, unique_level_gauges

def observed_handler(gauges, dates, allowance, climate):
    '''ingests a list of gauges and user defined parameters
    pulls gauge data using relevant states API, calcualtes and analyses EWRs
    returns dictionary of raw data results and result summary
    '''
    
    # Classify gauges:
    flow_gauges, level_gauges = categorise_gauges(gauges)
    # Call state API for flow and level gauge data, then combine to single dataframe
    
    flows = gg.gauge_pull(flow_gauges, start_time_user = dates['start_date'], end_time_user = dates['end_date'], var = 'F')
    levels = gg.gauge_pull(level_gauges, start_time_user = dates['start_date'], end_time_user = dates['end_date'], var = 'L')
    # Clean observed data:
    df_F = observed_cleaner(flows, dates)
    df_L = observed_cleaner(levels, dates)
    # Calculate EWRs
    detailed_results = {}
    gauge_results = {}
    gauge_events = {}
    detailed_events = {}
    all_locations = df_F.columns.to_list() + df_L.columns.to_list()
    for gauge in all_locations:
        gauge_results[gauge], gauge_events[gauge] = evaluate_EWRs.calc_sorter(df_F, df_L, gauge, allowance, climate)
        
    detailed_results['observed'] = gauge_results
    detailed_events['observed'] = gauge_events
    # Summarise the results:
    summary_results = summarise_results.summarise(detailed_results, detailed_events)

    return detailed_results, summary_results


def remove_data_with_bad_QC(input_dataframe, qc_codes):
    '''Takes in a dataframe of flow and a list of bad qc codes, removes the poor quality data from 
    the timeseries, returns this dataframe'''
    for qc in qc_codes:
        input_dataframe.loc[input_dataframe.QUALITYCODE == qc, 'VALUE'] = None
        
    return input_dataframe

def one_gauge_per_column(input_dataframe, gauge_iter):
    '''Takes in a dataframe and the name of a gauge, extracts this one location to a new dataframe, 
    cleans this and returns the dataframe with only the selected gauge data'''
    
    is_in = input_dataframe['SITEID']== gauge_iter
    single_df = input_dataframe[is_in]
    single_df = single_df.drop(['DATASOURCEID','SITEID','SUBJECTID','QUALITYCODE','DATETIME'],
                               axis = 1)
    single_df = single_df.set_index('Date')
    single_df = single_df.rename(columns={'VALUE': str(gauge_iter)})
    
    return single_df

def observed_cleaner(input_df, dates):
    '''Takes in raw dataframe consolidated from state websites, removes poor quality data.
    returns a dataframe with a date index and one flow column per gauge location.'''
    
    
    df_index = pd.date_range(dates['start_date'],dates['end_date']-timedelta(days=1),freq='d')
    gauge_data_df = pd.DataFrame()
    gauge_data_df['Date'] = df_index
    gauge_data_df['Date'] = pd.to_datetime(gauge_data_df['Date'], format = '%Y-%m-%d')
    gauge_data_df = gauge_data_df.set_index('Date')

    input_df["VALUE"] = pd.to_numeric(input_df["VALUE"])#, downcast="float")
    
    
    input_df['Date'] = pd.to_datetime(input_df['DATETIME'], format = '%Y-%m-%d')
    # Check with states for more codes:
    bad_data_codes = data_inputs.get_bad_QA_codes()
    input_df = remove_data_with_bad_QC(input_df, bad_data_codes)
    
    site_list = set(input_df['SITEID'])
    
    for gauge in site_list:
        # Seperate out to one gauge per column and add this to the gauge_data_df made above:
        single_gauge_df = one_gauge_per_column(input_df, gauge)
        gauge_data_df = pd.merge(gauge_data_df, single_gauge_df, left_index=True, right_index=True, how="outer")

    # Drop the non unique values:
    gauge_data_df = gauge_data_df[~gauge_data_df.index.duplicated(keep='first')]
    return gauge_data_df


class ObservedHandler:
    
    def __init__(self, gauges:List, dates:Dict , allowance:Dict, climate:str):
        self.gauges = gauges
        self.dates = dates
        self.allowance = allowance
        self.climate = climate
        self.EWR_TABLE = None
        self.yearly_events = None
        self.pu_ewr_statistics = None
        self.summary_results = None

    def process_gauges(self):
        '''ingests a list of gauges and user defined parameters
        pulls gauge data using relevant states API, calculates and analyses EWRs
        returns dictionary of raw data results and result summary
        '''
        
        # Classify gauges:
        flow_gauges, level_gauges = categorise_gauges(self.gauges)
        # Call state API for flow and level gauge data, then combine to single dataframe
        
        flows = gg.gauge_pull(flow_gauges, start_time_user = self.dates['start_date'], end_time_user = self.dates['end_date'], var = 'F')
        levels = gg.gauge_pull(level_gauges, start_time_user = self.dates['start_date'], end_time_user = self.dates['end_date'], var = 'L')
        # Clean observed data:
        df_F = observed_cleaner(flows, self.dates)
        df_L = observed_cleaner(levels, self.dates)
        # Calculate EWRs
        detailed_results = {}
        gauge_results = {}
        gauge_events = {}
        detailed_events = {}
        all_locations = df_F.columns.to_list() + df_L.columns.to_list()
        for gauge in all_locations:
            gauge_results[gauge], gauge_events[gauge] = evaluate_EWRs.calc_sorter(df_F, df_L, gauge, self.allowance, self.climate)
            
        detailed_results['observed'] = gauge_results
        detailed_events['observed'] = gauge_events
        
        self.pu_ewr_statistics = detailed_results
        self.yearly_events = detailed_events


    def get_all_events(self)-> pd.DataFrame:

        if not self.yearly_events:
            self.process_gauges()
        
        events_to_process = summarise_results.get_events_to_process(self.yearly_events)
        all_events = summarise_results.process_all_events_results(events_to_process)
        return all_events

    def get_yearly_ewr_results(self)-> pd.DataFrame:

        if not self.pu_ewr_statistics:
            self.process_gauges()

        to_process = summarise_results.pu_dfs_to_process(self.pu_ewr_statistics)
        yearly_ewr_results = summarise_results.process_df_results(to_process)
        return yearly_ewr_results

    def get_ewr_results(self) -> pd.DataFrame:
        
        if not self.pu_ewr_statistics:
            self.process_gauges()

        return summarise_results.summarise(self.pu_ewr_statistics , self.yearly_events)