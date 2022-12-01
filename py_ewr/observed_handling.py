from datetime import timedelta, date, datetime
from typing import Dict, List
import logging

import pandas as pd
from tqdm import tqdm

from . import data_inputs, evaluate_EWRs, summarise_results
from mdba_gauge_getter import gauge_getter as gg

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def categorise_gauges(gauges: list) -> tuple:
    '''Seperate gauges into level, flow, or both
    
    Args:
        gauges (list): List of user defined gauges
    
    Results:
        tuple[list, list]: A list of flow gauges; A list of water level gauges
    
    '''
    _level_gauges, weirpool_gauges = data_inputs.get_level_gauges()
    multi_gauges = data_inputs.get_multi_gauges('gauges')
    simultaneous_gauges = data_inputs.get_simultaneous_gauges('gauges')
    
    level_gauges = []
    flow_gauges = []
    stage_gauges = []

    # Loop through once to get the special gauges:
    for gauge in gauges:
        if gauge in multi_gauges.keys():
            flow_gauges.append(gauge)
            flow_gauges.append(multi_gauges[gauge])
        if gauge in simultaneous_gauges:
            flow_gauges.append(gauge)
            flow_gauges.append(simultaneous_gauges[gauge])
        if gauge in _level_gauges:
            level_gauges.append(gauge)
        if gauge in weirpool_gauges.keys(): # need level and flow gauges
            flow_gauges.append(gauge)
            level_gauges.append(weirpool_gauges[gauge])

            if '414209' in level_gauges:
                # 414209 returns 100.00 instead of 130.00
                level_gauges.remove('414209')
                stage_gauges.append('414209')

    # Then loop through again and allocate remaining gauges to the flow category
    for gauge in gauges:
        if ((gauge not in level_gauges) and (gauge not in stage_gauges) and (gauge not in flow_gauges)):
            # Otherwise, assume its a flow gauge and add
            flow_gauges.append(gauge)

    unique_flow_gauges = list(set(flow_gauges))
    unique_level_gauges = list(set(level_gauges))

    return unique_flow_gauges, unique_level_gauges, stage_gauges

def remove_data_with_bad_QC(input_dataframe: pd.DataFrame, qc_codes: list) -> pd.DataFrame:
    '''Takes in a dataframe of flow and a list of bad qc codes, removes the poor quality data from 
    the timeseries, returns this dataframe
    
    Args:
        input_dataframe (pd.DataFrame): flow/water level dataframe
        qc_codes (list): list of quality codes to filter out
    Results:
        pd.DataFrame: flow/water level dataframe with "None" assigned to the poor quality data days
    
    '''
    for qc in qc_codes:
        input_dataframe.loc[input_dataframe.QUALITYCODE == qc, 'VALUE'] = None
        
    return input_dataframe

def one_gauge_per_column(input_dataframe: pd.DataFrame, gauge_iter: str) -> pd.DataFrame:
    '''Takes in a dataframe and the name of a gauge, extracts this one location to a new dataframe, 
    cleans this and returns the dataframe with only the selected gauge data
    
    Args:
        input_dataframe (pd.DataFrame): Raw dataframe returned from water portal API
        gauge_iter (str): unique gauge ID
    Returns:
        pd.DataFrame: A dataframe daily flow/water level data matching the gauge_iter
    
    '''
    
    is_in = input_dataframe['SITEID']== gauge_iter
    single_df = input_dataframe[is_in]
    single_df = single_df.drop(['DATASOURCEID','SITEID','SUBJECTID','QUALITYCODE','DATETIME'],
                               axis = 1)
    single_df = single_df.set_index('Date')
    single_df = single_df.rename(columns={'VALUE': str(gauge_iter)})
    
    return single_df

def observed_cleaner(input_df: pd.DataFrame, dates: dict) -> pd.DataFrame:
    '''Takes in raw dataframe consolidated from state websites, removes poor quality data.
    returns a dataframe with a date index and one flow column per gauge location.
    
    Args:
        input_df (pd.DataFrame): Raw dataframe returned from water portal API
        dates (dict): Dictionary with the start and end dates from user request
    Results:
        pd.DataFrame: Daily flow/water level, gauges as the column ID's
    
    '''
    
    start_date = datetime(dates['start_date'].year, dates['start_date'].month, dates['start_date'].day)
    end_date = datetime(dates['end_date'].year, dates['end_date'].month, dates['end_date'].day)
    
    df_index = pd.date_range(start=start_date,end=end_date - timedelta(days=1)).to_period()
    gauge_data_df = pd.DataFrame()
    gauge_data_df['Date'] = df_index
    gauge_data_df = gauge_data_df.set_index('Date')

    input_df["VALUE"] = pd.to_numeric(input_df["VALUE"])#, downcast="float")
    
    
    input_df['Date'] = pd.to_datetime(input_df['DATETIME'], format = '%Y-%m-%d')
    input_df['Date'] = input_df['Date'].apply(lambda x: x.to_period(freq='D'))

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
    
    def __init__(self, gauges:List, dates:Dict , allowance:Dict, climate:str, parameter_sheet:str = None):
        self.gauges = gauges
        self.dates = dates
        self.allowance = allowance
        self.climate = climate
        self.yearly_events = None
        self.pu_ewr_statistics = None
        self.summary_results = None
        self.parameter_sheet = parameter_sheet
        self.flow_data = None
        self.level_data = None

    def process_gauges(self):
        '''ingests a list of gauges and user defined parameters
        pulls gauge data using relevant states API, calculates and analyses EWRs
        returns dictionary of raw data results and result summary
        '''
        
        # Classify gauges:
        flow_gauges, level_gauges, stage_gauges = categorise_gauges(self.gauges)
        # Call state API for flow and level gauge data, then combine to single dataframe

        log.info(f'Including gauges: {flow_gauges} {level_gauges} {stage_gauges}')
        
        flows = gg.gauge_pull(flow_gauges, start_time_user = self.dates['start_date'], end_time_user = self.dates['end_date'], var = 'F')
        levels = gg.gauge_pull(level_gauges, start_time_user = self.dates['start_date'], end_time_user = self.dates['end_date'], var = 'LL')
        stage = gg.gauge_pull(stage_gauges, start_time_user=self.dates['start_date'],
                               end_time_user=self.dates['end_date'], var='L')
        # Clean observed data:
        df_F = observed_cleaner(flows, self.dates)
        df_L = observed_cleaner(levels, self.dates)
        df_S = observed_cleaner(stage, self.dates)
        # Append stage values to level df
        df_L = pd.concat([df_L, df_S], axis=1)
        # Calculate EWRs
        detailed_results = {}
        gauge_results = {}
        gauge_events = {}
        detailed_events = {}
        all_locations = df_F.columns.to_list() + df_L.columns.to_list()
        EWR_table, bad_EWRs = data_inputs.get_EWR_table(self.parameter_sheet)
        for gauge in all_locations:
            gauge_results[gauge], gauge_events[gauge] = evaluate_EWRs.calc_sorter(df_F, df_L, gauge, self.allowance, self.climate, EWR_table)
            
        detailed_results['observed'] = gauge_results
        detailed_events['observed'] = gauge_events
        
        self.pu_ewr_statistics = detailed_results
        self.yearly_events = detailed_events

        self.flow_data = df_F
        self.level_data = df_L


    def get_all_events(self)-> pd.DataFrame:

        if not self.yearly_events:
            self.process_gauges()
        
        events_to_process = summarise_results.get_events_to_process(self.yearly_events)
        all_events = summarise_results.process_all_events_results(events_to_process)

        all_events = summarise_results.join_ewr_parameters(cols_to_add=['Multigauge'],
                                left_table=all_events,
                                left_on=['gauge','pu','ewr'],
                                selected_columns= ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                        'eventDuration', 'eventLength', 
                                        'Multigauge'],
                                parameter_sheet_path=self.parameter_sheet)

        all_events = summarise_results.filter_duplicate_start_dates(all_events)

        return all_events

    def get_all_interEvents(self)-> pd.DataFrame:
        
        if not self.yearly_events:
            self.process_gauges()
        
        events_to_process = summarise_results.get_events_to_process(self.yearly_events)
        all_events_temp = summarise_results.process_all_events_results(events_to_process)

        all_events_temp = summarise_results.join_ewr_parameters(cols_to_add=['Multigauge'],
                        left_table=all_events_temp,
                        left_on=['gauge','pu','ewr'],
                        selected_columns= ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                'eventDuration', 'eventLength', 
                                'Multigauge'],
                        parameter_sheet_path=self.parameter_sheet)
                    
        all_events_temp = summarise_results.filter_duplicate_start_dates(all_events_temp)

        # Get start and end date of the timeseries.
        date0 = self.flow_data.index[0]
        date1 = self.flow_data.index[-1]
        start_date = date(date0.year, date0.month, date0.day)
        end_date = date(date1.year, date1.month, date1.day)
        
        all_interEvents = summarise_results.events_to_interevents(start_date, end_date, all_events_temp)

        return all_interEvents

    def get_all_successful_events(self)-> pd.DataFrame:

        if not self.yearly_events:
            self.process_gauges()
        
        events_to_process = summarise_results.get_events_to_process(self.yearly_events)
        all_events_temp1 = summarise_results.process_all_events_results(events_to_process)

        all_events_temp1 = summarise_results.join_ewr_parameters(cols_to_add=['Multigauge'],
                        left_table=all_events_temp1,
                        left_on=['gauge','pu','ewr'],
                        selected_columns= ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                'eventDuration', 'eventLength', 
                                'Multigauge'],
                        parameter_sheet_path=self.parameter_sheet)

        all_events_temp1 = summarise_results.filter_duplicate_start_dates(all_events_temp1)

        all_successfulEvents = summarise_results.filter_successful_events(all_events_temp1) 

        return all_successfulEvents

    def get_all_successful_interEvents(self)-> pd.DataFrame:

        if not self.yearly_events:
            self.process_gauges()
        
        events_to_process = summarise_results.get_events_to_process(self.yearly_events)
        all_events_temp2 = summarise_results.process_all_events_results(events_to_process)

        all_events_temp2 = summarise_results.join_ewr_parameters(cols_to_add=['Multigauge'],
                        left_table=all_events_temp2,
                        left_on=['gauge','pu','ewr'],
                        selected_columns= ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                'eventDuration', 'eventLength', 
                                'Multigauge'],
                        parameter_sheet_path=self.parameter_sheet)

        all_events_temp2 = summarise_results.filter_duplicate_start_dates(all_events_temp2)

        # Part 1 - Get only the successful events:
        all_successfulEvents = summarise_results.filter_successful_events(all_events_temp2) 

        # Part 2 - Now we have a dataframe of only successful events, pull down the interevent periods
        # Get start and end date of the timeseries.
        date0 = self.flow_data.index[0]
        date1 = self.flow_data.index[-1]

        start_date = date(date0.year, date0.month, date0.day)
        end_date = date(date1.year, date1.month, date1.day)

        all_successful_interEvents = summarise_results.events_to_interevents(start_date, end_date, all_successfulEvents)

        return all_successful_interEvents

    def get_yearly_ewr_results(self)-> pd.DataFrame:

        if not self.pu_ewr_statistics:
            self.process_gauges()

        to_process = summarise_results.pu_dfs_to_process(self.pu_ewr_statistics)
        yearly_ewr_results = summarise_results.process_df_results(to_process)

        yearly_ewr_results = summarise_results.join_ewr_parameters(cols_to_add=['Multigauge'],
                                left_table=yearly_ewr_results,
                                left_on=['gauge','pu','ewrCode'],
                                selected_columns= ['Year', 'eventYears', 'numAchieved', 'numEvents', 'numEventsAll',# 'maxInterEventDays', 'maxInterEventDaysAchieved', 
                                            'eventLength', 'eventLengthAchieved', 'totalEventDays', 'totalEventDaysAchieved',
                                            'maxEventDays', 'maxRollingEvents', 'maxRollingAchievement',
                                            'missingDays', 'totalPossibleDays', 'ewrCode',
                                            'scenario', 'gauge', 'pu', 'Multigauge'],
                                parameter_sheet_path=self.parameter_sheet)

        # Setting up the dictionary of yearly rolling maximum interevent periods:
        events_to_process = summarise_results.get_events_to_process(self.yearly_events)
        all_events_temp = summarise_results.process_all_events_results(events_to_process)

        all_events_temp = summarise_results.join_ewr_parameters(cols_to_add=['Multigauge'],
                                                                left_table=all_events_temp,
                                                                left_on=['gauge', 'pu', 'ewr'],
                                                                selected_columns=['scenario', 'gauge', 'pu', 'ewr',
                                                                                  'waterYear', 'startDate', 'endDate',
                                                                                  'eventDuration', 'eventLength',
                                                                                  'Multigauge'],
                                                                parameter_sheet_path=self.parameter_sheet)
        
        all_events_temp = summarise_results.filter_duplicate_start_dates(all_events_temp)

        # Filter out the unsuccessful events:
        all_successfulEvents = summarise_results.filter_successful_events(all_events_temp)
        
        # Get start and end date of the timeseries.
        date0 = self.flow_data.index[0]
        date1 = self.flow_data.index[-1]
        start_date = date(date0.year, date0.month, date0.day)
        end_date = date(date1.year, date1.month, date1.day)
        df = summarise_results.events_to_interevents(start_date, end_date, all_successfulEvents)
        rolling_max_interevents_dict = summarise_results.get_rolling_max_interEvents(df, self.dates['start_date'], self.dates['end_date'],
                                                                                     yearly_ewr_results)

        # Add the rolling max interevents to the yearly dataframe:
        yearly_ewr_results = summarise_results.add_interevent_to_yearly_results(yearly_ewr_results,
                                                                                rolling_max_interevents_dict)
        # Drop temporary ID column

        # yearly_ewr_results.drop('ID', axis=1, inplace=True)
        return yearly_ewr_results

    def get_ewr_results(self) -> pd.DataFrame:
        
        if not self.pu_ewr_statistics:
            self.process_gauges()

        return summarise_results.summarise(self.pu_ewr_statistics , self.yearly_events)