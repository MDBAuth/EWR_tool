from typing import Dict, List
import csv
import os
import urllib
import re
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from . import data_inputs, evaluate_EWRs, summarise_results
#----------------------------------- Scenario testing handling functions--------------------------#

def scenario_handler(scenarios, model_format, allowance, climate, parameter_sheet:str = None):
    '''Takes in scenarios, send to functions to clean, calculate ewrs, and
    summarise results, returns a results summary as a dataframe and a 
    dictionary with yearly results.
    TODO: data profiling to check quality of model inputs prior to running
    '''
    
    # Analyse all scenarios for EWRs
    detailed_results = {}
    detailed_events = {}
    for scenario in tqdm(scenarios, position = 0, leave = True, 
                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                         desc= 'Evaluating scenarios'):
        
        if model_format == 'Bigmod - MDBA':
            data, header = unpack_model_file(scenarios[scenario], 'Dy', 'Field')
            data = build_MDBA_columns(data, header)
            df_clean = cleaner_MDBA(data)
            df_F, df_L = match_MDBA_nodes(df_clean, data_inputs.get_MDBA_codes())

        elif model_format == 'IQQM - NSW 10,000 years':
            df_unpacked = unpack_IQQM_10000yr(scenarios[scenario])
            df_F, df_L = cleaner_IQQM_10000yr(df_unpacked)

        elif model_format == 'Source - NSW (res.csv)':
            data, header = unpack_model_file(scenarios[scenario], 'Date', 'Field')
            data = build_NSW_columns(data, header)
            df_clean = cleaner_NSW(data)
            df_F, df_L = match_NSW_nodes(df_clean, data_inputs.get_NSW_codes())
            
        gauge_results = {}
        gauge_events = {}
        all_locations = df_F.columns.to_list() + df_L.columns.to_list()
        EWR_table, bad_EWRs = data_inputs.get_EWR_table(parameter_sheet)
        for gauge in all_locations:
            gauge_results[gauge], gauge_events[gauge] = evaluate_EWRs.calc_sorter(df_F, df_L, gauge, allowance, climate, EWR_table)
        detailed_results[scenario] = gauge_results
        detailed_events[scenario] = gauge_events

    summary_results = summarise_results.summarise(detailed_results, detailed_events)
    
    return detailed_results, summary_results

def unpack_IQQM_10000yr(csv_file):
    '''Ingesting scenario file locations with the NSW specific format for 10,000 year flow timeseries
    returns a dictionary of flow dataframes with their associated header data'''
    
    df = pd.read_csv(csv_file, index_col = 'Date')
    siteList = []
    for location in df.columns:
        gauge = extract_gauge_from_string(location)
        siteList.append(gauge)
    # Save over the top of the column headings with the new list containing only the gauges
    df.columns = siteList
    
    return df
    

def unpack_model_file(csv_file, main_key, header_key):
    '''Ingesting scenario file locations with bigmod format, seperates the flow data and header data
    returns a dictionary of flow dataframes with their associated header data'''
    if csv_file[-3:] != 'csv':
        raise ValueError('''Incorrect file type selected, bigmod format requires a csv file.
                         Rerun the program and try again.''')
    
    #--------functions for pulling main data-------#
    
    def mainData_url(url, line,**kwargs):
        '''Get daily data (excluding the header data); remote file upload'''
        response = urllib.request.urlopen(url)
        lines = [l.decode('utf-8') for l in response.readlines()]
        cr = csv.reader(lines)
        pos = 0
        for row in cr:
            if row[0].startswith(line):
                headerVal = pos
            pos = pos + 1
        if main_key == 'Dy':
            df = pd.read_csv(url, header=headerVal, dtype={'Dy':'int', 'Mn': 'int', 'Year': 'int'}, skiprows=range(headerVal+1, headerVal+2))
        elif main_key == 'Date':
            df = pd.read_csv(url, header=headerVal, skiprows=range(headerVal+1, headerVal+2))
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        return df, headerVal
        
    def mainData(file, line,**kwargs):
        '''Get daily data (excluding the header data); local file upload'''
        if os.stat(file).st_size == 0:
            raise ValueError("File is empty")
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file) #, delimiter=','
            line_count = 0
            for row in csv_reader:
                if row[0].startswith(line):
                    headerVal = line_count
                line_count = line_count + 1
        if main_key == 'Dy':
            df = pd.read_csv(file, header=headerVal, dtype={'Dy':'int', 'Mn': 'int', 'Year': 'int'}, skiprows=range(headerVal+1, headerVal+2))
        elif main_key == 'Date':
            df = pd.read_csv(file, header=headerVal, skiprows=range(headerVal+1, headerVal+2))
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        return df, headerVal
        
    #--------Functions for pulling header data---------#
    
    def headerData_url(url, line, endLine, **kwargs):
        '''Get header data for a remote file upload'''
        response = urllib.request.urlopen(url)
        lines = [l.decode('utf-8') for l in response.readlines()]
        cr = csv.reader(lines)
        pos = 0
        for row in cr:
            if row[0].startswith(line):
                headerVal = pos
            pos = pos + 1
        junkRows = headerVal # Junk rows because rows prior to this value will be discarded
        df = pd.read_csv(url, header=headerVal, nrows = (endLine-junkRows-1), dtype={'Site':'str', 'Measurand': 'str', 'Quality': 'str'})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        return df
        
    def headerData(file, line, endLine, **kwargs):
        '''Get header data for local file upload'''
        if os.stat(file).st_size == 0:
            raise ValueError("File is empty")
            
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if row[0].startswith(line):
                    headerVal = line_count
                    
                    # Then get column length:
                    num_cols = num_cols = list(range(0,len(row),1))
                    
                line_count = line_count + 1
            junkRows = headerVal # Junk rows because rows prior to this value will be discarded
            
        df = pd.read_csv(file, header=headerVal, usecols = num_cols, 
                         nrows = (endLine-junkRows-1), dtype={'Site':'str', 'Measurand': 'str', 'Quality': 'str'},
                        encoding = 'utf-8', sep =',')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
        return df
    
    if 'http' in csv_file:
        mainData_df, endLine = mainData_url(csv_file, main_key, sep=",")
        headerData_df = headerData_url(csv_file, header_key, endLine, sep=",")
    else:
        mainData_df, endLine = mainData(csv_file, main_key, sep=",")
        headerData_df = headerData(csv_file, header_key, endLine, sep=",")
    
    return mainData_df, headerData_df

def build_MDBA_columns(input_data, input_header):
    '''Takes in the header data file, trims it, and then renames the column headings with the full reference code
    returns a the dataframe with updated column headers'''
    # Clean dataframe
    numRows = int(input_header['Field'].iloc[1]) 
    df = input_header.drop([0,1])     
    df = df.astype(str)
    df['Site'] = df['Site'].map(lambda x: x.strip("'"))
    df['Measurand'] = df['Measurand'].map(lambda x: x.strip("'"))
    df['Quality'] = df['Quality'].map(lambda x: x.strip("'"))

    # Construct refs and save to list:
    listOfCols = []
    for i in range(0, numRows):
        colName = str(df['Site'].iloc[i] + '-' + df['Measurand'].iloc[i] + '-' + df['Quality'].iloc[i])
        listOfCols.append(colName)
    dateList = ['Dy', 'Mn', 'Year']
    listOfCols = dateList + listOfCols

    input_data.columns = listOfCols
    
    return input_data

def build_NSW_columns(input_data, input_header):
    '''Takes in the header data file, trims it, and then renames the column headings with the full reference code
    returns a the dataframe with updated column headers'''
    # Extract unique column ID's from the header:
    
    new_cols = input_header['Name'].to_list()
    new_cols = new_cols[2:]
    new_cols.insert(0, 'Date')
    input_data.columns = new_cols
    
    return input_data

def cleaner_MDBA(input_df):
    '''Ingests dataframe, removes junk columns, fixes date,
    returns formatted dataframe'''
    
    cleaned_df = input_df.rename(columns={'Mn': 'Month', 'Dy': 'Day'})
    cleaned_df['Date'] = pd.to_datetime(cleaned_df[['Year', 'Month', 'Day']], format = '%Y-%m-%d')
    cleaned_df['Date'] = cleaned_df['Date'].apply(lambda x: x.to_period(freq='D'))
    cleaned_df = cleaned_df.drop(['Day', 'Month', 'Year'], axis = 1)
    cleaned_df = cleaned_df.set_index('Date')
    
    return cleaned_df

def cleaner_NSW(input_df):
    '''Convert dates to datetime format, save this to the dataframe index'''
    try:
        input_df['Date'] = pd.to_datetime(input_df['Date'], format = '%d/%m/%Y')
    except ValueError:
        print('Attempted and failed to read in dates in format: dd/mm/yyyy, attempting to look for dates in format: yyyy-mm-dd')
        try:
            input_df['Date'] = pd.to_datetime(input_df['Date'], format = '%Y-%m-%d')
        except ValueError:
            raise ValueError('New date format detected. Cannot read in data')
        print('successfully read in data with yyyy-mm-dd formatting')
    input_df = input_df.set_index('Date')
    
    return input_df

def cleaner_IQQM_10000yr(input_df):
    '''Ingests dataframe, removes junk columns, fixes date, allocates gauges to either flow/level'''
    
    cleaned_df = input_df.copy(deep=True)
    
    try:
        date_start = datetime.strptime(cleaned_df.index[0], '%d/%m/%Y')
        date_end = datetime.strptime(cleaned_df.index[-1], '%d/%m/%Y')
    except ValueError:    
        print('Attempted and failed to read in dates in format: dd/mm/yyyy, attempting to look for dates in format: yyyy-mm-dd')
        try:
            date_start = datetime.strptime(cleaned_df.index[0], '%Y-%m-%d')
            date_end = datetime.strptime(cleaned_df.index[-1], '%Y-%m-%d')
        except ValueError:
            raise ValueError('New date format detected. Cannot read in data')
        print('successfully read in data with yyyy-mm-dd formatting')
    
    date_range = pd.period_range(date_start, date_end, freq = 'D')
    cleaned_df['Date'] = date_range
    cleaned_df = cleaned_df.set_index('Date')
    
    # Split gauges into flow and level, allocate to respective dataframe
    flow_gauges = data_inputs.get_gauges('flow gauges')
    level_gauges = data_inputs.get_gauges('level gauges')
    df_flow = pd.DataFrame(index = cleaned_df.index)
    df_level = pd.DataFrame(index = cleaned_df.index)
    for gauge in cleaned_df.columns:
        if gauge in flow_gauges:
            df_flow[gauge] = cleaned_df[gauge].copy(deep=True)
        if gauge in level_gauges:
            df_level[gauge] = cleaned_df[gauge].copy(deep=True)
    return df_flow, df_level

def extract_gauge_from_string(input_string):
    '''Takes in a string, pulls out the gauge number from this string'''
    found = re.findall(r'\d+\w', input_string)
    if found:
        for i in found:
            if len(i) >= 6:
                gauge = i
                return gauge
    else:
        return None

def match_MDBA_nodes(input_df, model_metadata):
    '''Checks if the source file columns have EWRs available, returns a flow and level dataframe with only 
    the columns with EWRs available. Renames columns to gauges'''

    flow_gauges = data_inputs.get_gauges('flow gauges')
    level_gauges = data_inputs.get_gauges('level gauges')
    measurands = ['1', '35']
    df_flow = pd.DataFrame(index = input_df.index)
    df_level = pd.DataFrame(index = input_df.index)
    for col in input_df.columns:
        col_clean = col.replace(' ', '')
        site = col_clean.split('-')[0]
        measure = col_clean.split('-')[1]
        if ((measure in measurands) and (model_metadata['SITEID'] == site).any()):
            subset = model_metadata.query("SITEID==@site")
            gauge = subset["AWRC"].iloc[0]
            if gauge in flow_gauges and measure == '1':
                df_flow[gauge] = input_df[col]
            if gauge in level_gauges and measure == '35':
                df_level[gauge] = input_df[col]

    return df_flow, df_level

def match_NSW_nodes(input_df, model_metadata):
    '''Checks if the source file columns have EWRs available, returns a flow and level dataframe with only 
    the columns with EWRs available. Renames columns to gauges'''
    flow_gauges = data_inputs.get_gauges('flow gauges')
    level_gauges = data_inputs.get_gauges('level gauges')
    df_flow = pd.DataFrame(index = input_df.index)
    df_level = pd.DataFrame(index = input_df.index)
    for col in input_df.columns:
        if not (model_metadata[model_metadata['SITEID'].str.contains(col, na=False)]).empty:
            subset = model_metadata.query("SITEID==@col")
            gauge = subset["AWRC"].iloc[0]
            if gauge in flow_gauges:
                df_flow[gauge] = input_df[col]
            if gauge in level_gauges:
                df_level[gauge] = input_df[col]

    return df_flow, df_level


class ScenarioHandler:
    
    def __init__(self, scenario_files: List[str], model_format:str, allowance:Dict, climate:str, parameter_sheet:str = None):
        self.scenario_files = scenario_files
        self.model_format = model_format
        self.allowance = allowance
        self.climate = climate
        self.yearly_events = None
        self.pu_ewr_statistics = None
        self.summary_results = None
        self.parameter_sheet = parameter_sheet

    def _get_file_names(self, loaded_files):

        file_locations = {}
        for file in loaded_files:
            full_name = file.split('/')
            name_exclude_extension = full_name[-1].split('.csv')[0]
            file_locations[str(name_exclude_extension)] = file
            
        return file_locations
        
    def process_scenarios(self):

        scenarios = self._get_file_names(self.scenario_files)

        # Analyse all scenarios for EWRs
        detailed_results = {}
        detailed_events = {}
        for scenario in tqdm(scenarios, position = 0, leave = True, 
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                            desc= 'Evaluating scenarios'):
        
            if self.model_format == 'Bigmod - MDBA':
                data, header = unpack_model_file(scenarios[scenario], 'Dy', 'Field')
                data = build_MDBA_columns(data, header)
                df_clean = cleaner_MDBA(data)
                df_F, df_L = match_MDBA_nodes(df_clean, data_inputs.get_MDBA_codes())

            elif self.model_format == 'IQQM - NSW 10,000 years':
                df_unpacked = unpack_IQQM_10000yr(scenarios[scenario])
                df_F, df_L = cleaner_IQQM_10000yr(df_unpacked)

            elif self.model_format == 'Source - NSW (res.csv)':
                data, header = unpack_model_file(scenarios[scenario], 'Date', 'Field')
                data = build_NSW_columns(data, header)
                df_clean = cleaner_NSW(data)
                df_F, df_L = match_NSW_nodes(df_clean, data_inputs.get_NSW_codes())
            
            gauge_results = {}
            gauge_events = {}
            all_locations = df_F.columns.to_list() + df_L.columns.to_list()
            EWR_table, bad_EWRs = data_inputs.get_EWR_table(self.parameter_sheet)
            for gauge in all_locations:
                gauge_results[gauge], gauge_events[gauge] = evaluate_EWRs.calc_sorter(df_F, df_L, gauge, self.allowance, self.climate, EWR_table)
            detailed_results[scenario] = gauge_results
            detailed_events[scenario] = gauge_events

            self.pu_ewr_statistics = detailed_results
            self.yearly_events = detailed_events


    def get_all_events(self)-> pd.DataFrame:

        if not self.yearly_events:
            self.process_scenarios()
        
        events_to_process = summarise_results.get_events_to_process(self.yearly_events)
        all_events = summarise_results.process_all_events_results(events_to_process)

        all_events = summarise_results.join_ewr_parameters(cols_to_add=['multigauge'],
                        left_table=all_events,
                        left_on=['gauge','pu','ewr'],
                        selected_columns= ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                'eventDuration', 'eventLength', 
                                'multigauge'],
                        parameter_sheet_path=self.parameter_sheet)

        return all_events

    def get_yearly_ewr_results(self)-> pd.DataFrame:

        if not self.pu_ewr_statistics:
            self.process_scenarios()

        to_process = summarise_results.pu_dfs_to_process(self.pu_ewr_statistics)
        yearly_ewr_results = summarise_results.process_df_results(to_process)

        yearly_ewr_results = summarise_results.join_ewr_parameters(cols_to_add=['multigauge'],
                                left_table=yearly_ewr_results,
                                left_on=['gauge','pu','ewrCode'],
                                selected_columns= ['Year', 'eventYears', 'numAchieved', 'numEvents', 'numEventsAll', 'maxInterEventDays',
                                            'maxInterEventDaysAchieved', 'eventLength', 'eventLengthAchieved', 'totalEventDays', 'totalEventDaysAchieved',
                                            'maxEventDays', 'maxRollingEvents', 'maxRollingAchievement',
                                            'daysBetweenEvents', 'missingDays', 'totalPossibleDays', 'ewrCode',
                                            'scenario', 'gauge', 'pu', 'multigauge'],
                                parameter_sheet_path=self.parameter_sheet)
                                
        return yearly_ewr_results

    def get_ewr_results(self) -> pd.DataFrame:
        
        if not self.pu_ewr_statistics:
            self.process_scenarios()

        return summarise_results.summarise(self.pu_ewr_statistics , self.yearly_events)