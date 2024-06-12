from typing import Dict, List
import csv
import os
import urllib
import re
from datetime import datetime, date
import logging

import pandas as pd
from tqdm import tqdm
 
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


from . import data_inputs, evaluate_EWRs, summarise_results
#----------------------------------- Scenario testing handling functions--------------------------#

def unpack_model_file(csv_file: str, main_key: str, header_key: str) -> tuple:
    '''Ingesting scenario file locations of model files with all formats (excluding standard timeseries format), seperates the flow data and header data
    returns a dictionary of flow dataframes with their associated header data
    
    Args:
        csv_file (str): location of model file
        main_key (str): unique identifier for the start of the flow data (dependent on model format type being uploaded)
        header_key (str): unique identifier for the start of the header data (dependent on model format type being uploaded)
    
    Results:
        tuple[pd.DataFrame, pd.DataFrame]: flow dataframe; header dataframe
    
    '''
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
                break
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
                    break
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
                break
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
                    break
                    
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

def build_MDBA_columns(input_data: pd.DataFrame, input_header: pd.DataFrame) -> pd.DataFrame:
    '''Takes in the header data file, trims it, and then renames the column headings with the full reference code
    returns a the dataframe with updated column headers (MDBA model formats).
    
    Args:
        input_data (pd.DataFrame): flow/water level dataframe
        input_header (pd.DataFrame): header dataframe

    Results:
        pd.DataFrame: the flow dataframe with updated header data
    
    '''
    # Clean dataframe
    numRows = int(input_header['Field'].iloc[1]) 
    df = input_header.drop([0,1])     
    df = df.astype(str)
    # Remove rogue quotes, spaces, and apostrophes
    df['Site'] = df['Site'].map(lambda x: x.replace("'", ""))
    df['Site'] = df['Site'].map(lambda x: x.replace('"', ''))
    df['Site'] = df['Site'].map(lambda x: x.replace(" ", ""))

    df['Measurand'] = df['Measurand'].map(lambda x: x.replace("'", ""))
    df['Measurand'] = df['Measurand'].map(lambda x: x.replace('"', ''))
    df['Measurand'] = df['Measurand'].map(lambda x: x.replace(" ", ""))

    df['Quality'] = df['Quality'].map(lambda x: x.replace("'", ""))
    df['Quality'] = df['Quality'].map(lambda x: x.replace('"', ''))
    df['Quality'] = df['Quality'].map(lambda x: x.replace(" ", ""))

    # Construct refs and save to list:
    listOfCols = []
    for i in range(0, numRows):
        colName = str(df['Site'].iloc[i] + '-' + df['Measurand'].iloc[i] + '-' + df['Quality'].iloc[i])
        listOfCols.append(colName)
    dateList = ['Dy', 'Mn', 'Year']
    listOfCols = dateList + listOfCols

    input_data.columns = listOfCols
    
    return input_data

def build_NSW_columns(input_data: pd.DataFrame, input_header: pd.DataFrame) -> pd.DataFrame:
    '''Takes in the header data file, trims it, and then renames the column headings with the full reference code
    returns a the dataframe with updated column headers (NSW res.csv model format).
    
    Args:
        input_data (pd.DataFrame): flow/water level dataframe
        input_header (pd.DataFrame): header dataframe

    Results:
        pd.DataFrame: the flow/water level dataframe with updated header data

    '''
    # Extract unique column ID's from the header:
    
    new_cols = input_header['Name'].to_list()
    new_cols = new_cols[2:]
    new_cols.insert(0, 'Date')
    input_data.columns = new_cols
    
    return input_data

def cleaner_MDBA(input_df: pd.DataFrame) -> pd.DataFrame:
    '''Ingests dataframe, removes junk columns, fixes date,
    returns formatted dataframe
    
    Args:
        input_df (pd.DataFrame): flow/water level dataframe
    Results:
        pd.DataFrame: Cleaned flow/water level dataframe
    
    '''
    
    cleaned_df = input_df.rename(columns={'Mn': 'Month', 'Dy': 'Day'})
    cleaned_df['Date'] = pd.to_datetime(cleaned_df[['Year', 'Month', 'Day']], format = '%Y-%m-%d')
    cleaned_df['Date'] = cleaned_df['Date'].apply(lambda x: x.to_period(freq='D'))
    cleaned_df = cleaned_df.drop(['Day', 'Month', 'Year'], axis = 1)
    cleaned_df = cleaned_df.set_index('Date')
    
    return cleaned_df

def cleaner_NSW(input_df: pd.DataFrame) -> pd.DataFrame:
    '''Convert dates to datetime format, save this to the dataframe index
    
    Args:
        input_df (pd.DataFrame): flow/water level dataframe
    Results:
        pd.DataFrame: Cleaned flow/water level dataframe

    '''
    
    cleaned_df = input_df.copy(deep=True)
    
    try:
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'], format = '%d/%m/%Y')
        cleaned_df['Date'] = cleaned_df['Date'].apply(lambda x: x.to_period(freq='D'))
    except ValueError:
        log.info('''Attempted and failed to read in dates in format: dd/mm/yyyy, 
        attempting to look for dates in format: yyyy-mm-dd''')
        try:
            cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'], format = '%Y-%m-%d')
            cleaned_df['Date'] = cleaned_df['Date'].apply(lambda x: x.to_period(freq='D'))
        except ValueError:
            raise ValueError('New date format detected. Cannot read in data')
        log.info('successfully read in data with yyyy-mm-dd formatting')
    cleaned_df = cleaned_df.set_index('Date')
    
    return cleaned_df

def cleaner_standard_timeseries(input_df: pd.DataFrame, ewr_table_path: str = None) -> pd.DataFrame:
    '''Ingests dataframe, removes junk columns, fixes date, allocates gauges to either flow/level
    
    Args:
        input_df (pd.DataFrame): flow/water level dataframe

    Results:
        tuple[pd.DataFrame, pd.DataFrame]: Cleaned flow dataframe; cleaned water level dataframe

    '''

    cleaned_df = input_df.copy(deep=True)
    try:
        cleaned_df.index = pd.to_datetime(cleaned_df.index, format = '%d/%m/%Y')
    except ValueError:
        log.info('''Attempted and failed to read in dates in format: dd/mm/yyyy, attempting
        to look for dates in format: yyyy-mm-dd''')
        try:
            cleaned_df.index = pd.to_datetime(cleaned_df.index, format = '%Y-%m-%d')
        except ValueError:
            raise ValueError('''New date format detected. Cannot read in data''')
        log.info('''Successfully read in data with yyyy-mm-dd formatting''')

    # If there are missing dates, add in the dates and fill with NaN values
    dates = pd.date_range(start = cleaned_df.index[0], end=cleaned_df.index[-1])
    cleaned_df = cleaned_df.reindex(dates)

    # TODO: Optional: add in gap filling code if user selects preference for gap filling

    df_flow = pd.DataFrame(index = cleaned_df.index)
    df_level = pd.DataFrame(index = cleaned_df.index)
    df_flow.index.name = 'Date'
    df_level.index.name = 'Date'

    for gauge in cleaned_df.columns:
        gauge_only = extract_gauge_from_string(gauge)
        if 'flow' in gauge:
            df_flow[gauge_only] = cleaned_df[gauge].copy(deep=True)
        if 'level' in gauge:
            df_level[gauge_only] = cleaned_df[gauge].copy(deep=True)
        if not gauge_only:
            log.info('Could not identify gauge in column name:', gauge, ', skipping analysis of data in this column.')
    return df_flow, df_level


def cleaner_ten_thousand_year(input_df: pd.DataFrame, ewr_table_path: str = None) -> pd.DataFrame:
    '''Ingests dataframe, removes junk columns, fixes date, allocates gauges to either flow/level
    
    Args:
        input_df (pd.DataFrame): flow/water level dataframe

    Results:
        tuple[pd.DataFrame, pd.DataFrame]: Cleaned flow dataframe; cleaned water level dataframe

    '''
    
    cleaned_df = input_df.copy(deep=True)
    
    try:
        date_start = datetime.strptime(cleaned_df.index[0], '%d/%m/%Y')
        date_end = datetime.strptime(cleaned_df.index[-1], '%d/%m/%Y')
    except ValueError:    
        log.info('Attempted and failed to read in dates in format: dd/mm/yyyy, attempting to look for dates in format: yyyy-mm-dd')
        try:
            date_start = datetime.strptime(cleaned_df.index[0], '%Y-%m-%d')
            date_end = datetime.strptime(cleaned_df.index[-1], '%Y-%m-%d')
        except ValueError:
            raise ValueError('New date format detected. Cannot read in data')
        log.info('successfully read in data with yyyy-mm-dd formatting')
    date_range = pd.period_range(date_start, date_end, freq = 'D')
    cleaned_df['Date'] = date_range
    cleaned_df = cleaned_df.set_index('Date')

    df_flow = pd.DataFrame(index = cleaned_df.index)
    df_level = pd.DataFrame(index = cleaned_df.index)

    for gauge in cleaned_df.columns:
        gauge_only = extract_gauge_from_string(gauge)
        if 'flow' in gauge:
            df_flow[gauge_only] = cleaned_df[gauge].copy(deep=True)
        if 'level' in gauge:
            df_level[gauge_only] = cleaned_df[gauge].copy(deep=True)
        if not gauge_only:
            log.info('Could not identify gauge in column name:', gauge, ', skipping analysis of data in this column.')
    return df_flow, df_level

def extract_gauge_from_string(input_string: str) -> str:
    '''Takes in a strings, pulls out the gauge number from this string
    
    Args:
        input_string (str): string which may contain a gauge number

    Returns:
        str: Gauge number as a string if found, None if not found
    '''
    gauge = input_string.split('_')[0]
    return gauge

def match_MDBA_nodes(input_df: pd.DataFrame, model_metadata: pd.DataFrame, ewr_table_path: str) -> tuple:
    '''Checks if the source file columns have EWRs available, returns a flow and level dataframe with only 
    the columns with EWRs available. Renames columns to gauges
    
    Args:
        input_df (pd.DataFrame): flow/water level dataframe
        model_metadata (pd.DataFrame): dataframe linking model nodes to gauges

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: flow dataframe, water level dataframe

    '''

    flow_gauges = data_inputs.get_gauges('flow gauges', ewr_table_path=ewr_table_path)
    level_gauges = data_inputs.get_gauges('level gauges', ewr_table_path=ewr_table_path)
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
    if df_flow.empty:
        raise ValueError('No relevant gauges and or measurands found in dataset, the EWR tool cannot evaluate this model output file')      
    return df_flow, df_level

def match_NSW_nodes(input_df: pd.DataFrame, model_metadata: pd.DataFrame) -> tuple:
    '''Checks if the source file columns have EWRs available, returns a flow and level dataframe with only 
    the columns with EWRs available. Renames columns to gauges
    
    Args:
        input_df (pd.DataFrame): flow/water level dataframe
        model_metadata (pd.DataFrame): dataframe linking model nodes to gauges

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: flow dataframe, water level dataframe
    
    '''
    flow_gauges = data_inputs.get_gauges('flow gauges')
    level_gauges = data_inputs.get_gauges('level gauges')
    df_flow = pd.DataFrame(index = input_df.index)
    df_level = pd.DataFrame(index = input_df.index)
    for col in input_df.columns:
        if not (model_metadata[model_metadata['SITEID'].str.contains(col, na=False, regex=False)]).empty:
            subset = model_metadata.query("SITEID==@col")
            gauge = subset["AWRC"].iloc[0]
            if gauge in flow_gauges:
                df_flow[gauge] = input_df[col]
            if gauge in level_gauges:
                df_level[gauge] = input_df[col]

    return df_flow, df_level

def any_cllmm_to_process(gauge_results: dict)->bool:
    cllmm_gauges = data_inputs.get_cllmm_gauges()
    processed_gauges = data_inputs.get_scenario_gauges(gauge_results)
    return any(gauge in processed_gauges for gauge in cllmm_gauges)

class ScenarioHandler:
    
    def __init__(self, scenario_file: str, model_format:str, parameter_sheet:str = None,
                calc_config_path:str = None):
        self.scenario_file = scenario_file
        self.model_format = model_format
        self.yearly_events = None
        self.pu_ewr_statistics = None
        self.summary_results = None
        self.parameter_sheet = parameter_sheet
        self.calc_config_path = calc_config_path
        self.flow_data = None
        self.level_data = None

    def _get_file_names(self, loaded_files):

        file_locations = {}
        # for file in loaded_files:
        if '/' in loaded_files:
            full_name = loaded_files.split('/')
        elif ('\\' in loaded_files):
            full_name = loaded_files.split('\\')
        else:
            full_name = loaded_files
        name_exclude_extension = full_name[-1].split('.csv')[0]
        file_locations[str(name_exclude_extension)] = loaded_files
            
        return file_locations
        
    def process_scenarios(self):

        scenarios = self._get_file_names(self.scenario_file)

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
                df_F, df_L = match_MDBA_nodes(df_clean, data_inputs.get_MDBA_codes(), self.parameter_sheet)
                               
            elif self.model_format == 'Standard time-series':
                df = pd.read_csv(scenarios[scenario], index_col = 'Date')
                df_F, df_L = cleaner_standard_timeseries(df, self.parameter_sheet)

            elif self.model_format == 'Source - NSW (res.csv)':
                data, header = unpack_model_file(scenarios[scenario], 'Date', 'Field')
                data = build_NSW_columns(data, header)
                df_clean = cleaner_NSW(data)
                df_F, df_L = match_NSW_nodes(df_clean, data_inputs.get_NSW_codes())

            elif self.model_format == 'ten thousand year':
                df = pd.read_csv(scenarios[scenario], index_col = 'Date')
                df_F, df_L = cleaner_ten_thousand_year(df, self.parameter_sheet)
            
            gauge_results = {}
            gauge_events = {}

            all_locations = set(df_F.columns.to_list() + df_L.columns.to_list())
            EWR_table, bad_EWRs = data_inputs.get_EWR_table(self.parameter_sheet)
            calc_config = data_inputs.get_ewr_calc_config(self.calc_config_path)
            for gauge in all_locations:
                gauge_results[gauge], gauge_events[gauge] = evaluate_EWRs.calc_sorter(df_F, df_L, gauge,
                                                                                        EWR_table, calc_config) 
            detailed_results[scenario] = gauge_results
            detailed_events[scenario] = gauge_events
            self.pu_ewr_statistics = detailed_results
            self.yearly_events = detailed_events
            
            self.flow_data = df_F
            self.level_data = df_L

    def get_all_events(self)-> pd.DataFrame:

        if not self.yearly_events:
            self.process_scenarios()
        
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
            self.process_scenarios()
        
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
            self.process_scenarios()
        
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

        all_successfulEvents = summarise_results.filter_successful_events(all_events_temp1, self.parameter_sheet) 

        return all_successfulEvents

    def get_all_successful_interEvents(self)-> pd.DataFrame:

        if not self.yearly_events:
            self.process_scenarios()
        
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
        all_successfulEvents = summarise_results.filter_successful_events(all_events_temp2, self.parameter_sheet) 

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
            self.process_scenarios()

        # Setting up the yearly results
        to_process = summarise_results.pu_dfs_to_process(self.pu_ewr_statistics)
        yearly_ewr_results = summarise_results.process_df_results(to_process)
                                
        yearly_ewr_results = summarise_results.join_ewr_parameters(cols_to_add=['Multigauge'],
                                left_table=yearly_ewr_results,
                                left_on=['gauge','pu','ewrCode'],
                                selected_columns= ['Year', 'eventYears', 'numAchieved', 'numEvents', 'numEventsAll',
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
                        left_on=['gauge','pu','ewr'],
                        selected_columns= ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                'eventDuration', 'eventLength', 
                                'Multigauge'],
                        parameter_sheet_path=self.parameter_sheet)
        all_events_temp = summarise_results.filter_duplicate_start_dates(all_events_temp)
        #Filter out the unsuccessful events:
        all_successfulEvents = summarise_results.filter_successful_events(all_events_temp, self.parameter_sheet)

        # Get start and end date of the timeseries.
        date0 = self.flow_data.index[0]
        date1 = self.flow_data.index[-1]
        start_date = date(date0.year, date0.month, date0.day)
        end_date = date(date1.year, date1.month, date1.day)
        df = summarise_results.events_to_interevents(start_date, end_date, all_successfulEvents)
        rolling_max_interevents_dict = summarise_results.get_rolling_max_interEvents(df, start_date, end_date, yearly_ewr_results, self.parameter_sheet)
        # Add the rolling max interevents to the yearly dataframe:
        yearly_ewr_results = summarise_results.add_interevent_to_yearly_results(yearly_ewr_results, rolling_max_interevents_dict)
        
        # Calculate the rolling achievement of the interevent, append this to a new column
        yearly_ewr_results = summarise_results.add_interevent_check_to_yearly_results(yearly_ewr_results, self.parameter_sheet)

        return yearly_ewr_results




    def get_ewr_results(self) -> pd.DataFrame:
        
        if not self.pu_ewr_statistics:
            self.process_scenarios()

        return summarise_results.summarise(self.pu_ewr_statistics , self.yearly_events, parameter_sheet_path=self.parameter_sheet)