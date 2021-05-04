import pandas as pd
import re
import numpy as np
from datetime import date, timedelta
from datetime import time 
import gauge_getter, gauge_getter_level
from tqdm import tqdm
import data_inputs
import os 
import urllib
import csv

#------------------------------- Handling functions---------------------------------------------#

def scenario_handler(file_locations, request_list, ewr_table, model_format, bigmod_info, toleranceDict, climate_file):
    '''Takes in scenarios, send to functions to clean, calculate ewrs, and summarise results,
    returns a results summary as a dataframe and a dictionary with yearly results'''
    
    dict_of_model_runs = {}
    for file in tqdm(file_locations, position = 0, leave = True,
                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc= 'Unpacking scenario data'):
        header_dict = {}
        data_dict = {}
        if model_format == 'Bigmod':
            data_df = unpack_bigmod_data(file_locations[file])    
            combined_data = {'flow data': data_df}
        elif model_format == 'Source':
            data_df = unpack_source_data(file_locations[file])
            combined_data = {'flow data': data_df}
        elif model_format == 'IQQM':
            print('Yet to set up IQQM handling')
        elif model_format == 'NSW 10,000 years':
            data_df = unpack_nsw_data(file_locations[file])
            combined_data = {'flow data': data_df}
        dict_of_model_runs[file] = combined_data
        
    # evaluate scenarios:
    dict_scenario_results = {}
    for scenario in tqdm(dict_of_model_runs, position = 0, leave = True,
                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc= 'Evaluating scenarios'):
        dict_scenario_results[scenario] = calculation_distributor(dict_of_model_runs[scenario],
                                                                  model_format,
                                                                  ewr_table,
                                                                  None,
                                                                  bigmod_info,
                                                                  toleranceDict,
                                                                  climate_file
                                                                 )
    # Analyse and summarise the results:
    std_results_table = summarise_results(dict_scenario_results, request_list, ewr_table)
    
    # Return yearly results, results summary
    return dict_scenario_results, std_results_table
    
def realtime_handler(gauges_list, request_list, input_params_g, ewr_table, toleranceDict, climate_file):
    '''ingests a list of gauges and user defined parameters
    pulls gauge data using relevant states API, calcualtes and analyses ewrs
    returns dictionary of raw data results and result summary'''
    # Need to determine which gauges require the level data to be pulled:
    menindeeGauges, wpGauges = data_inputs.getLevelGauges()
    multiGauges = data_inputs.getMultiGauges('gauges')
    simultaneousGauges = data_inputs.getSimultaneousGauges('gauges')
    levelGauges = []
    flowGauges = []
    for gauge in gauges_list:
        if gauge in multiGauges.keys():
            flowGauges.append(gauge)
            flowGauges.append(multiGauges[gauge])
        elif gauge in simultaneousGauges:
            flowGauges.append(gauge)
            flowGauges.append(simultaneousGauges[gauge])
        elif gauge in menindeeGauges:
            levelGauges.append(gauge)
        elif gauge in wpGauges.keys(): # If a weirpool gauge, we want to get both the level and flow gauges:
            flowGauges.append(gauge)
            levelGauges.append(wpGauges[gauge])
        else: # Otherwise it must be a flow gauge:
            flowGauges.append(gauge)
    # Retrieve the gauge data using relevant Basin states API:
    flow_gauge_data_df = gauge_getter.gaugePull(flowGauges, 
                                           input_params_g['start time'], 
                                           input_params_g['end time'])

    # Retrieve the level gauge data (if there are any weirpool or menindee locations tagged in the request):
    level_gauge_data_df = gauge_getter_level.gaugePull(levelGauges, 
                                           input_params_g['start time'], 
                                           input_params_g['end time'])
#     level_gauge_data_df = level_gauge_data_df.drop_duplicates()
    # Combine the flow and level dataframes:
    gauge_data_df = pd.concat([flow_gauge_data_df, level_gauge_data_df])

    # evaluate scenarios:
    dict_scenario_results = {}
    dict_scenario_results['gauge data'] = calculation_distributor(gauge_data_df, 
                                                                  'Realtime data', 
                                                                  ewr_table,
                                                                  input_params_g,
                                                                  None,
                                                                  toleranceDict,
                                                                  climate_file
                                                                 )
    
    # Analyse and summarise the results:
    std_results_table = summarise_results(dict_scenario_results, request_list, ewr_table)
    
    # return yearly results and results summary:
    return dict_scenario_results, std_results_table

def calculation_distributor(input_df, data_source, ewr_table, user_params, bigmod_info, toleranceDict, climate_file):
    '''Pass to different dataframe cleaning function, 
    then iterates over the locations and passes these with the dataframe to get calculated
    Returns a dataframes with results of the binary ewr check'''
    
    if data_source == 'Bigmod':
        flow_df_subset = match_model_code_gauge(input_df['flow data'],
                                               ewr_table,
                                               bigmod_info
                                              ) 
        flow_df_clean = bigmod_cleaner(flow_df_subset)# clean model inputs
        dict_of_results = {}
        for gauge_ID  in flow_df_clean:
            calculated_col = EWR_calculator(flow_df_clean, gauge_ID, ewr_table, toleranceDict, climate_file)
            dict_of_results[gauge_ID] = calculated_col

        return dict_of_results
    
    elif data_source == 'Source':
        flow_df_clean = source_cleaner(input_df['flow data'])
        dict_of_results = {}
        for gauge_ID  in flow_df_clean:
            calculated_col = EWR_calculator(flow_df_clean, gauge_ID, ewr_table, toleranceDict, climate_file)
            dict_of_results[gauge_ID] = calculated_col

        return dict_of_results
        
    elif data_source == 'IQQM':
        print('yet to introduce this file type handling')

    elif data_source == 'NSW 10,000 years':
        flow_df_clean = nswData_cleaner(input_df['flow data'])
        dict_of_results = {}
        for gauge_ID in flow_df_clean:
            calculated_col = EWR_calculator(flow_df_clean, gauge_ID, ewr_table, toleranceDict, climate_file)
            dict_of_results[gauge_ID] = calculated_col
        
        return dict_of_results
        
    elif data_source == 'Realtime data':
        gauge_df_clean, minDatesDict = realtime_cleaner(input_df, user_params) # clean gauge inputs
        dict_of_results = {}
        for gauge in gauge_df_clean:
            calculated_col = EWR_calculator(gauge_df_clean, gauge, ewr_table, toleranceDict, climate_file)
            filtered_col = filter_earlyDates(calculated_col, minDatesDict[gauge])
            dict_of_results[gauge] = filtered_col
        return dict_of_results   

def filter_earlyDates(input_df, minDate):
    for planningUnit in input_df:
        input_df[planningUnit][(input_df[planningUnit].index < (int(minDate)-1))] = None
    
    return input_df
    
#------------------------- Real time ewr functions------------------------------------------#

def convert_date_type(date_to_convert):
    ''' Converts input date to datetime format'''
    
    new_date = pd.to_datetime(str(date_to_convert[0:4]+'-'
                                  + date_to_convert[4:6]+'-'
                                  + date_to_convert[6:8]),
                              format = '%Y-%m-%d')
    return new_date

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
    single_df = single_df.drop(['DATASOURCEID', 
                                'SITEID',
                                'SUBJECTID',
                                'QUALITYCODE',
                                'DATETIME'],
                               axis = 1
                              )
    single_df = single_df.set_index('Date')
    single_df = single_df.rename(columns={'VALUE': gauge_iter})
    
    return single_df

def get_minDate(input_df, input_gauge):
    goodData = input_df[input_df[input_gauge].notnull()]
    minDate = goodData.index.year.min()
    
    return minDate

def realtime_cleaner(input_df, input_parameters):
    '''Takes in raw dataframe consolidated from state websites, removes poor quality data.
    returns a dataframe with a date index and one flow column per gauge location.'''
    
    start_date = convert_date_type(input_parameters['start time'])
    end_date = convert_date_type(input_parameters['end time'])
    
    df_index = pd.date_range(start_date,end_date-timedelta(days=1),freq='d')
    gauge_data_df = pd.DataFrame()
    gauge_data_df['Date'] = df_index
    gauge_data_df['Date'] = pd.to_datetime(gauge_data_df['Date'], format = '%Y-%m-%d')
    gauge_data_df = gauge_data_df.set_index('Date')

    input_df["VALUE"] = pd.to_numeric(input_df["VALUE"], downcast="float")
    input_df['Date'] = pd.to_datetime(input_df['DATETIME'], format = '%Y-%m-%d')
    # Check with states for more codes:
    bad_data_codes = [201, 202, 204, 205, 207, 255]
    input_df = remove_data_with_bad_QC(input_df, bad_data_codes)
    
    site_list = set(input_df['SITEID'])
    minDates = {}
    
    for gauge in site_list:
        # Seperate out to one gauge per column and add this to the gauge_data_df made above:
        single_gauge_df = one_gauge_per_column(input_df, gauge)
        minDates[gauge] = get_minDate(single_gauge_df, gauge)
        gauge_data_df = pd.merge(gauge_data_df, single_gauge_df, left_index=True, right_index=True, how="outer")  # pd.concat([gauge_data_df, single_gauge_df], axis = 1)

    # Drop the non unique values:
    gauge_data_df = gauge_data_df[~gauge_data_df.index.duplicated(keep='first')]
    return gauge_data_df, minDates  

#----------------------------------- Scenario testing ewr functions----------------------------------#
def unpack_nsw_data(csv_file):
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
    

def unpack_bigmod_data(csv_file):
    '''Ingesting scenario file locations with bigmod format, seperates the flow data and header data
    returns a dictionary of flow dataframes with their associated header data'''
    if csv_file[-3:] != 'csv':
        raise ValueError('''Incorrect file type selected, bigmod format requires a csv file.
                         Rerun the program and try again.''')
    
    #--------functions for pulling main data-------#
    def mainData_url(url, line,**kwargs):
        response = urllib.request.urlopen(url)
        lines = [l.decode('utf-8') for l in response.readlines()]
        cr = csv.reader(lines)
        pos = 0
        for row in cr:
            if row[0].startswith(line):
                headerVal = pos
            pos = pos + 1

        df = pd.read_csv(url, header=headerVal)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        return df, headerVal
        
    def mainData(fle, line,**kwargs):
        if os.stat(fle).st_size == 0:
            raise ValueError("File is empty")
        with open(fle) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if row[0].startswith(line):
                    headerVal = line_count
                line_count = line_count + 1
        df = pd.read_csv(fle, header=headerVal)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]            

        return df, headerVal
        
    #--------Functions for pulling header data---------#
    
    def headerData_url(url, line, endLine, **kwargs):
        response = urllib.request.urlopen(url)
        lines = [l.decode('utf-8') for l in response.readlines()]
        cr = csv.reader(lines)
        pos = 0
        for row in cr:
            if row[0].startswith(line):
                headerVal = pos
            pos = pos + 1
        junkRows = headerVal
        df = pd.read_csv(url, header=headerVal, nrows = (endLine-junkRows-1), dtype={'Site':'str', 'Measurand': 'str', 'Quality': 'str'})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        return df
        
    def headerData(fle, line, endLine, **kwargs):
        if os.stat(fle).st_size == 0:
            raise ValueError("File is empty")
            
        with open(fle) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if row[0].startswith(line):
                    headerVal = line_count
                line_count = line_count + 1
            junkRows = headerVal
        df = pd.read_csv(fle, header=headerVal, nrows = (endLine-junkRows-1), dtype={'Site':'str', 'Measurand': 'str', 'Quality': 'str'}, **kwargs) 
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
        return df

    #----------Constructing the full reference codes---------------#
    
    def bigmodHeaderContruction(input_header):
        '''Takes in the header data file, trims it, and then renames the column headings with the full reference code
        returns a list of headings'''
        # Clean dataframe
        numRows = int(input_header['Field'].iloc[1]) 
        df = input_header.drop([0,1])     
        df = df.astype(str)
        df['Site'] = df['Site'].map(lambda x: x.lstrip("'").rstrip("'"))
        df['Measurand'] = df['Measurand'].map(lambda x: x.lstrip("'").rstrip("'"))
        df['Quality'] = df['Quality'].map(lambda x: x.lstrip("'").rstrip("'"))
        # Get the EOC value for the number of rows we're going to look at then drop the row
  
        # Construct refs and save to list:
        listOfCols = []
        for i in range(0, numRows):
            colName = str(df['Site'].iloc[i] + '-' + df['Measurand'].iloc[i] + '-' + df['Quality'].iloc[i])
            listOfCols.append(colName)
        dateList = ['Dy', 'Mn', 'Year']
        listOfCols = dateList + listOfCols
        
        return listOfCols
    
    #-------------------Get main and header data--------------------#
    
    mainKeyLine = 'Dy'
    if 'http' in csv_file:
        mainData_df, endLine = mainData_url(csv_file, mainKeyLine, sep=",")        
    else:
        mainData_df, endLine = mainData(csv_file, mainKeyLine, sep=",")
    
    headerKeyLine = 'Field'
    if 'http' in csv_file:
        headerData_df = headerData_url(csv_file, headerKeyLine, endLine, sep=",")        
    else:
        headerData_df = headerData(csv_file, headerKeyLine, endLine, sep=",")
    
    #---------Save full model ref codes as columns headings---------#
    columnHeadingsFull = bigmodHeaderContruction(headerData_df)
    mainData_df.columns = columnHeadingsFull
    mainData_df = mainData_df.drop([0])      
    
    return mainData_df # Dont need header data anymore, just return main flow data:
    
def unpack_source_data(text_file):
    '''Ingesting scenario file locations with source format, seperates the flow data and header data
    returns a dictionary of flow dataframes with their associated header data'''    
    if text_file[-3:] != 'txt':
        raise ValueError('''Incorrect file type selected, Source format requires a txt file.
                         Rerun the program and try again.''')
    flow_data = pd.read_csv(text_file, sep="      ", header=1)
    
    # Pull the first line out with the location details:
    with open(text_file) as f:
        header_data = f.readline()

    # Send to get gauge from the header data:
    gauge = extract_gauge_from_string(header_data)
    
    flow_data.columns = [gauge]
    
    return flow_data

def bigmod_cleaner(input_df):
    '''Ingests dataframe, removes junk columns, fixes date,
    returns formatted dataframe'''
    
    cleaned_df = input_df.rename(columns={'Mn': 'Month', 'Dy': 'Day'})
    cleaned_df['Date'] = pd.to_datetime(cleaned_df[['Year', 'Month', 'Day']], format = '%Y-%m-%d')
    cleaned_df = cleaned_df.drop(['Day', 'Month', 'Year'], axis = 1)
    cleaned_df = cleaned_df.set_index('Date')
    
    return cleaned_df

def source_cleaner(input_df):
    '''Ingests dataframe, removes junk columns, fixes date,
    returns formatted dataframe'''
    
    cleaned_df = input_df.copy(deep=True)
    
    date_start = cleaned_df.index[0]
    date_end = cleaned_df.index[-1]

    date_range = pd.period_range(date_start, date_end, freq = 'D')
    cleaned_df['Date'] = date_range
    cleaned_df = cleaned_df.set_index('Date')
    
    return cleaned_df
  
def nswData_cleaner(input_df):
    '''Ingests dataframe, removes junk columns, fixes date,
    returns formatted dataframe'''
    
    cleaned_df = input_df.copy(deep=True)
    
    date_start = cleaned_df.index[0]
    date_end = cleaned_df.index[-1]

    date_range = pd.period_range(date_start, date_end, freq = 'D')
    cleaned_df['Date'] = date_range
    cleaned_df = cleaned_df.set_index('Date')
 
    return cleaned_df
    
def get_unit_from_string(input_string):
    '''Takes in a string, returns the measurement unit found in that string'''
    
    if '(ML/d)' in input_string:
        found_unit = 'ML/d'
    elif '(ML)' in input_string:
        found_unit = 'ML'
    elif '(GL)' in input_string:
        found_unit = 'GL'
    elif '(m)' in input_string:
        found_unit = 'm'
    else:
        found_unit = None
    return found_unit

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

def construct_refCode(site, measurand, quality):
    '''Takes in site name, the measurand and the quality strings, constructs the
    reference code used in the model column names, returns this reference code'''
    
    if site[-2] == ' ':
        new_site = site[1:-2]
    else:
        new_site = site[1:-1]
        
    siteRef = (new_site + '-' + str(int(measurand)) + '-' + quality[1:-1])
    
    return siteRef

def combine_metadata_columns(input_header_df):
    '''Combines all the potential columns containing metadata in the model output files '''
    
    name_desc = input_header_df['Name'] + input_header_df['Description'] 
            
    return name_desc

def match_model_code_gauge(input_df, ewr_table, bigmod_metadata):
    '''Checks if the bigmod file columns have ewrs available,
    returns a dataframe containing only the columns with ewrs available
    renames the column headers as the gauge location'''
    # get the two lists of level gauges:
    menindeeGauges, wpGauges = data_inputs.getLevelGauges()
    wpGauges = list(wpGauges.values()) # Gets returned as a dictionary of flow to level gauges, we just need the lavel gauges here
    multipleGauges = data_inputs.getMultiGauges('gauges')
    simultaneousGauges = data_inputs.getSimultaneousGauges('gauges')
    multiGauges = list(multipleGauges.values())
    simulGauges = list(simultaneousGauges.values())
    df = input_df[['Year', 'Mn', 'Dy']].copy()

    for col in input_df.columns[3:]:
        clean_col = col.replace(' ', '') # clean the ones with spaces
        col_split = clean_col.split('-') # split into the 3 components
        site = str(col_split[0]) # get the site code
        measurand = col_split[1] # get measurand code
        if measurand == '1': # Check for flow measurand 
            try:
                gauge_num = bigmod_metadata.loc[bigmod_metadata['SITEID'] == site, 'AWRC'].to_numpy()[0]
                if gauge_num != None:
                    if gauge_num not in menindeeGauges:
                        # Check if its in either gauge column (flow or weir pool gauges)
                        if ewr_table['gauge'].str.contains(gauge_num).any(): 
                            df[gauge_num]=input_df[col]
                    if gauge_num in multiGauges:
                        df[gauge_num]=input_df[col]
                    if gauge_num in simulGauges:
                        df[gauge_num]=input_df[col]
            except IndexError:
                continue
        elif measurand == '35': # check for level measurand
            try:
                gauge_num = bigmod_metadata.loc[bigmod_metadata['SITEID'] == site, 'AWRC'].to_numpy()[0]
                if gauge_num != None:
                    if ((gauge_num in menindeeGauges) or (gauge_num in wpGauges)):
                        if ewr_table['gauge'].str.contains(gauge_num).any():
                            df[gauge_num]=input_df[col]
                        elif ewr_table['weirpool gauge'].str.contains(gauge_num).any():
                            df[str(gauge_num)]=input_df[col]
            except IndexError:
                continue
    
    return df

#----------------------------- The ewr calculcator: --------------------------------#

#----------------------------- Getting EWRs from the database ----------------------#

def getEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for standard flow ewrs******** '''
    ewrs = {}
    ewrs['start_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                            (ewr_table['code'] == ewr_name)&\
                                            (ewr_table['PlanningUnitID'] == planning_unit)
                                           )]['start month'])[0])
    ewrs['end_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['end month'])[0])
    ewrs['minThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold min'])[0])*toleranceDict['minThreshold'], 0))
    ewrs['maxThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold max'])[0])*toleranceDict['maxThreshold'], 0))
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit)
                                                    )]['duration'])[0])*toleranceDict['duration'], 0))
    ewrs['gapTolerance'] = int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                               (ewr_table['code'] == ewr_name)&\
                                               (ewr_table['PlanningUnitID'] == planning_unit)
                                              )]['within event gap tolerance'])[0])
    ewrs['events per year']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['events per year'])[0])
    ewrs['min event']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['min event'])[0])  
    
    return ewrs
    
def getLowFlowEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):  
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for low flow ewrs only******** '''
    ewrs = {}
    
    ewrs['start_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                            (ewr_table['code'] == ewr_name)&\
                                            (ewr_table['PlanningUnitID'] == planning_unit)
                                           )]['start month'])[0])
    ewrs['end_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['end month'])[0])
    ewrs['minThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold min'])[0])*toleranceDict['minThreshold'], 0))
    ewrs['maxThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold max'])[0])*toleranceDict['maxThreshold'], 0))
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit)
                                                    )]['duration'])[0])*toleranceDict['duration'], 0))
    # these ewrs also have a very dry caveat, save the very dry duration to a variable for later:
    try:
        veryDry_ewr_code = str(ewr_name + '_VD')
        ewrs['veryDry_duration'] = int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == veryDry_ewr_code)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['duration'])[0]) 
    except IndexError:
        ewrs['veryDry_duration'] = None
    
    return ewrs
    
def getCtfEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for ctf ewrs only******** '''
    ewrs = {}
    ewrs['start_month'] = 7
    ewrs['end_month'] = 6
    ewrs['minThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold min'])[0])*toleranceDict['minThreshold'], 0))
    ewrs['maxThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit))]['flow threshold max'])[0])*toleranceDict['maxThreshold'], 0))
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit))]['duration'])[0])*toleranceDict['duration'], 0))
    # these ewrs also have a very dry caveat, save the very dry duration to a variable for later:
    try:
        veryDry_ewr_code = str(ewr_name + '_VD')
        ewrs['veryDry_duration'] = int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == veryDry_ewr_code)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['duration'])[0]) 
    except IndexError:
        ewrs['veryDry_duration'] = None
        
    return ewrs
        
def getCumulVolEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for cumulative duration ewrs only******** '''  
    ewrs = {}
    ewrs['start_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                            (ewr_table['code'] == ewr_name)&\
                                            (ewr_table['PlanningUnitID'] == planning_unit)
                                           )]['start month'])[0])
    ewrs['end_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['end month'])[0])
    ewrs['minThresholdV']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                    (ewr_table['code'] == ewr_name)&\
                                                    (ewr_table['PlanningUnitID'] == planning_unit)
                                                   )]['volume threshold'])[0])*toleranceDict['minThreshold'], 0))
    ewrs['duration']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                   (ewr_table['code'] == ewr_name)&\
                                                   (ewr_table['PlanningUnitID'] == planning_unit)
                                                  )]['duration'])[0])*toleranceDict['duration'], 0))
    ewrs['events per year']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['events per year'])[0])
    try:
        ewrs['minThresholdF']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                    (ewr_table['code'] == ewr_name)&\
                                                    (ewr_table['PlanningUnitID'] == planning_unit)
                                                   )]['flow threshold min'])[0])
    except (IndexError, ValueError) as e:
        ewrs['minThresholdF'] = 0
    
    return ewrs
    
def getLakeEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):  
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for ewrs measured against levels rather than flows******** '''    
    ewrs = {}
    ewrs['start_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                            (ewr_table['code'] == ewr_name)&\
                                            (ewr_table['PlanningUnitID'] == planning_unit)
                                           )]['start month'])[0])
    ewrs['end_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['end month'])[0])
    ewrs['minThresholdL']=float(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                               (ewr_table['code'] == ewr_name)&\
                                               (ewr_table['PlanningUnitID'] == planning_unit)
                                              )]['level threshold min'])[0])*toleranceDict['minThreshold']
    ewrs['maxThresholdL']=float(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                               (ewr_table['code'] == ewr_name)&\
                                               (ewr_table['PlanningUnitID'] == planning_unit)
                                              )]['level threshold max'])[0])*toleranceDict['maxThreshold']
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit)
                                                    )]['duration'])[0])*toleranceDict['duration'], 0))
    ewrs['events per year']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['events per year'])[0])
    try:
        maxDrawdown = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                    (ewr_table['code'] == ewr_name)&\
                                                    (ewr_table['PlanningUnitID'] == planning_unit)
                                                   )]['drawdown rate'])[0])*toleranceDict['drawdownTolerance'],0))
        ewrs['maxDrawdown'] = maxDrawdown/100
    except ValueError:
        ewrs['maxDrawdown'] = 1000000
        
    return ewrs

def getWPewrs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for WP ewrs only******** '''    
    ewrs = {}
    ewrs['start_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                            (ewr_table['code'] == ewr_name)&\
                                            (ewr_table['PlanningUnitID'] == planning_unit)
                                           )]['start month'])[0])
    ewrs['end_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['end month'])[0])
    ewrs['minThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold min'])[0])*toleranceDict['minThreshold'], 0))
    ewrs['maxThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold max'])[0])*toleranceDict['maxThreshold'], 0))
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit)
                                                    )]['duration'])[0])*toleranceDict['duration'], 0))
    minThresh=(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                               (ewr_table['code'] == ewr_name)&\
                               (ewr_table['PlanningUnitID'] == planning_unit)
                              )]['level threshold min'])[0])
    maxThresh=(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                               (ewr_table['code'] == ewr_name)&\
                               (ewr_table['PlanningUnitID'] == planning_unit)
                              )]['level threshold max'])[0])
    # Determine if its a weirpool raising or weirpool drawdown:
    if minThresh == '?':
        ewrs['wpType'] = 'wpDrawdown'
        ewrs['maxThresholdL'] = float(maxThresh)*toleranceDict['maxThreshold']
        
    elif maxThresh == '?':
        ewrs['wpType'] = 'wpRaising' 
        ewrs['minThresholdL'] = float(minThresh)*toleranceDict['minThreshold']
        
    # :getting the max drawdown in cm and converting to m
    maxDrawdown = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['drawdown rate'])[0])*toleranceDict['drawdownTolerance'],0))
    ewrs['maxDrawdown'] = maxDrawdown/100
    ewrs['weirpoolGauge']=str(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                              (ewr_table['code'] == ewr_name)&\
                                              (ewr_table['PlanningUnitID'] == planning_unit)
                                             )]['weirpool gauge'])[0])
    ewrs['events per year']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['events per year'])[0])
                       
    return ewrs

def getComplexEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):
    '''Takes in a gauge number and the ewr.
    returns the ewr values *******Complex EWRs only******** '''
    ewrs = {}
    ewrs['start_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                            (ewr_table['code'] == ewr_name)&\
                                            (ewr_table['PlanningUnitID'] == planning_unit)
                                           )]['start month'])[0])
    ewrs['end_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['end month'])[0])
    ewrs['minThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold min'])[0])*toleranceDict['minThreshold'], 0))
    ewrs['maxThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold max'])[0])*toleranceDict['maxThreshold'], 0))
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit)
                                                    )]['duration'])[0])*toleranceDict['duration'], 0))
    ewrs['events per year']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['events per year'])[0])
    # Currently defined within the tool. We will move to have this in a seperate table ideally:
    if gauge_number == '409025':
        if ((ewr_name == 'OB2_S') or (ewr_name == 'OB2_P')):
            ewrs['postEventThreshold'] = int(round(9000*toleranceDict['minThreshold'], 0))
            ewrs['postEventDuration'] = int(round(105*toleranceDict['duration'], 0))
            ewrs['start_month'] = 7
            ewrs['end_month'] = 6
            ewrs['gapTolerance'] = 7
        elif ((ewr_name == 'OB3_S') or (ewr_name == 'OB3_P')):
            ewrs['outsideEventThreshold'] = int(round(15000*toleranceDict['minThreshold'], 0))
            ewrs['outsideEventDuration'] = int(round(90*toleranceDict['duration'], 0))
            ewrs['start_month'] = 7
            ewrs['end_month'] = 6
            ewrs['gapTolerance'] = 7
    return ewrs

def getNestEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for WP ewrs only******** '''    
    ewrs = {}
    startDate = str(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                    (ewr_table['code'] == ewr_name)&\
                                    (ewr_table['PlanningUnitID'] == planning_unit)
                                   )]['start month'])[0])
    if '.' in startDate:
        ewrs['start_day'] = int(startDate.split('.')[1]) # Get the day from the float
        ewrs['start_month'] = int(startDate.split('.')[0]) # Get the month from the float
    else:
        ewrs['start_day'] = None
        ewrs['start_month'] = int(startDate)
    endDate = str(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                  (ewr_table['code'] == ewr_name)&\
                                  (ewr_table['PlanningUnitID'] == planning_unit)
                                 )]['end month'])[0])
    if '.' in endDate:  
        ewrs['end_day'] = int(endDate.split('.')[1]) # Get the day from the float
        ewrs['end_month'] = int(endDate.split('.')[0]) # Get the month from the float
    else:
        ewrs['end_day'] = None
        ewrs['end_month'] = int(endDate)
    ewrs['minThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold min'])[0])*toleranceDict['minThreshold'], 0))
    ewrs['maxThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold max'])[0])*toleranceDict['maxThreshold'], 0))
    minThreshL=(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                (ewr_table['code'] == ewr_name)&\
                                (ewr_table['PlanningUnitID'] == planning_unit)
                               )]['level threshold min'])[0])
    maxThreshL=(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                (ewr_table['code'] == ewr_name)&\
                                (ewr_table['PlanningUnitID'] == planning_unit)
                               )]['level threshold max'])[0])
    if minThreshL != '?':
        ewrs['minThresholdL'] = float(minThreshL)*toleranceDict['minThreshold']
    if maxThreshL != '?':
        ewrs['maxThresholdL'] = float(maxThreshL)*toleranceDict['maxThreshold']
        
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit)
                                                    )]['duration'])[0])*toleranceDict['duration'], 0))
    ewrs['maxDrawdown'] = (list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                           (ewr_table['code'] == ewr_name)&\
                                           (ewr_table['PlanningUnitID'] == planning_unit)
                                          )]['drawdown rate'])[0])
    if '%' not in ewrs['maxDrawdown']:
        ewrs['weirpoolGauge']=str(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                  (ewr_table['code'] == ewr_name)&\
                                                  (ewr_table['PlanningUnitID'] == planning_unit)
                                                 )]['weirpool gauge'])[0])
        ewrs['maxDrawdown'] = int(round(int(ewrs['maxDrawdown'])*toleranceDict['drawdownTolerance'],0))
    else:
        # remove the % sign, perform the tolerance transformation, return to int
        valueOnly = int(round(int(ewrs['maxDrawdown'].replace('%', ''))*toleranceDict['drawdownTolerance'],0))
        # put the % back in:
        ewrs['maxDrawdown'] = str(str(valueOnly)+'%')
        
    # Certain Nest EWRs have trigger days, these have been hard coded in until the ewr database is updated:
    if ((gauge_number == '409025') and (ewr_name == 'NestS1')):
        ewrs['triggerDay'] = 15
        ewrs['triggerMonth'] = 9
    elif ((gauge_number == '409207') and (ewr_name == 'NestS1')):
        ewrs['triggerDay'] = 1
        ewrs['triggerMonth'] = 10
        ewrs['start_month'] = 10 # Remove once database is updated
        ewrs['start_day'] = None # Remove once database is updated
    else:
        ewrs['triggerDay'] = None
        ewrs['triggerMonth'] = None
        
    ewrs['events per year']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['events per year'])[0])
    
    return ewrs

def getMultiGaugeEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for multi gauge ewrs******** '''
    
    ewrs = {}
    ewrs['start_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                            (ewr_table['code'] == ewr_name)&\
                                            (ewr_table['PlanningUnitID'] == planning_unit)
                                           )]['start month'])[0])
    ewrs['end_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['end month'])[0])
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit)
                                                    )]['duration'])[0])*toleranceDict['duration'], 0))
    ewrs['gapTolerance'] = int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                               (ewr_table['code'] == ewr_name)&\
                                               (ewr_table['PlanningUnitID'] == planning_unit)
                                              )]['within event gap tolerance'])[0])
    ewrs['events per year']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['events per year'])[0])
    # these ewrs also have a very dry caveat, save the very dry duration to a variable for later:
    try:
        veryDry_ewr_code = str(ewr_name + '_VD')
        ewrs['veryDry_duration'] = int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == veryDry_ewr_code)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['duration'])[0])
    except IndexError:
        ewrs['veryDry_duration'] = None    
    
    try:
        ewrs['minThresholdV']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                    (ewr_table['code'] == ewr_name)&\
                                                    (ewr_table['PlanningUnitID'] == planning_unit)
                                                   )]['volume threshold'])[0])*toleranceDict['minThreshold'], 0))
    
    except (IndexError, ValueError) as e:
        ewrs['minThresholdV'] = None
    
    
    try:
        ewrs['minThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold min'])[0])*toleranceDict['minThreshold'], 0))
        ewrs['maxThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == ewr_name)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['flow threshold max'])[0])*toleranceDict['maxThreshold'], 0))
        ewrs['min event']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit))]['min event'])[0])
    except (IndexError, ValueError) as e:
        ewrs['minThresholdF'] = 0
        ewrs['maxThresholdF'] = 1000000
        ewrs['min event'] = 0
    
    
    
    
    multiGaugeDict = data_inputs.getMultiGauges('all')
    
    ewrs['second gauge'] = multiGaugeDict[planning_unit][gauge_number]
    
    return ewrs

def getSimultaneousEWRs(planning_unit, gauge_number, ewr_name, ewr_table, toleranceDict):
    ''' Takes in a gauge number and the ewr. 
    Returns the ewr values *******for multi gauge ewrs******** '''
    
    ewrs = {}
    ewrs['start_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                            (ewr_table['code'] == ewr_name)&\
                                            (ewr_table['PlanningUnitID'] == planning_unit)
                                           )]['start month'])[0])
    ewrs['end_month']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['end month'])[0])
    try:
        ewrs['minThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                           (ewr_table['code'] == ewr_name)&\
                                                           (ewr_table['PlanningUnitID'] == planning_unit)
                                                          )]['flow threshold min'])[0])*toleranceDict['minThreshold'], 0))
        ewrs['maxThresholdF']=int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                           (ewr_table['code'] == ewr_name)&\
                                                           (ewr_table['PlanningUnitID'] == planning_unit)
                                                          )]['flow threshold max'])[0])*toleranceDict['maxThreshold'], 0))
    except ValueError:
        ewrs['minThresholdF']=None
        ewrs['maxThresholdF']=None
        
    ewrs['duration'] = int(round(int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                     (ewr_table['code'] == ewr_name)&\
                                                     (ewr_table['PlanningUnitID'] == planning_unit)
                                                    )]['duration'])[0])*toleranceDict['duration'], 0))
    ewrs['gapTolerance'] = int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                               (ewr_table['code'] == ewr_name)&\
                                               (ewr_table['PlanningUnitID'] == planning_unit)
                                              )]['within event gap tolerance'])[0])
    ewrs['events per year']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                (ewr_table['code'] == ewr_name)&\
                                                (ewr_table['PlanningUnitID'] == planning_unit)
                                               )]['events per year'])[0])
    ewrs['min event']=int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                          (ewr_table['code'] == ewr_name)&\
                                          (ewr_table['PlanningUnitID'] == planning_unit)
                                         )]['min event'])[0])
    
    # these ewrs also have a very dry caveat, save the very dry duration to a variable for later:
    try:
        veryDry_ewr_code = str(ewr_name + '_VD')
        ewrs['veryDry_duration'] = int(list(ewr_table[((ewr_table['gauge'] == gauge_number)&\
                                                       (ewr_table['code'] == veryDry_ewr_code)&\
                                                       (ewr_table['PlanningUnitID'] == planning_unit)
                                                      )]['duration'])[0])
    except IndexError:
        ewrs['veryDry_duration'] = None
    
    
    simultaneousGaugeDict = data_inputs.getSimultaneousGauges('all')
    
    ewrs['second gauge'] = simultaneousGaugeDict[planning_unit][gauge_number]
    
    return ewrs

#----------------------------- Masking dataframes to required dates ----------------------#

def get_month_mask(start, end, input_df):
    ''' takes in a start date, end date, and dataframe,
    masks the dataframe to these dates'''
    
    if start > end:
        month_mask = (input_df.index.month >= start) | (input_df.index.month <= end)
    elif start <= end:
        month_mask = (input_df.index.month >= start) & (input_df.index.month <= end)  
        
    input_df_timeslice = input_df.loc[month_mask]
    
    return input_df_timeslice

def get_day_month_mask(startDay, startMonth, endDay, endMonth, input_df):
    ''' for the ewrs with a day and month requirement, takes in a start day, start month, 
    end day, end month, and dataframe,
    masks the dataframe to these dates'''

    if startMonth > endMonth:
        month_mask = (((input_df.index.month >= startMonth) & (input_df.index.day >= startDay)) |\
                      ((input_df.index.month <= endMonth) & (input_df.index.day <= endDay)))
        input_df_timeslice = input_df.loc[month_mask]
        
    elif startMonth <= endMonth:
        #Filter the first and last month, and then get the entirety of the months in between
        month_mask1 = ((input_df.index.month == startMonth) & (input_df.index.day >= startDay))
        month_mask2 = ((input_df.index.month == endMonth) & (input_df.index.day <= endDay))
        month_mask3 = ((input_df.index.month > startMonth) & (input_df.index.month < endMonth))
        
        input_df_timeslice1 = input_df.loc[month_mask1]
        input_df_timeslice2 = input_df.loc[month_mask2]
        input_df_timeslice3 = input_df.loc[month_mask3]
        frames = [input_df_timeslice1, input_df_timeslice2, input_df_timeslice3]
        input_df_timeslice4 = pd.concat(frames)
        input_df_timeslice = input_df_timeslice4.sort_index()
        
    return input_df_timeslice

#------------------------------ Creating water year timeseries -------------------------------#

def waterYear_daily(input_df, ewrs):
    '''Creating a daily water year timeseries'''
    # check if water years needs to change:

    years = input_df.index.year.values
    months = input_df.index.month.values

    def appenderStandard(year, month):
        if month < 7:
            year = year - 1
        return year
    
    def appenderNonStandard(year, month):
        if month < ewrs['start_month']:
            year = year - 1
        return year
    
    if ((ewrs['start_month'] <= 6) and (ewrs['end_month'] >= 7)):
        waterYears = np.vectorize(appenderNonStandard)(years, months)
    else:     
        waterYears = np.vectorize(appenderStandard)(years, months)    
    
    return waterYears

#----------------------------- Checking EWRs -----------------------------------#

def ewrCheck(minThreshold, maxThreshold, duration, minEvent, gapTolerance, flow, eventList, 
             yearlyDict, year, noEventCounter, noEventList, gapToleranceTracker, fullEvent):
    '''Checks daily flow against ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list and event dictionary'''
    if ((flow >= minThreshold) and (flow <= maxThreshold)):
        eventList.append(flow)
        fullEvent.append(flow)
        gapToleranceTracker = gapTolerance # reset the gapTolerance after threshold is reached
    else:
        if gapToleranceTracker > 0:
            gapToleranceTracker = gapToleranceTracker - 1
            fullEvent.append(flow)
        else:
            if len(eventList) >= minEvent:
                yearlyDict[year].append(eventList)
                noEventList.append(noEventCounter-len(fullEvent))
            eventList = []
            fullEvent = []
    noEventCounter = noEventCounter + 1

    return eventList, yearlyDict, noEventCounter, noEventList, gapToleranceTracker, fullEvent

def lowFlowCheck(minThreshold, maxThreshold, flow, eventList, yearlyDict, year, noEventCounter, noEventList):
    '''Checks daily flow against the low flow ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list and event dictionary'''    
    if ((flow >= minThreshold) and (flow <= maxThreshold)):
        eventList.append(flow)
    else:
        if len(eventList) > 0:
            yearlyDict[year].append(eventList)
        #-----------Days between events section---------------#
            noEventCounter = noEventCounter - len(eventList) # count to the start of the event
            noEventList.append(noEventCounter)
            noEventCounter = 1 # revert the counter back to one as we are into the next phase
        else:
            noEventCounter = noEventCounter + 1 # gap between events grows by 1
        #-----------End dats between events section----------#
        eventList = []
        
    return eventList, yearlyDict, noEventCounter, noEventList

def ctfCheck(threshold, flow, eventList, yearlyDict, year, noEventCounter, noEventList):
    '''Checks daily flow against the cease to flow ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list and event dictionary'''  
    if flow <= threshold:
        eventList.append(flow)
    else:
        if len(eventList) > 0:
            yearlyDict[year].append(eventList)
        #----------Days between events section----------#
            noEventCounter = noEventCounter - len(eventList)
            noEventList.append(noEventCounter)
            noEventCounter = 1
        else:
            noEventCounter = noEventCounter + 1
        #--------End dats between events section--------#
        eventList = []
    
    return eventList, yearlyDict, noEventCounter, noEventList

def levelCheck(threshold, height, maxDrawdown, deltaHeight, duration, eventList, yearlyDict, 
               year, noEventCounter, noEventList):
    '''Checks daily level against the ewr level threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list and event dictionary'''      
    if ((height >= threshold) and (deltaHeight <= maxDrawdown)):
        eventList.append(height)
    else:
        if (len(eventList) >= duration):
            yearlyDict[year].append(eventList)
        #----Days between events section----#
            noEventCounter = noEventCounter - len(eventList)
            noEventList.append(noEventCounter)
            noEventCounter = 1
            eventList = []
        else:
            noEventCounter = noEventCounter + 1
            eventList = []
        #------End dates between events section-------# 

    return eventList, yearlyDict, noEventCounter, noEventList

def simultaneousEwrCheck(minThreshold, minThreshold2, maxThreshold, maxThreshold2, duration, duration2, minEvent, minEvent2, flow, flow2, eventList, 
             yearlyDict, year, noEventCounter, noEventList):
    '''Checking flow EWRs with simultaneous requirements '''
    if ((flow >= minThreshold) and (flow <= maxThreshold) and (flow2 >= minThreshold2) and (flow2 <= maxThreshold2)):
        eventList.append(flow)
    else:
        if len(eventList) >= minEvent:
            yearlyDict[year].append(eventList)
        eventList = []

    return eventList, yearlyDict, noEventCounter, noEventList

def simultaneousLowFlowCheck(minThreshold, minThreshold2, maxThreshold, maxThreshold2, flow, flow2, eventList, eventList2, yearlyDict, yearlyDict2, year, noEventCounter, noEventList):
    '''Perform a check to see if the low flow requirements have been met for both of the locations'''
    # Check the first location:
    if ((flow >= minThreshold) and (flow <= maxThreshold)):
        eventList.append(flow)
    else:
        if len(eventList) > 0:
            yearlyDict[year].append(eventList)
        #-----------Days between events section---------------#
            noEventCounter = noEventCounter - len(eventList) # count to the start of the event
            noEventList.append(noEventCounter)
            noEventCounter = 1 # revert the counter back to one as we are into the next phase
        else:
            noEventCounter = noEventCounter + 1 # gap between events grows by 1
        #-----------End dats between events section----------#
        eventList = []
        
    # Check the second location:
    if ((flow2 >= minThreshold2) and (flow2 <= maxThreshold2)):
        eventList2.append(flow)
    else:
        if len(eventList2) > 0:
            yearlyDict2[year].append(eventList)
        #-----------Days between events section---------------#
            noEventCounter = noEventCounter - len(eventList2) # count to the start of the event
            noEventList.append(noEventCounter)
            noEventCounter = 1 # revert the counter back to one as we are into the next phase
        else:
            noEventCounter = noEventCounter + 1 # gap between events grows by 1
        #-----------End dats between events section----------#
        eventList2 = []       
        
    
    return eventList, eventList2, yearlyDict, yearlyDict2, noEventCounter, noEventList

def simultaneousCtfCheck(threshold, threshold2, flow, flow2, eventList, eventList2, yearlyDict, yearlyDict2, year, noEventCounter, noEventList):
    '''Simultaneous ctf check'''
    # Check first location:
    if flow <= threshold:
        eventList.append(flow)
    else:
        if len(eventList) > 0:
            yearlyDict[year].append(eventList)
        #----------Days between events section----------#
            noEventCounter = noEventCounter - len(eventList)
            noEventList.append(noEventCounter)
            noEventCounter = 1
        else:
            noEventCounter = noEventCounter + 1
        #--------End dats between events section--------#
        eventList = []
        
    # Check second location:
    if flow2 <= threshold2:
        eventList2.append(flow2)
    else:
        if len(eventList2) > 0:
            yearlyDict2[year].append(eventList2)
        #----------Days between events section----------#
            noEventCounter = noEventCounter - len(eventList2)
            noEventList.append(noEventCounter)
            noEventCounter = 1
        else:
            noEventCounter = noEventCounter + 1
        #--------End dats between events section--------#
        eventList2 = []    
    
    return eventList, eventList2, yearlyDict, yearlyDict2, noEventCounter, noEventList

#---------------------- Functions for handling and distributing the different EWR categories -----------------------#

def complexEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict):
    '''Handling complex EWRs (complex EWRs are hard coded into the tool).
    returns a dataframe yearly results for each ewr within the planning unit'''
    # Get EWRs:
    ewr_dict = dict()
    ewr_dict = getComplexEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)
    # Slice relevant time from flow dataframe:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'], 
                                        ewr_dict['end_month'],
                                        input_df)  
    # Append water year column to the dataframe:
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    # Save relevant columns to variables to iterate through:
    flowSeries = input_df_timeslice[gauge_ID].values 
    waterYears = sorted(set(wySeries))
    # Check the flow timeseries against the ewr
    if calculationType == 'flowDurPostReq':
        eventDict = postDurationCalc(ewr_dict, flowSeries, wySeries, waterYears)
        statsRequest = ['number of events', 'years with events']
        stats = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)
        
    elif calculationType == 'flowDurOutsideReq':
        eventDict = outsideDurationCalc(ewr_dict, flowSeries, wySeries)
        statsRequest = ['number of events', 'years with events']
        PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)
    
    return PU_df

def flowEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict):
    '''Handling standard flow type EWRs.
    returns a dataframe yearly results for each ewr within the planning unit'''
    # Get ewr details:
    ewr_dict = dict()
    ewr_dict = getEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)  
    # Extract relevant timeslice of flows for the location:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'],
                                        ewr_dict['end_month'],
                                        input_df)
    # Append water year column to the dataframe:
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    # Save the relevant flow timeseries columns to variables:
    flowSeries = input_df_timeslice[gauge_ID].values
    dateSeries = input_df_timeslice.index
    waterYears = sorted(set(wySeries))
    # Check the flow timeseries against the ewr
    if calculationType == 'flow':
        if ((ewr_dict['start_month'] == 7) and (ewr_dict['end_month'] == 6)):
            eventDict, noneList = flowCalcAnytime(ewr_dict, flowSeries, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events', 'average length of events'] # , 'average time between events'
            PU_df = resultsStatsFlow(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears, noneList)
        else:
            eventDict, noneList = flowCalc(ewr_dict, flowSeries, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events', 'average length of events'] #, 'average time between events'
            PU_df = resultsStatsFlow(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears, noneList)
    return PU_df 
        
def lowFlowEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict, climate_file):
    '''Handling low flow type EWRs
    returns a dataframe yearly results for each ewr within the planning unit'''        
    # Get ewr details:
    ewr_dict = dict()
    ewr_dict = getLowFlowEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)
    # Extract relevant timeslice of flows for the location:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'], 
                                        ewr_dict['end_month'],
                                        input_df)  
    # Append water year column to the dataframe:
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    # Allocate climate to flow timeseries:
    catchment_name = data_inputs.gauge_to_catchment(gauge_ID) # catchment name for checking climate
    climateSeries = data_inputs.wy_to_climate(input_df_timeslice, catchment_name, climate_file)  
    # Save the relevant flow timeseries columns to variables:
    flowSeries = input_df_timeslice[gauge_ID].values
    waterYears = sorted(set(wySeries))        
    
    # Check the flow timeseries against the ewr
    if calculationType == 'low flow':
        PU_df = lowFlowCalc(ewr_dict, flowSeries, wySeries, climateSeries, PU_df, ewr, gauge_ID)
        
    return PU_df
        
def cfFlowEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict, climate_file):
    '''Handling cease to flow type EWRs
    returns a dataframe yearly results for each ewr within the planning unit'''  
    ewr_dict = dict()
    ewr_dict = getCtfEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)
    
    # Extract relevant timeslice of flows for the location:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'],
                                        ewr_dict['end_month'],
                                        input_df)   
    # Append water year column to the dataframe:
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    
    # Allocate climate to flow timeseries:
    catchment_name = data_inputs.gauge_to_catchment(gauge_ID) # catchment name for checking climate
    
    # Put the climate on the dataseries
    climateSeries = data_inputs.wy_to_climate(input_df_timeslice, catchment_name, climate_file)     
    
    # Save the relevant flow timeseries columns to variables:
    flowSeries = input_df_timeslice[gauge_ID].values
    waterYears = sorted(set(wySeries))
    # Check the flow timeseries against the ewr
    if calculationType == 'cease to flow':
        PU_df = cfCalc(ewr_dict, flowSeries, wySeries, climateSeries, PU_df, ewr)
    
    return PU_df    

def cumulVolEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict):
    '''Handling cumulative volume type EWRs
    returns a dataframe yearly results for each ewr within the planning unit'''
    # Get ewr details:
    ewr_dict = dict()
    ewr_dict = getCumulVolEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)
    # Extract relevant timeslice of flows for the location:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'],
                                        ewr_dict['end_month'],
                                        input_df)
    # Append water year column to the dataframe:
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    # Save the relevant flow timeseries columns to variables:
    flowSeries = input_df_timeslice[gauge_ID].values
    dateSeries = input_df_timeslice.index
    waterYears = sorted(set(wySeries))
    # Check the flow timeseries against the ewr
    if calculationType == 'cumulative volume':
        if ((ewr_dict['start_month'] == 7) and (ewr_dict['end_month'] == 6)):
            eventDict, noneList = cumulVolCalcAnytime(ewr_dict, flowSeries, wySeries)
            statsRequest = ['number of events', 'years with events', ]
            PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)
        else:
            eventDict, noneList = cumulVolCalc(ewr_dict, flowSeries, wySeries)
            statsRequest = ['number of events', 'years with events']
            PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)              
    
    return PU_df

def lakeEWRhandler(calculationType, planningUnit,  gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict):
    '''Handling lake type EWRs.
    returns a dataframe yearly results for each ewr within the planning unit'''
    # Get ewr details:
    ewr_dict = dict()
    ewr_dict = getLakeEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)
    # Extract relevant timeslice of flows for the location:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'],
                                        ewr_dict['end_month'],
                                        input_df) 
    # Append water year column to the dataframe:
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    # Save the relevant flow timeseries columns to variables:
    heightSeries = input_df_timeslice[gauge_ID].values
    waterYears = sorted(set(wySeries))
    # Check the flow timeseries against the ewr
    if calculationType == 'lake level':
        eventDict = lakeLevelCalc(ewr_dict, heightSeries, wySeries)
        statsRequest = ['number of events', 'years with events', 'average length of events']
        PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)   
    
    return PU_df
    
def wpEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict, climate_file): 
    '''Handling of weirpool ewrs.
    returns a dataframe yearly results for each ewr within the planning unit'''
    ewr_dict = dict()
    ewr_dict = getWPewrs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)
    # Extract relevant timeslice of flows for the location:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'], 
                                        ewr_dict['end_month'],
                                        input_df)
    # Append water year column to the dataframe:
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    # Allocate climate to flow timeseries:
    catchment_name = data_inputs.gauge_to_catchment(gauge_ID) # catchment name for checking climate
    climateSeries = data_inputs.wy_to_climate(input_df_timeslice, catchment_name, climate_file)    
    # Save the relevant flow timeseries columns to variables:
    dateSeries = input_df_timeslice.index
    flowSeries = input_df_timeslice[gauge_ID].values
    waterYears = sorted(set(wySeries))   
    
    try:
        heightSeries = input_df_timeslice[ewr_dict['weirpoolGauge']]
        # Check the flow timeseries against the ewr
        if calculationType == 'weir pool':
            eventDict = wpCalc(ewr_dict, flowSeries, heightSeries, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events']
            PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)
        
        return PU_df
    
    except KeyError:
        print('''Cannot evaluate this weirpool ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge_ID, ewr, ewr_dict['weirpoolGauge']))
        
        return PU_df

def nestEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict):
    '''Handling Nest type EWRs.
    returns a dataframe yearly results for each ewr within the planning unit'''
    # Get ewr details:
    ewr_dict = dict()
    ewr_dict = getNestEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)
    # Extract relevant timeslice of flows for the location:
    if ((ewr_dict['start_day'] == None) and (ewr_dict['end_day'] == None)):
        input_df_timeslice = get_month_mask(ewr_dict['start_month'],
                                            ewr_dict['end_month'],
                                            input_df)
    else:
        input_df_timeslice = get_day_month_mask(ewr_dict['start_day'],
                                            ewr_dict['start_month'],
                                            ewr_dict['end_day'],
                                            ewr_dict['end_month'],
                                            input_df)   
    # Append water year column to the dataframe:
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    # Save the relevant flow timeseries columns to variables:
    flowSeries = input_df_timeslice[gauge_ID].values
    dateSeries = input_df_timeslice.index
    waterYears = sorted(set(wySeries))
    # Check the flow timeseries against the ewr
    if calculationType == 'nest':
        if ((ewr_dict['triggerDay'] != None) and (ewr_dict['triggerMonth'] != None)): 
            eventDict = nestCalcTrigger(ewr_dict, flowSeries, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events']
            PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)  
        elif ((ewr_dict['triggerDay'] == None) and (ewr_dict['triggerMonth'] == None)):
            eventDict = nestCalc(ewr_dict, flowSeries, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events']
            PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)
    
    return PU_df

def multiGaugeEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict, climate_file):
    '''Handling those EWRs that require flow at two gauges summed together'''
    # Get EWRs
    ewr_dict = dict()
    ewr_dict = getMultiGaugeEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)
    # Slice relevant time from flow dataframe:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'], 
                                        ewr_dict['end_month'],
                                        input_df)      
    # Append water year column to the dataframe:
    np.set_printoptions(threshold=np.inf)
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    
    # Allocate climate to flow timeseries:
    catchment_name = data_inputs.gauge_to_catchment(gauge_ID) # catchment name for checking climate
    climateSeries = data_inputs.wy_to_climate(input_df_timeslice, catchment_name, climate_file)  
    
    # Save relevant columns to variables to iterate through:
    firstFlowSeries = input_df_timeslice[gauge_ID].values
    try:
        secondFlowSeries = input_df_timeslice[ewr_dict['second gauge']].values 
    except KeyError: # if no flow data exists for the second gauge required, let the user know its missing and return the empty results dataframe
        print('''Cannot evaluate this ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge_ID, ewr, ewr_dict['second gauge']))
        return PU_df
    flowSeries = firstFlowSeries + secondFlowSeries
    dateSeries = input_df_timeslice.index
    waterYears = sorted(set(wySeries))
    # Send to relevant function for checking depending on EWR characteristics:
    if calculationType == 'flow':
        if ((ewr_dict['start_month'] == 7) and (ewr_dict['end_month'] == 6)):
            eventDict, noneList = flowCalcAnytime(ewr_dict, flowSeries, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events', 'average length of events'] # , 'average time between events'
            PU_df = resultsStatsFlow(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears, noneList)
        else:
            eventDict, noneList = flowCalc(ewr_dict, flowSeries, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events', 'average length of events'] #, 'average time between events'
            PU_df = resultsStatsFlow(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears, noneList)
    if calculationType == 'low flow':
        PU_df = lowFlowCalc(ewr_dict, flowSeries, wySeries, climateSeries, PU_df, ewr, gauge_ID)   
    if calculationType == 'cease to flow':
        PU_df = cfCalc(ewr_dict, flowSeries, wySeries, climateSeries, PU_df, ewr)
    if calculationType == 'cumulative volume':
        if ((ewr_dict['start_month'] == 7) and (ewr_dict['end_month'] == 6)):
            eventDict, noneList = cumulVolCalcAnytime(ewr_dict, flowSeries, wySeries)
            statsRequest = ['number of events', 'years with events']
            PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)
        else:
            eventDict, noneList = cumulVolCalc(ewr_dict, flowSeries, wySeries)
            statsRequest = ['number of events', 'years with events']
            PU_df = resultsStats(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears)  
        
    return PU_df

def simultaneousEWRhandler(calculationType, planningUnit, gauge_ID, ewr, ewrTableExtract, input_df, PU_df, toleranceDict, ewr_table, climate_file):
    '''For those EWRs that need to be met at two locations at the same time'''
    # Get EWRs
    ewr_dict = dict()
    ewr_dict = getSimultaneousEWRs(planningUnit, gauge_ID, ewr, ewrTableExtract, toleranceDict)    
    # extract the other other gauge from the master ewr table:
    gauge2table = ewr_table[ewr_table['gauge'] == ewr_dict['second gauge']]
    gauge2table = gauge2table[gauge2table['PlanningUnitID'] == planningUnit]
    
    ewr_dict2 = getSimultaneousEWRs(planningUnit, ewr_dict['second gauge'], ewr, gauge2table, toleranceDict)
    # Slice relevant time from flow dataframe:
    input_df_timeslice = get_month_mask(ewr_dict['start_month'], 
                                        ewr_dict['end_month'],
                                        input_df) # The date range must be the same for the simultaneous gauges. The script just uses the date range from the first
    # Get a daily water year series
    wySeries = waterYear_daily(input_df_timeslice, ewr_dict)
    waterYears = sorted(set(wySeries))
    # Get the catchment name, and then create a series with a daily climate category for the catchment
    catchment_name = data_inputs.gauge_to_catchment(gauge_ID) # catchment name for checking climate
    climateSeries = data_inputs.wy_to_climate(input_df_timeslice, catchment_name, climate_file)     
    # Save relevant columns to variables to iterate through:
    flowSeries = input_df_timeslice[gauge_ID].values 
    dateSeries = input_df_timeslice.index
    try:
        flowSeries2 = input_df_timeslice[ewr_dict['second gauge']].values
    except KeyError: # if no flow data exists for the second gauge required, let the user know its missing and return the empty results dataframe
        print('''Cannot evaluate this ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge_ID, ewr, ewr_dict['second gauge']))
        return PU_df
        
    # Send to relevant function for checking depending on EWR characteristics. So far there have only been three categories of simultaneous ewrs, with the options below:
    if calculationType == 'cease to flow':
        PU_df = simultaneousCfCalc(ewr_dict, ewr_dict2, flowSeries, flowSeries2, wySeries, climateSeries, PU_df, ewr)
    if calculationType == 'low flow':
        PU_df = simultaneousLowFlowCalc(ewr_dict, ewr_dict2, flowSeries, flowSeries2, wySeries, climateSeries, PU_df, ewr, gauge_ID)
    if calculationType == 'flow':
        if ((ewr_dict['start_month'] == 7) and (ewr_dict['end_month'] == 6)):
            eventDict, noneList = simultaneousFlowCalcAnytime(ewr_dict, ewr_dict2, flowSeries, flowSeries2, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events', 'average length of events'] #, 'average time between events'
            PU_df = resultsStatsFlow(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears, noneList)
        else:
            eventDict, noneList = simultaneousFlowCalc(ewr_dict, ewr_dict2, flowSeries, flowSeries2, wySeries, dateSeries)
            statsRequest = ['number of events', 'years with events', 'average length of events'] #, 'average time between events'
            PU_df = resultsStatsFlow(statsRequest, eventDict, ewr_dict, PU_df, ewr, waterYears, noneList)
    
    return PU_df


#------------------- Main calculation functions -----------------#    

def nestCalcTrigger(ewr_dict, flowSeries, wySeries, dateSeries):
    '''Calculate the Nest style EWRs with a trigger day (i.e. if the flow is between x and y ML/day apply this ewr check).    
    '''
    # Decalre variables:
    eList = []
    eventDict = {}
    noneCounter = 0
    noneList = []
    linesToSkip = 0
    gapToleranceTracker = 0
    # Saving dates as nparrays to increase speed of processing:
    days = dateSeries.day.values
    months = dateSeries.month.values    

    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    # Cycle through the flow timeseries:    
    for iteration, flow in enumerate(flowSeries):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            wyCurrent = wySeries[iteration]  
            # if there is is a date trigger:
            if ((int(days[iteration]) == int(ewr_dict['triggerDay'])) and (int(months[iteration]) == int(ewr_dict['triggerMonth']))):
                if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF'])):
                    futureSubset = flowSeries[iteration:(iteration+ewr_dict['duration'])]
                    if 'weirpoolGauge' in ewr_dict.keys():
                        eList = [] 
                        for i, value in enumerate(futureSubset):
                            if i != len(futureSubset)-1:
                                check = futureSubset[i]-futureSubset[i+1]
                                eList.append(check)
                        if ((max(eList) <= ewr_dict['maxDrawdown']) and (min(futureSubset) >= ewr_dict['minThresholdF']) and (max(futureSubset) <= ewr_dict['maxThresholdF'])):
                            eventDict[wyCurrent].append(eList)
                            linesToSkip = len(eList)
                    elif '%' in ewr_dict['maxDrawdown']:
                        eList = [] 
                        for i, value in enumerate(futureSubset):
                            if i != len(futureSubset)-1:
                                check = ((futureSubset[i]-futureSubset[i+1])/futureSubset[i])*100
                                eList.append(check)
                        if ((max(eList) <= int(ewr_dict['maxDrawdown'][:-1])) and (min(futureSubset) >= ewr_dict['minThresholdF']) and (max(futureSubset) <= ewr_dict['maxThresholdF'])):
                            eventDict[wyCurrent].append(futureSubset)
                            linesToSkip = ewr_dict['duration']                              
    return eventDict

def nestCalc(ewr_dict, flowSeries, wySeries, dateSeries):
    '''Calculate the standard Nest style EWRs'''
    # Decalre variables:
    eventDict = {}
    noneCounter = 0
    noneList = []
    linesToSkip = 0
    gapToleranceTracker = 0
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []

    # Cycle through the flow timeseries:    
    for iteration, flow in enumerate(flowSeries[:-ewr_dict['duration']]):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            wyCurrent = wySeries[iteration] # check for transformation    

            if wySeries[iteration] == wySeries[iteration+ewr_dict['duration']-1]:
                if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF'])):
                    futureSubset = flowSeries[iteration:(iteration+ewr_dict['duration'])]
                    if 'weirpoolGauge' in ewr_dict.keys():
                        eList = []
                        for i, value in enumerate(futureSubset):
                            if i != len(futureSubset)-1:
                                check = futureSubset[i]-futureSubset[i+1]
                                eList.append(check)
                        if ((max(eList) <= ewr_dict['maxDrawdown']) and (min(futureSubset) >= ewr_dict['minThresholdF']) and (max(futureSubset) <= ewr_dict['maxThresholdF'])):
                            eventDict[wyCurrent].append(futureSubset)
                            linesToSkip = ewr_dict['duration']
                    elif '%' in ewr_dict['maxDrawdown']:
                        eList = [] 
                        for i, value in enumerate(futureSubset):
                            if i != len(futureSubset)-1:
                                check = ((futureSubset[i]-futureSubset[i+1])/futureSubset[i])*100
                                eList.append(check)
                        if ((max(eList) <= int(ewr_dict['maxDrawdown'][:-1])) and (min(futureSubset) >= ewr_dict['minThresholdF']) and (max(futureSubset) <= ewr_dict['maxThresholdF'])):
                            eventDict[wyCurrent].append(futureSubset)
                            linesToSkip = ewr_dict['duration']  
            else:
                continue # Not enough time to complete the event
    return eventDict


def wpCalc(ewr_dict, flowSeries, heightSeries, wySeries, dateSeries):
    '''Evaluating the weirpool type ewrs. These require information from a flow gauge,
    river level gauge and have a drawdown rate'''
    # Decalre variables:
    eventDict = {}
    sub_days_tracker = 0
    days_tracker = 0
    noneCounter = 0
    noneList = []
    linesToSkip = 0
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Cycle through the flow timeseries
    for iteration, flow in enumerate(flowSeries[:ewr_dict['duration']]):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:           
            wyCurrent = wySeries[iteration]
            
            if wySeries[iteration] == wySeries[iteration+ewr_dict['duration']-1]:
                if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF'])):
                    if ewr_dict['wpType'] == 'wpRaising':
                        if heightSeries[iteration] >= ewr_dict['minThresholdL']:
                            # Check for length of duration into the future for raising rate
                            futureSubsetLevel = heightSeries[iteration:iteration+ewr_dict['duration']]
                            futureSubsetFlow = flowSeries[iteration:iteration+ewr_dict['duration']]
                            eList = [] 
                            for i, value in enumerate(futureSubsetLevel):
                                if i != len(futureSubsetLevel)-1:
                                    check = futureSubsetLevel[i]-futureSubsetLevel[i+1]
                                    eList.append(check)
                            # Check to see if change in level, minimum weirpool height, and flow thresholds are met over the duration:
                            if ((max(eList) <= ewr_dict['maxDrawdown']) and (min(futureSubsetLevel) >= ewr_dict['minThresholdL']) \
                                and (min(futureSubsetFlow) >= ewr_dict['minThresholdF']) and (max(futureSubsetFlow) <= ewr_dict['maxThresholdF'])):
                                eventDict[wyCurrent].append(futureSubsetLevel)
                                linesToSkip = ewr_dict['duration']
                    elif ewr_dict['wpType'] == 'wpDrawdown':
                        if heightSeries[iteration] <= ewr_dict['maxThresholdL']:
                            # Check for length of duration into the future for drawdown rate
                            futureSubsetLevel = heightSeries[iteration:iteration+ewr_dict['duration']]
                            futureSubsetFlow = flowSeries[iteration:iteration+ewr_dict['duration']]
                            eList = [] 
                            for i, value in enumerate(futureSubsetLevel):
                                if i != len(futureSubsetLevel)-1:
                                    check = futureSubsetLevel[i]-futureSubsetLevel[i+1]
                                    eList.append(check)
                            # Check to see if change in level, maximum weirpool height, and flow thresholds are met over the duration:
                            if ((max(eList) <= ewr_dict['maxDrawdown']) and (max(futureSubsetLevel) <= ewr_dict['maxThresholdL']) \
                                and (min(futureSubsetFlow) >= ewr_dict['minThresholdF']) and (max(futureSubsetFlow) <= ewr_dict['maxThresholdF'])):
                                eventDict[wyCurrent].append(futureSubsetLevel)
                                linesToSkip = ewr_dict['duration']
            else:
                continue # Not enough time left to complete the EWR requirements
    return eventDict
    
def postDurationCalc(ewr_dict, flowSeries, wySeries):
    '''Calculate when there is a requirement to meet a duration and threshold
    after the ewr has been met -  one of the 'complex EWR' requirements'''
    # Declare variables:
    noneCounter = 0
    noneList = []
    linesToSkip = 0
    gapToleranceTracker = 0
    eList = []
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Cycle through the flow timeseries:
    for iteration, flow in enumerate(flowSeries):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            wyCurrent = wySeries[iteration]      
            # Look for first part of EWR (first threshold/duration requirement):    
            if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF'])):
                eList.append(flow)
                gapToleranceTracker = ewr_dict['gapTolerance']
                if len(eList) >= ewr_dict['duration']:
                    # When part 1 is achieved, look for the second part (next threshold/duration requirement)
                    postEventList = []
                    postGapToleranceTracker = 0
                    for futureIter in flowSeries[(iteration+1):(iteration+1+ewr_dict['postEventDuration'])]:
                        if futureIter >= ewr_dict['postEventThreshold']:
                            postEventList.append(futureIter)
                            postGapToleranceTracker = ewr_dict['gapTolerance']
                            if len(postEventList) >= ewr_dict['postEventDuration']:
                                # EWR achieved for parts 1 and 2, save the combined event:
                                totalEvent = eList + postEventList
                                # Get the year in which the majority of the event falls in: 
                                # Put in function:
                                startIteration = iteration - len(eList)
                                endIteration = iteration + len(postEventList)
                                middleIteration = int((startIteration + endIteration)/2)
                                waterYearMiddle = wySeries[middleIteration]
                                if waterYearMiddle not in eventDict.keys():
                                    eventDict[waterYearMiddle] = []
                                # Save the event to this year:    
                                eventDict[waterYearMiddle].append(totalEvent)
                                # Skip over the part 2 of the event so we don't double count it:
                                linesToSkip = len(postEventList)
                        else: 
                            if postGapToleranceTracker > 0:
                                postGapToleranceTracker = postGapToleranceTracker - 1
                                postEventList.append(futureIter)
                            else:
                                postEventList = []
            else:
                if gapToleranceTracker > 0:
                    eList.append(flow)
                    gapToleranceTracker = gapToleranceTracker - 1   
                else:
                    eList = []
                    
    return eventDict
    
def outsideDurationCalc(ewr_dict, flowSeries, wySeries):
    '''Calculate when there is a requirement to meet a duration and threshold
    either before or after the ewr has been met - one of the 'complex EWR' requirements'''
    # Decalre variables:
    noneCounter = 0
    noneList = []
    linesToSkip = 0
    gapToleranceTracker, futureGapTolerance, pastGapTolerance = 0, 0, 0
    eventDict = {}
    eList = []
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Cycle through the flow timeseries:
    for iteration, flow in enumerate(flowSeries):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            wyCurrent = wySeries[iteration]        
            # Look for first part of EWR (first threshold/duration requirement):   
            if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF'])):
                eList.append(flow)
                gapToleranceTracker = ewr_dict['gapTolerance']
                if len(eList) >= ewr_dict['duration']:
                    # When part 1 is achieved, look for the second part (next threshold/duration requirement)
                    postEventList = []  
                    for futureIter in flowSeries[(iteration+1):\
                                                 (iteration+1+ewr_dict['outsideEventDuration'])]:
                        if futureIter >= ewr_dict['outsideEventThreshold']:
                            postEventList.append(futureIter)
                            futureGapTolerance = ewr_dict['gapTolerance']
                            if len(postEventList) >= ewr_dict['outsideEventDuration']:
                                # parts 1 and 2 achieved for the EWR:
                                totalEvent = eList + postEventList
                                # Get the year where the majority of the event fell:
                                # Send to function?
                                startIteration = iteration - len(eList)
                                endIteration = iteration + len(postEventList)
                                middleIteration = int((startIteration + endIteration)/2)
                                waterYearMiddle = wySeries[middleIteration]
                                if waterYearMiddle not in eventDict.keys():
                                    eventDict[waterYearMiddle] = []
                                eventDict[waterYearMiddle].append(totalEvent)
                                linesToSkip = len(postEventList)
                            else: 
                                if futureGapTolerance > 0:
                                    futureGapTolerance = futureGapTolerance - 1
                                    postEventList.append(futureIter)
                                else:
                                    postEventList = []
                    # Only trigger the backwards look if the post event didnt pass the test:
                    if len(postEventList) < ewr_dict['outsideEventDuration']:
                        preEventList = []
                        for pastIter in flowSeries[(iteration-len(eList)+1-ewr_dict['outsideEventDuration'])\
                                                   :iteration-len(eList)+1]:
                            if pastIter >= ewr_dict['outsideEventThreshold']:
                                preEventList.append(futureIter)
                                pastGapTolerance = ewr_dict['gapTolerance']
                                if len(preEventList) >= ewr_dict['outsideEventDuration']:
                                    # parts 1 and 2 achieved for the EWR:
                                    totalEvent = eList + preEventList
                                    # Get the year where the majority of the event occurred
                                    # Put in function:
                                    startIteration = iteration - len(eList)-len(preEventList)
                                    endIteration = iteration
                                    middleIteration = int((startIteration + endIteration)/2)
                                    waterYearMiddle = wySeries[middleIteration]
                                    if waterYearMiddle not in eventDict.keys():
                                        eventDict[waterYearMiddle] = []
                                    eventDict[waterYearMiddle].append(totalEvent)
                            else:
                                if pastGapTolerance > 0:
                                    eList.append(flow)
                                    pastGapTolerance = pastGapTolerance - 1 
                                else:
                                    eList = []
    
    return eventDict

def multiYearEvent(flow, ewr_dict, flowSeries, iteration, maxForward, maxBackward, wy_type):
    '''For finding when the event spanning multiple years falls and details about the event'''
    
    preEventList = []
    postEventList = []
    tracker = 0
    if wy_type == 'end':
        preEventList.append(flow) # start with this in the list
    elif wy_type == 'start':
        postEventList.append(flow)
    while True:
        tracker = tracker + 1
        if tracker == maxBackward: # only search back a year, unlikely to be activated
            break
        eventTrack = flowSeries[(iteration - tracker)]
        if ((eventTrack < ewr_dict['minThresholdF']) or (eventTrack > ewr_dict['maxThresholdF'])):
            break            
        else:
            preEventList.insert(0, eventTrack)
    tracker = 0
    while True:
        tracker = tracker + 1
        if tracker == maxForward: # only search forward a year
            break
        eventTrack = flowSeries[(iteration + tracker)]
        if ((eventTrack < ewr_dict['minThresholdF']) or (eventTrack > ewr_dict['maxThresholdF'])):
            break
        else:
            postEventList.append(eventTrack)
    eList = preEventList + postEventList
        
    return eList, preEventList, postEventList
    
def flowCalcAnytime(ewr_dict, flowSeries, wySeries, dateSeries):
    '''The calculation section for flow type ewrs with no time constraints,
    these have been seperated due to their ability to cross water year boundaries'''
    # Declare variables:
    eList = []
    eventDict = {}
    noneCounter = 0
    noneList = []
    linesToSkip = 0
    gapToleranceTracker = 0
    fullEvent = []

    # Saving dates as nparrays to increase speed of processing:
    days = dateSeries.month.values
    months = dateSeries.day.values
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Hit the first iteration outside the loop
    eList, eventDict, noneCounter, noneList, gapToleranceTracker, fullEvent = \
        ewrCheck(ewr_dict['minThresholdF'],
                 ewr_dict['maxThresholdF'],
                 ewr_dict['duration'],
                 ewr_dict['min event'],
                 ewr_dict['gapTolerance'],
                 flowSeries[0], 
                 eList,
                 eventDict,
                 wySeries[0],
                 noneCounter,
                 noneList,
                 gapToleranceTracker,
                 fullEvent
                )      
    # Cycle through the remaining flow timeseries except the last one:
    for iteration, flow in enumerate(flowSeries[1:-1]):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            wyCurrent = wySeries[iteration]      
            if (months[iteration] == 6) and (days[iteration] == 30): # last day in WY
                # Determine the max window (if the date is close to the start or end of the timeseries), otherwise leave the max window to the full year
                if iteration > len(flowSeries) - 365:
                    maxForwardWindow = len(flowSeries) - iteration
                else:
                    maxForwardWindow = 365
                if iteration < 365:
                    maxBackwardWindow = iteration
                else:
                    maxBackwardWindow = 365
                if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF'])):
                    eList, preEventList, postEventList = multiYearEvent(flow, 
                                                                        ewr_dict, 
                                                                        flowSeries, 
                                                                        iteration,
                                                                        maxForwardWindow,
                                                                        maxBackwardWindow,
                                                                        'end'
                                                                       )
                    if len(eList) >= ewr_dict['min event']:
                        if len(preEventList) >= len(postEventList):
                            eventDict[wyCurrent].append(eList)
                            linesToSkip = len(postEventList)

                eList = [] # Reset the event list at the end of the water year  

            elif (months[iteration] == 7) and (days[iteration] == 1): # start of WY
                # Determine the max window (if the date is close to the start or end of the timeseries), otherwise leave the max window to the full year
                if iteration > len(flowSeries) - 365:
                    maxForwardWindow = len(flowSeries) - iteration
                else:
                    maxForwardWindow = 365
                if iteration < 365:
                    maxBackwardWindow = iteration
                else:
                    maxBackwardWindow = 365
                if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF'])):
                    eList, preEventList, postEventList = multiYearEvent(flow, 
                                                                        ewr_dict, 
                                                                        flowSeries, 
                                                                        iteration,
                                                                        maxForwardWindow,
                                                                        maxBackwardWindow,
                                                                        'start'
                                                                       )
                    if len(eList) >= ewr_dict['min event']:
                        if len(preEventList) < len(postEventList):
                            eventDict[wyCurrent].append(eList)
                            linesToSkip = len(postEventList)-1
                eList = []
            else:
                eList, eventDict, noneCounter, noneList, gapToleranceTracker, fullEvent = \
                ewrCheck(ewr_dict['minThresholdF'],
                         ewr_dict['maxThresholdF'],
                         ewr_dict['duration'],
                         ewr_dict['min event'],
                         ewr_dict['gapTolerance'],
                         flow, 
                         eList,
                         eventDict,
                         wyCurrent,
                         noneCounter,
                         noneList,
                         gapToleranceTracker,
                         fullEvent
                        )
    
    # Hit the final iteration
    eList, eventDict, noneCounter, noneList, gapToleranceTracker, fullEvent = \
        ewrCheck(ewr_dict['minThresholdF'],
                 ewr_dict['maxThresholdF'],
                 ewr_dict['duration'],
                 ewr_dict['min event'],
                 ewr_dict['gapTolerance'],
                 flowSeries[-1], 
                 eList,
                 eventDict,
                 wySeries[-1],
                 noneCounter,
                 noneList,
                 gapToleranceTracker,
                 fullEvent
                )
    if len(eList) >= ewr_dict['min event']:
        eventDict[wyCurrent].append(eList) 
    
    return eventDict, noneList
    
def flowCalc(ewr_dict, flowSeries, wySeries, dateSeries):
    '''For standard flow EWRs with time constraints'''
    # Decalre variables:
    eList = []
    eventDict = {}
    noneCounter = 0
    noneList = []
    linesToSkip = 0
    gapToleranceTracker = 0
    fullEvent = []
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Cycle through the flow timeseries:    
    for iteration, flow in enumerate(flowSeries[:-1]):
        wyCurrent = wySeries[iteration] # check for transformation    
        # Check ewr completion:
        eList, eventDict, noneCounter, noneList, gapToleranceTracker, fullEvent =\
                ewrCheck(ewr_dict['minThresholdF'],
                         ewr_dict['maxThresholdF'],
                         ewr_dict['duration'],
                         ewr_dict['min event'],
                         ewr_dict['gapTolerance'],
                         flow,
                         eList,
                         eventDict,
                         wyCurrent,
                         noneCounter,
                         noneList,
                         gapToleranceTracker,
                         fullEvent
                        )
        if wySeries[iteration] != wySeries[iteration+1]:
            if len(eList) >= ewr_dict['min event']: 
                eventDict[wyCurrent].append(eList)
            eList = [] # reset the event list at the end of the water year

    # Check ewr completion for last day:
    eList, eventDict, noneCounter, noneList, gapToleranceTracker, fullEvent =\
            ewrCheck(ewr_dict['minThresholdF'],
                     ewr_dict['maxThresholdF'],
                     ewr_dict['duration'],
                     ewr_dict['min event'],
                     ewr_dict['gapTolerance'],
                     flowSeries[-1], 
                     eList,
                     eventDict,
                     wySeries[-1],
                     noneCounter,
                     noneList,
                     gapToleranceTracker,
                     fullEvent
                    )
    if len(eList) >= ewr_dict['min event']: 
        eventDict[wyCurrent].append(eList)
    #--- End last day check ---#
    
    return eventDict, noneList

def lowFlowCalc(ewr_dict, flowSeries, wySeries, climateSeries, PU_df, ewr, gauge):
    '''For calculating low flow ewrs. These have no consecutive requirement on their durations'''        
    # Decalre variables:
    eList = []
    eventDict = {}
    noneCounter = 0
    noneList = []                 
    yearsWithEvents = []
    avLowFlowDays = []
    waterYears = sorted(set(wySeries))
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    # Cycle through the flow timeseries:
    for iteration, flow in enumerate(flowSeries[:-1]): 
        wyCurrent = wySeries[iteration] # check for transformation   
        # EWR check:
        eList, eventDict, noneCounter, noneList = \
            lowFlowCheck(ewr_dict['minThresholdF'],
                         ewr_dict['maxThresholdF'],
                             flow,
                             eList,
                             eventDict,
                             wyCurrent,
                             noneCounter,
                             noneList
                            )
        if wySeries[iteration] != wySeries[iteration+1]:
            if ((climateSeries[iteration] == 'Very Dry') and (ewr_dict['veryDry_duration'] !=None)):
                ewr_duration = ewr_dict['veryDry_duration']
            else:
                ewr_duration = ewr_dict['duration']
            # If there are more elements in the final list, save them to the results dictionary:    
            if len(eList) > 0:
                eventDict[wyCurrent].append(eList)
                noneCounter = noneCounter - len(eList)#count to start of event
                noneList.append(noneCounter)
                noneCounter = 1
            # Check year for event, add result to the list
            calcList = ['years with events','number of low flow days']
            yearsWithEvents, avLowFlowDays = resultsStatsLowFlow(calcList,
                                                                 eventDict, 
                                                                 wyCurrent, 
                                                                 yearsWithEvents, 
                                                                 avLowFlowDays,
                                                                 ewr_duration)
            eList = [] # Reset at the end of the water year
            
    #--- Check final iteration: ---#
    eList, eventDict, noneCounter, noneList = \
        lowFlowCheck(ewr_dict['minThresholdF'],
                     ewr_dict['maxThresholdF'],
                         flowSeries[-1],
                         eList,
                         eventDict,
                         wySeries[-1],
                         noneCounter,
                         noneList
                        )      
    if ((climateSeries[-1] == 'Very Dry') and \
        (ewr_dict['veryDry_duration'] !=None)):
        ewr_duration = ewr_dict['veryDry_duration']
    else:
        ewr_duration = ewr_dict['duration']
    # If there are more elements in the final list, save them to the results dictionary:  
    if len(eList) > 0:
        eventDict[wySeries[-1]].append(eList)
        noneCounter = noneCounter - len(eList)
        noneList.append(noneCounter)
        noneCounter = 1
    #--- End checking final iteration ---#
    
    # Check year for event, add result to the list
    calcList = ['years with events','number of low flow days']
    yearsWithEvents, avLowFlowDays = resultsStatsLowFlow(calcList,
                                                         eventDict, 
                                                         wySeries[-1], 
                                                         yearsWithEvents, 
                                                         avLowFlowDays,
                                                         ewr_duration)
    
    addSeries = pd.Series(yearsWithEvents)
    addSeries.index = waterYears
    
    addSeries1 =  pd.Series(avLowFlowDays)
    addSeries1.index = waterYears
    
    # Average length of time between events:
#     averageBetween = sum(noneList)/len(waterYears)
#     addSeries2 = [averageBetween] * len(waterYears)
    
    # Saving the results to the dataframe: 
    PU_df[str(ewr + '_eventYears')] = addSeries
    PU_df[str(ewr + '_avLowFlowDays')] = addSeries1
#     PU_df[str(ewr + '_avDaysBetween')] = addSeries2
    
    return PU_df

def cfCalc(ewr_dict, flowSeries, wySeries, climateSeries, PU_df, ewr):
    '''For calculating cease to flow type ewrs'''
    # Decalre variables:
    eList = []
    eventDict = {}
    noneCounter = 0
    noneList = []
    counter = 0
    overallCounter = 0
    eventCounter = 0   
    #Variables to save yearly results to:
    yearsWithEvents = []
    numEvents = []
    ctfDaysPerYear = []
    avLenCtfSpells = []
    waterYears = sorted(set(wySeries))
    # Cycle through the flow timeseries:
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    for iteration, flow in enumerate(flowSeries[:-1]): 
        wyCurrent = wySeries[iteration] # check for transformation 
        eList, eventDict, noneCounter, noneList = ctfCheck(ewr_dict['minThresholdF'],
                                                    flow,
                                                    eList,
                                                    eventDict,
                                                    wyCurrent,
                                                    noneCounter,
                                                    noneList)                                  
        if wySeries[iteration] != wySeries[iteration+1]:
            if ((climateSeries[iteration] == 'Very Dry') and (ewr_dict['veryDry_duration'] !=None)):
                ewr_duration = ewr_dict['veryDry_duration']
            else:
                ewr_duration = ewr_dict['duration']
            # Add in the current event list:
            if len(eList) > 0:
                eventDict[wyCurrent].append(eList)
            # Send to get evaluated:  
            yearsWithEvents,numEvents,ctfDaysPerYear,avLenCtfSpells=resultsStatsCF(eventDict, 
                                                                                   wyCurrent,
                                                                                   yearsWithEvents,
                                                                                   numEvents,
                                                                                   ctfDaysPerYear,
                                                                                   avLenCtfSpells,
                                                                                   ewr_duration)
            eList = []
            
    #--- Handling the final element in the series ---#
    eList, eventDict, noneCounter, noneList = ctfCheck(ewr_dict['minThresholdF'],
                                                flowSeries[-1],
                                                eList,
                                                eventDict,
                                                wySeries[-1],
                                                noneCounter,
                                                noneList)     
    
    if ((climateSeries[-1] == 'Very Dry') and \
        (ewr_dict['veryDry_duration'] !=None)):
        ewr_duration = ewr_dict['veryDry_duration']
    else:
        ewr_duration = ewr_dict['duration']
    if len(eList) > 0:
        eventDict[wySeries[-1]].append(eList)
    # Send to get evaluated:
    yearsWithEvents,numEvents,ctfDaysPerYear,avLenCtfSpells=resultsStatsCF(eventDict, 
                                                                           wySeries[-1],
                                                                           yearsWithEvents,
                                                                           numEvents,
                                                                           ctfDaysPerYear,
                                                                           avLenCtfSpells,
                                                                           ewr_duration)
    #--- End handling the final element ---#
    addSeries = pd.Series(yearsWithEvents)
    addSeries.index = waterYears
    
    addSeries1 =  pd.Series(numEvents)
    addSeries1.index = waterYears
    
    addSeries2 = pd.Series(ctfDaysPerYear)
    addSeries2.index = waterYears
    
    addSeries3 =  pd.Series(avLenCtfSpells)
    addSeries3.index = waterYears

    # Average length of time between events:
#     averageBetween = sum(noneList)/len(waterYears)
#     addSeries4 = [averageBetween] * len(waterYears)
    
    # Save results to dataframe:
    PU_df[str(ewr + '_eventYears')] = addSeries
    PU_df[str(ewr + '_numEvents')] = addSeries1
    PU_df[str(ewr + '_ctfDaysPerYear')] = addSeries2
    PU_df[str(ewr + '_avLenCtfSpells')] = addSeries3   
#     PU_df[str(ewr + '_avDaysBetween')] = addSeries4
    
    return PU_df

def cumulVolCalcAnytime(ewr_dict, flowSeries, wySeries):
    '''Calculates cumulative volume EWRs with no time constraints'''
    # Decalre variables:
    eList = []
    eventDict = {}
    linesToSkip = 0
    noneCounter = 0
    noneList = []
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []

    # Cycle through the flow timeseries:
    for iteration, flow in enumerate(flowSeries[:-ewr_dict['duration']+1]):
        wyCurrent = wySeries[iteration]
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            subset = flowSeries[iteration:iteration+(ewr_dict['duration']+1)] 
            subsetSum = sum(filter(lambda x: x>=ewr_dict['minThresholdF'], subset)) #Only sum those values over a certain min flow threshold, if none exists, defaults to 0 ML/Day as the minimum
            if subsetSum >= ewr_dict['minThresholdV']:
                # Work out the middle of the event:
                middleIteration = ((iteration+iteration+(ewr_dict['duration']+1))/2)
                middleWY = wySeries[iteration]
                eventDict[middleWY].append(subset)
                linesToSkip = len(subset)
                noneList.append(noneCounter)
                noneCounter = 0
            else:
                noneCounter += 1
    # then iterate over the last of the flow series
    finalFlowSeries = flowSeries[-ewr_dict['duration']:]
    finalWYseries = wySeries[-ewr_dict['duration']:]
    for iteration, flow in enumerate(finalFlowSeries):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            wyCurrent = finalWYseries[iteration]
            # Only search to the end of the subset:
            subset=finalFlowSeries[iteration:]
            subsetSum = sum(filter(lambda x: x>=ewr_dict['minThresholdF'], subset)) # Only sum those values over a certain min flow threshold, if none exists, defaults to 0 ML/Day as the minimum
            if subsetSum >= ewr_dict['minThresholdV']:
                eventDict[wyCurrent].append(subset)
                linesToSkip = len(subset)
                noneList.append(noneCounter)
                noneCounter = 0
            else:
                noneCounter += 1
    return eventDict, noneList

def cumulVolCalc(ewr_dict, flowSeries, wySeries):
    '''Calculates cumulative volume EWRs with time constraints'''
    # Decalre variables:
    eList = []
    eventDict = {}
    linesToSkip = 0
    noneList = []
    noneCounter = 0
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Cycle through the flow timeseries, up until the final duration check:
    for iteration, flow in enumerate(flowSeries[:-ewr_dict['duration']]):
        wyCurrent = wySeries[iteration]
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            # only subset as far as the end of the water year.
            # If we are approaching the end of the water year, start reducing the size of the subset:
            if wySeries[iteration] != wySeries[iteration+ewr_dict['duration']]:
                for futureI in range(iteration, (iteration + ewr_dict['duration']), 1):
                    if wySeries[futureI] != wySeries[futureI+1]:
                        endOfSubset = futureI
                        break
                subset = flowSeries[iteration:(endOfSubset+1)] # flowSeries[iteration:(iteration+endOfSubset+1)]
                subsetSum = sum(filter(lambda x: x>=ewr_dict['minThresholdF'], subset)) # Only sum those values over a certain min flow threshold, if none exists, defaults to 0 ML/Day as the minimum
                if subsetSum >= ewr_dict['minThresholdV']:
                    eventDict[wyCurrent].append(subset)
                    linesToSkip = len(subset)
                    noneList.append(noneCounter)
                    noneCounter = 0
                else:
                    noneCounter += 1  
            else:
                subset=flowSeries[iteration:(iteration+ewr_dict['duration']+1)]
                subsetSum = sum(filter(lambda x: x>=ewr_dict['minThresholdF'], subset)) # Only sum those values over a certain min flow threshold, if none exists, defaults to 0 ML/Day as the minimum
                if subsetSum >= ewr_dict['minThresholdV']:
                    eventDict[wyCurrent].append(subset)
                    linesToSkip = len(subset)
                    noneList.append(noneCounter)
                    noneCounter = 0
                else:
                    noneCounter += 1                    
    
    # then iterate over the last of the flow series
    finalFlowSeries = flowSeries[-ewr_dict['duration']:]
    for iteration, flow in enumerate(finalFlowSeries):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            # Only search to the end of the subset:
            subset=finalFlowSeries[iteration:]
            subsetSum = sum(filter(lambda x: x>=ewr_dict['minThresholdF'], subset)) # Only sum those values over a certain min flow threshold, if none exists, defaults to 0 ML/Day as the minimum
            if subsetSum >= ewr_dict['minThresholdV']:
                eventDict[wyCurrent].append(subset)
                linesToSkip = len(subset)
                noneList.append(noneCounter)
                noneCounter = 0
            else:
                noneCounter += 1  
    
    return eventDict, noneList


def lakeLevelCalc(ewr_dict, heightSeries, wySeries):
    '''For use with lakes including the Menindee Lakes''' 
    # Decalre variables:
    eList = []
    eventDict = {}
    noneCounter = 0
    noneList = []
    eventDict = {}
    linesToSkip = 0
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Hit the first element in the series outside the loop:
    deltaHeight = 0 # First change define as 0
    
    eList, eventDict, noEventCounter, noEventList = \
    levelCheck(ewr_dict['minThresholdL'],
    heightSeries[0], 
    ewr_dict['maxDrawdown'],           
    deltaHeight,           
    ewr_dict['duration'],                            
    eList, 
    eventDict, 
    wySeries[0],
    noneCounter,
    noneList,                      
    )
    
    # Cycle through the flow timeseries (excluding first and last element):
    for iteration, height in enumerate(heightSeries[1:-1]):
        wyCurrent = wySeries[iteration] # check for transformation  
        # Calculate the change in height from the day before to today:
        deltaHeight = (heightSeries[iteration-1]-heightSeries[iteration])
        
        eList, eventDict, noEventCounter, noEventList = \
        levelCheck(ewr_dict['minThresholdL'],
        height, 
        ewr_dict['maxDrawdown'],           
        deltaHeight,           
        ewr_dict['duration'],                            
        eList, 
        eventDict, 
        wyCurrent,
        noneCounter,
        noneList,                      
        )
        if wySeries[iteration] != wySeries[iteration+1]:
            if len(eList) >= ewr_dict['duration']: 
                eventDict[wyCurrent].append(eList)
            eList = [] # Reset at the end of the water year

            
    # Hit the final element in the series:
    deltaHeight = (heightSeries[-2]-heightSeries[iteration-1])

    eList, eventDict, noEventCounter, noEventList = \
    levelCheck(ewr_dict['minThresholdL'],
    heightSeries[-1], 
    ewr_dict['maxDrawdown'],           
    deltaHeight,           
    ewr_dict['duration'],                            
    eList, 
    eventDict, 
    wySeries[-1],
    noneCounter,
    noneList,                      
    )
            
    if len(eList) >= ewr_dict['duration']: 
        eventDict[wyCurrent].append(eList)
    
    return eventDict 
 
def simultaneousMultiYearEvent(flow, flow2, ewr_dict, ewr_dict2, flowSeries, flowSeries2, iteration, maxForward, maxBackward, wy_type):
    '''For finding when the event spanning multiple years falls and details about the event'''
    
    preEventList = []
    postEventList = []
    tracker = 0
    if wy_type == 'end':
        preEventList.append(flow) # start with this in the list
    elif wy_type == 'start':
        postEventList.append(flow)
    while True:
        tracker = tracker + 1
        if tracker == maxBackward: # only search back a year, unlikely to be activated
            break
        eventTrack = flowSeries[(iteration - tracker)]
        eventTrack2 = flowSeries2[(iteration - tracker)]
        if ((eventTrack < ewr_dict['minThresholdF']) or (eventTrack > ewr_dict['maxThresholdF']) or (eventTrack2 < ewr_dict2['minThresholdF']) or (eventTrack2 > ewr_dict2['maxThresholdF'])):
            break            
        else:
            preEventList.insert(0, eventTrack)
    tracker = 0
    while True:
        tracker = tracker + 1
        if tracker == maxForward: # only search forward a year
            break
        eventTrack = flowSeries[(iteration + tracker)]
        eventTrack2 = flowSeries2[(iteration - tracker)]
        if ((eventTrack < ewr_dict['minThresholdF']) or (eventTrack > ewr_dict['maxThresholdF']) or (eventTrack2 < ewr_dict2['minThresholdF']) or (eventTrack2 > ewr_dict2['maxThresholdF'])):
            break
        else:
            postEventList.append(eventTrack)
    eList = preEventList + postEventList
        
    return eList, preEventList, postEventList

def simultaneousFlowCalcAnytime(ewr_dict, ewr_dict2, flowSeries, flowSeries2, wySeries, dateSeries):
    '''The calculation section for flow type ewrs with no time constraints,
    these have been seperated due to their ability to cross water year boundaries'''
    # Declare variables:
    eList = []
    eventDict = {}
    noneCounter = 0
    noneList = []
    linesToSkip = 0

    # Saving dates as nparrays to increase speed of processing:
    days = dateSeries.month.values
    months = dateSeries.day.values
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Hit the first iteration outside the loop
    eList, eventDict, noneCounter, noneList =\
            simultaneousEwrCheck(ewr_dict['minThresholdF'],
                     ewr_dict2['minThresholdF'],
                     ewr_dict['maxThresholdF'],
                     ewr_dict2['maxThresholdF'],
                     ewr_dict['duration'],
                     ewr_dict2['duration'],
                     ewr_dict['min event'],
                     ewr_dict2['min event'],
                     flowSeries[0],
                     flowSeries2[0],
                     eList,
                     eventDict,
                     wySeries[0],
                     noneCounter,
                     noneList
                    )    
    # Cycle through the remaining flow timeseries except the last one:
    for iteration, flow in enumerate(flowSeries[1:-1]):
        if linesToSkip > 0:
            linesToSkip = linesToSkip - 1
        else:
            wyCurrent = wySeries[iteration]      
            if (months[iteration] == 6) and (days[iteration] == 30): # last day in WY
                # Determine the max window (if the date is close to the start or end of the timeseries), otherwise leave the max window to the full year
                if iteration > len(flowSeries) - 365:
                    maxForwardWindow = len(flowSeries) - iteration
                else:
                    maxForwardWindow = 365
                if iteration < 365:
                    maxBackwardWindow = iteration
                else:
                    maxBackwardWindow = 365
                if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF']) and (flowSeries2[iteration] >= ewr_dict2['minThresholdF']) and (flowSeries2[iteration] <= ewr_dict2['mxThresholdF'])):
                    eList, preEventList, postEventList = simultaneousMultiYearEvent(flow,
                                                                        flowSeries2[iteration],
                                                                        ewr_dict,
                                                                        ewr_dict2,
                                                                        flowSeries,
                                                                        flowSeries2,
                                                                        iteration,
                                                                        maxForwardWindow,
                                                                        maxBackwardWindow,
                                                                        'end'
                                                                       )
                    if len(eList) >= ewr_dict['min event']:
                        if len(preEventList) >= len(postEventList):
                            eventDict[wyCurrent].append(eList)
                            linesToSkip = len(postEventList)

                eList = [] # Reset the event list at the end of the water year  

            elif (months[iteration] == 7) and (days[iteration] == 1): # start of WY
                # Determine the max window (if the date is close to the start or end of the timeseries), otherwise leave the max window to the full year
                if iteration > len(flowSeries) - 365:
                    maxForwardWindow = len(flowSeries) - iteration
                else:
                    maxForwardWindow = 365
                if iteration < 365:
                    maxBackwardWindow = iteration
                else:
                    maxBackwardWindow = 365
                if ((flow >= ewr_dict['minThresholdF']) and (flow <= ewr_dict['maxThresholdF']) and (flowSeries2[iteration] >= ewr_dict2['minThresholdF']) and (flowSeries2[iteration] <= ewr_dict2['maxThresholdF'])):
                    eList, preEventList, postEventList = simultaneousMultiYearEvent(flow,
                                                                        flowSeries[iteration],
                                                                        ewr_dict,
                                                                        ewr_dict2,
                                                                        flowSeries, 
                                                                        flowSeries2,
                                                                        iteration,
                                                                        maxForwardWindow,
                                                                        maxBackwardWindow,
                                                                        'start'
                                                                       )
                    if len(eList) >= ewr_dict['min event']:
                        if len(preEventList) < len(postEventList):
                            eventDict[wyCurrent].append(eList)
                            linesToSkip = len(postEventList)-1
                eList = []
            else:
                eList, eventDict, noneCounter, noneList =\
                        simultaneousEwrCheck(ewr_dict['minThresholdF'],
                                 ewr_dict2['minThresholdF'],
                                 ewr_dict['maxThresholdF'],
                                 ewr_dict2['maxThresholdF'],
                                 ewr_dict['duration'],
                                 ewr_dict2['duration'],
                                 ewr_dict['min event'],
                                 ewr_dict2['min event'],
                                 flow,
                                 flowSeries2[iteration],
                                 eList,
                                 eventDict,
                                 wySeries[iteration],
                                 noneCounter,
                                 noneList
                                )  
    
    # Hit the final iteration
    eList, eventDict, noneCounter, noneList =\
            simultaneousEwrCheck(ewr_dict['minThresholdF'],
                     ewr_dict2['minThresholdF'],
                     ewr_dict['maxThresholdF'],
                     ewr_dict2['maxThresholdF'],
                     ewr_dict['duration'],
                     ewr_dict2['duration'],
                     ewr_dict['min event'],
                     ewr_dict2['min event'],
                     flowSeries[-1],
                     flowSeries2[-1],
                     eList,
                     eventDict,
                     wySeries[-1],
                     noneCounter,
                     noneList
                    )  
    if len(eList) >= ewr_dict['min event']:
        eventDict[wyCurrent].append(eList) 
    
    return eventDict, noneList

def simultaneousFlowCalc(ewr_dict, ewr_dict2, flowSeries, flowSeries2, wySeries, dateSeries):
    '''For standard flow EWRs with time constraints'''
    # Decalre variables:
    eList = []
    eventDict = {}
    noneCounter = 0
    noneList = []
    linesToSkip = 0
    
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    
    # Cycle through the flow timeseries:    
    for iteration, flow in enumerate(flowSeries[:-1]):
        wyCurrent = wySeries[iteration] # check for transformation
        # Check ewr completion:
        eList, eventDict, noneCounter, noneList =\
                simultaneousEwrCheck(ewr_dict['minThresholdF'],
                         ewr_dict2['minThresholdF'],
                         ewr_dict['maxThresholdF'],
                         ewr_dict2['maxThresholdF'],
                         ewr_dict['duration'],
                         ewr_dict2['duration'],
                         ewr_dict['min event'],
                         ewr_dict2['min event'],
                         flow,
                         flowSeries2[iteration],
                         eList,
                         eventDict,
                         wyCurrent,
                         noneCounter,
                         noneList
                        )
        if wySeries[iteration] != wySeries[iteration+1]:
            if len(eList) >= ewr_dict['min event']: 
                eventDict[wyCurrent].append(eList)
            eList = [] # reset the event list at the end of the water year
            
    # Check ewr completion for last day:
    eList, eventDict, noneCounter, noneList =\
            simultaneousEwrCheck(ewr_dict['minThresholdF'],
                     ewr_dict2['minThresholdF'],
                     ewr_dict['maxThresholdF'],
                     ewr_dict2['maxThresholdF'],
                     ewr_dict['duration'],
                     ewr_dict2['duration'],
                     ewr_dict['min event'],
                     ewr_dict2['min event'],
                     flowSeries[-1],
                     flowSeries2[-1],
                     eList,
                     eventDict,
                     wySeries[-1],
                     noneCounter,
                     noneList
                    )
    if len(eList) >= ewr_dict['min event']: 
        eventDict[wyCurrent].append(eList)
    #--- End last day check ---#
    
    return eventDict, noneList

def simultaneousLowFlowCalc(ewr_dict, ewr_dict2, flowSeries, flowSeries2, wySeries, climateSeries, PU_df, ewr, gauge_ID):
    '''Evaluating the EWRs where a simultaneous EWR is required to be met'''
    # Decalre variables:
    eList, eList2 = [], []
    eventDict, eventDict2 = {}, {}
    noneCounter = 0
    noneList = []                 
    yearsWithEvents = []
    avLowFlowDays = []
    waterYears = sorted(set(wySeries))
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    eventDict2 = dict.fromkeys(wYears)
    for k, _ in eventDict2.items():
        eventDict2[k] = []
    # Cycle through the flow timeseries:
    for iteration, flow in enumerate(flowSeries[:-1]):
        wyCurrent = wySeries[iteration] 
        # EWR check:
        eList, eList2, eventDict, eventDict2, noneCounter, noneList = simultaneousLowFlowCheck(ewr_dict['minThresholdF'],
                                                                           ewr_dict2['minThresholdF'],
                                                                           ewr_dict['maxThresholdF'],
                                                                           ewr_dict2['maxThresholdF'],
                                                                           flow,
                                                                           flowSeries2[iteration],
                                                                           eList,
                                                                                               eList2,
                                                                           eventDict,
                                                                                               eventDict2,
                                                                           wyCurrent,
                                                                           noneCounter,
                                                                           noneList
                                                                          )
        if wySeries[iteration] != wySeries[iteration+1]:
            if ((climateSeries[iteration] == 'Very Dry') and (ewr_dict['veryDry_duration'] !=None)):
                ewr_duration = ewr_dict['veryDry_duration']
                ewr_duration2 = ewr_dict2['veryDry_duration']
            else:
                ewr_duration = ewr_dict['duration']
                ewr_duration2 = ewr_dict2['duration']
            # If there are more elements in the final list, save them to the results dictionary:    
            if len(eList) > 0:
                eventDict[wyCurrent].append(eList)
                noneCounter = noneCounter - len(eList)#count to start of event
                noneList.append(noneCounter)
                noneCounter = 1
            if len(eList2) > 0:
                eventDict2[wyCurrent].append(eList2)
                noneCounter= noneCounter - len(eList2)
                noneList.append(noneCounter)
                noneCounter = 1

            # Check year for event, add result to the list
            calcList = ['years with events','number of low flow days']
            yearsWithEvents, avLowFlowDays = simultaneousResultsStatsLowFlow(calcList,
                                                                 eventDict, 
                                                                 eventDict2,
                                                                 wyCurrent, 
                                                                 yearsWithEvents, 
                                                                 avLowFlowDays,
                                                                 ewr_duration,
                                                                 ewr_duration2)
            
            eList, eList2 = [], [] # Reset at the end of the water year

    #--- Check final iteration: ---#
    eList, eList2, eventDict, eventDict2, noneCounter, noneList = simultaneousLowFlowCheck(ewr_dict['minThresholdF'],
                                                                       ewr_dict2['minThresholdF'],
                                                                       ewr_dict['maxThresholdF'],
                                                                       ewr_dict2['maxThresholdF'],
                                                                       flowSeries[-1],
                                                                       flowSeries2[-1],
                                                                       eList,
                                                                                           eList2,
                                                                       eventDict,
                                                                                           eventDict2,
                                                                       wySeries[-1],
                                                                       noneCounter,
                                                                       noneList
                                                                      )
            
    if ((climateSeries[-1] == 'Very Dry') and (ewr_dict['veryDry_duration'] !=None)):
        ewr_duration = ewr_dict['veryDry_duration']
        ewr_duration2 = ewr_dict2['veryDry_duration']
    else:
        ewr_duration = ewr_dict['duration']
        ewr_duration2 = ewr_dict2['duration']
    # If there are more elements in the final list, save them to the results dictionary:    
    if len(eList) > 0:
        eventDict[wySeries[-1]].append(eList)
        noneCounter = noneCounter - len(eList)#count to start of event
        noneList.append(noneCounter)
        noneCounter = 1
    if len(eList2) > 0:
        eventDict2[wySeries[-1]].append(eList2)
        noneCounter= noneCounter - len(eList2)
        noneList.append(noneCounter)
        noneCounter = 1

    # Check year for event, add result to the list
    calcList = ['years with events','number of low flow days']
    yearsWithEvents, avLowFlowDays = simultaneousResultsStatsLowFlow(calcList,
                                                         eventDict,
                                                         eventDict2,
                                                         wySeries[-1], 
                                                         yearsWithEvents, 
                                                         avLowFlowDays,
                                                         ewr_duration,
                                                         ewr_duration2)   

    addSeries = pd.Series(yearsWithEvents)
    addSeries.index = waterYears
    
    addSeries1 =  pd.Series(avLowFlowDays)
    addSeries1.index = waterYears
    
    # Average length of time between events:
#     averageBetween = sum(noneList)/len(waterYears)
#     addSeries2 = [averageBetween] * len(waterYears)
    
    # Saving the results to the dataframe: 
    PU_df[str(ewr + '_eventYears')] = addSeries
    PU_df[str(ewr + '_avLowFlowDays')] = addSeries1
#     PU_df[str(ewr + '_avDaysBetween')] = addSeries2
    
    return PU_df  

def simultaneousCfCalc(ewr_dict, ewr_dict2, flowSeries, flowSeries2, wySeries, climateSeries, PU_df, ewr):
    '''For calculating cease to flow type ewrs'''
    # Decalre variables:
    eList, eList2 = [], []
    eventDict, eventDict2 = {}, {}
    noneCounter = 0
    noneList = []
    counter = 0
    overallCounter = 0
    eventCounter = 0   
    #Variables to save yearly results to:
    yearsWithEvents = []
    numEvents = []
    ctfDaysPerYear = []
    avLenCtfSpells = []
    waterYears = sorted(set(wySeries))
    # Save spot in the dictionary of results for the years included:
    wYears = set(wySeries)
    eventDict = dict.fromkeys(wYears)
    for k, _ in eventDict.items():
        eventDict[k] = []
    eventDict2 = dict.fromkeys(wYears)
    for k, _ in eventDict2.items():
        eventDict2[k] = []
        
    for iteration, flow in enumerate(flowSeries[:-1]): 
        wyCurrent = wySeries[iteration] # check for transformation 
        eList, eList2, eventDict, eventDict2, noneCounter, noneList = simultaneousCtfCheck(ewr_dict['minThresholdF'],
                                                           ewr_dict2['minThresholdF'],
                                                           flow,
                                                           flowSeries2[iteration],
                                                           eList,
                                                           eList2,
                                                           eventDict,
                                                           eventDict2,
                                                           wyCurrent,
                                                           noneCounter,
                                                           noneList)
        if wySeries[iteration] != wySeries[iteration+1]:
            if ((climateSeries[iteration] == 'Very Dry') and (ewr_dict['veryDry_duration'] !=None)):
                ewr_duration = ewr_dict['veryDry_duration']
                ewr_duration2 = ewr_dict2['veryDry_duration']
            else:
                ewr_duration = ewr_dict['duration']
                ewr_duration2 = ewr_dict2['duration']
            # Add in the current event list:
            if len(eList) > 0:
                eventDict[wyCurrent].append(eList)
            if len(eList2) > 0:
                eventDict2[wyCurrent].append(eList2)
            # Send to get evaluated:
            yearsWithEvents,numEvents,ctfDaysPerYear,avLenCtfSpells=simultaneousResultsStatsCF(eventDict, 
                                                                                   eventDict2,
                                                                                   wyCurrent,
                                                                                   yearsWithEvents,
                                                                                   numEvents,
                                                                                   ctfDaysPerYear,
                                                                                   avLenCtfSpells,
                                                                                   ewr_duration,
                                                                                   ewr_duration2
                                                                                  )
            eList = []
    #--- Handling the final element in the series ---#
    eList, eList2, eventDict, eventDict2, noneCounter, noneList = simultaneousCtfCheck(ewr_dict['minThresholdF'],
                                                       ewr_dict2['minThresholdF'],
                                                       flowSeries[-1],
                                                       flowSeries2[-1],
                                                       eList,
                                                       eList2,
                                                       eventDict,
                                                       eventDict2,
                                                       wySeries[-1],
                                                       noneCounter,
                                                       noneList)
    if ((climateSeries[-1] == 'Very Dry') and \
        (ewr_dict['veryDry_duration'] !=None)):
        ewr_duration = ewr_dict['veryDry_duration']
        ewr_duration2 = ewr_dict2['veryDry_duration']
    else:
        ewr_duration = ewr_dict['duration']
        ewr_duration2 = ewr_dict2['duration']
    if len(eList) > 0:
        eventDict[wySeries[-1]].append(eList)
    if len(eList2) > 0:
        eventDict2[wyCurrent].append(eList2)    
    # Send to get evaluated:
    yearsWithEvents,numEvents,ctfDaysPerYear,avLenCtfSpells=simultaneousResultsStatsCF(eventDict, 
                                                                           eventDict2,
                                                                           wySeries[-1],
                                                                           yearsWithEvents,
                                                                           numEvents,
                                                                           ctfDaysPerYear,
                                                                           avLenCtfSpells,
                                                                           ewr_duration,
                                                                           ewr_duration2
                                                                          )
    #--- End handling the final element ---#
    addSeries = pd.Series(yearsWithEvents)
    addSeries.index = waterYears
    
    addSeries1 =  pd.Series(numEvents)
    addSeries1.index = waterYears
    
    addSeries2 = pd.Series(ctfDaysPerYear)
    addSeries2.index = waterYears
    
    addSeries3 =  pd.Series(avLenCtfSpells)
    addSeries3.index = waterYears

    # Average length of time between events:
#     averageBetween = sum(noneList)/len(waterYears)
#     addSeries4 = [averageBetween] * len(waterYears)
    
    # Save results to dataframe:
    PU_df[str(ewr + '_eventYears')] = addSeries
    PU_df[str(ewr + '_numEvents')] = addSeries1
    PU_df[str(ewr + '_ctfDaysPerYear')] = addSeries2
    PU_df[str(ewr + '_avLenCtfSpells')] = addSeries3   
#     PU_df[str(ewr + '_avDaysBetween')] = addSeries4
    
    return PU_df    

    
    
#------------------- Doing stats on the dictionary of events -----------------#
def resultsStatsFlow(calcList, eventDict, ewr_dict, PU_df, ewr, wyList, noneList):
    '''Taking in a dictionary of events from a flow timeseries, summarises these and returns in the results dataframe
    Only for calculating the final results, not on a yearly basis. For flow only, due to the inclusion of the 'min_event' being able to contribute to an event
    '''
    
    if 'years with events' in calcList:  
        yearsWithEvents = []
        for wYear in wyList:
            yearTracker = []
            minEventTracker = []
            for i in eventDict[wYear]:
                if len(i) >= ewr_dict['duration']:
                    yearTracker.append(1)
                else:
                    if ewr_dict['duration'] != ewr_dict['min event']:
                        # Check if the combined go over the min event
                        if len(i) >= ewr_dict['min event']:
                            minEventTracker = minEventTracker + i
                            if len(minEventTracker) >= ewr_dict['duration']:
                                yearTracker.append(1)
                                minEventTracker = [] # reset the minimum event tracker so further events are not lost
                            else:
                                yearTracker.append(0)
                        else:
                            yearTracker.append(0)
                    else:
                        yearTracker.append(0)
            if yearTracker != []:
                if len(yearTracker) >= ewr_dict['events per year']:
                    yearsWithEvents.append(max(yearTracker))
                else:
                    yearsWithEvents.append(0)
            else:
                yearsWithEvents.append(0)
        
        addSeries = pd.Series(yearsWithEvents)
        addSeries.index = wyList
        
        PU_df[str(ewr + '_eventYears')] = addSeries
            
    if 'number of events' in calcList:
        numEvents = []
        for wYear in wyList:
            yearTracker = []
            minEventTracker = []
            for i in eventDict[wYear]:
                if len(i) >= ewr_dict['duration']:  
                    yearTracker.append(1)
                else:
                    if ewr_dict['duration'] != ewr_dict['min event']:
                        # Check if the combined go over the min event
                        if len(i) >= ewr_dict['min event']:
                            minEventTracker = minEventTracker + i
                            if len(minEventTracker) >= ewr_dict['duration']:
                                yearTracker.append(1)
                                minEventTracker = [] # reset the minimum event tracker so further events are not lost
                            else:
                                yearTracker.append(0)
                        else:
                            yearTracker.append(0)
                    else:
                        yearTracker.append(0)
            numEvents.append(sum(yearTracker)) 
            
        addSeries = pd.Series(numEvents) 
        addSeries.index = wyList    
        PU_df[str(ewr + '_numEvents')] = addSeries
        
    if 'average length of events' in calcList:
        avLengthEvents = []
        for wYear in wyList:
            yearLenTracker = []
            for i in eventDict[wYear]:
                if len(i) >= ewr_dict['min event']:
                    yearLenTracker.append(len(i))
                else:
                    yearLenTracker.append(0)
            if sum(yearLenTracker) != 0:
                avLength = sum(yearLenTracker)/len(yearLenTracker)
            else:
                avLength = 0
            avLengthEvents.append(avLength)
        
        addSeries = pd.Series(avLengthEvents)
        addSeries.index = wyList
        
        PU_df[str(ewr + '_avLengthEvents')] = addSeries
         
    if 'average time between events' in calcList:
        if (len(noneList)>0):
            avDaysBetween = []
            avDays = sum(noneList)/len(noneList)
            years = len(PU_df.index)
            avDaysBetween = avDays/years
        else:
            avDaysBetween = 0 
            
        daysBetweenList = [avDaysBetween] * len(wyList)        
        
        addSeries = pd.Series(daysBetweenList)  
        addSeries.index = wyList
            
        PU_df[str(ewr + '_avDaysBetween')] = addSeries
        
    return PU_df

def resultsStats(calcList, eventDict, ewr_dict, PU_df, ewr, wyList):
    '''Taking in a dictionary of events, summarises these and returns in the 
    results dataframe
    Only for calculating the final results, not on a yearly basis'''
    # If there are years that were missed due to events spanning multiple years, add these spots into the dictionary:
    for wYear in wyList:
        if wYear not in eventDict.keys():
            eventDict[wYear] = []
    
    if 'years with events' in calcList:
        yearsWithEvents = []
        for wYear in wyList:
            yearTracker = []
            for i in eventDict[wYear]:
                if len(i) >= ewr_dict['duration']:
                    yearTracker.append(1)
                else:
                    yearTracker.append(0)
            if yearTracker != []:
                if len(yearTracker) >= ewr_dict['events per year']:
                    yearsWithEvents.append(max(yearTracker))
                else:
                    yearsWithEvents.append(0)
            else:
                yearsWithEvents.append(0)
        
        addSeries = pd.Series(yearsWithEvents)
        addSeries.index = wyList
        
        PU_df[str(ewr + '_eventYears')] = addSeries
        
            
    if 'number of events' in calcList:
        numEvents = []
        for wYear in wyList:
            yearTracker = []
            for i in eventDict[wYear]:
                if len(i) >= ewr_dict['duration']:  
                    yearTracker.append(1)
                else:
                    yearTracker.append(0)
            numEvents.append(sum(yearTracker)) 
            
        addSeries = pd.Series(numEvents) 
        addSeries.index = wyList    
        PU_df[str(ewr + '_numEvents')] = addSeries
        
    if 'average length of events' in calcList:
        avLengthEvents = []
        for wYear in wyList:
            yearLenTracker = []
            for i in eventDict[wYear]:
                if len(i) >= ewr_dict['duration']:  
                    yearLenTracker.append(len(i))
                else:
                    yearLenTracker.append(0)
            if sum(yearLenTracker) != 0:
                avLength = sum(yearLenTracker)/len(yearLenTracker)
            else:
                avLength = 0
            avLengthEvents.append(avLength)
        
        addSeries = pd.Series(avLengthEvents)
        addSeries.index = wyList
        
        PU_df[str(ewr + '_avLengthEvents')] = addSeries
         
    if 'average time between events' in calcList:
        if (len(noneList)>0):
            avDaysBetween = []
            avDays = sum(noneList)/len(noneList)
            years = len(PU_df.index)
            avDaysBetween = avDays/years
        else:
            avDaysBetween = 0 
            
        daysBetweenList = [avDaysBetween] * len(wyList)        
        
        addSeries = pd.Series(daysBetweenList)  
        addSeries.index = wyList
            
        PU_df[str(ewr + '_avDaysBetween')] = addSeries
        
    return PU_df

def resultsStatsLowFlow(calcList, eventDict, wyCurrent, yearsWithEvents, avLowFlowDays, ewr_duration):
    '''For low flows, needing to calculate the results on a year by year basis
    because climate needs to be used'''
    if 'years with events' in calcList:
        yearlyTracker = []
        for i in eventDict[wyCurrent]:
            yearlyTracker.extend(i)
        if len(yearlyTracker) >= ewr_duration:
            yearsWithEvents.append(1)
        else:
            yearsWithEvents.append(0)
    
    if 'number of low flow days' in calcList:
        yearlyTracker = []
        for i in eventDict[wyCurrent]:
            yearlyTracker.extend(i)
        avLowFlowDays.append(len(yearlyTracker))
    
    return yearsWithEvents, avLowFlowDays
    
def resultsStatsCF(eventDict, wyCurrent, yearsWithEvents, numEvents, ctfDaysPerYear, 
                   avLenCtfSpells, ewr_duration):
    '''Getting yearly statistics on the cease to flow EWRs'''
    yearTracker = []
    for i in eventDict[wyCurrent]:
        if len(i) >= ewr_duration:
            yearTracker.append(1)
        else:
            yearTracker.append(0)
    if yearTracker != []:
        yearsWithEvents.append(max(yearTracker))
    else:
        yearsWithEvents.append(0)
    numEvents.append(sum(yearTracker)) 
    # Num ctf days per year:
    yearlyCounter = 0
    for i in eventDict[wyCurrent]:
        yearlyCounter = yearlyCounter + len(i)
    ctfDaysPerYear.append(yearlyCounter)
    # Average length of ctf spells for the year:
    if len(eventDict[wyCurrent])>0:  
        lenEvents = []
        for i in eventDict[wyCurrent]:
            lenEvents.append(len(i)) 
        avLength = sum(lenEvents)/len(lenEvents)
    else:
        avLength = 0
    avLenCtfSpells.append(avLength)
    
    return yearsWithEvents, numEvents, ctfDaysPerYear, avLenCtfSpells

def simultaneousResultsStatsLowFlow(calcList, eventDict, eventDict2, wyCurrent, yearsWithEvents, avLowFlowDays, ewr_duration, ewr_duration2):
    '''For low flows, needing to calculate the results on a year by year basis
    because climate needs to be used'''
    if 'years with events' in calcList:
        yearlyTracker, yearlyTracker2 = [], []
        for i in eventDict[wyCurrent]:
            yearlyTracker.extend(i)
        for i in eventDict2[wyCurrent]:
            yearlyTracker2.extend(i)
        if ((len(yearlyTracker) >= ewr_duration) and (len(yearlyTracker2) >= ewr_duration2)):
            yearsWithEvents.append(1)
        else:
            yearsWithEvents.append(0)
    
    if 'number of low flow days' in calcList: # Just get the low flow stats for the gauge we are iterating over.
        yearlyTracker = []
        for i in eventDict[wyCurrent]:
            yearlyTracker.extend(i)
        avLowFlowDays.append(len(yearlyTracker))
    
    return yearsWithEvents, avLowFlowDays

def simultaneousResultsStatsCF(eventDict, eventDict2, wyCurrent, yearsWithEvents, numEvents, ctfDaysPerYear, avLenCtfSpells, ewr_duration, ewr_duration2):
    '''Getting yearly statistics on the cease to flow EWRs'''
    yearTracker = []
    for i in eventDict[wyCurrent]:
        if len(i) >= ewr_duration:
            yearTracker.append(1)
        else:
            yearTracker.append(0)
    for i in eventDict2[wyCurrent]:
        if len(i) >= ewr_duration2:
            yearTracker.append(1)
        else:
            yearTracker.append(0)
    if yearTracker != []:
        yearsWithEvents.append(max(yearTracker))
    else:
        yearsWithEvents.append(0)
    numEvents.append(sum(yearTracker)) # If there is an event on at the same time at both sites this will be recorded as two events
    # Num ctf days per year:
    yearlyCounter = 0
    for i in eventDict[wyCurrent]:
        yearlyCounter = yearlyCounter + len(i)
    for i in eventDict[wyCurrent]:
        yearlyCounter = yearlyCounter + len(i)
    ctfDaysPerYear.append(yearlyCounter) # If there is an event on at the same time at both sites this will be doubled in the total days
    # Average length of ctf spells for the year:
    # First site:
    if len(eventDict[wyCurrent])>0:
        lenEvents = []
        for i in eventDict[wyCurrent]:
            lenEvents.append(len(i))
        avLength = sum(lenEvents)/len(lenEvents)
    else:
        avLength = 0
    # Second site:
    if len(eventDict2[wyCurrent])>0:
        lenEvents2 = []
        for i in eventDict2[wyCurrent]:
            lenEvents2.append(len(i))
        avLength2 = sum(lenEvents)/len(lenEvents)
    else:
        avLength2 = 0        
    if ((avLength == 0) and (avLength2 == 0)):
        avLength = (avLength + avLength2)/2
    else:
        avLength = 0
        
    avLenCtfSpells.append(avLength) # average length of ctf events for this type are worked out seperately for each site, then averaged out. i.e. site 1 has 10 days, site 2 has 5 days. The av length would be 7.5 days
    
    return yearsWithEvents, numEvents, ctfDaysPerYear, avLenCtfSpells

#------------------ Calculation master function -------------------------#

def EWR_calculator(df, gauge_ID, ewr_table, toleranceDict, climate_file):
    '''Sends to handling functions to get calculated depending on the type of EWR''' 
    # Get menindee and weirpool gauges, gauges with combined flow requirements, simultaneous ewr requirements, and those with 'other complex requirements'
    menindeeGauges, wpGauges = data_inputs.getLevelGauges()
    multiGaugeCalc = data_inputs.getMultiGauges('all')
    simultaneousGaugesCalc = data_inputs.getSimultaneousGauges('all')
    complexCalc = data_inputs.getComplexCalcs()
    # Extract relevant sections of the EWR table:
    sub_ewr_table = ewr_table[ewr_table['gauge'] == gauge_ID]
    # Get unique list of planning units to iterate over in the first instance:
    planningUnits = set(sub_ewr_table['PlanningUnitID'])
    # save the planning unit dataframes to this dictionary:
    locationDict = {}
    for planUnit in planningUnits:
        # Extract the planning unit table:
        planUnitTable = sub_ewr_table[sub_ewr_table['PlanningUnitID'] == planUnit]
        # Extract the relevant columns from the table that will be referred to below:
        ewr_type_col = planUnitTable['flow level volume'].values
        code_col = planUnitTable['code']
        # Get clean version of the dataframe template from above section to save results to:
        PU_df = pd.DataFrame()
        # Iterate over the ewrs for that planning unit:
        for num, ewr in enumerate(tqdm(code_col, position = 0, leave = False,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', 
                                       desc= str('Evaluating ewrs for '+ gauge_ID))):
            if ('_VD' not in ewr): # Skip very dry (VD) ewrs, these will be called upon later 
                # Check to see if this ewr is classified as 'complex' (requiring special functions to check)
                # If not complex, now start looking through standard EWR categories for how to analyse:
                if ((gauge_ID in complexCalc) and (ewr in complexCalc[gauge_ID])):
                    PU_df = complexEWRhandler(complexCalc[gaugeOnly][ewr],
                                              planUnit,
                                                   gaugeOnly,
                                                   ewr,
                                                   planUnitTable,
                                                   df,
                                                   PU_df,
                                                   toleranceDict
                                                  )
                elif ((planUnit in multiGaugeCalc) and (gauge_ID in multiGaugeCalc[planUnit]) and (ewr_type_col[num] == 'F')):
                    if 'CF' in ewr:
                        PU_df = multiGaugeEWRhandler('cease to flow',
                                                     planUnit,
                                                     gauge_ID,
                                                     ewr,
                                                     planUnitTable,
                                                     df,
                                                     PU_df,
                                                     toleranceDict,
                                                     climate_file)
                    if (('VF' in ewr) or ('BF' in ewr)):
                        PU_df = multiGaugeEWRhandler('low flow',
                                                    planUnit,
                                                    gauge_ID,
                                                    ewr,
                                                    planUnitTable,
                                                    df,
                                                    PU_df,
                                                    toleranceDict,
                                                    climate_file)
                    if (('SF' in ewr) or ('LF' in ewr) or ('BK' in ewr) or ('OB' in ewr)):
                        PU_df = multiGaugeEWRhandler('flow',
                                                    planUnit,
                                                    gauge_ID,
                                                    ewr,
                                                    planUnitTable,
                                                    df,
                                                    PU_df,
                                                    toleranceDict,
                                                    climate_file)
                elif ((planUnit in multiGaugeCalc) and (gauge_ID in multiGaugeCalc[planUnit]) and (ewr_type_col[num] == 'V')):
                    if ('OB' in ewr):
                        PU_df = multiGaugeEWRhandler('cumulative volume',
                                                     planUnit,
                                                     gauge_ID,
                                                     ewr,
                                                     planUnitTable,
                                                     df,
                                                     PU_df,
                                                     toleranceDict,
                                                     climate_file)
                
                elif ((planUnit in simultaneousGaugesCalc) and (gauge_ID in simultaneousGaugesCalc[planUnit]) and (ewr_type_col[num] == 'F')):
                    if 'CF' in ewr:
                        PU_df = simultaneousEWRhandler('cease to flow',
                                                            planUnit,
                                                    gauge_ID,
                                                    ewr,
                                                    planUnitTable,
                                                    df,
                                                    PU_df,
                                                    toleranceDict,
                                                    ewr_table,
                                                    climate_file)
                    if (('VF' in ewr) or ('BF' in ewr)):
                        PU_df = simultaneousEWRhandler('low flow',
                                                            planUnit,
                                                    gauge_ID,
                                                    ewr,
                                                    planUnitTable,
                                                    df,
                                                    PU_df,
                                                    toleranceDict,
                                                    ewr_table,
                                                    climate_file)
                    if (('SF' in ewr) or ('LF' in ewr) or ('BK' in ewr) or ('OB' in ewr)):
                        PU_df = simultaneousEWRhandler('flow',
                                                            planUnit,
                                                    gauge_ID,
                                                    ewr,
                                                    planUnitTable,
                                                    df,
                                                    PU_df,
                                                    toleranceDict,
                                                    ewr_table,
                                                    climate_file)
                    
                elif ((ewr_type_col[num] == 'F') and (gauge_ID not in wpGauges.values())):
                    # If standard flow EWRs:
                    if (('SF' in ewr) or ('LF' in ewr) or ('BK' in ewr) or ('OB' in ewr) or ('AC' in ewr)\
                       or ('WL' in ewr)):
                        PU_df = flowEWRhandler('flow',
                                               planUnit,
                                               gauge_ID,
                                               ewr,
                                               planUnitTable,
                                               df,
                                               PU_df,
                                               toleranceDict
                                              )
                    # If low flow type EWRs:
                    elif (('BF' in ewr) or ('VF' in ewr)):
                        PU_df = lowFlowEWRhandler('low flow',
                                                  planUnit,
                                               gauge_ID,
                                               ewr,
                                               planUnitTable,
                                               df,
                                               PU_df,
                                               toleranceDict,
                                               climate_file
                                              )
                    # If cease to flow type EWRs:
                    elif ('CF' in ewr):         
                        PU_df = cfFlowEWRhandler('cease to flow',
                                                 planUnit,
                                               gauge_ID,
                                               ewr,
                                               planUnitTable,
                                               df,
                                               PU_df,
                                               toleranceDict,
                                               climate_file
                                              )
                    elif ('Nest' in ewr):
                        PU_df = nestEWRhandler('nest',
                                               planUnit,
                                               gauge_ID,
                                               ewr,
                                               planUnitTable,
                                               df,
                                               PU_df,
                                               toleranceDict
                                              )
                    elif ('WP' in ewr):
                        PU_df = wpEWRhandler('weir pool',
                                             planUnit,
                                               gauge_ID,
                                               ewr,
                                               planUnitTable,
                                               df,
                                               PU_df,
                                               toleranceDict,
                                               climate_file
                                              )
                    else:
                        continue

                # Option for cumulative volume type EWRs:
                elif ewr_type_col[num] == 'V':
                    if (('LF' in ewr) or ('OB' in ewr) or ('WL' in ewr)):
                        PU_df = cumulVolEWRhandler('cumulative volume',
                                                   planUnit,
                                               gauge_ID,
                                               ewr,
                                               planUnitTable,
                                               df,
                                               PU_df,
                                               toleranceDict
                                              )
                
                elif ewr_type_col[num] == 'L':
                    #--- Menindee lakes and WP ewrs ---#
                    #---- Requires a trigger and a max drawdown rate ----#
                    if (('LLLF' in ewr) or ('MLLF' in ewr) or ('HLLF' in ewr)  or ('VHLL' in ewr)):
                        PU_df = lakeEWRhandler('lake level',
                                               planUnit,
                                               gauge_ID,
                                               ewr,
                                               planUnitTable,
                                               df,
                                               PU_df,
                                               toleranceDict
                                              )    
                    
        # Save the dataframe with all the locations to the dictionary
        locationDict[planUnit] = PU_df
        
    return locationDict  

#--------------------------------Making the results summary-------------------------------#
def get_count(list_of_events):
    '''Takes in list of numbers, returns the count of the list'''
    checkSeries = list_of_events.dropna()
    result = checkSeries.sum()
    result = int(result)
    
    return result

def get_frequency(list_of_events):
    '''Takes in list of numbers, returns the frequency of years they occur in'''
    checkSeries = list_of_events.dropna()
    result = (int(checkSeries.sum())/int(checkSeries.count()))*100
    result = int(round(result, 0))
    
    return result 

def get_maxDry(list_of_events):
    '''Takes in list of numbers, returns the maximum number of sequential 0's (dry years)'''
    checkSeries = list_of_events.dropna()
    max_dry_tracker = 0
    dry_spell_counter = 0
    for i in checkSeries:
        if i == 0:
            dry_spell_counter += 1
        elif i == 1:
            dry_spell_counter =0
        if dry_spell_counter > max_dry_tracker:
            max_dry_tracker = dry_spell_counter
            
    return max_dry_tracker

def get_last_event(list_of_events):
    '''Counts back through the years, returns how many years have passed since an event'''
    checkSeries = list_of_events.dropna()
    years = 0
    reversed_list = checkSeries.tolist()
    reversed_list = reversed(reversed_list)

    for year in reversed_list:
        if year == 1:
            
            return years
        
        else:
            years = years + 1

def get_numEvents(list_of_events):
    '''Takes in a list containing events per year,
    returns the total amount of events'''
    checkSeries = list_of_events.dropna()
    numEvents = int(sum(checkSeries))
    
    return numEvents

def get_avEvents(list_of_events):
    '''Takes in a list containing the average length of events per year,
    returns the overall average'''
    checkSeries = list_of_events.dropna()
    avEvents = sum(checkSeries)/len(checkSeries)
    
    return avEvents

def get_avNumEvents(list_of_events):
    '''Takes in a list containing the events per year,
    returns an average events per year over the series'''
    checkSeries = list_of_events.dropna()
    avNumEvent = sum(checkSeries)/len(checkSeries)
    
    return avNumEvent

def get_avDaysBetweenEvents(list_of_events):
    ''' '''
    checkSeries = list_of_events.dropna()
    avDays = sum(checkSeries)/len(checkSeries)
    
    return avDays

def get_avLowFlowDays(list_of_events):
    ''' '''
    checkSeries = list_of_events.dropna()
    avLowFlowDays = sum(checkSeries)/len(checkSeries)
    
    return avLowFlowDays

def get_avCtfDaysPerYear(list_of_events):
    ''' '''
    checkSeries = list_of_events.dropna()
    avCtfDaysPerYear = sum(checkSeries)/len(checkSeries)
    
    return avCtfDaysPerYear

def get_avLenCtfSpells(list_of_events):
    ''' '''
    checkSeries = list_of_events.dropna()
    avLenCtfSpells = sum(checkSeries)/len(checkSeries)
    
    return avLenCtfSpells

def initialise_summary_df_columns(input_dict, things_to_calc):
    '''Ingest a dictionary of ewr yearly results and a list of statistical tests to perform
    initialises a dataframe with these as a multilevel heading and returns this'''
    
    column_list = list()
    list_of_arrays = list()
    for scenario, scenario_results in input_dict.items():
        for sub_col in things_to_calc:
            column_list = tuple((scenario, sub_col))
            list_of_arrays.append(column_list)
    
    array_of_arrays =tuple(list_of_arrays)    
    multi_col_df = pd.MultiIndex.from_tuples(array_of_arrays, names = ['scenario', 'type'])

    return multi_col_df
    
def initialise_summary_df_rows(input_dict):
    '''Ingests a dictionary of ewr yearly results
    pulls the location information and the assocaited ewrs at each location,
    saves these as respective indexes and return the multi-level index'''
    
    index_list_level_1 = list()
    index_list_level_2 = list()
    index_list_level_3 = list()
    list_of_indexes = list()
    #Remove:
    counter = 0
    for scenario, scenario_results in input_dict.items():
        if counter >0:
            break
    # Get unique col list:
    dictCheck = {}
    for site, site_results in scenario_results.items():
        for planUnit in site_results:
            siteList = []
            for col in site_results[planUnit]:
                if ('_eventYears' in col):
                    ewrCode = col[:-len('_eventYears')]
                elif ('_numEvents' in col):
                    ewCode = col[:-len('_numEvents')]
                elif ('_avLengthEvents' in col):
                    ewCode = col[:-len('_avLengthEvents')]
                elif ('_avLowFlowDays' in col):
                    ewCode = col[:-len('_avLowFlowDays')]
                elif ('_ctfDaysPerYear' in col):
                    ewCode = col[:-len('_ctfDaysPerYear')]            
                elif ('_avLenCtfSpells' in col):
                    ewCode = col[:-len('_avLenCtfSpells')]   
                elif ('_avDaysBetween' in col):
                    ewrCode = col[:-len('_avDaysBetween')]
                if ewrCode in siteList:
                    continue
                else:
                    siteList.append(ewrCode)
                    index_list_level_1.append(site)
                    index_list_level_2.append(planUnit)
                    index_list_level_3.append(ewrCode)
    counter +=1
    combined_lists = list((index_list_level_1, index_list_level_2, index_list_level_3))
    tuples = list(zip(*combined_lists))
    index = pd.MultiIndex.from_tuples(tuples, names = ['location', 'planning unit', 'EWR'])

    return index

def summarise_results(input_dict, things_to_calc, ewrs):
    '''Ingests a dictionary with ewr pass/fails
    summarises these results and returns a single summary dataframe'''
    
    # Initialise dataframe with multi level column heading and multi-index:
    multi_col_df = initialise_summary_df_columns(input_dict, things_to_calc)
    index = initialise_summary_df_rows(input_dict)
    
    # Assign them to the dataframe:
    df_multi_level = pd.DataFrame(index = index, columns=multi_col_df)

    # Run the analysis and add the results to the dataframe created above:
    for scenario, scenario_results in input_dict.items():
        for site, site_results in scenario_results.items():
            for planUnit in site_results:
                for col in site_results[planUnit]:
                    if ('_eventYears' in col):
                        trunCol = col[:-len('_eventYears')]
                        idx = pd.IndexSlice
                        if 'Years with events' in things_to_calc:    
                            count_of_events = get_count(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 'Years with events']]= \
                            count_of_events
                        if 'Frequency' in things_to_calc:
                            frequency_of_events = get_frequency(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 'Frequency']] = \
                            frequency_of_events
                        if 'Target frequency' in things_to_calc:
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 'Target frequency']] = \
                            list(ewrs[((ewrs['gauge'] == site)&(ewrs['code'] == trunCol))]['frequency'])[0]
                        if 'Max dry' in things_to_calc:
                            maxDry_of_events = get_maxDry(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 'Max dry']] = \
                            maxDry_of_events
                        if 'Years since last event' in things_to_calc: 
                            last_event = get_last_event(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 'Years since last event']]=\
                            last_event
                    elif ('_numEvents' in col):
                        trunCol = col[:-len('_numEvents')]
                        if 'Number of events' in things_to_calc:
                            sumEvents = get_numEvents(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 'Number of events']]=\
                            sumEvents
                        if 'Average events per year' in things_to_calc:
                            meanNumEvents = get_avNumEvents(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 'Average events per year']]=\
                            meanNumEvents
                    elif ('_avLengthEvents' in col):
                        trunCol = col[:-len('_avLengthEvents')]
                        if 'Average event length' in things_to_calc:
                            avEvents = get_avEvents(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 'Average event length']]=\
                            avEvents
                    elif ('_avDaysBetween' in col):
                        trunCol = col[:-len('_avDaysBetween')]
                        if 'Average time between events' in things_to_calc:
                            avDaysBetweenEvents = get_avDaysBetweenEvents(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, \
                                                                          'Average time between events']]=\
                            avDaysBetweenEvents
                    # Low flow specific stats:
                    elif ('_avLowFlowDays' in col):
                        trunCol = col[:-len('_avLowFlowDays')]
                        if 'Average days above low flow' in things_to_calc:
                            avEventDays = get_avNumEvents(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario,
                                                                           'Average days above low flow']]=\
                            avEventDays
                    # CF specific stats:
                    elif ('_ctfDaysPerYear' in col):
                        trunCol = col[:-len('_ctfDaysPerYear')]
                        if 'Average CtF days per year' in things_to_calc:
                            avCtfDaysPerYear = get_avCtfDaysPerYear(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 
                                                                           'Average CtF days per year']]=\
                            avCtfDaysPerYear
                    elif ('_avLenCtfSpells' in col):
                        trunCol = col[:-len('_avLenCtfSpells')]
                        if 'Average length CtF spells' in things_to_calc:
                            avLenCtfSpells = get_avLenCtfSpells(site_results[planUnit][col])
                            df_multi_level.loc[idx[[site], [planUnit], [trunCol]], idx[scenario, 
                                                                           'Average length CtF spells']]=\
                            avLenCtfSpells

    return df_multi_level