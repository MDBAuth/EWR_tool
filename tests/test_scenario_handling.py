from pathlib import Path
from datetime import date, datetime
import re
import numpy as np

import pandas as pd
from pandas._testing import assert_frame_equal
import pytest

from py_ewr import scenario_handling, data_inputs

def test_match_MDBA_nodes():
    '''
    1. Ensure dataframe with flows and levels is split into two dataframes (one flow and one level dataframe)
    '''
    # Set up input data and pass to test function:
    model_metadata = data_inputs.get_MDBA_codes()
    data_df = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                'EUSTDS-1-8': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                'EUSTUS-35-8': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                }
    df = pd.DataFrame(data = data_df)
    df = df.set_index('Date')
    
    df_F, df_L = scenario_handling.match_MDBA_nodes(df, model_metadata, 'py_ewr/parameter_metadata/parameter_sheet.csv')
    
    # Set up expected outputs and test:
    data_expected_df_L = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                            '414209': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                            }
    expected_df_L = pd.DataFrame(data_expected_df_L)
    expected_df_L = expected_df_L.set_index('Date') 
    data_expected_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                            '414203': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                            }
    expected_df_F = pd.DataFrame(data_expected_df_F)
    expected_df_F = expected_df_F.set_index('Date')
    
    assert_frame_equal(df_F, expected_df_F)
    assert_frame_equal(df_L, expected_df_L)

def test_match_NSW_nodes():
    '''
    1. Check NSW model nodes are mapped correctly to their gauges
    '''
    # Set up input data and pass to test function:
    model_metadata = data_inputs.get_NSW_codes()
    data_df = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                'Gauge: 410134 Billabong Creek at Darlot: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                'Gauge: 410016 Billabong Creek at Jerilderie: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                }
    df = pd.DataFrame(data = data_df)
    df = df.set_index('Date')
    
    df_F, df_L = scenario_handling.match_NSW_nodes(df, model_metadata)
    
    # Set up expected outputs and test:
    data_expected_df_L = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))}
    expected_df_L = pd.DataFrame(data_expected_df_L)
    expected_df_L = expected_df_L.set_index('Date')
    
    data_expected_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                            '410134': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                            '410016': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                            }
    expected_df_F = pd.DataFrame(data_expected_df_F)
    expected_df_F = expected_df_F.set_index('Date')
    
    assert_frame_equal(df_F, expected_df_F)
    assert_frame_equal(df_L, expected_df_L, check_column_type=False)

def test_extract_gauge_from_string():
    '''
    1. Check gauge string is pulled from various strings likely encountered during standard program run
    '''
    # Test 1
    input_string = '409025'
    return_gauge = scenario_handling.extract_gauge_from_string(input_string)
    expected_gauge = '409025'
    assert return_gauge == expected_gauge
    # Test 2 - TODO - is this test necessary?
    # input_string = ' 409025 '
    # return_gauge = scenario_handling.extract_gauge_from_string(input_string)
    # expected_gauge = '409025'
    # assert return_gauge == expected_gauge
    # Test 3
    # input_string = '409025---999'
    # return_gauge = scenario_handling.extract_gauge_from_string(input_string)
    # expected_gauge = '409025'
    # assert return_gauge == expected_gauge

    

## changed to 100 year
def test_cleaner_standard_timeseries():
    '''
    1. Check date formatting and correct allocationo of gauge data to either flow/level dataframes
    2. TODO: Add test to assess when there are missing dates
    3. TODO: Add test for both date types
    '''
    # CHECK 1
    # Set up input data and pass to test function:
    date_start = '1900-07-01'
    date_end = '2000-06-30'
    date_range = pd.date_range(start= datetime.strptime(date_start, '%Y-%m-%d'), end = datetime.strptime(date_end, '%Y-%m-%d'), freq='D')
 
    data_for_input_df = {'Date': date_range, '409025_flow': [50]*len(date_range)}
    input_df = pd.DataFrame(data_for_input_df)
    input_df = input_df.set_index('Date')
    df_f, df_l = scenario_handling.cleaner_standard_timeseries(input_df)

    # Set up expected data and test:
    expected_df_flow = input_df.copy(deep=True)
    expected_df_flow.columns = ['409025']
    expected_df_level = pd.DataFrame(index = expected_df_flow.index)

    expected_df_flow.index = date_range
    expected_df_flow.index.name = 'Date'

    assert_frame_equal(expected_df_level, df_l)
    assert_frame_equal(expected_df_flow, df_f)

    # CHECK 2

    date_start = '1900-07-01'
    date_end = '2000-06-30'
    date_range = pd.date_range(start= datetime.strptime(date_start, '%Y-%m-%d'), end = datetime.strptime(date_end, '%Y-%m-%d'), freq='D')

    data_for_input_df = {'Date': date_range, '409025_flow': [50]*len(date_range)}
    input_df = pd.DataFrame(data_for_input_df)
    input_df = input_df.set_index('Date')
    cut_df = input_df.drop([datetime(1900, 7, 5),datetime(1900,7,8)], axis = 0)

    df_f, df_l = scenario_handling.cleaner_standard_timeseries(cut_df)
    
    expected_df_flow = cut_df.copy(deep = True)
    expected_df_flow.loc[datetime(1900, 7, 5), '409025_flow'] = np.nan
    expected_df_flow.loc[datetime(1900, 7, 8), '409025_flow'] = np.nan
    expected_df_flow.sort_index(inplace=True)
    expected_df_flow.columns = ['409025']
    expected_df_level = pd.DataFrame(index = expected_df_flow.index)

    expected_df_flow.index = date_range
    expected_df_flow.index.name = 'Date'

    assert_frame_equal(expected_df_level, df_l)
    assert_frame_equal(expected_df_flow, df_f)

    
def test_cleaner_NSW():
    '''
    1. Test date formatting is applied correctly
    '''
    # Set up input data and pass to test function:
    model_metadata = data_inputs.get_NSW_codes()
    dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
    date_strings = dates.format(formatter=lambda x: x.strftime('%Y-%m-%d'))
    data_df = {'Date': date_strings,
                'Gauge: YCB_410134_BillabongCreek@Darlot: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                'Gauge: YCB_410016 Billabong Creek @ Jerilderie: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                }
    df = pd.DataFrame(data = data_df)
    df_clean = scenario_handling.cleaner_NSW(df)
    # Set up expected output data and test:
    expected_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
    data_expected_df_F = {'Date': expected_dates,
                            'Gauge: YCB_410134_BillabongCreek@Darlot: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                            'Gauge: YCB_410016 Billabong Creek @ Jerilderie: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                            }
    expected_df_F = pd.DataFrame(data_expected_df_F)
    expected_df_F['Date'] = expected_df_F['Date'].apply(lambda x: x.to_period(freq='D'))
    expected_df_F = expected_df_F.set_index('Date')
    assert_frame_equal(df_clean, expected_df_F)
    
    
def test_cleaner_MDBA():
    '''
    1. Test date formatting is applied correctly
    '''
    # Set up input data and pass to test function:
    data_df = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                'EUSTDS-1-8': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                'EUSTUS-35-8': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                }
    df = pd.DataFrame(data = data_df)
    df['Dy'], df['Mn'], df['Year'] = df['Date'].dt.day, df['Date'].dt.month, df['Date'].dt.year
    df = df.drop(['Date'], axis = 1)
    
    df_clean = scenario_handling.cleaner_MDBA(df)
    # Set up expected output data and test:
    data_expected_df = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                            'EUSTDS-1-8': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                            'EUSTUS-35-8': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                            }
    expected_df = pd.DataFrame(data_expected_df)
    expected_df = expected_df.set_index('Date')
    
    assert_frame_equal(df_clean, expected_df)
    
    
def test_build_NSW_columns():
    '''
    1. Check data columns are renamed based on header data
    '''
    # Load in test input data and pass to test function:
    input_file = 'unit_testing_files/NSW_source_res_test_file.csv'
    data, header = scenario_handling.unpack_model_file(input_file, 'Date', 'Field')
    data = scenario_handling.build_NSW_columns(data, header)
    # Load in test expected output data and test:
    expected_result = pd.read_csv('unit_testing_files/NSW_source_res_test_file_flow_result.csv')
    assert_frame_equal(data, expected_result)
    
    
def test_build_MDBA_columns():
    '''
    1. Ensure data column names are correctly taken from the header data
    '''
    # Load input data and send to test function:
    input_file = 'unit_testing_files/MDBA_bigmod_test_file_buggy_ID.csv'
    data, header = scenario_handling.unpack_model_file(input_file, 'Dy', 'Field')
    data = scenario_handling.build_MDBA_columns(data, header)
    data = data.astype({'Dy': 'int32', 'Mn': 'int32', 'Year': 'int32'})
    # Load expected output data, format, and test:
    expected_result = pd.read_csv('unit_testing_files/MDBA_bigmod_test_file_flow_result.csv')
    expected_result = expected_result.astype({'Dy': 'int32', 'Mn': 'int32', 'Year': 'int32'})
    assert_frame_equal(data, expected_result)        


def test_unpack_model_file():
    '''
    1. Test MDBA style file ingestion
    2. Test NSW style file ingestion
    '''
    
    # Test 1
    # Load test data and expected output data
    file_to_pass = 'unit_testing_files/MDBA_bigmod_test_file.csv'
    expected_header = pd.read_csv('unit_testing_files/MDBA_bigmod_test_file_header_result.csv', dtype={'Site':'str', 'Measurand': 'str', 'Quality': 'str'})
    expected_flow = pd.read_csv('unit_testing_files/MDBA_bigmod_test_file_flow_result.csv', dtype={'Dy':'int', 'Mn': 'int', 'Year': 'int'})
    # Pass to test function and test
    flow, header = scenario_handling.unpack_model_file(file_to_pass, 'Dy', 'Field')
    assert_frame_equal(flow, expected_flow)
    assert_frame_equal(header, expected_header)
    #------------------------------------------
    # Test 2
    # Load test data and expected output data:
    file_to_pass = 'unit_testing_files/NSW_source_res_test_file.csv'
    expected_header = pd.read_csv('unit_testing_files/NSW_source_res_test_file_header_result.csv')
    expected_flow = pd.read_csv('unit_testing_files/NSW_source_res_test_file_flow_result.csv')
    expected_flow.columns = ['Date', '1>Data Sources>Data Sources@Climate Data@FAO56_res_csv@049050_SILO_FAO56.csv', '2>Data Sources>Data Sources@Climate Data@Mwet_res_csv@049050_SILO_Mwet.csv']
    # Pass to test function and test
    flow, header = scenario_handling.unpack_model_file(file_to_pass, 'Date', 'Field')  
    assert_frame_equal(flow, expected_flow)
    assert_frame_equal(header, expected_header)

    
# def test_gauge_only_column():
#     '''
#     1. Check gauge strings are pulled from column names and saved in place of the original string 
#     '''
#     file_to_pass ='unit_testing_files/NSW_10000yr_test_file.csv'
#     flow = scenario_handling.gauge_only_column(file_to_pass)
    
#     expected_flow = pd.read_csv('unit_testing_files/NSW_10000yr_test_file.csv', index_col = 'Date')
#     expected_flow.columns = ['418013']
    
#     assert_frame_equal(flow, expected_flow)

def test_scenario_handler_class(scenario_handler_expected_detail, scenario_handler_instance):
   
    detailed = scenario_handler_instance.pu_ewr_statistics
    
    detailed['Low_flow_EWRs_Bidgee_410007']['410007']['Upper Yanco Creek'].index = detailed['Low_flow_EWRs_Bidgee_410007']['410007']['Upper Yanco Creek'].index.astype('int64')

    assert_frame_equal(detailed['Low_flow_EWRs_Bidgee_410007']['410007']['Upper Yanco Creek'], scenario_handler_expected_detail)


def test_get_all_events(scenario_handler_instance):

    all_events = scenario_handler_instance.get_all_events()
    assert type(all_events) == pd.DataFrame
    assert all_events.shape == (26, 10)
    assert all_events.columns.to_list() == ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                     'eventDuration', 'eventLength', 'Multigauge']
        
def test_get_all_interEvents(scenario_handler_instance):

    all_interEvents = scenario_handler_instance.get_all_interEvents()
    assert type(all_interEvents) == pd.DataFrame
    assert all_interEvents.shape == (26, 7)
    assert all_interEvents.columns.to_list() == ['scenario', 'gauge', 'pu', 'ewr', 'startDate', 'endDate', 'interEventLength']

def test_get_all_successful_events(scenario_handler_instance):

    all_successful_events = scenario_handler_instance.get_all_successful_events()
    assert type(all_successful_events) == pd.DataFrame
    assert all_successful_events.shape == (24, 10)
    assert all_successful_events.columns.to_list() == ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                                       'eventDuration', 'eventLength', 'Multigauge']

def test_get_all_successful_interEvents(scenario_handler_instance):

    all_successful_interEvents = scenario_handler_instance.get_all_successful_interEvents()
    assert type(all_successful_interEvents) == pd.DataFrame
    assert all_successful_interEvents.shape == (24, 7)
    assert all_successful_interEvents.columns.to_list() == ['scenario', 'gauge', 'pu', 'ewr', 'startDate', 'endDate', 'interEventLength']

def test_get_yearly_ewr_results(scenario_handler_instance):

    yearly_results = scenario_handler_instance.get_yearly_ewr_results()

    assert type(yearly_results) == pd.DataFrame
    assert yearly_results.shape == (126, 21)
    assert yearly_results.columns.to_list() == ['Year', 'eventYears', 'numAchieved', 'numEvents', 'numEventsAll',
        'eventLength', 'eventLengthAchieved',
       'totalEventDays', 'totalEventDaysAchieved','maxEventDays', 'maxRollingEvents', 'maxRollingAchievement', 'missingDays',
       'totalPossibleDays', 'ewrCode', 'scenario', 'gauge', 'pu', 'Multigauge', 
       'rollingMaxInterEvent', 'rollingMaxInterEventAchieved'] 

def test_get_ewr_results(scenario_handler_instance):

    ewr_results = scenario_handler_instance.get_ewr_results()
    assert type(ewr_results) == pd.DataFrame
    assert ewr_results.shape == (21, 19)
    assert ewr_results.columns.to_list() == ['Scenario', 'Gauge', 'PlanningUnit', 'EwrCode', 'Multigauge','EventYears',
       'Frequency', 'TargetFrequency', 'AchievementCount',
       'AchievementPerYear', 'EventCount', 'EventCountAll', 'EventsPerYear', 'EventsPerYearAll',
       'AverageEventLength', 'ThresholdDays', 
       'MaxInterEventYears', 'NoDataDays', 'TotalDays']
    
def test_any_cllmm_to_process(gauge_results):
    result = scenario_handling.any_cllmm_to_process(gauge_results)
    assert result == True

    