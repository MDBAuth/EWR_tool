from pathlib import Path
from datetime import date, datetime
import re

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
    
    df_F, df_L = scenario_handling.match_MDBA_nodes(df, model_metadata)
    
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
                'Gauge: YCB_410134_BillabongCreek@Darlot: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                'Gauge: YCB_410016 Billabong Creek @ Jerilderie: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
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
    assert_frame_equal(df_L, expected_df_L)

def test_extract_gauge_from_string():
    '''
    1. Check gauge string is pulled from various strings likely encountered during standard program run
    '''
    # Test 1
    input_string = '409025'
    return_gauge = scenario_handling.extract_gauge_from_string(input_string)
    expected_gauge = '409025'
    assert return_gauge == expected_gauge
    # Test 2
    input_string = ' 409025 '
    return_gauge = scenario_handling.extract_gauge_from_string(input_string)
    expected_gauge = '409025'
    assert return_gauge == expected_gauge
    # Test 3
    input_string = '409025---999'
    return_gauge = scenario_handling.extract_gauge_from_string(input_string)
    expected_gauge = '409025'
    assert return_gauge == expected_gauge

    

## changed to 100 year
def test_cleaner_IQQM_100yr():
    '''
    1. Check date formatting and correct allocationo of gauge data to either flow/level dataframes
    '''
    # Set up input data and pass to test function:
    date_start = '1900-07-01'
    date_end = '2000-06-30'
    date_range = pd.period_range(date_start, date_end, freq = 'D')
    data_for_input_df = {'Date': date_range, '409025': [50]*len(date_range)}
    input_df = pd.DataFrame(data_for_input_df)
    str_df = input_df.copy(deep=True)
    str_df['Date'] = str_df['Date'].astype('str')
    def add_0 (row):
        j = row.split('-')
        if len(j[0]) < 4:
            new_row = '0'+ row
        else:
            new_row = row
        return new_row
    str_df['Date'] = str_df['Date'].apply(add_0)
    str_df = str_df.set_index('Date')
    df_f, df_l = scenario_handling.cleaner_IQQM_10000yr(str_df)
    
    # Set up expected data and test:
    expected_df_flow = input_df.copy(deep=True)
    expected_df_flow = expected_df_flow.set_index('Date')
    expected_df_flow.columns = ['409025']
    expected_df_level = pd.DataFrame(index = expected_df_flow.index)
    
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
    data_expected_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                            'Gauge: YCB_410134_BillabongCreek@Darlot: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10,
                            'Gauge: YCB_410016 Billabong Creek @ Jerilderie: Downstream Flow': [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10
                            }
    expected_df_F = pd.DataFrame(data_expected_df_F)
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
    data_expected_df = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
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
    input_file = 'unit_testing_files/MDBA_bigmod_test_file.csv'
    data, header = scenario_handling.unpack_model_file(input_file, 'Dy', 'Field')
    data = scenario_handling.build_MDBA_columns(data, header)
    data = data.astype({'Dy': 'float64'})
    # Load expected output data, format, and test:
    expected_result = pd.read_csv('unit_testing_files/MDBA_bigmod_test_file_flow_result.csv')
    expected_result = expected_result.astype({'Dy': 'float64'})
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

    
def test_unpack_IQQM_10000yr():
    '''
    1. Check gauge strings are pulled from column names and saved in place of the original string 
    '''
    file_to_pass ='unit_testing_files/NSW_10000yr_test_file.csv'
    flow = scenario_handling.unpack_IQQM_10000yr(file_to_pass)
    
    expected_flow = pd.read_csv('unit_testing_files/NSW_10000yr_test_file.csv', index_col = 'Date')
    expected_flow.columns = ['418013']
    
    assert_frame_equal(flow, expected_flow)

# @pytest.mark.xfail
def test_scenario_handler():
    '''things to test here:
    1. Ensure all parts of the function generate expected output
    '''
    # Testing the MDBA bigmod format:
    # Input params
    scenarios = {'Low_flow_EWRs_Bidgee_410007': 'unit_testing_files/Low_flow_EWRs_Bidgee_410007.csv'}
    model_format = 'Bigmod - MDBA'
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass to the function
    detailed, summary = scenario_handling.scenario_handler(scenarios, model_format, allowance, climate)
    # Expected output params
    expected_detailed_results = pd.read_csv('unit_testing_files/detailed_results_test.csv', index_col=0)
    expected_detailed_results.index = expected_detailed_results.index.astype('object')
    print(expected_detailed_results.index.astype('object'))
    cols = expected_detailed_results.columns[expected_detailed_results.columns.str.contains('eventLength')]
    expected_detailed_results[cols] = expected_detailed_results[cols].astype('float64')
    for col in expected_detailed_results:
        if 'daysBetweenEvents' in col:
            for i, val in enumerate(expected_detailed_results[col]):
                new = expected_detailed_results[col].iloc[i]
                if new == '[]':
                    new_list = []
                else:
                    new = re.sub('\[', '', new)
                    new = re.sub('\]', '', new)
                    new = new.split(',')
                    new_list = []
                    for days in new:
                        new_days = days.strip()
                        new_days = int(new_days)
                        new_list.append(new_days)

                expected_detailed_results[col].iloc[i] = new_list
    # Test
    assert_frame_equal(detailed['Low_flow_EWRs_Bidgee_410007']['410007']['Upper Yanco Creek'], expected_detailed_results)

# @pytest.mark.xfail
def test_scenario_handler_class(scenario_handler_expected_detail, scenario_handler_instance):
   
    detailed = scenario_handler_instance.pu_ewr_statistics
    
    # Test
    assert_frame_equal(detailed['Low_flow_EWRs_Bidgee_410007']['410007']['Upper Yanco Creek'], scenario_handler_expected_detail)


# @pytest.mark.xfail(raises=TypeError, reason="yearly events on Nest ewr missing the date")
def test_get_all_events(scenario_handler_instance):

    all_events = scenario_handler_instance.get_all_events()
    assert type(all_events) == pd.DataFrame
    # assert all_events.shape == (56, 9)
    assert all_events.columns.to_list() == ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                            'eventDuration', 'eventLength']

def test_get_yearly_ewr_results(scenario_handler_instance):

    yearly_results = scenario_handler_instance.get_yearly_ewr_results()
    assert type(yearly_results) == pd.DataFrame
    assert yearly_results.shape == (126, 16)
    assert yearly_results.columns.to_list() == ['Year', 'eventYears', 'numAchieved', 'numEvents', 'eventLength',
       'totalEventDays', 'maxEventDays', 'maxRollingEvents', 'maxRollingAchievement','daysBetweenEvents', 'missingDays',
       'totalPossibleDays', 'ewrCode', 'scenario', 'gauge', 'pu']

def test_get_ewr_results(scenario_handler_instance):

    ewr_results = scenario_handler_instance.get_ewr_results()
    assert type(ewr_results) == pd.DataFrame
    assert ewr_results.shape == (21, 18)
    assert ewr_results.columns.to_list() == ['Scenario', 'Gauge', 'PlanningUnit', 'EwrCode', 'EventYears',
       'Frequency', 'TargetFrequency', 'AchievementCount',
       'AchievementPerYear', 'EventCount', 'totalEvents', 'EventsPerYear',
       'AverageEventLength', 'ThresholdDays', 'InterEventExceedingCount',
       'MaxInterEventYears', 'NoDataDays', 'TotalDays']