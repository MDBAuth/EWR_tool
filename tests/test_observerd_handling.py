from pathlib import Path
import copy
from datetime import date
import re

import pandas as pd
from pandas._testing import assert_frame_equal
import pytest

from py_ewr import observed_handling, data_inputs

BASE_PATH = Path(__file__).resolve().parents[1]

def test_observed_cleaner():
    '''
    1. Run sample data through and compare the expected output sample data
    '''
    # Test 1
    # Load sample input data and pass to function
    input_df = pd.read_csv( 'unit_testing_files/observed_flows_test_input.csv')
    dates = {'start_date': date(2014, 1, 1), 'end_date': date(2020, 1, 1)}
    result = observed_handling.observed_cleaner(input_df, dates)
    # Load sample output data and test
    output_df = 'unit_testing_files/observed_flows_test_output.csv'
    expected_result = pd.read_csv(output_df, index_col = 'Date')
    expected_result.index = pd.to_datetime(expected_result.index, format='%Y-%m-%d')
    expected_result.columns = ['419039']
    assert_frame_equal(result, expected_result)

def test_one_gauge_per_column():
    '''
    1. Run sample data through and compare to the expected output sample data
    '''
    # Test 1
    # load and format expected output data
    output_df = 'unit_testing_files/observed_flows_test_output.csv'
    expected_result = pd.read_csv(output_df, index_col = 'Date')
    expected_result.index = pd.to_datetime(expected_result.index, format='%Y-%m-%d')
    expected_result.columns = ['419039']
    # Load input data and pass to function
    input_dataframe = pd.read_csv('unit_testing_files/observed_flows_test_input.csv')
    gauge_iter = 419039
    input_dataframe['Date'] = expected_result.index
    result = observed_handling.one_gauge_per_column(input_dataframe, gauge_iter)
    # assert equal test
    assert_frame_equal(result, expected_result)
    
    
def test_remove_data_with_bad_QC():
    '''
    1. Run sample data through and compare to the expected output sample data to ensure bad data is removed
    '''        
    # Test 1
    # Load sample input data:
    input_dataframe = pd.read_csv('unit_testing_files/observed_flows_test_input_QC.csv', index_col = 'Date')
    gauge_iter = 419039
    qc_codes = data_inputs.get_bad_QA_codes()
    # Load output sample data
    expected_df = pd.read_csv('unit_testing_files/observed_flows_test_output_QC.csv', index_col = 'Date')

    # Throw to the function
    df = observed_handling.remove_data_with_bad_QC(input_dataframe, qc_codes)
    
    # Test for equality
    assert_frame_equal(df, expected_df)


def test_categorise_gauges():
    '''
    1. gauges in all categories
    2. gauges outside cats
    '''
    
    level = ['425020', '425022', '425023', '414209', '4260501', '4260508','4260506']
    flow = ['414203', '425010', '4260507', '4260505',  '421090', '421088', '421088', '421090', '421090', '421088', '421088', '421090', '421090', '421088', '421088', '421090', '421090', '421088', '421088', '421090', '409023', '409003']
    all_gauges = level + flow
    
    f, l = observed_handling.categorise_gauges(all_gauges)

    expected_level = copy.deepcopy(level)
    expected_flow = copy.deepcopy(flow)
    expected_flow = expected_flow + ['421022'] # Add in this one as it will be getting picked up for being associated with a simultaneious gauge
    assert set(f) == set(expected_flow)
    assert set(l) == set(expected_level)
  
def test_observed_handler():
    '''
    1. Test each part of the function are working correctly and producing an overall expected output
    '''
    
    # Set up input parameters and pass to test function
    gauges = ['419039']
    dates = {'start_date': date(2020, 7, 1), 'end_date': date(2021, 6, 30)}
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'

    detailed, summary = observed_handling.observed_handler(gauges, dates, allowance, climate)
    
    # Load and format expected results
    expected_detailed_results = pd.read_csv('unit_testing_files/detailed_results_observed.csv', index_col = 0)
    expected_detailed_results.index = expected_detailed_results.index.astype('object')
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
    
    assert_frame_equal(detailed['observed']['419039']['Boggabri to Wee Waa'], expected_detailed_results)


def test_observed_handler_class(observed_handler_expected_detail, observed_handler_instance):

    observed_handler_instance.process_gauges()

    detailed = observed_handler_instance.pu_ewr_statistics

    assert_frame_equal(detailed['observed']['419039']['Boggabri to Wee Waa'], observed_handler_expected_detail)

def test_get_all_events(observed_handler_instance):

    all_events = observed_handler_instance.get_all_events()
    assert type(all_events) == pd.DataFrame
    assert all_events.shape == (56, 9)
    assert all_events.columns.to_list() == ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                            'eventDuration', 'eventLength']

def test_get_yearly_ewr_results(observed_handler_instance):

    yearly_results = observed_handler_instance.get_yearly_ewr_results()
    assert type(yearly_results) == pd.DataFrame
    assert yearly_results.shape == (24, 14)
    assert yearly_results.columns.to_list() == ['Year', 'eventYears', 'numAchieved', 'numEvents', 'eventLength',
       'totalEventDays', 'maxEventDays', 'daysBetweenEvents', 'missingDays',
       'totalPossibleDays', 'ewrCode', 'scenario', 'gauge', 'pu']

def test_get_ewr_results(observed_handler_instance):

    ewr_results = observed_handler_instance.get_ewr_results()
    assert type(ewr_results) == pd.DataFrame
    assert ewr_results.shape == (24, 18)
    assert ewr_results.columns.to_list() == ['Scenario', 'Gauge', 'PlanningUnit', 'EwrCode', 'EventYears',
       'Frequency', 'TargetFrequency', 'AchievementCount',
       'AchievementPerYear', 'EventCount', 'totalEvents', 'EventsPerYear',
       'AverageEventLength', 'ThresholdDays', 'InterEventExceedingCount',
       'MaxInterEventYears', 'NoDataDays', 'TotalDays']
