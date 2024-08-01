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
    expected_result.index = expected_result.index.to_period()
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
    gauges = ['422015', '422030', '416001', '421088', '414203', 'A4260527', 'A4260633', '425020', 'A4261002', 'A4261002']

    flow = ['422015', '422030', '421088', '414203', '416001', 'A4261002']
    level = ['422015', '422030', '414209', 'A4260527', 'A4260633']
    lake_level = ['A4260527', 'A4260633','425020']

    expected_flow = ['422015', '422030', '421088', '421090', '414203', '416001', 'A4261002']
    expected_level = ['422015', '422030', '414209', 'A4260527','A4261133', 'A4260524', 'A4260574', 'A4260575', 'A4260633','A4261209', 'A4261165']
    expected_lake_level = ['425020', 'A4260527','A4261133', 'A4260524', 'A4260574', 'A4260575', 'A4260633','A4261209', 'A4261165']

    all_gauges = level + flow + lake_level
    
    f, l, ll = observed_handling.categorise_gauges(all_gauges)

    # expected_level = copy.deepcopy(level)
    # expected_flow = copy.deepcopy(flow)
    # expected_lake_level = copy.deepcopy(lake_level)

    assert set(f) == set(expected_flow)
    assert set(l) == set(expected_level)
    assert set(ll) == set(expected_lake_level)


def test_observed_handler_class(observed_handler_expected_detail, observed_handler_instance):

    observed_handler_instance.process_gauges()

    detailed = observed_handler_instance.pu_ewr_statistics

    detailed['observed']['419039']['Boggabri to Wee Waa'].index = detailed['observed']['419039']['Boggabri to Wee Waa'].index.astype('int64')

    assert_frame_equal(detailed['observed']['419039']['Boggabri to Wee Waa'], observed_handler_expected_detail)

def test_get_all_events(observed_handler_instance):

    all_events = observed_handler_instance.get_all_events()
    assert type(all_events) == pd.DataFrame
    assert all_events.shape == (76, 10)
    assert all_events.columns.to_list() == ['scenario', 'gauge', 'pu', 'ewr', 'waterYear', 'startDate', 'endDate',
                                     'eventDuration', 'eventLength', 'Multigauge']

def test_get_yearly_ewr_results(observed_handler_instance):

    yearly_results = observed_handler_instance.get_yearly_ewr_results()
    assert type(yearly_results) == pd.DataFrame
    assert yearly_results.shape == (24, 21)
    assert yearly_results.columns.to_list() == ['Year', 'eventYears', 'numAchieved', 'numEvents', 'numEventsAll',
       'eventLength', 'eventLengthAchieved',
       'totalEventDays', 'totalEventDaysAchieved','maxEventDays', 'maxRollingEvents', 'maxRollingAchievement','missingDays',
       'totalPossibleDays', 'ewrCode', 'scenario', 'gauge', 'pu', 'Multigauge', 'rollingMaxInterEvent', 'rollingMaxInterEventAchieved']

def test_get_ewr_results(observed_handler_instance):

    ewr_results = observed_handler_instance.get_ewr_results()
    assert type(ewr_results) == pd.DataFrame
    assert ewr_results.shape == (24, 19)
    assert ewr_results.columns.to_list() == ['Scenario', 'Gauge', 'PlanningUnit', 'EwrCode', 'Multigauge','EventYears',
       'Frequency', 'TargetFrequency', 'AchievementCount',
       'AchievementPerYear', 'EventCount', 'EventCountAll', 'EventsPerYear', 'EventsPerYearAll',
       'AverageEventLength', 'ThresholdDays',
       'MaxInterEventYears', 'NoDataDays', 'TotalDays']
