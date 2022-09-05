from datetime import date
import pandas as pd
from pandas._testing import assert_frame_equal
import pytest


from py_ewr import data_inputs, summarise_results


def test_get_frequency():
    '''
    Series input to get_frequency function will only be made up of combination of 1's or 0's.
    1. Test to see if function is returning frequency of 1 occurrence out of all years. 
    2. Test to see if function handles no occurrence of 1's
    '''
    # Test 1
    input_series = pd.Series(index=[1895, 1896, 1897, 1898, 1899], data=[0,1,1,1,0])
    f = summarise_results.get_frequency(input_series)
    expected_f = 60
    assert f == expected_f
    # ----------------------------
    # Test 2
    input_series = pd.Series(index=[1895, 1896, 1897, 1898, 1899], data=[0,0,0,0,0])
    f = summarise_results.get_frequency(input_series)
    expected_f = 0
    assert f == expected_f

@pytest.mark.parametrize("ewr,cols,expected",
        [("CF1", ["CF1_foo","CF1_bar","CF2_foo","CF2_bar"],["CF1_foo","CF1_bar"]),
        ("CF3", ["CF1_foo","CF1_bar","CF2_foo","CF2_bar"],[])],
)
def test_get_ewr_columns(ewr, cols, expected):
    result = summarise_results.get_ewr_columns(ewr, cols)
    assert result == expected

@pytest.mark.parametrize("cols,expected",
        [(["CF1_foo","CF1_bar","CF2_foo","CF2_bar"],["foo","bar","foo","bar"]),
        ],
)
def test_get_columns_attributes(cols, expected):
    result = summarise_results.get_columns_attributes(cols)

    assert result == expected


def test_get_ewrs(pu_df):
    result = summarise_results.get_ewrs(pu_df)
    assert result == ["CF1_a"]


def test_pu_dfs_to_process(detailed_results, pu_df):
    result = summarise_results.pu_dfs_to_process(detailed_results)
    assert result == [{ "scenario" : 'observed',
                         "gauge" : '419001',
                         "pu" : 'Keepit to Boggabri',
                         "pu_df" : pu_df },
                         { "scenario" : 'observed',
                         "gauge" : '419002',
                         "pu" : 'Keepit to Boggabri',
                         "pu_df" : pu_df }]

def test_process_df(item_to_process):
    result = summarise_results.process_df(**item_to_process)
    columns = result.columns.to_list()
    assert columns == ['Year', 'eventYears', 'numAchieved', 'numEvents', 'numEventsAll','eventLength', 'eventLengthAchieved',
       'totalEventDays', 'totalEventDaysAchieved','daysBetweenEvents', 'missingDays',
       'totalPossibleDays', 'ewrCode', 'scenario', 'gauge', 'pu']
    assert result.shape == (2, 16)

def test_process_df_results(items_to_process):
    result = summarise_results.process_df_results(items_to_process)
    columns = result.columns.to_list()
    assert columns == ['Year', 'eventYears', 'numAchieved', 'numEvents', 'numEventsAll','eventLength', 'eventLengthAchieved',
       'totalEventDays', 'totalEventDaysAchieved','daysBetweenEvents', 'missingDays',
       'totalPossibleDays', 'ewrCode', 'scenario', 'gauge', 'pu']
    assert result.shape == (4, 16)

def test_get_events_to_process(gauge_events):
    result = summarise_results.get_events_to_process(gauge_events)
    assert result == [ { "scenario" : 'observed',
                          "gauge" : '419001',
                          "pu" : 'Keepit to Boggabri',
                          "ewr": 'CF1_a',
                          "ewr_events" : {2010: [],
                            2011: [],
                            2012: [],
                            2013: [],
                            2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}
                            },
                       {"scenario" : 'observed',
                          "gauge" : '419002',
                          "pu" : 'Keepit to Boggabri',
                          "ewr": 'CF1_a',
                          "ewr_events" : {2010: [],
                            2011: [],
                            2012: [],
                            2013: [],
                            2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}}    
                       ]

def test_count_events(yearly_events):
    result = summarise_results.count_events(yearly_events)
    assert result == 1

def test_sum_events(yearly_events):
    result = summarise_results.sum_events(yearly_events)
    assert result == 6

def test_process_yearly_events(event_item_to_process):
    result = summarise_results.process_yearly_events(**event_item_to_process)
    assert type(result) == pd.DataFrame
    assert result.to_dict() == {'scenario': {0: 'observed'},
                                            'gauge': {0: '419001'},
                                            'pu': {0: 'Keepit to Boggabri'},
                                            'ewrCode': {0: 'CF1_a'},
                                            'totalEvents': {0: 1},
                                            'totalEventDays': {0: 6},
                                            'averageEventLength': {0: 6.0}}

def test_process_ewr_events_stats(event_items_to_process):
    result = summarise_results.process_ewr_events_stats(event_items_to_process)
    assert type(result) == pd.DataFrame
    assert result.shape == (2, 7)
    assert result.to_dict() == {'scenario': {0: 'observed', 1: 'observed'},
                                'gauge': {0: '419001', 1: '419002'},
                                'pu': {0: 'Keepit to Boggabri', 1: 'Keepit to Boggabri'},
                                'ewrCode': {0: 'CF1_a', 1: 'CF1_a'},
                                'totalEvents': {0: 1, 1: 1},
                                'totalEventDays': {0: 6, 1: 6},
                                'averageEventLength': {0: 6.0, 1: 6.0}}

def test_summarise(detailed_results, gauge_events):
    result = summarise_results.summarise(input_dict=detailed_results, events=gauge_events)
    assert type(result) == pd.DataFrame

   
def test_process_all_yearly_events(event_item_to_process):
    result = summarise_results.process_all_yearly_events(**event_item_to_process)
    assert result.to_dict() == {'scenario': {0: 'observed'},
                                'gauge': {0: '419001'},
                                'pu': {0: 'Keepit to Boggabri'},
                                'ewr': {0: 'CF1_a'},
                                'waterYear': {0: 2014},
                                'startDate': {0: date(2020, 11, 30)},
                                'endDate': {0:   date(2020, 12, 5)},
                                'eventDuration': {0: 6},
                                'eventLength': {0: 6}}

def test_process_all_events_results(event_items_to_process):
    result = summarise_results.process_all_events_results(event_items_to_process)
    assert result.to_dict() == {'scenario': {0: 'observed', 1: 'observed'},
                                'gauge': {0: '419001', 1: '419002'},
                                'pu': {0: 'Keepit to Boggabri', 1: 'Keepit to Boggabri'},
                                'ewr': {0: 'CF1_a', 1: 'CF1_a'},
                                'waterYear': {0: 2014, 1: 2014},
                                'startDate': {0: date(2020, 11, 30), 1: date(2020, 11, 30)},
                                'endDate': {0: date(2020, 12, 5), 1: date(2020, 12, 5)},
                                'eventDuration': {0: 6, 1: 6},
                                'eventLength': {0: 6, 1: 6}}