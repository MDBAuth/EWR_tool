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
       'totalEventDays', 'totalEventDaysAchieved', "rollingMaxInterEventAchieved", 'missingDays',
       'totalPossibleDays', 'ewrCode', 'scenario', 'gauge', 'pu']
    assert result.shape == (2, 16)

def test_process_df_results(items_to_process):
    result = summarise_results.process_df_results(items_to_process)
    columns = result.columns.to_list()
    assert columns == ['Year', 'eventYears', 'numAchieved', 'numEvents', 'numEventsAll','eventLength', 'eventLengthAchieved',
       'totalEventDays', 'totalEventDaysAchieved', "rollingMaxInterEventAchieved", 'missingDays',
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


def test_events_to_interevents(interEvent_item_to_process):

    all_events_df = pd.DataFrame(data = interEvent_item_to_process)
    start_date = date(1901,7,1)
    end_date = date(1905,6,30)
    all_interevents_df = summarise_results.events_to_interevents(start_date, end_date, all_events_df)

    expected_data = {'scenario': ['example_scenario']*12, 
                    'gauge': ['409025']*8+['410007']*4, 
                    'pu': ['Murray River - Yarrawonga to Barmah']*8+['Upper Yanco Creek']*4, 
                    'ewr': ['VF']*4+['LF2']*4+['SF2']*4,
                    'startDate': [date(1901,7,1), date(1901, 9, 1), date(1901, 12, 16), date(1904, 4, 1), date(1901,7,1), date(1901, 8, 26), date(1901, 12, 11), date(1904, 2, 16), date(1901, 7, 1), date(1901, 8, 16), date(1901, 12, 9), date(1904, 2, 6)],
                    'endDate': [date(1901, 7, 31), date(1901, 11, 30), date(1904, 1, 30), date(1905,6,30), date(1901, 8, 4), date(1901, 11, 30), date(1904, 1, 30), date(1905,6,30), date(1901, 8, 9), date(1901, 12, 5), date(1904, 1, 30), date(1905,6,30)],
                    'interEventLength': [31, 91, 776, 456, 35, 97, 781, 501, 40, 112, 783, 511]
    }
    expected_all_interevents_df = pd.DataFrame(data = expected_data)

    assert all_interevents_df.to_dict() == expected_all_interevents_df.to_dict()

def test_filter_successful_events(successfulEvent_item_to_process):
    all_events_df = pd.DataFrame(data = successfulEvent_item_to_process)
    all_interevents_df = summarise_results.filter_successful_events(all_events_df)

    expected_data = {'scenario': ['example_scenario']*6,
            'gauge': ['409025']*5+['410007']*1, 
            'pu': ['Murray River - Yarrawonga to Barmah']*5+['Upper Yanco Creek']*1, 
            'ewr': ['VF']*3+['LF2']*2+['SF2']*1,
            'waterYear': ['1901', '1901', '1904', '1901', '1904', '1904'], 
            'startDate': [date(1901, 8, 1), date(1901, 12, 1), date(1904, 1, 31), date(1901, 8, 5), date(1904, 1, 31), date(1904, 1, 31)], 
            'endDate': [date(1901, 8, 31), date(1901, 12, 2), date(1904, 3, 31), date(1901, 8, 25), date(1904, 2, 15), date(1904, 3, 5)], 
            'eventDuration': [31, 2, 61, 21, 16, 34],
            'eventLength': [31, 2, 61, 21, 16, 34],
            'Multigauge': ['']*6}
    expected_all_interevents_df = pd.DataFrame(data = expected_data)

    assert all_interevents_df.to_dict() == expected_all_interevents_df.to_dict()

def test_filter_duplicate_start_dates(duplicate_event_item_to_process):
    all_events_df = pd.DataFrame(data = duplicate_event_item_to_process)
    df = summarise_results.filter_duplicate_start_dates(all_events_df)

    expected_data = {'scenario': ['example_scenario']*7,
            'gauge': ['409025']*5+['410007']*2, 
            'pu': ['Murray River - Yarrawonga to Barmah']*5+['Upper Yanco Creek']*2, 
            'ewr': ['VF']*2+['LF2']*3+['SF2']*2,
            'waterYear': ['1901', '1904', '1901', '1901', '1904','1901', '1904'], 
            'startDate': [date(1901, 8, 1), date(1904, 1, 31), date(1901, 8, 5), date(1901, 12, 1), date(1904, 1, 31), date(1901, 8, 10), date(1901, 12, 6)], 
            'endDate': [date(1901, 12, 2), date(1904, 3, 31), date(1901, 8, 25), date(1901, 12, 9), date(1904, 2, 15), date(1901, 8, 15), date(1904, 3, 5)], 
            'eventDuration': [124, 61, 21, 9, 16, 6, 821],
            'eventLength': [124, 61, 21, 9, 16, 6, 821],
            'Multigauge': ['']*7}
    
    expected_df = pd.DataFrame(data=expected_data)

    assert df.reset_index(drop=True).to_dict() == expected_df.reset_index(drop=True).to_dict()


@pytest.mark.parametrize("events_date_range, start_date, end_date, expected_result",[
    (
         [
            ( date(2000,12,14), date(2007,1,2) ),
            ( date(2004,3,19), date(2009,12,25)),
            ( date(2013,4,23), date(2020,1,24) )
         ],
          date(2000,1,1),
          date(2023,12,31),

        [(date(2000, 1, 1), date(2000, 12, 13)), 
         (date(2009, 12, 26), date(2013, 4, 22)), 
         (date(2020, 1, 25), date(2023, 12, 31))]
    ),
    (
         [
            ( date(2000,12,14), date(2007,1,2) ),
            ( date(2004,3,19), date(2009,12,25)),
            ( date(2013,4,23), date(2020,1,24) )
         ],
          date(2000,12,14),
          date(2023,12,31),

        [
         (date(2009, 12, 26), date(2013, 4, 22)), 
         (date(2020, 1, 25), date(2023, 12, 31))]
    ),
    (
         [
            ( date(2000,12,14), date(2007,1,2) ),
            ( date(2004,3,19), date(2009,12,25)),
            ( date(2013,4,23), date(2020,1,24) )
         ],
          date(2000,12,14),
          date(2020,1,24),

        [
         (date(2009, 12, 26), date(2013, 4, 22))
        ]
    ),
    (
         [
            ( date(2000,12,14), date(2007,1,2) ),
            ( date(2004,3,19), date(2009,12,25))
         ],
          date(2000,12,14),
          date(2009,12,25),

        [
            
        ]
    ),
   
])
def test_get_inter_events_date_ranges(events_date_range, start_date, end_date, expected_result):
    result = summarise_results.get_inter_events_date_ranges(events_date_range, start_date, end_date)
    assert result == expected_result

    