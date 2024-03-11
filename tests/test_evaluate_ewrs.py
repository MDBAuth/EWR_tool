from datetime import datetime, date, timedelta

import pandas as pd
from pandas._testing import assert_frame_equal
import pytest
import numpy as np

from py_ewr import evaluate_EWRs, data_inputs

def test_ctf_handle():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # set up input data
    PU = 'PU_0000283'
    gauge = '410007'
    EWR = 'CF1'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: [0]*1+[0]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*10+[0]*345+[0]*1+[0]*9 + [0]*5+[0]*351+[0]*10}
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    PU_df = pd.DataFrame()
    # Send input data to test function:
    PU_df, events = evaluate_EWRs.ctf_handle(PU, gauge, EWR, EWR_table, df_F, PU_df)
    # Setting up expected output - PU_df
    data = {'CF1_eventYears': [0,0,0,1], 'CF1_numAchieved': [0,0,0,1], 'CF1_numEvents': [0,0,0,1], 'CF1_numEventsAll': [0,0,0,1], 
      'CF1_maxInterEventDays': [0,0,0,0],  'CF1_maxInterEventDaysAchieved': [1,1,1,1], 'CF1_eventLength': [0.0,0.0,0.0,1461.0], 'CF1_eventLengthAchieved': [0.0,0.0,0.0,1461.0], 
    'CF1_totalEventDays': [0,0,0,1461], 'CF1_totalEventDaysAchieved': [0,0,0,1461], 'CF1_maxEventDays': [0,0,0,1461], 'CF1_maxRollingEvents': [365, 730, 1095, 1461], 'CF1_maxRollingAchievement': [1, 1, 1, 1],
    'CF1_missingDays': [0,0,0,0], 'CF1_totalPossibleDays': [365,365,365,366]} 
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')
    assert_frame_equal(PU_df, expected_PU_df)
    # Setting up expected output - events
    expected_events = {2012:[], 2013:[], 2014:[], 2015:[[(date(2012, 7, 1)+timedelta(days=i),0) for i in range(1461)]]}
    expected_events = tuple([expected_events])



    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i] 

    
def test_lowflow_handle():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000283'
    gauge = '410007'
    EWR = 'BF1_a'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: [0]*1+[249]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[249]*345+[0]*1+[249]*17 + [0]*5+[249]*351+[249]*10}
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    PU_df = pd.DataFrame()
    # Send input data to test function
    PU_df, events = evaluate_EWRs.lowflow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df)
    # Setting up expected output data - PU_df, and testing
    data = {'BF1_a_eventYears': [0,0,0,0], 'BF1_a_numAchieved': [0,0,0,0], 'BF1_a_numEvents': [0,0,0,0], 'BF1_a_numEventsAll': [0,0,0,0],
            'BF1_a_maxInterEventDays': [0,0,0,0], 
           'BF1_a_maxInterEventDaysAchieved': [1,1,1,1],  
            'BF1_a_eventLength': [0.0,0.0,0.0,0.0], 'BF1_a_eventLengthAchieved': [0.0,0.0,0.0,0.0],
            'BF1_a_totalEventDays': [0,0,0,0], 'BF1_a_totalEventDaysAchieved': [0,0,0,0],  
            'BF1_a_maxEventDays': [0,0,0,0], 'BF1_a_maxRollingEvents': [0, 0, 0, 0], 'BF1_a_maxRollingAchievement': [0, 0, 0, 0],
            'BF1_a_missingDays': [0,0,0,0], 'BF1_a_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')
    assert_frame_equal(PU_df, expected_PU_df)

    # Setting up expected output - events, and testing
    expected_events = {2012:[], 2013:[], 2014:[], 2015:[]}
    expected_events = tuple([expected_events])

    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]  

def test_flow_handle():
    '''Things to calc in this function:
    1. Ensure all parts of the function generate expected output
    '''
    # Setting up input data
    PU = 'PU_0000283'
    gauge = '410007'
    EWR = 'SF1_S'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: [0]*1+[250]*350+[450]*10+[0]*4 + 
                               [0]*360+[450]*5 + 
                               [450]*5+[250]*345+[0]*1+[450]*14 + 
                               [0]*5+[450]*10+[0]*1+[450]*10+[250]*330+[450]*10
                               }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    PU_df = pd.DataFrame()
    # Send input data to test function
    PU_df, events = evaluate_EWRs.flow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df)
    # Setting up expected output - PU_df - and testing
    data = {'SF1_S_eventYears': {2012: 0, 2013: 0, 2014: 0, 2015: 1}, 
            'SF1_S_numAchieved': {2012: 0, 2013: 0, 2014: 0, 2015: 1}, 
            'SF1_S_numEvents': {2012: 1, 2013: 0, 2014: 1, 2015: 3}, 
            'SF1_S_numEventsAll': {2012: 1, 2013: 1, 2014: 2, 2015: 3}, 
            'SF1_S_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
            'SF1_S_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
            'SF1_S_eventLength': {2012: 10.0, 2013: 5.0, 2014: 9.5, 2015: 10.0}, 
            'SF1_S_eventLengthAchieved': {2012: 10.0, 2013: 0.0, 2014: 14.0, 2015: 10.0}, 
            'SF1_S_totalEventDays': {2012: 10, 2013: 5, 2014: 19, 2015: 30}, 
            'SF1_S_totalEventDaysAchieved': {2012: 10, 2013: 0, 2014: 14, 2015: 30}, 
            'SF1_S_maxEventDays': {2012: 10, 2013: 5, 2014: 14, 2015: 10}, 
            'SF1_S_maxRollingEvents': {2012: 10, 2013: 5, 2014: 14, 2015: 10},
            'SF1_S_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 1, 2015: 1}, 
            'SF1_S_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
            'SF1_S_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}}
    expected_PU_df = pd.DataFrame(data)
    assert data == PU_df.to_dict()
    # Setting up expected output - events - and testing
    expected_events = {2012: [[(date(2013, 6, 17) + timedelta(days=i), 450) for i in range(10)]], 
                        2013: [[(date(2014, 6, 26) + timedelta(days=i), 450) for i in range(5)]], 
                        2014: [[(date(2014, 7, 1) + timedelta(days=i), 450) for i in range(5)],
                                [(date(2015, 6, 17) + timedelta(days=i), 450) for i in range(14)]],
                        2015: [[(date(2015, 7, 6) + timedelta(days=i), 450) for i in range(10)],
                            [(date(2015, 7, 17) + timedelta(days=i), 450) for i in range(10)],     
                            [(date(2016, 6, 21) + timedelta(days=i), 450) for i in range(10)]]}
    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i] 

def test_cumulative_handle():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000040'
    gauge = '418068'
    gauge_flows = ([0]*1+[0]*350+[10000]*1+[3000]*4 +[0]*9 + 
                   [0]*360+[450]*3+[19000]*1+[1000]*1 + 
                   [450]*5+[250]*345+[0]*1+[0]*13+[5000]*1 + 
                   [5000]*4+[450]*10+[0]*2+[450]*10+[250]*330+[450]*10)
    EWR = 'OB3_S'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                        gauge: gauge_flows}
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    PU_df = pd.DataFrame()
    # Send input data to test function
    PU_df, events = evaluate_EWRs.cumulative_handle(PU, gauge, EWR, EWR_table, df_F, PU_df)
    # Setting up expected output - PU_df - and testing
    data = {'OB3_S_eventYears': [1,0,0,0], 'OB3_S_numAchieved': [1,0,0,0], 'OB3_S_numEvents': [1,0,0,0], 'OB3_S_numEventsAll': [1,0,0,0], 
            'OB3_S_maxInterEventDays': [0, 0, 0, 0], 
           'OB3_S_maxInterEventDaysAchieved': [1, 1, 1, 1],'OB3_S_eventLength': [1.0,0.0,0.0,0.0], 'OB3_S_eventLengthAchieved': [1.0,0.0,0.0,0.0], 
            'OB3_S_totalEventDays': [1,0,0,0], 'OB3_S_totalEventDaysAchieved': [1,0,0,0], 'OB3_S_maxEventDays': [1,0,0,0],'OB3_S_maxRollingEvents': [1, 0, 0, 0], 
            'OB3_S_maxRollingAchievement': [1, 1, 1, 1],'OB3_S_missingDays': [0,0,0,0], 
            'OB3_S_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')
    assert_frame_equal(PU_df, expected_PU_df)
    # Setting up expected output - events - and testing 
    expected_events = {2012:[[(date(2013, 6, 21), 22000)]], 2013:[], 2014:[], 2015:[]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

@pytest.mark.parametrize("expected_events, expected_PU_df_data", [
    (
          {2012: [[(date(2012, 7, 10) + timedelta(days=i), 25000) for i in range(51)]], 
                2013: [], 
                2014: [], 
                2015: [] },

{'BBR2_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_numAchieved': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_numEvents': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_numEventsAll': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
 'BBR2_eventLength': {2012: 51.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
 'BBR2_eventLengthAchieved': {2012: 0.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
 'BBR2_totalEventDays': {2012: 51, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_totalEventDaysAchieved': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_maxEventDays': {2012: 51, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_maxRollingEvents': {2012: 51, 2013: 0, 2014: 0, 2015: 0}, 
 'BBR2_maxRollingAchievement': {2012: 0, 2013: 0, 2014: 0, 2015: 0},
   'BBR2_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
   'BBR2_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}},
    )
])
def test_cumulative_handle_qld(qld_parameter_sheet,expected_events, expected_PU_df_data):
    # Set up input data
    PU = 'PU_0000991'
    gauge = '422016'
    EWR = 'BBR2'

    EWR_table = qld_parameter_sheet

    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        '422016': ( [2500]*10+[0]*355   + 
                                    [0]*365 + 
                                    [0]*365 + 
                                    [0]*366  )} 
    

    df_F = pd.DataFrame(data = data_for_df_F)

    df_F = df_F.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.cumulative_handle_qld(PU, gauge, EWR, EWR_table, df_F, PU_df)

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

def test_level_handle():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000266'
    gauge = '425022'
    EWR = 'LLLF'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_L = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: [0]*1+[0]*260+[56]*90+[0]*1+[0]*4+[0]*9 + 
                               [56]*45+[55.9]*1+[56]*45+[0]*269+[0]*3+[19000]*1+[1000]*1 + 
                               [0]*5+[0]*345+[0]*1+[0]*13+[56]*1 + 
                               [56]*89+[0]*4+[0]*10+[0]*3+[0]*10+[0]*230+[0]*20}
    df_L = pd.DataFrame(data = data_for_df_L)
    df_L = df_L.set_index('Date')
    PU_df = pd.DataFrame()
    # Send input data to test function
    PU_df, events = evaluate_EWRs.level_handle(PU, gauge, EWR, EWR_table, df_L, PU_df)
    # Setting up expected output - PU_df and test
    data = {'LLLF_eventYears': [1,0,0,1], 'LLLF_numAchieved': [1,0,0,1], 'LLLF_numEvents': [1,0,0,1], 'LLLF_numEventsAll': [1,0,0,1], 
            'LLLF_maxInterEventDays': [0, 0, 0, 0], 
            'LLLF_maxInterEventDaysAchieved': [1, 1, 1, 1],'LLLF_eventLength': [90.0,0.0,0.0,90.0], 'LLLF_eventLengthAchieved': [90.0,0.0,0.0,90.0], 
            'LLLF_totalEventDays': [90,0,0,90], 'LLLF_totalEventDaysAchieved': [90,0,0,90], 
            'LLLF_maxEventDays': [90,0,0,90], 'LLLF_maxRollingEvents': [90, 0, 1, 90],'LLLF_maxRollingAchievement': [1, 0, 0, 1],
            'LLLF_missingDays': [0,0,0,0], 'LLLF_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')
    assert_frame_equal(PU_df, expected_PU_df)
    # Setting up expected output - events - and test
    expected_events = {2012:[[(date(2013, 3, 19) + timedelta(days=i), 56) for i in range(90)]], 
                        2013:[], 
                        2014:[], 
                        2015:[[(date(2015, 6, 30) + timedelta(days=i), 56) for i in range(90)]]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i] 

def test_nest_handle():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000253'
    gauge = '409025'
    EWR = 'NestS1'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    # input data up df_L:
    # flows declining at acceptable rate:
    acceptable_flows = [10000]*10
    reduction_max = 5
    for i, flow in enumerate(acceptable_flows):
        acceptable_flows[i] = acceptable_flows[i-1]-(reduction_max/100*acceptable_flows[i-1])
    acceptable_flows = acceptable_flows + [5900]*50
    # flows declining at unnacceptable rate:
    unnacceptable_flows = [10000]*10
    reduction_max = 7
    for i, flow in enumerate(unnacceptable_flows):
        unnacceptable_flows[i] = unnacceptable_flows[i-1]-(reduction_max/100*unnacceptable_flows[i-1])
    unnacceptable_flows = unnacceptable_flows + [5300]*50
    # flows declining at acceptable rate but going below the threshold
    threshold_flows = [10000]*10
    reduction_max = 5
    for i, flow in enumerate(threshold_flows):
        threshold_flows[i] = threshold_flows[i-1]-(reduction_max/100*threshold_flows[i-1])
    threshold_flows = threshold_flows + [5300]*50
    # input data for df_F:

    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: ([0]*76+acceptable_flows+[0]*229 + 
                                [0]*76+unnacceptable_flows+[0]*229 + 
                                [0]*76+threshold_flows+[0]*229 + 
                                [0]*77+threshold_flows+[0]*229)}
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    # Pass input data to test function:
    PU_df, events = evaluate_EWRs.nest_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df)
    # Setting up expected output - PU_df - and testing
    data = {'NestS1_eventYears': [1,0,0,0], 'NestS1_numAchieved': [1,0,0,0], 'NestS1_numEvents': [1,0,0,0], 'NestS1_numEventsAll': [1,2,2,2], 
            'NestS1_maxInterEventDays': [0, 0, 0, 0], 
            'NestS1_maxInterEventDaysAchieved': [1, 1, 1, 1],'NestS1_eventLength': [60.0, 25.5, 29.5, 29.5], 'NestS1_eventLengthAchieved':  [60.0, 0.0, 0.0, 0.0], 
            'NestS1_totalEventDays': [60,51,59,59], 'NestS1_totalEventDaysAchieved': [60, 0, 0, 0],
            'NestS1_maxEventDays':[60,50,49,49],'NestS1_maxRollingEvents': [60, 50, 49, 49], 'NestS1_maxRollingAchievement': [1, 0, 0, 0],
            'NestS1_missingDays': [0,0,0,0], 'NestS1_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')

    assert_frame_equal(PU_df, expected_PU_df)

    acceptable_flows_results = [(date(2012, 9, 15) + timedelta(days=i), f) for i, f in enumerate(acceptable_flows)]
    unnacceptable_flows_results = [(date(2013, 9, 15) + timedelta(days=i), f) for i, f in enumerate(unnacceptable_flows)]
    threshold_flows_results_1 = [(date(2014, 9, 15) + timedelta(days=i), f) for i, f in enumerate(threshold_flows)]
    threshold_flows_results_2 = [(date(2015, 9, 15) + timedelta(days=i), f) for i, f in enumerate(threshold_flows)]

    expected_events = {2012:[acceptable_flows_results], 
                       2013:[[(date(2013, 9, 15), 9300.0)], [(date(2013, 9, 25) + timedelta(days=i), 5300.0) for i in range(50)]], 
                       2014:[[(date(2014, 9, 15), 9500.0), (date(2014, 9, 16), 9025.0), (date(2014, 9, 17), 8573.75), 
                              (date(2014, 9, 18), 8145.0625), (date(2014, 9, 19), 7737.809375), (date(2014, 9, 20), 7350.91890625), 
                              (date(2014, 9, 21), 6983.3729609375), (date(2014, 9, 22), 6634.204312890624), 
                              (date(2014, 9, 23), 6302.494097246094), (date(2014, 9, 24), 5987.369392383789)],
                              [(date(2014, 9, 26) + timedelta(days=i), 5300.0) for i in range(49)]
                              ], 
                       2015:[[(date(2015, 9, 16), 9500.0), (date(2015, 9, 17), 9025.0), (date(2015, 9, 18), 8573.75), 
                              (date(2015, 9, 19), 8145.0625), (date(2015, 9, 20), 7737.809375), (date(2015, 9, 21), 7350.91890625), 
                              (date(2015, 9, 22), 6983.3729609375), (date(2015, 9, 23), 6634.204312890624), 
                              (date(2015, 9, 24), 6302.494097246094), (date(2015, 9, 25), 5987.369392383789)],
                              [(date(2015, 9, 27) + timedelta(days=i), 5300.0) for i in range(49)]
                              ]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]


def test_flow_handle_multi():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000130'
    gauge1 = '421090'
    gauge2 = '421088'
    EWR = 'LF1'
    gauge1_flows = ([0]*76+[1250]*5+[0]*229+[0]*55 + [0]*76+[0]*55+[0]*231+[1250]*3 + [1250]*3+[0]*76+[0]*50+[1250]*5+[0]*231 + [0]*77+[1250]*5+[0]*229+[0]*55)
    gauge2_flows = ([0]*76+[1250]*5+[0]*229+[0]*55 + [0]*76+[0]*55+[0]*231+[1250]*3 + [1250]*3+[0]*76+[0]*50+[1250]*5+[0]*231 + [0]*76+[1250]*5+[0]*230+[0]*55)
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge1: gauge1_flows,
                        gauge2: gauge2_flows
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    # Send input data to test function
    PU_df, events = evaluate_EWRs.flow_handle_multi(PU, gauge1, EWR, EWR_table, df_F, PU_df)
    # Setting up expected output - PU_df - and testing
    data = {'LF1_eventYears': [1,0,1,0], 'LF1_numAchieved': [1,0,1,0], 'LF1_numEvents': [1,0,1,0], 'LF1_numEventsAll': [1, 1, 2, 1], 
            'LF1_maxInterEventDays': [0, 0, 0, 0], 
            'LF1_maxInterEventDaysAchieved': [1, 1, 1, 1],'LF1_eventLength': [5.0, 3.0, 4.0, 4.0], 'LF1_eventLengthAchieved': [5.0, 0.0, 5.0, 0.0], 
            'LF1_totalEventDays': [5,3,8,4], 'LF1_totalEventDaysAchieved': [5, 0, 5, 0],
            'LF1_maxEventDays':[5, 3, 5, 4], 'LF1_maxRollingEvents': [5, 3, 5, 4], 'LF1_maxRollingAchievement': [1, 0, 1, 0],
            'LF1_missingDays': [0,0,0,0], 'LF1_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')
    assert_frame_equal(PU_df, expected_PU_df)    
    # Setting up expected output - events - and testing
    expected_events = {2012:[[(date(2012, 9, 15) + timedelta(days=i), 2500) for i in range(5)]], 
                       2013:[[(date(2014, 6, 28) + timedelta(days=i), 2500) for i in range(3)]], 
                       2014:[ [(date(2014, 7, 1) + timedelta(days=i), 2500) for i in range(3)], [(date(2014, 11, 7) + timedelta(days=i), 2500) for i in range(5)]], 
                       2015:[[(date(2015, 9, 16) + timedelta(days=i), 2500) for i in range(4)]]}
    
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

def test_lowflow_handle_multi():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Input data
    PU = 'PU_0000130'
    gauge1 = '421090'
    gauge2 = '421088'
    EWR = 'BF1_a'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge1: [40]*76+[1250]*5+[40]*229+[40]*15+[0]*40 + [40]*3+[0]*76+[0]*50+[0]*5+[0]*231 + [40]*75+[0]*50+[40]*230+[40]*10 + [0]*77+[40]*5+[0]*229+[40]*55,
                        gauge2: [40]*76+[1250]*5+[40]*229+[0]*40+[40]*15 + [40]*3+[0]*76+[0]*50+[0]*5+[0]*231 + [40]*75+[0]*50+[40]*230+[40]*10 + [0]*76+[40]*5+[0]*230+[40]*55
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.lowflow_handle_multi(PU, gauge1, EWR, EWR_table, df_F, PU_df)
    # Setting up expected output - PU_df - and testing
    data = {'BF1_a_eventYears': [0,0,0,0], 'BF1_a_numAchieved': [0,0,0,0], 'BF1_a_numEvents': [0,0,0,0], 'BF1_a_numEventsAll': [1,0,0,0], 
            'BF1_a_maxInterEventDays': [0, 0, 0, 0], 
            'BF1_a_maxInterEventDaysAchieved': [1, 1, 1, 1],'BF1_a_eventLength': [5.0, 0.0, 0.0, 0.0], 'BF1_a_eventLengthAchieved': [5.0, 0.0, 0.0, 0.0], 
            'BF1_a_totalEventDays': [5, 0, 0, 0], 'BF1_a_totalEventDaysAchieved': [5, 0, 0, 0],
            'BF1_a_maxEventDays':[5, 0, 0, 0], 'BF1_a_maxRollingEvents': [5, 0, 0, 0], 'BF1_a_maxRollingAchievement': [0, 0, 0, 0],
            'BF1_a_missingDays': [0,0,0,0], 'BF1_a_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')
    assert_frame_equal(PU_df, expected_PU_df)    
    # Setting up expected output - events - and testing
    expected_events = {2012:[[(date(2012, 9, 15) + timedelta(days=i), 2500) for i in range(5)]], 
    2013:[], 2014:[], 2015:[]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

def test_ctf_handle_multi():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up the input data
    PU = 'PU_0000130'
    gauge1 = '421090'
    gauge2 = '421088'
    EWR = 'CF'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge1: [0]*1+[2]*350+[0]*9+[0]*5 + [2]*360+[0]*5 + [0]*10+[2]*345+[0]*1+[2]*9 + [0]*5+[0]*351+[0]*10,
                        gauge2: [0]*1+[2]*350+[0]*9+[0]*5 + [2]*360+[0]*5 + [0]*10+[2]*345+[0]*1+[2]*9 + [0]*5+[0]*351+[0]*10
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    # Pass input data to the test function
    PU_df, events = evaluate_EWRs.ctf_handle_multi(PU, gauge1, EWR, EWR_table, df_F, PU_df)
    # Setting up expected output - PU_df - and testing
    data = {'CF_eventYears': [1,0,1,1], 'CF_numAchieved': [2,0,2,1], 'CF_numEvents': [2,0,2,1], 'CF_numEventsAll': [2,0,2,1],
            'CF_maxInterEventDays': [0, 0, 0, 0], 
            'CF_maxInterEventDaysAchieved': [1, 1, 1, 1], 'CF_eventLength': [7.5,0.0,8.0,366.0], 'CF_eventLengthAchieved': [7.5,0.0,8.0,366.0], 
            'CF_totalEventDays': [15,0,16,366], 'CF_totalEventDaysAchieved': [15,0,16,366],
            'CF_maxEventDays':[14, 0, 15, 366], 'CF_maxRollingEvents': [14, 5, 15, 366], 'CF_maxRollingAchievement': [1, 1, 1, 1],
            'CF_missingDays': [0,0,0,0], 'CF_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')
    assert_frame_equal(PU_df, expected_PU_df)    
    # Setting up expected output - events - and testing
    expected_events = {2012:[[(date(2012, 7, 1), 0)], 
                              [(date(2013, 6, 17) + timedelta(days=i), 0) for i in range(14)]], 
                       2013:[], 
                       2014:[ [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(15)], 
                       [(date(2015, 6, 21), 0)]], 
                       2015:[[(date(2015, 7, 1) + timedelta(days=i), 0) for i in range(366)]]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

def test_cumulative_handle_multi():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000132'
    gauge1 = '421090'
    gauge2 = '421088'
    gauge1_flows = ([0]*1+[0]*260+[334]*90+[0]*5+[0]*9 + 
                    [0]*310+[0]*3+[0]*1+[0]*1+[500]*50 + 
                    [500]*40+[0]*310+[0]*1+[0]*13+[0]*1 + 
                    [5000]*4+[500]*90+[500]*90+[450]*10+ [0]*2+[450]*10+[250]*150+[450]*10)
    gauge2_flows = ([0]*1+[0]*260+[334]*90+[0]*5+[0]*9 + 
                    [0]*310+[0]*3+[0]*1+[0]*1+[500]*50 + 
                    [500]*40+[0]*310+[0]*1+[0]*13+[0]*1 + 
                    [5000]*4+[500]*90+[500]*90+[450]*10+[0]*2+ [450]*10+[250]*150+[450]*10)
    EWR = 'OB_WS1_S'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                        gauge1: gauge1_flows,
                        gauge2: gauge2_flows
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.cumulative_handle_multi(PU, gauge1, EWR, EWR_table, df_F, PU_df)
    # Setting up expected output - PU_df - and testing
    data = {'OB_WS1_S_eventYears': [1,0,0,1], 'OB_WS1_S_numAchieved': [1,0,0,1], 'OB_WS1_S_numEvents': [1,0,0,1], 'OB_WS1_S_numEventsAll': [1,0,0,1],
            'OB_WS1_S_maxInterEventDays': [0, 0, 0, 0],
            'OB_WS1_S_maxInterEventDaysAchieved': [1, 1, 1, 1], 'OB_WS1_S_eventLength': [1,0.0,0.0,235.0], 'OB_WS1_S_eventLengthAchieved': [1,0.0,0.0,235.0],
            'OB_WS1_S_totalEventDays': [1,0,0,235], 'OB_WS1_S_totalEventDaysAchieved': [1,0,0,235],
            'OB_WS1_S_maxEventDays':[1,0,0,235], 'OB_WS1_S_maxRollingEvents':  [1,0,0,235],
            'OB_WS1_S_maxRollingAchievement': [1,1,1,1],
            'OB_WS1_S_missingDays': [0,0,0,0], 'OB_WS1_S_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')

    assert_frame_equal(PU_df, expected_PU_df)   
    # Setting up expected output - events - and testing
    expected_events = {2012:[[(date(2013, 6, 16), 60120)]], 
                       2013:[], 
                       2014:[], 
                       2015:[[(date(2015, 7, 24), 60000), (date(2015, 7, 25), 61000), (date(2015, 7, 26), 62000), (date(2015, 7, 27), 63000), (date(2015, 7, 28), 64000), (date(2015, 7, 29), 65000), (date(2015, 7, 30), 66000), (date(2015, 7, 31), 67000), (date(2015, 8, 1), 68000), (date(2015, 8, 2), 69000), (date(2015, 8, 3), 70000), (date(2015, 8, 4), 71000), (date(2015, 8, 5), 72000), (date(2015, 8, 6), 73000), (date(2015, 8, 7), 74000), (date(2015, 8, 8), 75000), (date(2015, 8, 9), 76000), (date(2015, 8, 10), 77000), (date(2015, 8, 11), 78000), (date(2015, 8, 12), 79000), (date(2015, 8, 13), 80000), (date(2015, 8, 14), 81000), (date(2015, 8, 15), 82000), (date(2015, 8, 16), 83000), (date(2015, 8, 17), 84000), (date(2015, 8, 18), 85000), (date(2015, 8, 19), 86000), (date(2015, 8, 20), 87000), (date(2015, 8, 21), 88000), (date(2015, 8, 22), 89000), (date(2015, 8, 23), 90000), (date(2015, 8, 24), 91000), (date(2015, 8, 25), 92000), (date(2015, 8, 26), 93000), (date(2015, 8, 27), 94000), (date(2015, 8, 28), 95000), (date(2015, 8, 29), 96000), (date(2015, 8, 30), 97000), (date(2015, 8, 31), 98000), (date(2015, 9, 1), 99000), (date(2015, 9, 2), 100000), (date(2015, 9, 3), 101000), (date(2015, 9, 4), 102000), (date(2015, 9, 5), 103000), (date(2015, 9, 6), 104000), (date(2015, 9, 7), 105000), (date(2015, 9, 8), 106000), (date(2015, 9, 9), 107000), (date(2015, 9, 10), 108000), (date(2015, 9, 11), 109000), (date(2015, 9, 12), 110000), (date(2015, 9, 13), 111000), (date(2015, 9, 14), 112000), (date(2015, 9, 15), 113000), (date(2015, 9, 16), 114000), (date(2015, 9, 17), 115000), (date(2015, 9, 18), 116000), (date(2015, 9, 19), 117000), (date(2015, 9, 20), 118000), (date(2015, 9, 21), 119000), (date(2015, 9, 22), 120000), (date(2015, 9, 23), 121000), (date(2015, 9, 24), 122000), (date(2015, 9, 25), 123000), (date(2015, 9, 26), 124000), (date(2015, 9, 27), 125000), (date(2015, 9, 28), 126000), (date(2015, 9, 29), 117000), (date(2015, 9, 30), 108000), (date(2015, 10, 1), 99000), (date(2015, 10, 2), 90000), (date(2015, 10, 3), 90000), (date(2015, 10, 4), 90000), (date(2015, 10, 5), 90000), (date(2015, 10, 6), 90000), (date(2015, 10, 7), 90000), (date(2015, 10, 8), 90000), (date(2015, 10, 9), 90000), (date(2015, 10, 10), 90000), (date(2015, 10, 11), 90000), (date(2015, 10, 12), 90000), (date(2015, 10, 13), 90000), (date(2015, 10, 14), 90000), (date(2015, 10, 15), 90000), (date(2015, 10, 16), 90000), (date(2015, 10, 17), 90000), (date(2015, 10, 18), 90000), (date(2015, 10, 19), 90000), (date(2015, 10, 20), 90000), (date(2015, 10, 21), 90000), (date(2015, 10, 22), 90000), (date(2015, 10, 23), 90000), (date(2015, 10, 24), 90000), (date(2015, 10, 25), 90000), (date(2015, 10, 26), 90000), (date(2015, 10, 27), 90000), (date(2015, 10, 28), 90000), (date(2015, 10, 29), 90000), (date(2015, 10, 30), 90000), (date(2015, 10, 31), 90000), (date(2015, 11, 1), 90000), (date(2015, 11, 2), 90000), (date(2015, 11, 3), 90000), (date(2015, 11, 4), 90000), (date(2015, 11, 5), 90000), (date(2015, 11, 6), 90000), (date(2015, 11, 7), 90000), (date(2015, 11, 8), 90000), (date(2015, 11, 9), 90000), (date(2015, 11, 10), 90000), (date(2015, 11, 11), 90000), (date(2015, 11, 12), 90000), (date(2015, 11, 13), 90000), (date(2015, 11, 14), 90000), (date(2015, 11, 15), 90000), (date(2015, 11, 16), 90000), (date(2015, 11, 17), 90000), (date(2015, 11, 18), 90000), (date(2015, 11, 19), 90000), (date(2015, 11, 20), 90000), (date(2015, 11, 21), 90000), (date(2015, 11, 22), 90000), (date(2015, 11, 23), 90000), (date(2015, 11, 24), 90000), (date(2015, 11, 25), 90000), (date(2015, 11, 26), 90000), (date(2015, 11, 27), 90000), (date(2015, 11, 28), 90000), (date(2015, 11, 29), 90000), (date(2015, 11, 30), 90000), (date(2015, 12, 1), 90000), (date(2015, 12, 2), 90000), (date(2015, 12, 3), 90000), (date(2015, 12, 4), 90000), (date(2015, 12, 5), 90000), (date(2015, 12, 6), 90000), (date(2015, 12, 7), 90000), (date(2015, 12, 8), 90000), (date(2015, 12, 9), 90000), (date(2015, 12, 10), 90000), (date(2015, 12, 11), 90000), (date(2015, 12, 12), 90000), (date(2015, 12, 13), 90000), (date(2015, 12, 14), 90000), (date(2015, 12, 15), 90000), (date(2015, 12, 16), 90000), (date(2015, 12, 17), 90000), (date(2015, 12, 18), 90000), (date(2015, 12, 19), 90000), (date(2015, 12, 20), 90000), (date(2015, 12, 21), 90000), (date(2015, 12, 22), 90000), (date(2015, 12, 23), 90000), (date(2015, 12, 24), 90000), (date(2015, 12, 25), 90000), (date(2015, 12, 26), 90000), (date(2015, 12, 27), 90000), (date(2015, 12, 28), 90000), (date(2015, 12, 29), 90000), (date(2015, 12, 30), 90000), (date(2015, 12, 31), 90000), (date(2016, 1, 1), 89900), (date(2016, 1, 2), 89800), (date(2016, 1, 3), 89700), (date(2016, 1, 4), 89600), (date(2016, 1, 5), 89500), (date(2016, 1, 6), 89400), (date(2016, 1, 7), 89300), (date(2016, 1, 8), 89200), (date(2016, 1, 9), 89100), (date(2016, 1, 10), 89000), (date(2016, 1, 11), 88000), (date(2016, 1, 12), 87000), (date(2016, 1, 13), 86900), (date(2016, 1, 14), 86800), (date(2016, 1, 15), 86700), (date(2016, 1, 16), 86600), (date(2016, 1, 17), 86500), (date(2016, 1, 18), 86400), (date(2016, 1, 19), 86300), (date(2016, 1, 20), 86200), (date(2016, 1, 21), 86100), (date(2016, 1, 22), 86000), (date(2016, 1, 23), 85500), (date(2016, 1, 24), 85000), (date(2016, 1, 25), 84500), (date(2016, 1, 26), 84000), (date(2016, 1, 27), 83500), (date(2016, 1, 28), 83000), (date(2016, 1, 29), 82500), (date(2016, 1, 30), 82000), (date(2016, 1, 31), 81500), (date(2016, 2, 1), 81000), (date(2016, 2, 2), 80500), (date(2016, 2, 3), 80000), (date(2016, 2, 4), 79500), (date(2016, 2, 5), 79000), (date(2016, 2, 6), 78500), (date(2016, 2, 7), 78000), (date(2016, 2, 8), 77500), (date(2016, 2, 9), 77000), (date(2016, 2, 10), 76500), (date(2016, 2, 11), 76000), (date(2016, 2, 12), 75500), (date(2016, 2, 13), 75000), (date(2016, 2, 14), 74500), (date(2016, 2, 15), 74000), (date(2016, 2, 16), 73500), (date(2016, 2, 17), 73000), (date(2016, 2, 18), 72500), (date(2016, 2, 19), 72000), (date(2016, 2, 20), 71500), (date(2016, 2, 21), 71000), (date(2016, 2, 22), 70500), (date(2016, 2, 23), 70000), (date(2016, 2, 24), 69500), (date(2016, 2, 25), 69000), (date(2016, 2, 26), 68500), (date(2016, 2, 27), 68000), (date(2016, 2, 28), 67500), (date(2016, 2, 29), 67000), (date(2016, 3, 1), 66500), (date(2016, 3, 2), 66000), (date(2016, 3, 3), 65500), (date(2016, 3, 4), 65000), (date(2016, 3, 5), 64500), (date(2016, 3, 6), 64000), (date(2016, 3, 7), 63500), (date(2016, 3, 8), 63000), (date(2016, 3, 9), 62500), (date(2016, 3, 10), 62000), (date(2016, 3, 11), 61500), (date(2016, 3, 12), 61000), (date(2016, 3, 13), 60500), (date(2016,3,14), 60000)]]}
    expected_events = tuple([expected_events])

    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

@pytest.mark.parametrize("date,water_year",
        [ (date(2022,6,29), 2021),
         (date(2022,6,20), 2021),
         (date(2022,7,1), 2022),],
)
def test_water_year(date, water_year):
    result = evaluate_EWRs.water_year(date)
    assert result == water_year

@pytest.mark.parametrize("start_date,end_date,water_years",
        [ ( date(2022,6,1), date(2022,6,29), [2021]),
          ( date(2022,6,1), date(2022,7,29), [2021,2022]),
          (date(2022,6,1),date(2023,7,29), [2021,2022,2023]),
        ],
)
def test_water_year_touches(start_date, end_date, water_years):
    result = evaluate_EWRs.water_year_touches(start_date, end_date)
    assert result == water_years

@pytest.mark.parametrize("gauge_events,events_info",
        [ ( {2012:[ [(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                    [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
            2013:[[(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                  [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)]], 
            2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                  [(date(2015, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
            2015:[[(date(2015, 11, 1) + timedelta(days=i), 0) for i in range(5)]]}, 
            [(date(2012, 11, 1), date(2012, 11, 5), 5, [2012]),
             (date(2013, 6, 26), date(2013, 6, 30), 5, [2012]),
             (date(2013, 11, 2), date(2013, 11, 4), 3, [2013]),
             (date(2014, 6, 26), date(2014, 6, 28), 3, [2013]),
             (date(2014, 11, 1), date(2014, 11, 5), 5, [2014]),
             (date(2015, 6, 26), date(2015, 6, 30), 5, [2014]),
             (date(2015, 11, 1), date(2015, 11, 5), 5, [2015])]),
        ( {2012:[ [(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                    [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(7)]], 
            2013:[[(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                  [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)]], 
            2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                  [(date(2015, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
            2015:[[(date(2015, 11, 1) + timedelta(days=i), 0) for i in range(5)]]}, 
            [(date(2012, 11, 1), date(2012, 11, 5), 5, [2012]),
             (date(2013, 6, 26), date(2013, 7, 2), 7, [2012,2013]),
             (date(2013, 11, 2), date(2013, 11, 4), 3, [2013]),
             (date(2014, 6, 26), date(2014, 6, 28), 3, [2013]),
             (date(2014, 11, 1), date(2014, 11, 5), 5, [2014]),
             (date(2015, 6, 26), date(2015, 6, 30), 5, [2014]),
             (date(2015, 11, 1), date(2015, 11, 5), 5, [2015])]),
          ( {2012:[ [(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)] ], 
            2013:[[(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                  [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)],
                  [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(374)]], 
            2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                  [(date(2015, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
            2015:[[(date(2015, 11, 1) + timedelta(days=i), 0) for i in range(5)]]}, 
            [(date(2012, 11, 1), date(2012, 11, 5), 5, [2012]),
             (date(2013, 11, 2), date(2013, 11, 4), 3, [2013]),
             (date(2014, 6, 26), date(2014, 6, 28), 3, [2013]),
             (date(2013, 6, 26), date(2014, 7, 4), 374, [2012, 2013, 2014]),
             (date(2014, 11, 1), date(2014, 11, 5), 5, [2014]),
             (date(2015, 6, 26), date(2015, 6, 30), 5, [2014]),
             (date(2015, 11, 1), date(2015, 11, 5), 5, [2015])])
        ],
)
def test_return_events_list_info(gauge_events, events_info):
    result = evaluate_EWRs.return_events_list_info(gauge_events)
    assert result == events_info

@pytest.mark.parametrize("events_info,water_year_maxs",
        [ (  [(date(2012, 11, 1), date(2012, 11, 5), 5, [2012]),
             (date(2013, 6, 26), date(2013, 6, 30), 5, [2012]),
             (date(2013, 11, 2), date(2013, 11, 4), 3, [2013]),
             (date(2014, 6, 26), date(2014, 6, 28), 3, [2013]),
             (date(2014, 11, 1), date(2014, 11, 5), 5, [2014]),
             (date(2015, 6, 26), date(2015, 6, 30), 5, [2014]),
             (date(2015, 11, 1), date(2015, 11, 5), 5, [2015])] ,
             {2012: [5, 5], 2013: [3, 3], 2014: [5, 5], 2015: [5]} ),
           (  [(date(2012, 11, 1), date(2012, 11, 5), 5, [2012]),
             (date(2013, 6, 26), date(2013, 7, 2), 7, [2012,2013]),
             (date(2013, 11, 2), date(2013, 11, 4), 3, [2013]),
             (date(2014, 6, 26), date(2014, 6, 28), 3, [2013]),
             (date(2014, 11, 1), date(2014, 11, 5), 5, [2014]),
             (date(2015, 6, 26), date(2015, 6, 30), 5, [2014]),
             (date(2015, 11, 1), date(2015, 11, 5), 5, [2015])] ,
             {2012: [5, 5], 2013: [7, 3, 3], 2014: [5, 5], 2015: [5]} )
        ],
)
def test_lengths_to_years(events_info,water_year_maxs):
    result = evaluate_EWRs.lengths_to_years(events_info)
    assert result == water_year_maxs

@pytest.mark.parametrize("event,expected_event_info",[
    ([(date(2012, 6, 25) + timedelta(days=i), 0) for i in range(11)],
    (date(2012, 6, 25), date(2012, 7, 5), 11, [2011, 2012])),
    ([(date(2012, 6, 25) + timedelta(days=i), 0) for i in range(376)],
    (date(2012, 6, 25), date(2013, 7, 5), 376, [2011, 2012, 2013])),
    ([(date(2012, 6, 25) + timedelta(days=i), 0) for i in range(5)],
    (date(2012, 6, 25), date(2012, 6, 29), 5, [2011])),
],)
def test_return_event_info(event, expected_event_info):
    result = evaluate_EWRs.return_event_info(event)
    assert result == expected_event_info


@pytest.mark.parametrize("event_info,expected_years_lengths_list",[
    ((date(2012, 6, 25), date(2012, 7, 5), 11, [2011, 2012]),
    [6,5]),
    ((date(2012, 6, 25), date(2013, 7, 5), 376, [2011, 2012, 2013]),
    [6,371,5]),
    ((date(2012, 6, 25), date(2012, 7, 29), 5, [2011]),
    [5]),
],)
def test_years_lengths(event_info, expected_years_lengths_list):
    result = evaluate_EWRs.years_lengths(event_info)
    assert result == expected_years_lengths_list

@pytest.mark.parametrize("gauge_events,unique_water_years,max_consecutive_events",
        [ ( {2012:[ [(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                    [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
            2013:[[(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                  [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)]], 
            2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                  [(date(2015, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
            2015:[[(date(2015, 11, 1) + timedelta(days=i), 0) for i in range(5)]]} ,
             [i for i in range(2012,2015+1)],
             [5, 3, 5, 5] ),
         ({2012:[ [(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                    [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(7)]], 
            2013:[[(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                  [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)]], 
            2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                  [(date(2015, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
            2015:[[(date(2015, 11, 1) + timedelta(days=i), 0) for i in range(5)]]},
            [i for i in range(2012,2015+1)],
            [5, 7, 5, 5]),
          ({2012:[ [(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)] ], 
            2013:[[(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                  [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)],
                  [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(374)]], 
            2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                  [(date(2015, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
            2015:[[(date(2015, 11, 1) + timedelta(days=i), 0) for i in range(5)]]},
            [i for i in range(2012,2015+1)],
            [5, 370, 374, 5]),
          
        ],
)
def test_get_max_consecutive_event_days(gauge_events, unique_water_years, max_consecutive_events):
    result = evaluate_EWRs.get_max_consecutive_event_days(gauge_events,unique_water_years)
    assert result == max_consecutive_events


@pytest.mark.parametrize("durations,max_consecutive_days,duration_achievement",
        [ ( [5,5,5,5],
            [1,5,6,0],
            [0,1,1,0]
              ),
              ],
)
def test_get_max_rolling_duration_achievement(durations, max_consecutive_days,duration_achievement):
    result = evaluate_EWRs.get_max_rolling_duration_achievement(durations, max_consecutive_days)
    assert result == duration_achievement

@pytest.mark.parametrize("iteration,flows,expected_result",[
    (1,[100,80],-20),
    (1,[0,80], 0),
    (0,[80,50], 0),
    (1,[100,120], 20),

],)
def test_calc_flow_percent_change(iteration, flows, expected_result):
    result = evaluate_EWRs.calc_flow_percent_change(iteration,flows)
    assert result == pytest.approx(expected_result)


@pytest.mark.parametrize("flow_percent_change,flow,expected_result",[
    (-10,5, True),
    (-20, 10, True),
    (-40, 11, True),
    (-40, 9, False),
    (10, 10,True),

],)
def test_check_nest_percent_drawdown(flow_percent_change, flow, expected_result):
    EWR_info = {'max_flow':10, 'drawdown_rate':"15%"}

    result = evaluate_EWRs.check_nest_percent_drawdown(flow_percent_change, EWR_info, flow)

    assert result == expected_result


@pytest.mark.parametrize("EWR_info,iteration,expected_result",[
    ({'end_month': 10, 'end_day': None}, 0, date(2012, 10, 31)),
    ({'end_month': 9, 'end_day': None}, 0, date(2012, 9, 30)),
    ({'end_month': 12, 'end_day': None}, 0, date(2012, 12, 31)),
    ({'end_month': 4, 'end_day': None}, 366, date(2013, 4, 30)),
    ({'end_month': 4, 'end_day': 15}, 366, date(2013, 4, 15)),
],)
def test_calc_nest_cut_date(EWR_info, iteration,expected_result):
    dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
    result = evaluate_EWRs.calc_nest_cut_date(EWR_info,iteration, dates)
    assert result == expected_result


@pytest.mark.parametrize("levels,EWR_info,iteration,event_length,expected_result",[
    ( [10, 10, 9.3], {"drawdown_rate_week" : "0.3"},2, 2,False),
    ( [10, 10], {"drawdown_rate_week" : "0.3"},1, 1, True),
    ( [10, 10, 10, 10, 9.8, 9.7, 9.65], {"drawdown_rate_week" : "0.3"}, 6, 6, False),
    ( [10, 10, 10, 10, 9.8, 9.7, 9.7], {"drawdown_rate_week" : "0.3"}, 6, 6, False),
    ( [10, 10, 10, 9.8, 9.7, 9.7], {"drawdown_rate_week" : "0.3"}, 5, 5, False),
    ( [10 , 10, 10, 10, 10, 10, 9.8, 9.7, 9.8], {"drawdown_rate_week" : "0.3"}, 8, 8, True),
    ( [10], {"drawdown_rate_week" : "0.3"}, 0, 0, True),
],)
def test_check_weekly_drawdown(levels, EWR_info, iteration, event_length, expected_result):
    result = evaluate_EWRs.check_weekly_drawdown(levels, EWR_info, iteration, event_length)
    assert result == expected_result


@pytest.mark.parametrize("gauge",[
    ("425010"),
],)
def test_calc_sorter_wp(wp_df_F_df_L, wp_EWR_table, ewr_calc_config, gauge):
    
    df_F, df_L = wp_df_F_df_L
    print(df_F)

    location_results, _ = evaluate_EWRs.calc_sorter(df_F, df_L, gauge, wp_EWR_table, ewr_calc_config)

    pu_df = location_results['Murray River - Lock 10 to Lock 9']

    data_result =  pu_df.to_dict()

    assert data_result['SF_WP/WP3_eventYears'] == {1896: 1, 1897: 1, 1898: 1, 1895: 1} 
    assert data_result['LF2_WP/WP4_eventYears'] == {1896: 1, 1897: 1, 1898: 1, 1895: 1} 


@pytest.mark.parametrize("wp_freshes,freshes_eventYears,wp_eventYears,merged_eventYears",[
    (["SF_WP","LF2_WP"],
    {
    'SF_WP_eventYears': {1896: 1, 1897: 1, 1898: 0, 1895: 0}, 
    'LF2_WP_eventYears': {1896: 0, 1897: 0, 1898: 1, 1895: 1}
    },
    {
    'WP3_eventYears': {1896: 1, 1897: 1, 1898: 0, 1895: 0}, 
    'WP4_eventYears': {1896: 0, 1897: 0, 1898: 1, 1895: 1}
    },
    {
    'SF_WP/WP3_eventYears': [1,1,0,0], 
    'LF2_WP/WP4_eventYears': [0,0,1,1]
    }
    ),
    (["SF_WP"],
    {
    'SF_WP_eventYears': {1896: 0, 1897: 0, 1898: 0, 1895: 0}, 
    'LF2_WP_eventYears': {1896: 0, 1897: 0, 1898: 1, 1895: 1}
    },
    {
    'WP3_eventYears': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
    'WP4_eventYears': {1896: 0, 1897: 0, 1898: 1, 1895: 1}
    },
    {
    'SF_WP/WP3_eventYears': [1,1,1,1], 
    'LF2_WP/WP4_eventYears': [0,0,1,1]
    }
    ),
],)
def test_merge_weirpool_with_freshes(PU_df_wp, wp_freshes, freshes_eventYears, wp_eventYears, merged_eventYears):
    weirpool_pair = {'SF_WP':'WP3',
                      'LF2_WP': 'WP4' }

    pu_df_data = PU_df_wp.to_dict()

    test_pu_data = {**pu_df_data, **freshes_eventYears, **wp_eventYears}

    expeted_pu_df = evaluate_EWRs.merge_weirpool_with_freshes(wp_freshes, pd.DataFrame(test_pu_data))

    for fresh in wp_freshes:
        merged_column = f"{fresh}/{weirpool_pair[fresh]}_eventYears"
        assert merged_column in expeted_pu_df.columns
        assert  expeted_pu_df[merged_column].to_list() == merged_eventYears[merged_column]
    
    assert expeted_pu_df.shape[0] == PU_df_wp.shape[0]


@pytest.mark.parametrize("data_for_df_F,EWR,main_gauge,expected_events,pu_df_data", [
    ({'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        'A4261002': (
                               [5000]*62 + [16500]*122 + [5000]*181 + 
                                [5000]*62 + [16500]*122 + [5000]*181 +
                                [5000]*62 + [16500]*122 + [5000]*181 +
                                [5000]*62 + [16500]*122 + [5000]*182
                        )
                        },
                        'CLLMM1a_P',
                        'A4261002' ,
                        { 2012:[[(date(2013,6,30) , 3228000)]], 
                        2013:[[(date(2014,6,30) , 3228000)]], 
                        2014:[[(date(2015,6,30) , 3228000)]], 
                        2015:[[(date(2016,6,30) , 3233000)]]},
                        {'CLLMM1a_P_eventYears': [1,1,1,1], 'CLLMM1a_P_numAchieved': [1,1,1,1], 'CLLMM1a_P_numEvents': [1,1,1,1], 
                            'CLLMM1a_P_numEventsAll': [1, 1, 1, 1], 'CLLMM1a_P_maxInterEventDays': [0, 0, 0, 0], 
                            'CLLMM1a_P_maxInterEventDaysAchieved': [1, 1, 1, 1],'CLLMM1a_P_eventLength': [1.0, 1.0, 1.0, 1.0], 
                            'CLLMM1a_P_eventLengthAchieved':  [1.0, 1.0, 1.0, 1.0], 'CLLMM1a_P_totalEventDays': [1, 1, 1, 1], 
                            'CLLMM1a_P_totalEventDaysAchieved': [1, 1, 1, 1],'CLLMM1a_P_maxEventDays':[1, 1, 1, 1],
                            'CLLMM1a_P_maxRollingEvents': [1, 1, 1, 1], 'CLLMM1a_P_maxRollingAchievement': [1, 1, 1, 1],
                            'CLLMM1a_P_missingDays': [0,0,0,0], 'CLLMM1a_P_totalPossibleDays': [365,365,365,366]}
                        ),
    ({'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        'A4261002': (
                               [5000]*62 + [16500]*122 + [5000]*181 + 
                                [5000]*62 + [16500]*122 + [5000]*181 +
                                [5000]*62 + [16500]*122 + [5000]*181 +
                                [5000]*62 + [16500]*122 + [5000]*182
                        )},
                        'CLLMM1b',
                        'A4261002' ,
                        { 2012:[], 
                        2013:[], 
                        2014:[[(date(2015,6,30) , 9684000)]], 
                        2015:[[(date(2016,6,30) , 9689000)]]},
                        {'CLLMM1b_eventYears': [0,0,1,1], 'CLLMM1b_numAchieved': [0,0,1,1], 'CLLMM1b_numEvents': [0,0,1,1], 
                            'CLLMM1b_numEventsAll': [0,0,1,1], 'CLLMM1b_maxInterEventDays': [0, 0, 0, 0], 
                            'CLLMM1b_maxInterEventDaysAchieved': [1, 1, 1, 1],'CLLMM1b_eventLength': [0.0, 0.0, 1.0, 1.0], 
                            'CLLMM1b_eventLengthAchieved':  [0.0, 0.0, 1.0, 1.0], 'CLLMM1b_totalEventDays': [0, 0, 1, 1], 
                            'CLLMM1b_totalEventDaysAchieved': [0, 0, 1, 1],'CLLMM1b_maxEventDays':[0, 0, 1, 1],
                            'CLLMM1b_maxRollingEvents': [0, 0, 1, 1], 'CLLMM1b_maxRollingAchievement': [0, 0, 1, 1],
                            'CLLMM1b_missingDays': [0,0,0,0], 'CLLMM1b_totalPossibleDays': [365,365,365,366]}
                        ),
])
def test_barrage_flow_handle(data_for_df_F, EWR, main_gauge, expected_events, pu_df_data, sa_parameter_sheet):

    # Set up input data
    PU = 'PU_0000029'

    EWR_table = sa_parameter_sheet
	 
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')

    PU_df = pd.DataFrame()
    # Pass input data to test function:

    PU_df, events = evaluate_EWRs.barrage_flow_handle(PU, main_gauge, EWR, EWR_table, df_F, PU_df)
    
    # Setting up expected output - PU_df - and testing
    index = pd.Index([2012, 2013, 2014, 2015])
    expected_PU_df = pd.DataFrame(index = index, data = pu_df_data)
    expected_PU_df.index = expected_PU_df.index.astype('int64')
    PU_df.index = PU_df.index.astype('int64')
    assert_frame_equal(PU_df, expected_PU_df)
    
    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]



@pytest.mark.parametrize("expected_events,expected_PU_df_data",[
    (
    {   2012:[[(date(2013, 6, 30), 0.8)]], 
        2013:[], 
        2014:[], 
        2015:[]},
    {'CLLMM1c_P_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'CLLMM1c_P_numAchieved': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'CLLMM1c_P_numEvents': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'CLLMM1c_P_numEventsAll': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'CLLMM1c_P_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
     'CLLMM1c_P_maxInterEventDaysAchieved':{2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
     'CLLMM1c_P_eventLength': {2012: 1.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
     'CLLMM1c_P_eventLengthAchieved': {2012: 1.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
     'CLLMM1c_P_totalEventDays': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'CLLMM1c_P_totalEventDaysAchieved': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'CLLMM1c_P_maxEventDays': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'CLLMM1c_P_maxRollingEvents': {2012: 1, 2013: 0, 2014: 0, 2015: 0},
       'CLLMM1c_P_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0},
       'CLLMM1c_P_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
       'CLLMM1c_P_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}}  
    )
])
def test_barrage_level_handle(sa_parameter_sheet, expected_events, expected_PU_df_data):
    # Set up input data
    PU = 'PU_0000029'
    gauge = 'A4260527'
    barrage_gauges =  ['A4260527','A4261133', 'A4260524', 'A4260574', 'A4260575']
    EWR = 'CLLMM1c_P'
    gauge_levels = (  [.55]*66 + [.8]*5 + [.6]*115 + [.55]*179 + 
                            [0]*365 + 
                            [0]*365 + 
                            [0]*366
                ) 
    gauge_levels_data = { gauge:gauge_levels for gauge in barrage_gauges }

    EWR_table = sa_parameter_sheet
    DATE = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()}
    
    data_for_df_L = {**DATE, **gauge_levels_data}
    df_L = pd.DataFrame(data = data_for_df_L)
    df_L = df_L.set_index('Date')

    PU_df = pd.DataFrame()
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.barrage_level_handle(PU, gauge, EWR, EWR_table, df_L, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data
    
    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]


@pytest.mark.parametrize("expected_events,expected_PU_df_data",[
    (
    {   2012:[[(date(2012,9,1) + timedelta(days=i), 12001) for i in range(60)] 
              + [(date(2012,10,31) + timedelta(days=i), 10000) for i in range(1)]], 
        2013:[], 
        2014:[], 
        2015:[]},
   {'IC1_P_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
    'IC1_P_numAchieved': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
    'IC1_P_numEvents': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
    'IC1_P_numEventsAll': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
    'IC1_P_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0},
    'IC1_P_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1},
    'IC1_P_eventLength': {2012: 61.0, 2013: 0.0, 2014: 0.0, 2015: 0.0},
    'IC1_P_eventLengthAchieved': {2012: 61.0, 2013: 0.0, 2014: 0.0, 2015: 0.0},
    'IC1_P_totalEventDays': {2012: 61, 2013: 0, 2014: 0, 2015: 0},
    'IC1_P_totalEventDaysAchieved': {2012: 61, 2013: 0, 2014: 0, 2015: 0},
    'IC1_P_maxEventDays': {2012: 61, 2013: 0, 2014: 0, 2015: 0}, 
    'IC1_P_maxRollingEvents': {2012: 61, 2013: 0, 2014: 0, 2015: 0}, 
    'IC1_P_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
    'IC1_P_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
    'IC1_P_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}}  
    )
])
def test_flow_handle_sa(sa_parameter_sheet, expected_events, expected_PU_df_data):
     # Set up input data
    PU = 'PU_0000027'
    gauge = 'A4261001'
    EWR = 'IC1_P'

    EWR_table = sa_parameter_sheet

    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: (
                                [0]*31+ [400 + i*400 for i in range(30)] + [12001]*61 + [10000] + 
                                [9900 - i*200 for i in range(30)] + [0]*212 + 
                                [0]*365 + 
                                [0]*365 + 
                                [0]*366
                        )} 
    df_F = pd.DataFrame(data = data_for_df_F)

    df_F = df_F.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.flow_handle_sa(PU, gauge, EWR, EWR_table, df_F, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data
    
    

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]


@pytest.mark.parametrize("flows, iteration, period, expected_result",[
    (
    [0]*10 +[90]*90 +[2000]*265 +  
    [0]*365 +
    [0]*365 + 
    [0]*366,
    99,
    3,
    True
    ),
      (
    [2]*10 +[90]*90 +[2000]*265 +  
    [0]*365 +
    [0]*365 + 
    [0]*366,
    99,
    3,
    False
    ),
      (
    [0]*10 +[90]*90 +[2000]*265 +  
    [0]*365 +
    [0]*365 + 
    [0]*366,
    105,
    3,
    False
    ),
])
def test_check_cease_flow_period(flows, iteration, period, expected_result):
    result = evaluate_EWRs.check_cease_flow_period(flows, iteration, period)
    assert result == expected_result


@pytest.mark.parametrize("expected_events,expected_PU_df_data",[
    (
    {   2012:[], 
        2013:[], 
        2014:[ [(date(2012,7,1) + timedelta(days=i), 0) for i in range(365)] +
		   	 [(date(2013,7,1) + timedelta(days=i), 19) for i in range(10)] +
			 [(date(2013,7,11) + timedelta(days=i), 0) for i in range(365)]], 
        2015:[]},
{'FD1_eventYears': {2012: 0, 2013: 0, 2014: 1, 2015: 0}, 
 'FD1_numAchieved': {2012: 0, 2013: 0, 2014: 1, 2015: 0}, 
 'FD1_numEvents': {2012: 0, 2013: 0, 2014: 1, 2015: 0}, 
 'FD1_numEventsAll': {2012: 0, 2013: 0, 2014: 1, 2015: 0}, 
 'FD1_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
 'FD1_eventLength': {2012: 0.0, 2013: 0.0, 2014: 740.0, 2015: 0.0}, 
 'FD1_eventLengthAchieved': {2012: 0.0, 2013: 0.0, 2014: 740.0, 2015: 0.0}, 
 'FD1_totalEventDays': {2012: 0, 2013: 0, 2014: 740, 2015: 0}, 
 'FD1_totalEventDaysAchieved': {2012: 0, 2013: 0, 2014: 740, 2015: 0}, 
 'FD1_maxEventDays': {2012: 0, 2013: 0, 2014: 740, 2015: 0}, 
 'FD1_maxRollingEvents': {2012: 365, 2013: 730, 2014: 740, 2015: 0},
 'FD1_maxRollingAchievement': {2012: 1, 2013: 1, 2014: 1, 2015: 0}, 
 'FD1_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}}
    )
])
def test_flow_handle_check_ctf(qld_parameter_sheet, expected_events, expected_PU_df_data):
     # Set up input data
    PU = 'PU_0000991'
    gauge = '422015'
    EWR = 'FD1'

    EWR_table = qld_parameter_sheet

    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: (  [0]*365 + # first dry spell
                                [19]*10 + # in between
                                [0]*365 + # second dry spell
                                [6]*355 +		      
                                [0]*366
                                )
                        } 
    df_F = pd.DataFrame(data = data_for_df_F)

    df_F = df_F.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.flow_handle_check_ctf(PU, gauge, EWR, EWR_table, df_F, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]


@pytest.mark.parametrize("expected_events,expected_PU_df_data",[
    (
    {   2012:[[(date(2012,7,1) + timedelta(days=i), 15400) for i in range(14)]], 
        2013:[], 
        2014:[], 
        2015:[]},
    {'BBR1_a_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_numAchieved': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_numEvents': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_numEventsAll': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
     'BBR1_a_eventLength': {2012: 14.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
     'BBR1_a_eventLengthAchieved': {2012: 0.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
     'BBR1_a_totalEventDays': {2012: 14, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_totalEventDaysAchieved': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_maxEventDays': {2012: 14, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_maxRollingEvents': {2012: 14, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_maxRollingAchievement': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
     'BBR1_a_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}}
    )
])
def test_cumulative_handle_bbr(qld_parameter_sheet, expected_events, expected_PU_df_data):
     # Set up input data
    PU = 'PU_0000991'
    gauge = '422016'
    EWR = 'BBR1_a'

    EWR_table = qld_parameter_sheet

    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: (
                                [15400]*20+[0]*345 + 
                                [0]*365 + 
                                [0]*365 + 
                                [0]*366
                        )} 
    df_F = pd.DataFrame(data = data_for_df_F)

    df_F = df_F.set_index('Date')
    
    data_for_df_L = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        "422034": (
                               [1.]*10 +[1.3]*3+[1.]*5+[0]*347  + 
                                [0]*365 + 
                                [0]*365 + 
                                [0]*366
                        )} 
    df_L = pd.DataFrame(data = data_for_df_L)

    df_L = df_L.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.cumulative_handle_bbr(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]




@pytest.mark.parametrize("events, expected_result",[
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(5)] , 
        [(date(2012, 11, 2) + timedelta(days=i), 0) for i in range(3)]
        ],
        28 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(3)]
        ],
        0 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(31)]
        ],
        0 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(5)] , 
        [(date(2012, 11, 2) + timedelta(days=i), 0) for i in range(3)],
        [(date(2012, 11, 24) + timedelta(days=i), 0) for i in range(10)],
        ],
        20 
    ),
    (
       [],
        0 
    ),
       
])
def test_get_min_gap(events, expected_result):
    result = evaluate_EWRs.get_min_gap(events)
    assert result == expected_result

@pytest.mark.parametrize("events, expected_result",[
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(5)] , 
        [(date(2012, 11, 2) + timedelta(days=i), 0) for i in range(3)]
        ],
        28 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(3)]
        ],
        0 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(31)]
        ],
        0 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(5)] , 
        [(date(2012, 11, 2) + timedelta(days=i), 0) for i in range(3)],
        [(date(2012, 11, 24) + timedelta(days=i), 0) for i in range(10)],
        ],
        28 
    ),
    (
       [],
        0 
    ),
       
])
def test_get_max_gap(events, expected_result):
    result = evaluate_EWRs.get_max_gap(events)
    assert result == expected_result

@pytest.mark.parametrize("events, expected_result",[
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(5)] , 
        [(date(2012, 11, 2) + timedelta(days=i), 0) for i in range(3)]
        ],
        5 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(3)]
        ],
        3 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(31)]
        ],
        31 
    ),
    (
       [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(5)] , 
        [(date(2012, 11, 2) + timedelta(days=i), 0) for i in range(3)],
        [(date(2012, 11, 24) + timedelta(days=i), 0) for i in range(10)],
        ],
        10 
    ),
    (
       [],
        0 
    ),
       
])
def test_get_max_event_length(events, expected_result):
    result = evaluate_EWRs.get_max_event_length(events)
    assert result == expected_result



@pytest.mark.parametrize("event_years, expected_results",[
    (
    
    { 2012: [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(90)] , 
        [(date(2013, 1, 2) + timedelta(days=i), 0) for i in range(1)],
        [(date(2013, 1, 24) + timedelta(days=i), 0) for i in range(1)],
      ],
     2013: [
        [(date(2013, 10, 1) + timedelta(days=i), 0) for i in range(10)] , 
        [(date(2013, 11, 10) + timedelta(days=i), 0) for i in range(3)],
      ],
     2014: [
        [(date(2014, 10, 1) + timedelta(days=i), 0) for i in range(5)] , 
        [(date(2014, 10, 20) + timedelta(days=i), 0) for i in range(29)],
      ],
     2015: [],
    },
    [1,1,0,0]
    )
])
def test_get_event_years_connecting_events(event_years, expected_results):
    unique_water_years = [2012, 2013, 2014, 2015]
    result = evaluate_EWRs.get_event_years_connecting_events(event_years, unique_water_years)
    assert result == expected_results


@pytest.mark.parametrize("event_years, expected_results",[
    (
    
    { 2012: [
        [(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(90)] , 
        [(date(2013, 1, 2) + timedelta(days=i), 0) for i in range(1)],
        [(date(2013, 1, 24) + timedelta(days=i), 0) for i in range(1)],
      ],
     2013: [
        [(date(2013, 10, 1) + timedelta(days=i), 0) for i in range(10)] , 
        [(date(2013, 10, 31) + timedelta(days=i), 0) for i in range(3)],
        [(date(2013, 11, 30) + timedelta(days=i), 0) for i in range(3)],
        [(date(2013, 12, 31) + timedelta(days=i), 0) for i in range(3)]
      ],
     2014: [
        [(date(2014, 10, 1) + timedelta(days=i), 0) for i in range(5)] , 
        [(date(2014, 10, 20) + timedelta(days=i), 0) for i in range(29)],
      ],
     2015: [[(date(2012, 10, 1) + timedelta(days=i), 0) for i in range(90)]],
    },
    [1,3,0,1]
    )
])
def test_get_achievements_connecting_events(event_years, expected_results):
    unique_water_years = [2012, 2013, 2014, 2015]
    result = evaluate_EWRs.get_achievements_connecting_events(event_years, unique_water_years)
    assert result == expected_results


@pytest.mark.parametrize("expected_events, expected_PU_df_data", [
    (
      { 2012:[[(date(2012, 8, 1)+timedelta(days=i), 71) for i in range(9)],
              [(date(2012, 8, 2)+timedelta(days=i), 71) for i in range(9)]], 
        2013:[], 
        2014:[], 
        2015:[]},
 {'FrW2_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_numAchieved': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_numEvents': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_numEventsAll': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
  'FrW2_eventLength': {2012: 9.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
  'FrW2_eventLengthAchieved': {2012: 9.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
  'FrW2_totalEventDays': {2012: 18, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_totalEventDaysAchieved': {2012: 18, 2013: 0, 2014: 0, 2015: 0},
  'FrW2_maxEventDays': {2012: 9, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_maxRollingEvents': {2012: 9, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0},  
  'FrW2_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
  'FrW2_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}},
    )
])
def test_water_stability_handle(qld_parameter_sheet, expected_events, expected_PU_df_data):
     # Set up input data
    PU = 'PU_0000999'
    gauge = '416011'
    EWR = 'FrW2'

    EWR_table = qld_parameter_sheet

    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: (    [0]*31 + [71]*10 + [0]*324 + 
                                    [0]*365 + 
                                    [0]*365 + 
                                    [0]*366)} 
    df_F = pd.DataFrame(data = data_for_df_F)

    df_F = df_F.set_index('Date')
    
    data_for_df_L = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        "416011": (     [1]*365 + 
                                        [0]*365 + 
                                        [0]*365 + 
                                        [0]*366)} 
    df_L = pd.DataFrame(data = data_for_df_L)

    df_L = df_L.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.water_stability_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]


@pytest.mark.parametrize("expected_events, expected_PU_df_data", [
    (
      { 2012:[[(date(2012, 8, 1)+timedelta(days=i), 1) for i in range(9)],
              [(date(2012, 8, 2)+timedelta(days=i), 1) for i in range(9)]], 
        2013:[], 
        2014:[], 
        2015:[]},
 {'FrL2_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_numAchieved': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_numEvents': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_numEventsAll': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
  'FrL2_eventLength': {2012: 9.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
  'FrL2_eventLengthAchieved': {2012: 9.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
  'FrL2_totalEventDays': {2012: 18, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_totalEventDaysAchieved': {2012: 18, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_maxEventDays': {2012: 9, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_maxRollingEvents': {2012: 9, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
  'FrL2_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}
  },
    )
])
def test_water_stability_level_handle(qld_parameter_sheet, expected_events, expected_PU_df_data):
     # Set up input data
    PU = 'PU_0000991'
    gauge = '422015'
    EWR = 'FrL2'

    EWR_table = qld_parameter_sheet

    data_for_df_L = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        "422015": (     [2]*31 + [1]*10 + [2]*324 +
                                        [2]*365 + 
                                        [2]*365 + 
                                        [2]*366)} 
    df_L = pd.DataFrame(data = data_for_df_L)

    df_L = df_L.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.water_stability_level_handle(PU, gauge, EWR, EWR_table, df_L, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]


@pytest.mark.parametrize("expected_events, expected_PU_df_data", [
    (
      {2012: [[(date(2013, 6, 16)+timedelta(days=i),5600) for i in range(15+11)]], 
                2013: [], 
                2014: [], 
                2015: [] },

{'FD1_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_numAchieved': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_numEvents': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_numEventsAll': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
 'FD1_eventLength': {2012: 26.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
 'FD1_eventLengthAchieved': {2012: 26.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
 'FD1_totalEventDays': {2012: 26, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_totalEventDaysAchieved': {2012: 26, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_maxEventDays': {2012: 26, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_maxRollingEvents': {2012: 15, 2013: 26, 2014: 0, 2015: 0}, 
 'FD1_maxRollingAchievement': {2012: 1, 2013: 1, 2014: 0, 2015: 0}, 
 'FD1_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
 'FD1_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}},
    )
])
def test_flow_handle_anytime(qld_parameter_sheet, expected_events, expected_PU_df_data):
     # Set up input data
    PU = 'PU_0000999'
    gauge = '416011'
    EWR = 'FD1'

    EWR_table = qld_parameter_sheet

    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        '416011': (    [0]*350+[5600]*15 + 
	                                   [5600]*11+ [0]*354 + 
									   [0]*365 +
									   [0]*366)} 
    

    df_F = pd.DataFrame(data = data_for_df_F)

    df_F = df_F.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.flow_handle_anytime(PU, gauge, EWR, EWR_table, df_F, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]


@pytest.mark.parametrize("pu, gauge, ewr, gauge_data, expected_events, expected_PU_df_data", [
    (  'PU_0000189',
          '405203',
           'RFF' ,
    np.array(    [40,30,41,30] + [30]*361 + 
				  [30]*365 +
				  [30]*365 + 
				  [30]*366),
    {2012: [[(date(2012, 7, 2), 30)], [(date(2012, 7, 4), 30)]], 
                2013: [], 
                2014: [], 
                2015: [] },

{  'RFF_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_numAchieved': {2012: 2, 2013: 0, 2014: 0, 2015: 0},
   'RFF_numEvents': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_numEventsAll': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
   'RFF_eventLength': {2012: 1.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
   'RFF_eventLengthAchieved': {2012: 1.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
   'RFF_totalEventDays': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_totalEventDaysAchieved': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_maxEventDays': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_maxRollingEvents': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
   'RFF_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}},
    ),
    (  'PU_0000189',
          '405203',
           'RRF' ,
    np.array( [1, 2.1, 4.45] + [1000,2001, 3000, 4000, 5000, 15000] + [1000]*356 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
    {2012: [[(date(2012, 7, 4), 1000.0), (date(2012, 7, 5), 2001.0)],
				[(date(2012, 7, 9), 15000.0)]], 
                2013: [], 
                2014: [], 
                2015: [] },

    {   
    'RRF_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_numAchieved': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_numEvents': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_numEventsAll': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1},
    'RRF_eventLength': {2012: 1.5, 2013: 0.0, 2014: 0.0, 2015: 0.0},
    'RRF_eventLengthAchieved': {2012: 1.5, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
    'RRF_totalEventDays': {2012: 3, 2013: 0, 2014: 0, 2015: 0},
    'RRF_totalEventDaysAchieved': {2012: 3, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_maxEventDays': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_maxRollingEvents': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
    'RRF_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}
    },
    ),
    (  'PU_0000192',
          '405200',
           'RRL_su' ,
    np.array( [0]*153 + [1, 1.39, 1.8, 1, 1.39, 1.8 ] +  [0]*206 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
    {2012: [[(date(2012, 12, 1), 1.0), (date(2012, 12, 2), 1.39), (date(2012, 12, 3), 1.8)], 
            [(date(2012, 12, 5), 1.39), (date(2012, 12, 6), 1.8)]], 
                2013: [], 
                2014: [], 
                2015: [] },

    {
        'RRL_su_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_numAchieved': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_numEvents': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_numEventsAll': {2012: 2, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
        'RRL_su_eventLength': {2012: 2.5, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
        'RRL_su_eventLengthAchieved': {2012: 2.5, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
        'RRL_su_totalEventDays': {2012: 5, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_totalEventDaysAchieved': {2012: 5, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_maxEventDays': {2012: 3, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_maxRollingEvents': {2012: 3, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
        'RRL_su_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}}
    ),
    (  'PU_0000192',
          '405200',
           'RFL_su' ,
         np.array([0]*153 + [1.22, 1, .77 ] + [0]*209 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
    {2012: [[(date(2012, 12, 2), 1.0), (date(2012, 12, 3), 0.77), (date(2012, 12, 4), 0.0)]], 
                2013: [], 
                2014: [], 
                2015: [] },

   {
       'RFL_su_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_numAchieved': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_numEvents': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_numEventsAll': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
       'RFL_su_eventLength': {2012: 3.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
       'RFL_su_eventLengthAchieved': {2012: 3.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
       'RFL_su_totalEventDays': {2012: 3, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_totalEventDaysAchieved': {2012: 3, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_maxEventDays': {2012: 3, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_maxRollingEvents': {2012: 3, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
       'RFL_su_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}}
    ),
])
def test_rise_and_fall_handle(pu, gauge, ewr, gauge_data, expected_events, expected_PU_df_data, vic_parameter_sheet):
    EWR_table = vic_parameter_sheet

    data_for_df = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: gauge_data } 
    

    df_F = pd.DataFrame(data = data_for_df)
    df_L = pd.DataFrame(data = data_for_df)

    df_F = df_F.set_index('Date')
    df_L = df_L.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.rise_and_fall_handle(pu, gauge, ewr, EWR_table, df_F, df_L, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]


@pytest.mark.parametrize("pu, gauge, ewr, gauge_data, expected_events, expected_PU_df_data", [
    (  'PU_0000191',
          '405202',
           'F3' ,
        	np.array([1]*62 +[1., 1., 1., 1., 1., 1., 1.51, 2.01, 1.3] + [0]*294 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
    {2012: [[(date(2012, 9, 1), 1.0), (date(2012, 9, 2), 1.0), (date(2012, 9, 3), 1.0), (date(2012, 9, 4), 1.0), 
             (date(2012, 9, 5), 1.0), (date(2012, 9, 6), 1.0), (date(2012, 9, 7), 1.51), (date(2012, 9, 8), 2.01)]], 
                2013: [], 
                2014: [], 
                2015: [] },

   {
       'F3_eventYears': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_numAchieved': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_numEvents': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_numEventsAll': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_maxInterEventDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_maxInterEventDaysAchieved': {2012: 1, 2013: 1, 2014: 1, 2015: 1}, 
       'F3_eventLength': {2012: 8.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
       'F3_eventLengthAchieved': {2012: 8.0, 2013: 0.0, 2014: 0.0, 2015: 0.0}, 
       'F3_totalEventDays': {2012: 8, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_totalEventDaysAchieved': {2012: 8, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_maxEventDays': {2012: 8, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_maxRollingEvents': {2012: 8, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_maxRollingAchievement': {2012: 1, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_missingDays': {2012: 0, 2013: 0, 2014: 0, 2015: 0}, 
       'F3_totalPossibleDays': {2012: 365, 2013: 365, 2014: 365, 2015: 366}}
    ),
])
def test_level_change_handle(pu, gauge, ewr, gauge_data, expected_events, expected_PU_df_data, vic_parameter_sheet):
    
    EWR_table = vic_parameter_sheet

    data_for_df = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge: gauge_data } 
    

    df_L = pd.DataFrame(data = data_for_df)

    df_L = df_L.set_index('Date')

    PU_df = pd.DataFrame()
    
    # Pass input data to test function:
    
    PU_df, events = evaluate_EWRs.level_change_handle(pu, gauge, ewr, EWR_table, df_L, PU_df)

    assert PU_df.to_dict() == expected_PU_df_data

    expected_events = tuple([expected_events])
    for index, _ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]
    