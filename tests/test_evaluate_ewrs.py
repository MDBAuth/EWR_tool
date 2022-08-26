from datetime import datetime, date, timedelta

import pandas as pd
from pandas._testing import assert_frame_equal
import pytest

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
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Send input data to test function:
    PU_df, events = evaluate_EWRs.ctf_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
    # Setting up expected output - PU_df
    data = {'CF1_eventYears': [0,0,0,1], 'CF1_numAchieved': [0,0,0,1], 'CF1_numEvents': [0,0,0,1], 
      'CF1_maxInterEventDays': [0,0,0,0],  'CF1_maxInterEventDaysAchieved': [1,1,1,1],                                                               'CF1_eventLength': [0.0,0.0,0.0,1461.0], 
    'CF1_totalEventDays': [0,0,0,1461], 'CF1_maxEventDays': [0,0,0,1461], 'CF1_maxRollingEvents': [365, 730, 1095, 1461], 'CF1_maxRollingAchievement': [1, 1, 1, 1],
    'CF1_daysBetweenEvents': [[],[],[],[]],'CF1_missingDays': [0,0,0,0], 'CF1_totalPossibleDays': [365,365,365,366]} 
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
#         print(PU_df.head())
#         print(expected_PU_df.head())        
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
                        gauge: [0]*1+[250]*350+[0]*9+[0]*5 + [0]*360+[0]*5 + [0]*2+[250]*345+[0]*1+[250]*17 + [0]*5+[250]*351+[250]*10}
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Send input data to test function
    PU_df, events = evaluate_EWRs.lowflow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
    # Setting up expected output data - PU_df, and testing
    data = {'BF1_a_eventYears': [0,0,0,0], 'BF1_a_numAchieved': [0,0,0,0], 'BF1_a_numEvents': [0,0,0,0], 'BF1_a_maxInterEventDays': [0,0,0,1461], 
           'BF1_a_maxInterEventDaysAchieved': [1,1,1,0],  
            'BF1_a_eventLength': [0.0,0.0,0.0,0.0], 'BF1_a_totalEventDays': [0,0,0,0], 
            'BF1_a_maxEventDays': [0,0,0,0], 'BF1_a_maxRollingEvents': [0, 0, 0, 0], 'BF1_a_maxRollingAchievement': [0, 0, 0, 0],
            'BF1_a_daysBetweenEvents': [[],[],[],[1461]],
            'BF1_a_missingDays': [0,0,0,0], 'BF1_a_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
#         print(PU_df.head())
#         print(expected_PU_df.head())
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
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Send input data to test function
    PU_df, events = evaluate_EWRs.flow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
    # Setting up expected output - PU_df - and testing
    data = {'SF1_S_eventYears': [0,0,0,1], 'SF1_S_numAchieved': [0,0,0,1], 'SF1_S_numEvents': [1,0,1,3],
           'SF1_S_maxInterEventDays': [351, 0, 720, 330], 
           'SF1_S_maxInterEventDaysAchieved': [1, 1, 0, 1], 'SF1_S_eventLength': [10.0,0.0,14.0,10.0], 'SF1_S_totalEventDays': [10,0,14,30], 
            'SF1_S_maxEventDays': [10, 0, 14, 10], 'SF1_S_maxRollingEvents': [10, 0, 14, 10], 'SF1_S_maxRollingAchievement': [1, 0, 1, 1],
            'SF1_S_daysBetweenEvents': [[],[],[720],[]],'SF1_S_missingDays': [0,0,0,0], 'SF1_S_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
#         print(PU_df.head())
#         print(expected_PU_df.head())
    assert_frame_equal(PU_df, expected_PU_df)
    # Setting up expected output - events - and testing
    expected_events = {2012: [[(date(2013, 6, 17) + timedelta(days=i), 450) for i in range(10)]], 
                        2013: [], 
                        2014: [
                                [(date(2015, 6, 17) + timedelta(days=i), 450) for i in range(14)]],
                        2015: [[(date(2015, 7, 6) + timedelta(days=i), 450) for i in range(10)],
                            [(date(2015, 7, 17) + timedelta(days=i), 450) for i in range(10)],     
                            [(date(2016, 6, 21) + timedelta(days=i), 450) for i in range(10)]]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
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
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Send input data to test function
    PU_df, events = evaluate_EWRs.cumulative_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
    # Setting up expected output - PU_df - and testing
    data = {'OB3_S_eventYears': [0,0,0,0], 'OB3_S_numAchieved': [0,0,0,0], 'OB3_S_numEvents': [0,0,0,0], 
            'OB3_S_maxInterEventDays': [0, 0, 0, 1461], 
           'OB3_S_maxInterEventDaysAchieved': [1, 1, 1, 0],'OB3_S_eventLength': [0,0.0,0.0,0.0], 
            'OB3_S_totalEventDays': [0,0,0,0], 'OB3_S_maxEventDays': [0,0,0,0],'OB3_S_maxRollingEvents': [0, 0, 0, 0], 
            'OB3_S_maxRollingAchievement': [0, 0, 0, 0],'OB3_S_daysBetweenEvents': [[],[],[],[1461]],'OB3_S_missingDays': [0,0,0,0], 
            'OB3_S_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
    assert_frame_equal(PU_df, expected_PU_df)
    # Setting up expected output - events - and testing 
    expected_events = {2012:[], 2013:[], 2014:[], 2015:[]}
    expected_events = tuple([expected_events])
    print(events)
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

@pytest.mark.xfail(raises=ValueError, reason="could not convert string to float: '4%'")
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
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Send input data to test function
    PU_df, events = evaluate_EWRs.level_handle(PU, gauge, EWR, EWR_table, df_L, PU_df, allowance)
    # print(events)
    # Setting up expected output - PU_df and test
    data = {'LLLF_eventYears': [1,0,0,1], 'LLLF_numAchieved': [1,0,0,1], 'LLLF_numEvents': [1,0,0,1], 
            'LLLF_maxInterEventDays': [261, 0, 652, 277], 
            'LLLF_maxInterEventDaysAchieved': [1, 1, 1, 1],'LLLF_eventLength': [90.0,0.0,0.0,90.0], 'LLLF_totalEventDays': [90,0,0,90], 
            'LLLF_maxEventDays': [90,0,0,90], 'LLLF_maxRollingEvents': [90, 0, 1, 90],'LLLF_maxRollingAchievement': [1, 0, 0, 1],
            'LLLF_daysBetweenEvents': [[],[],[],[]],'LLLF_missingDays': [0,0,0,0], 'LLLF_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
#         print(PU_df.head())
#         print(expected_PU_df.head())
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


def test_weirpool_handle():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000260'
    gauge = '414203'
    wp_gauge = '414209'
    EWR = 'WP1'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    # input data for df_L:
    levels = [47.3]*100 
    reduction_max = 0.04
    for i, level in enumerate(levels):
        levels[i] = levels[i-1]-reduction_max # Levels declining at acceptable rate
    exceeding_levels = [47.3]*100 
    reduction_max = 0.05
    for i, level in enumerate(exceeding_levels):
        exceeding_levels[i] = exceeding_levels[i-1]-reduction_max # Levels exceeding the acceptable rate: 
    
    data_for_df_L = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                        wp_gauge: ([0]*187+[47.3]*90+[0]*88 + 
                                  [47.3]*90+[0]*275 + 
                                  [0]*187+ levels+[0]*78 + 
                                  [0]*187 + exceeding_levels + [0]*79 ) }
    df_L = pd.DataFrame(data = data_for_df_L)
    df_L = df_L.set_index('Date')
    # input data for df_F:
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                        gauge: [0]*187+[2500]*90+[0]*88 + [2500]*90+[0]*275 + [0]*187+[2500]*100+[0]*78 + [0]*187+[2500]*100+[0]*79}
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Passing input data to test function
    PU_df, events = evaluate_EWRs.weirpool_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance)
    
    # print(events)

    # Setting up expected output data - PU_df - and testing
    data = {'WP1_eventYears': [1,0,1,1], 'WP1_numAchieved': [1,0,1,1], 'WP1_numEvents': [1,0,1,1], 
            'WP1_maxInterEventDays': [187, 0, 640, 265], 
            'WP1_maxInterEventDaysAchieved': [1, 1, 1, 1],'WP1_eventLength': [90.0, 0.0, 100.0, 100.0], 'WP1_totalEventDays': [90,0,100,100], 
            'WP1_maxEventDays':[90,0,100,100], 'WP1_maxRollingEvents': [90, 0, 100, 100], 'WP1_maxRollingAchievement': [1, 0, 1, 1],
            'WP1_daysBetweenEvents': [[],[],[],[]],'WP1_missingDays': [0,0,0,0], 'WP1_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')

    assert_frame_equal(PU_df, expected_PU_df)
    # Setting up expected output - events - and testing
    expected_events = {2012:[[(date(2013, 1, 4) + timedelta(days=i), 2500) for i in range(90)]], 
                       2013:[], 
                       2014:[[(date(2015, 1, 4) + timedelta(days=i), 2500) for i in range(100)]], 
                       2015:[[(date(2016, 1, 4) + timedelta(days=i), 2500) for i in range(100)]]}

                        

    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            if year == 2015:
                print(events[index][year])
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
    reduction_max = 5.9
    for i, flow in enumerate(acceptable_flows):
        acceptable_flows[i] = acceptable_flows[i-1]-(reduction_max/100*acceptable_flows[i-1])
    acceptable_flows = acceptable_flows + [5300]*50
    # flows declining at unnacceptable rate:
    unnacceptable_flows = [10000]*10
    reduction_max = 7
    for i, flow in enumerate(unnacceptable_flows):
        unnacceptable_flows[i] = unnacceptable_flows[i-1]-(reduction_max/100*unnacceptable_flows[i-1])
    unnacceptable_flows = unnacceptable_flows + [5300]*50
    # flows declining at acceptable rate but going below the threshold
    threshold_flows = [10000]*10
    reduction_max = 6
    for i, flow in enumerate(threshold_flows):
        threshold_flows[i] = threshold_flows[i-1]-(reduction_max/100*threshold_flows[i-1])
    threshold_flows = threshold_flows + [5300]*50
    # input data for df_F:
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                        gauge: ([0]*76+acceptable_flows+[0]*229 + 
                                [0]*76+unnacceptable_flows+[0]*229 + 
                                [0]*76+threshold_flows+[0]*229 + 
                                [0]*77+threshold_flows+[0]*229)}
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to test function:
    PU_df, events = evaluate_EWRs.nest_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance)
    print(events)
    # Setting up expected output - PU_df - and testing
    data = {'NestS1_eventYears': [1,0,0,0], 'NestS1_numAchieved': [1,0,0,0], 'NestS1_numEvents': [1,0,0,0], 
            'NestS1_maxInterEventDays': [76, 0, 0, 1325], 
            'NestS1_maxInterEventDaysAchieved': [1, 1, 1, 0],'NestS1_eventLength': [60.0,0.0,0.0,0.0], 'NestS1_totalEventDays': [60,0,0,0],
            'NestS1_maxEventDays':[60,0,0,0],'NestS1_maxRollingEvents': [60, 0, 0, 0], 'NestS1_maxRollingAchievement': [1, 0, 0, 0],
            'NestS1_daysBetweenEvents': [[],[],[],[1325]],'NestS1_missingDays': [0,0,0,0], 'NestS1_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
    assert_frame_equal(PU_df, expected_PU_df)

    acceptable_flows_results = [(date(2012, 9, 15) + timedelta(days=i), f) for i, f in enumerate(acceptable_flows)]

    expected_events = {2012:[acceptable_flows_results], 2013:[], 2014:[], 2015:[]}
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
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Send input data to test function
    PU_df, events = evaluate_EWRs.flow_handle_multi(PU, gauge1, EWR, EWR_table, df_F, PU_df, allowance)
    # Setting up expected output - PU_df - and testing
    data = {'LF1_eventYears': [1,0,1,0], 'LF1_numAchieved': [1,0,1,0], 'LF1_numEvents': [1,0,1,0], 
            'LF1_maxInterEventDays': [76, 0, 778, 597], 
            'LF1_maxInterEventDaysAchieved': [1, 1, 0, 1],'LF1_eventLength': [5.0,0.0,5.0,0.0], 'LF1_totalEventDays': [5,0,5,0],
            'LF1_maxEventDays':[5, 0, 5, 0], 'LF1_maxRollingEvents': [5, 0, 5, 0], 'LF1_maxRollingAchievement': [1, 0, 1, 0],
            'LF1_daysBetweenEvents': [[],[],[778],[]],'LF1_missingDays': [0,0,0,0], 'LF1_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
    assert_frame_equal(PU_df, expected_PU_df)    
    # Setting up expected output - events - and testing
    expected_events = {2012:[[(date(2012, 9, 15) + timedelta(days=i), 2500) for i in range(5)]], 
                       2013:[], 
                       2014:[ [(date(2014, 11, 7) + timedelta(days=i), 2500) for i in range(5)]], 
                       2015:[]}
    
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
                        gauge1: [50]*76+[1250]*5+[50]*229+[50]*15+[0]*40 + [50]*3+[0]*76+[0]*50+[0]*5+[0]*231 + [50]*75+[0]*50+[50]*230+[50]*10 + [0]*77+[50]*5+[0]*229+[50]*55,
                        gauge2: [50]*76+[1250]*5+[50]*229+[0]*40+[50]*15 + [50]*3+[0]*76+[0]*50+[0]*5+[0]*231 + [50]*75+[0]*50+[50]*230+[50]*10 + [0]*76+[50]*5+[0]*230+[50]*55
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.lowflow_handle_multi(PU, gauge1, EWR, EWR_table, df_F, PU_df, allowance, climate)
    # Setting up expected output - PU_df - and testing
    data = {'BF1_a_eventYears': [0,0,0,0], 'BF1_a_numAchieved': [0,0,0,0], 'BF1_a_numEvents': [0,0,0,0], 
            'BF1_a_maxInterEventDays': [76, 0, 0, 1380], 
            'BF1_a_maxInterEventDaysAchieved': [0, 1, 1, 0],'BF1_a_eventLength': [5.0, 0.0, 0.0, 0.0], 'BF1_a_totalEventDays': [5, 0, 0, 0],
            'BF1_a_maxEventDays':[5, 0, 0, 0], 'BF1_a_maxRollingEvents': [5, 0, 0, 0], 'BF1_a_maxRollingAchievement': [0, 0, 0, 0],
            'BF1_a_daysBetweenEvents': [[76], [], [], [1380]],'BF1_a_missingDays': [0,0,0,0], 'BF1_a_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
#         print(PU_df.head())
#         print(expected_PU_df.head())
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
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to the test function
    PU_df, events = evaluate_EWRs.ctf_handle_multi(PU, gauge1, EWR, EWR_table, df_F, PU_df, allowance, climate)
    # Setting up expected output - PU_df - and testing
    data = {'CF_eventYears': [1,0,1,1], 'CF_numAchieved': [2,0,2,1], 'CF_numEvents': [2,0,2,1],
            'CF_maxInterEventDays': [350, 360, 345, 0], 
            'CF_maxInterEventDaysAchieved': [0, 0, 0, 1], 'CF_eventLength': [7.5,0.0,8.0,366.0], 'CF_totalEventDays': [15,0,16,366],
            'CF_maxEventDays':[14, 0, 15, 366], 'CF_maxRollingEvents': [14, 5, 15, 366], 'CF_maxRollingAchievement': [1, 1, 1, 1],
            'CF_daysBetweenEvents': [[350],[360],[345,9],[]], 'CF_missingDays': [0,0,0,0], 'CF_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
#         print(PU_df.head())
#         print(expected_PU_df.head())
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
    EWR = 'OB/WS1_S'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                        gauge1: gauge1_flows,
                        gauge2: gauge2_flows
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.cumulative_handle_multi(PU, gauge1, EWR, EWR_table, df_F, PU_df, allowance)
    # Setting up expected output - PU_df - and testing
    data = {'OB/WS1_S_eventYears': [1,0,0,1], 'OB/WS1_S_numAchieved': [1,0,0,1], 'OB/WS1_S_numEvents': [1,0,0,1],
            'OB/WS1_S_maxInterEventDays': [350, 0, 0, 768], 
            'OB/WS1_S_maxInterEventDaysAchieved': [1, 1, 1, 0], 'OB/WS1_S_eventLength': [1,0.0,0.0,233.0], 
            'OB/WS1_S_totalEventDays': [1,0,0,233], 'OB/WS1_S_maxEventDays':[1,0,0,233], 'OB/WS1_S_maxRollingEvents':  [1,0,0,233],
            'OB/WS1_S_maxRollingAchievement': [1,1,1,1],
            'OB/WS1_S_daysBetweenEvents': [[],[],[],[768]],'OB/WS1_S_missingDays': [0,0,0,0], 'OB/WS1_S_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
#         print(PU_df.head())
#         print(expected_PU_df.head())
    assert_frame_equal(PU_df, expected_PU_df)   
    # Setting up expected output - events - and testing
    expected_events = {2012:[[(date(2013, 6, 16), 60120)]], 
                       2013:[], 
                       2014:[], 
                       2015:[[(date(2015, 7, 25), 61000), (date(2015, 7, 26), 62000), (date(2015, 7, 27), 63000), (date(2015, 7, 28), 64000), (date(2015, 7, 29), 65000), (date(2015, 7, 30), 66000), (date(2015, 7, 31), 67000), (date(2015, 8, 1), 68000), (date(2015, 8, 2), 69000), (date(2015, 8, 3), 70000), (date(2015, 8, 4), 71000), (date(2015, 8, 5), 72000), (date(2015, 8, 6), 73000), (date(2015, 8, 7), 74000), (date(2015, 8, 8), 75000), (date(2015, 8, 9), 76000), (date(2015, 8, 10), 77000), (date(2015, 8, 11), 78000), (date(2015, 8, 12), 79000), (date(2015, 8, 13), 80000), (date(2015, 8, 14), 81000), (date(2015, 8, 15), 82000), (date(2015, 8, 16), 83000), (date(2015, 8, 17), 84000), (date(2015, 8, 18), 85000), (date(2015, 8, 19), 86000), (date(2015, 8, 20), 87000), (date(2015, 8, 21), 88000), (date(2015, 8, 22), 89000), (date(2015, 8, 23), 90000), (date(2015, 8, 24), 91000), (date(2015, 8, 25), 92000), (date(2015, 8, 26), 93000), (date(2015, 8, 27), 94000), (date(2015, 8, 28), 95000), (date(2015, 8, 29), 96000), (date(2015, 8, 30), 97000), (date(2015, 8, 31), 98000), (date(2015, 9, 1), 99000), (date(2015, 9, 2), 100000), (date(2015, 9, 3), 101000), (date(2015, 9, 4), 102000), (date(2015, 9, 5), 103000), (date(2015, 9, 6), 104000), (date(2015, 9, 7), 105000), (date(2015, 9, 8), 106000), (date(2015, 9, 9), 107000), (date(2015, 9, 10), 108000), (date(2015, 9, 11), 109000), (date(2015, 9, 12), 110000), (date(2015, 9, 13), 111000), (date(2015, 9, 14), 112000), (date(2015, 9, 15), 113000), (date(2015, 9, 16), 114000), (date(2015, 9, 17), 115000), (date(2015, 9, 18), 116000), (date(2015, 9, 19), 117000), (date(2015, 9, 20), 118000), (date(2015, 9, 21), 119000), (date(2015, 9, 22), 120000), (date(2015, 9, 23), 121000), (date(2015, 9, 24), 122000), (date(2015, 9, 25), 123000), (date(2015, 9, 26), 124000), (date(2015, 9, 27), 125000), (date(2015, 9, 28), 126000), (date(2015, 9, 29), 117000), (date(2015, 9, 30), 108000), (date(2015, 10, 1), 99000), (date(2015, 10, 2), 90000), (date(2015, 10, 3), 90000), (date(2015, 10, 4), 90000), (date(2015, 10, 5), 90000), (date(2015, 10, 6), 90000), (date(2015, 10, 7), 90000), (date(2015, 10, 8), 90000), (date(2015, 10, 9), 90000), (date(2015, 10, 10), 90000), (date(2015, 10, 11), 90000), (date(2015, 10, 12), 90000), (date(2015, 10, 13), 90000), (date(2015, 10, 14), 90000), (date(2015, 10, 15), 90000), (date(2015, 10, 16), 90000), (date(2015, 10, 17), 90000), (date(2015, 10, 18), 90000), (date(2015, 10, 19), 90000), (date(2015, 10, 20), 90000), (date(2015, 10, 21), 90000), (date(2015, 10, 22), 90000), (date(2015, 10, 23), 90000), (date(2015, 10, 24), 90000), (date(2015, 10, 25), 90000), (date(2015, 10, 26), 90000), (date(2015, 10, 27), 90000), (date(2015, 10, 28), 90000), (date(2015, 10, 29), 90000), (date(2015, 10, 30), 90000), (date(2015, 10, 31), 90000), (date(2015, 11, 1), 90000), (date(2015, 11, 2), 90000), (date(2015, 11, 3), 90000), (date(2015, 11, 4), 90000), (date(2015, 11, 5), 90000), (date(2015, 11, 6), 90000), (date(2015, 11, 7), 90000), (date(2015, 11, 8), 90000), (date(2015, 11, 9), 90000), (date(2015, 11, 10), 90000), (date(2015, 11, 11), 90000), (date(2015, 11, 12), 90000), (date(2015, 11, 13), 90000), (date(2015, 11, 14), 90000), (date(2015, 11, 15), 90000), (date(2015, 11, 16), 90000), (date(2015, 11, 17), 90000), (date(2015, 11, 18), 90000), (date(2015, 11, 19), 90000), (date(2015, 11, 20), 90000), (date(2015, 11, 21), 90000), (date(2015, 11, 22), 90000), (date(2015, 11, 23), 90000), (date(2015, 11, 24), 90000), (date(2015, 11, 25), 90000), (date(2015, 11, 26), 90000), (date(2015, 11, 27), 90000), (date(2015, 11, 28), 90000), (date(2015, 11, 29), 90000), (date(2015, 11, 30), 90000), (date(2015, 12, 1), 90000), (date(2015, 12, 2), 90000), (date(2015, 12, 3), 90000), (date(2015, 12, 4), 90000), (date(2015, 12, 5), 90000), (date(2015, 12, 6), 90000), (date(2015, 12, 7), 90000), (date(2015, 12, 8), 90000), (date(2015, 12, 9), 90000), (date(2015, 12, 10), 90000), (date(2015, 12, 11), 90000), (date(2015, 12, 12), 90000), (date(2015, 12, 13), 90000), (date(2015, 12, 14), 90000), (date(2015, 12, 15), 90000), (date(2015, 12, 16), 90000), (date(2015, 12, 17), 90000), (date(2015, 12, 18), 90000), (date(2015, 12, 19), 90000), (date(2015, 12, 20), 90000), (date(2015, 12, 21), 90000), (date(2015, 12, 22), 90000), (date(2015, 12, 23), 90000), (date(2015, 12, 24), 90000), (date(2015, 12, 25), 90000), (date(2015, 12, 26), 90000), (date(2015, 12, 27), 90000), (date(2015, 12, 28), 90000), (date(2015, 12, 29), 90000), (date(2015, 12, 30), 90000), (date(2015, 12, 31), 90000), (date(2016, 1, 1), 89900), (date(2016, 1, 2), 89800), (date(2016, 1, 3), 89700), (date(2016, 1, 4), 89600), (date(2016, 1, 5), 89500), (date(2016, 1, 6), 89400), (date(2016, 1, 7), 89300), (date(2016, 1, 8), 89200), (date(2016, 1, 9), 89100), (date(2016, 1, 10), 89000), (date(2016, 1, 11), 88000), (date(2016, 1, 12), 87000), (date(2016, 1, 13), 86900), (date(2016, 1, 14), 86800), (date(2016, 1, 15), 86700), (date(2016, 1, 16), 86600), (date(2016, 1, 17), 86500), (date(2016, 1, 18), 86400), (date(2016, 1, 19), 86300), (date(2016, 1, 20), 86200), (date(2016, 1, 21), 86100), (date(2016, 1, 22), 86000), (date(2016, 1, 23), 85500), (date(2016, 1, 24), 85000), (date(2016, 1, 25), 84500), (date(2016, 1, 26), 84000), (date(2016, 1, 27), 83500), (date(2016, 1, 28), 83000), (date(2016, 1, 29), 82500), (date(2016, 1, 30), 82000), (date(2016, 1, 31), 81500), (date(2016, 2, 1), 81000), (date(2016, 2, 2), 80500), (date(2016, 2, 3), 80000), (date(2016, 2, 4), 79500), (date(2016, 2, 5), 79000), (date(2016, 2, 6), 78500), (date(2016, 2, 7), 78000), (date(2016, 2, 8), 77500), (date(2016, 2, 9), 77000), (date(2016, 2, 10), 76500), (date(2016, 2, 11), 76000), (date(2016, 2, 12), 75500), (date(2016, 2, 13), 75000), (date(2016, 2, 14), 74500), (date(2016, 2, 15), 74000), (date(2016, 2, 16), 73500), (date(2016, 2, 17), 73000), (date(2016, 2, 18), 72500), (date(2016, 2, 19), 72000), (date(2016, 2, 20), 71500), (date(2016, 2, 21), 71000), (date(2016, 2, 22), 70500), (date(2016, 2, 23), 70000), (date(2016, 2, 24), 69500), (date(2016, 2, 25), 69000), (date(2016, 2, 26), 68500), (date(2016, 2, 27), 68000), (date(2016, 2, 28), 67500), (date(2016, 2, 29), 67000), (date(2016, 3, 1), 66500), (date(2016, 3, 2), 66000), (date(2016, 3, 3), 65500), (date(2016, 3, 4), 65000), (date(2016, 3, 5), 64500), (date(2016, 3, 6), 64000), (date(2016, 3, 7), 63500), (date(2016, 3, 8), 63000), (date(2016, 3, 9), 62500), (date(2016, 3, 10), 62000), (date(2016, 3, 11), 61500), (date(2016, 3, 12), 61000), (date(2016, 3, 13), 60500)]]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

# @pytest.mark.xfail(raises=IndexError, reason="data missing parameter sheet")
def test_flow_handle_sim():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000131'
    gauge1 = '421090'
    gauge2 = '421022'
    EWR = 'LF1_S'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge1: [0]*76+[1000]*5+[0]*229+[0]*55 + [0]*76+[0]*55+[0]*231+[1000]*3 + [1000]*3+[0]*76+[0]*50+[1000]*5+[0]*231 + [0]*77+[1000]*5+[0]*229+[0]*55,
                        gauge2: [0]*76+[1000]*5+[0]*229+[0]*55 + [0]*76+[0]*55+[0]*231+[1000]*3 + [1000]*3+[0]*76+[0]*50+[1000]*5+[0]*231 + [0]*76+[1000]*5+[0]*230+[0]*55
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.flow_handle_sim(PU, gauge1, EWR, EWR_table, df_F, PU_df, allowance)
    # Setting up expected output - PU_df - and testing
    data = {'LF1_S_eventYears': [1,0,1,0], 'LF1_S_numAchieved': [1,0,1,0], 'LF1_S_numEvents': [1,0,1,0], 
            'LF1_S_maxInterEventDays': [76, 0, 778, 597], 
            'LF1_S_maxInterEventDaysAchieved': [1, 1, 0, 1],'LF1_S_eventLength': [5.0,0.0,5.0,0.0], 
            'LF1_S_totalEventDays': [5,0,5,0],
            'LF1_S_maxEventDays':[5, 0, 5, 0],'LF1_S_maxRollingEvents':  [5, 0, 5, 0],  'LF1_S_maxRollingAchievement': [1, 0, 1, 0],
            'LF1_S_daysBetweenEvents': [[],[],[778],[]],'LF1_S_missingDays': [0,0,0,0], 'LF1_S_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
    assert_frame_equal(PU_df, expected_PU_df)
    # Setting up expected output - events - and testing
    expected_events = {2012:[[(date(2012, 9, 15) + timedelta(days=i), 1000) for i in range(5)]], 
                        2013:[], 
                        2014:[ 
                              [(date(2014, 11, 7) + timedelta(days=i), 1000) for i in range(5)]], 
                        2015:[]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

# @pytest.mark.xfail(raises=IndexError, reason="data missing parameter sheet")
def test_lowflow_handle_sim():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000131'
    gauge1 = '421090'
    gauge2 = '421022'
    EWR = 'BF1_a'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge1: [0]*76+[65]*280+[0]*9 + [0]*76+[0]*9+[65]*280 + [0]*80+[0]*9+[65]*276 + [65]*270+[0]*76+[0]*14+[65]*6,
                        gauge2: [0]*76+[65]*280+[0]*9 + [65]*280+[0]*76+[0]*9 + [0]*80+[0]*9+[65]*276 + [65]*270+[0]*76+[0]*14+[65]*6
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.lowflow_handle_sim(PU, gauge1, EWR, EWR_table, df_F, PU_df, allowance, climate)
    # Setting up expected output - PU_df - and test
    # Note the floats that get returned in the total event days series. This is because the totals of the two series are averaged.
    data = {'BF1_a_eventYears': [0,0,0,0], 'BF1_a_numAchieved': [0,0,0,0], 'BF1_a_numEvents': [0,0,0,0], 
             'BF1_a_eventLength': [0.0, 0.0, 0.0, 0.0],
             'BF1_a_totalEventDays': [0.0, 0.0, 0.0, 0.0], 
            'BF1_a_daysBetweenEvents': [[],[],[],[1461]], 'BF1_a_missingDays': [0,0,0,0], 'BF1_a_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
    assert_frame_equal(PU_df, expected_PU_df)
    # Setting up expected output - events - and testing
    expected_events1 = {2012:[], 2013:[], 2014:[], 2015:[]}
    expected_events2 = {2012:[], 2013:[], 2014:[], 2015:[]}
    expected_events = tuple([expected_events1, expected_events2])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) ==len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

# @pytest.mark.xfail(raises=IndexError, reason="data missing parameter sheet")
def test_ctf_handle_sim():
    '''
    1. Ensure all parts of the function generate expected output
    '''
    # Set up input data
    PU = 'PU_0000131'
    gauge1 = '421090'
    gauge2 = '421022'
    EWR = 'CF_a'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period(),
                        gauge1: [5]*123+[0]*5+[5]*232+[0]*5 + [0]*1+[5]*123+[0]*3+[5]*233+[0]*3+[5]*2 + [5]*123+[0]*5+[5]*232+[0]*5 + [5]*123+[0]*5+[5]*233+[5]*5,
                        gauge2: [5]*123+[0]*5+[5]*232+[0]*5 + [0]*1+[5]*123+[0]*3+[5]*233+[0]*3+[5]*2 + [5]*123+[0]*5+[5]*232+[0]*5 + [5]*123+[5]*5+[5]*233+[0]*5
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.ctf_handle_sim(PU, gauge1, EWR, EWR_table, df_F, PU_df, allowance, climate)
    # Setting up expected output - PU_df - and test
    # Note the floats that get returned in the total event days series. This is because the totals of the two series are averaged.
    data = {'CF_a_eventYears': [1,0,1,1], 'CF_a_numAchieved': [2,0,2,1], 'CF_a_numEvents': [2,0,2,1], 'CF_a_eventLength': [5.0, 2.3333333333333335, 5.0, 5.0], 
            'CF_a_totalEventDays': [10.0,7.0,10.0,5.0],
            'CF_a_daysBetweenEvents': [[123, 232],[123,233],[125,232],[123,238]],
            'CF_a_missingDays': [0,0,0,0], 'CF_a_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
    assert_frame_equal(PU_df, expected_PU_df) 
    # Setting up expected output - events - and test
    expected_events1 = {2012:[[(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                                [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
                                2013:[[(date(2013, 7, 1), 0)],
                                    [(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                                [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)]], 
                                2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                                [(date(2015, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
                                2015:[[(date(2015, 11, 1) + timedelta(days=i), 0) for i in range(5)]]}
    expected_events2 = {2012:[[(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                                [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
                                2013:[[(date(2013, 7, 1), 0)],
                                    [(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                                [(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)]], 
                                2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                                [(date(2015, 6, 26) + timedelta(days=i), 0) for i in range(5)]], 
                                2015:[[(date(2016, 6, 26) + timedelta(days=i), 0) for i in range(5)]]}
    expected_events = tuple([expected_events1, expected_events2])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            print(year)
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]

def test_complex_handle():
    '''
    1. Ensure all parts of the function generate expected output for OB2
    2. Ensure all parts of the function generate expected output for OB3
    '''
    # Test 1
    # Set up input data
    PU = 'PU_0000253'
    gauge = '409025'
    EWR = 'OB2a_S'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                        gauge: [19000]*45+[9000]*105+[0]*215 + [15000]*45+[0]*8+[9000]*105+[0]*207 + [15000]*15+[0]*6+[15000]*15+[0]*6+[15000]*15+[0]*6+[9000]*55+[0]*6+[9000]*50+[0]*150+[16000]*41 + \
                                        [0]*6+[16000]*4+[0]*6+[9000]*105+[0]*95+[18000]*45+[9000]*105,
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.complex_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
    # Setting up expected output - PU_df - and testing
    data = {'OB2a_S_eventYears': [1,0,1,1], 'OB2a_S_numAchieved': [1,0,1,2], 'OB2a_S_numEvents': [1,0,1,2], 
            'OB2a_S_maxInterEventDays': [1, 0, 580, 94], 
            'OB2a_S_maxInterEventDaysAchieved': [1, 1, 1, 1],'OB2a_S_eventLength': [150.0,0.0,150.0,150.0], 'OB2a_S_totalEventDays': [150,0,150,300],
            'OB2a_S_maxEventDays':[150, 0, 150, 150], 'OB2a_S_maxRollingEvents':[0, 0, 0, 0], 'OB2a_S_maxRollingAchievement': [0, 0, 0, 0],
            'OB2a_S_daysBetweenEvents': [[],[],[],[]],'OB2a_S_missingDays': [0,0,0,0], 'OB2a_S_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
    assert_frame_equal(PU_df, expected_PU_df) 
    # Setting up expected output - events - and testing
    expected_events = {2012:[[19000]*45+[9000]*105], 2013:[], 2014:[[15000]*15+[15000]*15+[15000]*15+[9000]*55+[9000]*50], 2015:[[16000]*41+[16000]*4+[9000]*105, [18000]*45+[9000]*105]}
    expected_events = tuple([expected_events])
    for index, tuple_ in enumerate(events):
        for year in events[index]:
            assert len(events[index][year]) == len(expected_events[index][year])
            for i, event in enumerate(events[index][year]):
                assert event == expected_events[index][year][i]
    #-------------------------------------------------------------------------------------
    # Test 2
    # Set up input data
    PU = 'PU_0000253'
    gauge = '409025'
    EWR = 'OB3a_S'
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    data_for_df_F = {'Date': pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')),
                        gauge:  [25000]*21+[15000]*90+[0]*254 + [25000]*21+[0]*8+[15000]*90+[0]*246 + [15000]*90+[0]*6+[25000]*21+[0]*8+[25000]*15+[0]*6+[25000]*6+[0]*6+[15000]*80+[0]*6+[15000]*10+[0]*94+[15000]*17 + \
                                        [0]*6+[15000]*73+[0]*6+[25000]*1+[25000]*20+[0]*149+[25000]*21+[15000]*90
                    }
    df_F = pd.DataFrame(data = data_for_df_F)
    df_F = df_F.set_index('Date')
    df_L = pd.DataFrame()
    PU_df = pd.DataFrame()
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    # Pass input data to test function
    PU_df, events = evaluate_EWRs.complex_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
    # Setting up expected output - PU_df - and testing
    data = {'OB3a_S_eventYears': [1,0,1,1], 'OB3a_S_numAchieved': [1,0,2,2], 'OB3a_S_numEvents': [1,0,2,2], 
            'OB3a_S_maxInterEventDays': [1, 0, 751, 379], 
            'OB3a_S_maxInterEventDaysAchieved': [1, 1, 1, 1],'OB3a_S_eventLength': [111.0,0.0,111.0,111.0], 'OB3a_S_totalEventDays': [111,0,222,222],
            'OB3a_S_maxEventDays':[111, 0, 111, 111], 'OB3a_S_maxRollingEvents':[0, 0, 0, 0], 'OB3a_S_maxRollingAchievement': [0, 0, 0, 0],
            'OB3a_S_daysBetweenEvents': [[],[],[],[]],'OB3a_S_missingDays': [0,0,0,0], 'OB3a_S_totalPossibleDays': [365,365,365,366]}
    index = [2012, 2013, 2014,2015]
    expected_PU_df = pd.DataFrame(index = index, data = data)
    expected_PU_df.index = expected_PU_df.index.astype('object')
    assert_frame_equal(PU_df, expected_PU_df) 
    # Setting up expected output - events - and testing
    expected_events = {2012:[[25000]*21+[15000]*90], 2013:[], 2014:[[15000]*90+[25000]*21, [25000]*15+[25000]*6+[15000]*80+[15000]*10], 2015:[[15000]*17+[15000]*73+[25000]*21, [25000]*21+[15000]*90]}
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
    print(result)
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
    print(result)
    assert result == pytest.approx(expected_result)


@pytest.mark.parametrize("flow_percent_change,flow,expected_result",[
    (-10,5, True),
    (-20, 10, False),
    (-40, 11, True),
    (-40, 9, False),
    (10, 10,True),

],)
def test_check_nest_percent_drawdown(flow_percent_change, flow, expected_result):
    EWR_info = {'max_flow':10, 'drawdown_rate':"15%"}

    result = evaluate_EWRs.check_nest_percent_drawdown(flow_percent_change, EWR_info, flow)
    assert result == expected_result


@pytest.mark.parametrize("EWR_info,iteration,expected_result",[
    ({'end_month': 10}, 0, date(2012, 10, 31)),
    ({'end_month': 9}, 0, date(2012, 9, 30)),
    ({'end_month': 12}, 0, date(2012, 12, 31)),
    ({'end_month': 4}, 366, date(2013, 4, 30)),
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