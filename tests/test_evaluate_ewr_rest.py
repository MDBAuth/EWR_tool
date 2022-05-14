from datetime import datetime

import pandas as pd
import numpy as np

from py_ewr import evaluate_EWRs, data_inputs, summarise_results

def test_component_pull():
        '''
        1. Test correct value is pulled from EWR dataset
        '''
        EWR_table, bad_EWRs = data_inputs.get_EWR_table()
        gauge = '409025'
        PU = 'PU_0000253'
        EWR = 'SF1_P'
        component = 'duration'
        assert  evaluate_EWRs.component_pull(EWR_table, gauge, PU, EWR, component) == '10'

def test_apply_correction():
        '''
        1. Test function that applies relaxation to parts of indicator
        '''
        info = 960
        correction = 1.2
        assert evaluate_EWRs.apply_correction(info, correction) == 1152.0

def test_get_EWRs():
        '''
        1. Ensure requested parts of EWR are returned
        '''
        EWR_table, bad_EWRs = data_inputs.get_EWR_table()
        PU = 'PU_0000283'
        gauge = '410007'
        EWR = 'SF1_P'
        minThreshold_tolerance = (100 - 0)/100
        maxThreshold_tolerance = (100 + 0)/100
        duration_tolerance = (100 - 0)/100
        drawdown_tolerance = (100 - 0)/100
        allowance ={'minThreshold': minThreshold_tolerance, 'maxThreshold': maxThreshold_tolerance,
                        'duration': duration_tolerance, 'drawdown': drawdown_tolerance}
        components = ['SM', 'EM']

        expected = {'gauge': '410007', 'planning_unit': 'PU_0000283', 'EWR_code': 'SF1_P', 'start_day': None, 'start_month': 10, 'end_day': None, 'end_month':4}
        assert evaluate_EWRs.get_EWRs(PU, gauge, EWR, EWR_table, allowance, components) == expected

def test_mask_dates():
        '''
        This testing function will also be testing the functions get_month_mask, get_day_month_mask, and get_day_month_mask
        1. Testing for no filtering (all year round EWR requirement)
        2. Testing for a month subset
        3. Testing for water year crossover EWR requirement
        4. Testing for start day and end day within same month inclusion in the EWR requirement
        5. Testing for start day and end day within different months inclusion in the EWR requirement:
        '''
        #------------ Dataframe to be passed to all testing functions here ----------#

        data = {'409102': list(range(0,3650,10)), '425012': list(range(0,3650,10))}
        index = pd.date_range(start='1/1/2019', end='31/12/2019')
        df = pd.DataFrame(data = data, index = index)

        #----------------------------------------------------------------------------#
        # Test 1
        EWR_info = {'start_day': None, 'end_day': None, 'start_month': 7, 'end_month': 6}
        masked_7to6 = set(pd.date_range(start='1/1/2019', end='31/12/2019'))
        assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_7to6
        #-----------------------------------------------------------------------
        # Test 2
        EWR_info = {'start_day': None, 'end_day': None, 'start_month': 7, 'end_month': 9}
        masked_7to9 = set(pd.date_range(start='2019-07-01', end='2019-09-30'))
        assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_7to9
        #-----------------------------------------------------------------------
        # Test 3
        EWR_info = {'start_day': None, 'end_day': None, 'start_month': 6, 'end_month': 8}
        masked_6to8 = set(pd.date_range(start='2019-06-01', end='2019-08-31'))
        assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_6to8
        #-----------------------------------------------------------------------
        # Test 4
        EWR_info = {'start_day': 12, 'end_day': 28, 'start_month': 6, 'end_month': 6}
        masked_612to628 = set(pd.date_range(start='2019-06-12', end='2019-06-28'))
        assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_612to628
        #-----------------------------------------------------------------------
        # Test 5
        EWR_info = {'start_day': 12, 'end_day': 18, 'start_month': 8, 'end_month': 9}
        masked_812to918 = set(pd.date_range(start='2019-08-12', end='2019-09-18'))
        assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_812to918


def test_wateryear_daily():
        '''
        1. Testing non standard water year part of function
        2. Testing standard water year part of function
        '''
        # Test 1
        EWR_info = {'start_day': None, 'end_day': None, 'start_month': 5, 'end_month': 8}
        data = {'409102': list(range(0,3650,10)), '425012': list(range(0,3650,10))}
        index = pd.date_range(start='1/1/2019', end='31/12/2019')
        df = pd.DataFrame(data = data, index = index)
        expected_2018 = [2018]*120
        expected_2019 = [2019]*245
        expected_array = np.array(expected_2018 + expected_2019)
        array = evaluate_EWRs.wateryear_daily(df, EWR_info)
        assert np.array_equal(array, expected_array)
        #-------------------------------------------------------
        # Test 2
        EWR_info = {'start_day': None, 'end_day': None, 'start_month': 7, 'end_month': 9}
        data = {'409102': list(range(0,3650,10)), '425012': list(range(0,3650,10))}
        index = pd.date_range(start='1/1/2019', end='31/12/2019')
        df = pd.DataFrame(data = data, index = index)
        expected_2018 = [2018]*181
        expected_2019 = [2019]*184
        expected_array = np.array(expected_2018 + expected_2019)
        array = evaluate_EWRs.wateryear_daily(df, EWR_info)
        assert np.array_equal(array, expected_array)

def test_which_wateryear():
        '''
        1. Testing for when there is an equal portion of the event falling in two water years
        2. Testing for when there is a non-equal portion of the event falling within each water year
        '''
        # Test 1
        i = 45
        event = [10]*10
        water_years = np.array([2018]*40+[2019]*10)
        water_year = evaluate_EWRs.which_water_year(i, len(event), water_years)
        assert water_year == 2019
        #----------------------------------
        # Test 2
        i = 42
        event = [10]*9
        water_years = np.array([2018]*40+[2019]*10)
        water_year = evaluate_EWRs.which_water_year(i, len(event), water_years)
        assert water_year == 2018

def test_get_duration():
        '''
        1. Test for very dry duration for EWR where a very dry duration requirement exists
        2. Test for very dry duration for EWR where a very dry duration requirement does not exists
        '''
        # Test 1
        EWR_info = {'duration_VD': 2, 'duration': 5}
        duration = evaluate_EWRs.get_duration('Very Dry', EWR_info)
        assert duration == 2
        #-----------------------------
        # Test 2
        EWR_info = {'duration_VD': None, 'duration': 5}
        duration = evaluate_EWRs.get_duration('Very Dry', EWR_info)
        assert duration == 5
        
def test_construct_event_dict():
        '''
        1. Test event dictionary with correct keys and values are returned
        '''
        water_years = [2018]*365+[2019]*365
        all_events = evaluate_EWRs.construct_event_dict(water_years)
        expected_result = {2018:[], 2019:[]}
        assert all_events == expected_result

def test_get_days_between():
        '''
        1. Testing low flow with more than 1 year interevent requirement
        2. Testing low flow with less than 1 year of interevent requirement
        3. Testing non low flow EWR with more than 1 year requirement
        4. Testing for EWR with only a subset of the water year available
        '''
        # Test 1
        no_events = {2012: [[735], [50], [2]], 2013: [[35], [50], [365]],
                        2014: [[35], [50], [2]], 2015: [[35], [280], [2]]}
        years_with_events = [0,0,0,1] # This will be used in the calculation
        EWR = 'BF'
        EWR_info = {'max_inter-event': 2}
        unique_water_years = [2012, 2013, 2014, 2015]
        water_years = [2012]*365+[2013]*365+[2014]*365+[2015]*365
        days_between = evaluate_EWRs.get_days_between(years_with_events, no_events, EWR, EWR_info, unique_water_years, water_years)
        expected_days_between = [[], [], [], [1095]]
        for i, v in enumerate(days_between):
                assert days_between[i] == expected_days_between[i]
        #--------------------------------------------------------------------
        # Test 2
        no_events = {2012: [[35], [50], [2]], 2013: [[35], [50], [2]],
                        2014: [[35], [50], [2]], 2015: [[35], [50], [2]]}
        years_with_events = [0,0,0,1] #This will be ignored in this calculation
        EWR = 'BF'
        EWR_info = {'max_inter-event': 0.04} # Equates to about 14-15 days
        unique_water_years = [2012, 2013, 2014, 2015]
        water_years = [2012]*365+[2013]*365+[2014]*365+[2015]*365
        days_between = evaluate_EWRs.get_days_between(years_with_events, no_events, EWR, EWR_info, unique_water_years, water_years)
        expected_days_between = [[35, 50], [35, 50], [35, 50], [35, 50]]
        for i, v in enumerate(days_between):
                assert days_between[i] == expected_days_between[i]
        #--------------------------------------------------------------------
        # Test 3
        no_events = {2012: [[35], [50], [2]], 2013: [[35], [50], [2]],
                        2014: [[35], [50], [2]], 2015: [[735], [2]]}
        years_with_events = [0,1,0,0] #This will be ignored in this calculation
        EWR = 'LF'
        EWR_info = {'max_inter-event': 2}
        unique_water_years = [2012, 2013, 2014, 2015]
        water_years = [2012]*365+[2013]*365+[2014]*365+[2015]*365
        days_between = evaluate_EWRs.get_days_between(years_with_events, no_events, EWR, EWR_info, unique_water_years, water_years)
        expected_days_between = [[], [], [], [735]]
        for i, v in enumerate(days_between):
                assert days_between[i] == expected_days_between[i]
        #--------------------------------------------------------------------
        # Test 4
        no_events = {2012: [[35], [122], [2]], 2013: [[35], [50], [2]],
                        2014: [[35], [50], [2]], 2015: [[730], [50], [121]]}
        years_with_events = [0,0,0,1] #This will be ignored in this calculation
        EWR = 'LF'
        EWR_info = {'max_inter-event': 2}
        unique_water_years = [2012, 2013, 2014, 2015]
        water_years = [2012]*365+[2013]*365+[2014]*365+[2015]*365
        days_between = evaluate_EWRs.get_days_between(years_with_events, no_events, EWR, EWR_info, unique_water_years, water_years)
        expected_days_between = [[], [], [], [730]]
        for i, v in enumerate(days_between):          
                assert days_between[i] == expected_days_between[i]


def test_get_event_years():
        '''
        Year 1: check 1 is returned when there are 3 events with 2 required per year
        Year 2: check 0 is returned when there is 1 event with 2 required per year
        Year 3: check 1 is returned when there are 4 events with 2 required per year
        Year 4: check 0 is returned when there are 0 events with 2 required per year
        '''
        EWR_info = {'events_per_year': 2}
        events = {2012: [[5]*5, [10]*5, [20*8]], 2013: [[50]*20],
                        2014: [[5]*5, [10]*5, [20*8], [20*8]], 2015: []}
        unique_water_years = [2012, 2013, 2014, 2015]
        durations = [5,5,5,5]
        min_events = [5,5,5,5]
        event_years = evaluate_EWRs.get_event_years(EWR_info, events, unique_water_years, durations, min_events)
        expected_event_years = [1,0,1,0]
        assert event_years == expected_event_years

def test_get_achievements():
        '''
        1. Testing 1 event per year requirement with four unique events per year ranges
        2. Testing 2 events per year requirement with four unique events per year ranges
        '''
        EWR_info = {'events_per_year': 1}
        events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
                        2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
        unique_water_years = [2012, 2013, 2014, 2015]
        durations = [5,5,5,5]
        min_events = [5,5,5,5]
        num_events = evaluate_EWRs.get_achievements(EWR_info, events, unique_water_years, durations, min_events)
        expected_num_events = [3,1,4,0]
        assert num_events == expected_num_events
        #-------------------------------------------------
        # Test 2
        EWR_info = {'events_per_year': 2}
        events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
                        2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
        unique_water_years = [2012, 2013, 2014, 2015]
        durations = [5,5,5,5]
        min_events = [5,5,5,5]
        num_events = evaluate_EWRs.get_achievements(EWR_info, events, unique_water_years, durations, min_events)
        expected_num_events = [1,0,2,0]
        assert num_events == expected_num_events

def test_get_number_events():
        '''
        1. Testing 1 event per year requirement with four unique events per year ranges
        2. Testing 2 events per year requirement with four unique events per year ranges
        '''
        # Test 1
        EWR_info = {'events_per_year': 1}
        events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
                        2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
        unique_water_years = [2012, 2013, 2014, 2015]
        durations = [5,5,5,5]
        min_events = [5,5,5,5]
        num_events = evaluate_EWRs.get_number_events(EWR_info, events, unique_water_years, durations, min_events)
        expected_num_events = [3,1,4,0]
        assert num_events == expected_num_events
        #--------------------------------------------------
        # Test 2
        EWR_info = {'events_per_year': 2}
        events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
                        2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
        unique_water_years = [2012, 2013, 2014, 2015]
        durations = [5,5,5,5]
        min_events = [5,5,5,5]
        num_events = evaluate_EWRs.get_number_events(EWR_info, events, unique_water_years, durations, min_events)
        expected_num_events = [3,1,4,0]
        assert num_events == expected_num_events

def test_get_average_event_length():
        '''
        1. Test yearly average event length for test years with between 0 and 4 total events
        '''
        events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
                        2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
        unique_water_years = [2012, 2013, 2014, 2015]
        average_length = evaluate_EWRs.get_average_event_length(events, unique_water_years)
        expected_average_length = [6,20,6.5,0]
        assert average_length == expected_average_length

def test_get_total_days():
        '''
        1. Test total yearly event length for test years with between 0 and 4 total events
        '''
        events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
                        2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
        unique_water_years = [2012, 2013, 2014, 2015]
        total_days = evaluate_EWRs.get_total_days(events, unique_water_years)
        expected_total_days = [18,20,26,0]
        assert total_days == expected_total_days

def test_get_data_gap():
        '''
        1. Check event gaps are accurate
        '''
        data = {'409102': list(range(0,3650,10)), '425012': list(range(0,3650,10))}
        index = pd.date_range(start='1/1/2019', end='31/12/2019')
        df = pd.DataFrame(data = data, index = index)
        df.iloc[0:4] = None
        df.iloc[57] = np.nan
        unique_water_years = [2018]*181+[2019]*184
        gauge = '409102'
        missing_list = evaluate_EWRs.get_data_gap(df, unique_water_years, gauge)
        expected_missing = [5,0]
        assert missing_list == expected_missing

def test_check_flow():
        '''
        1. Test flow threshold passes and event requirements just met
        2. TO-TEST: flow threshold below but event requirement passed
        3. TO-TEST: event requirements failed but there is some gap track remaining
        4. TO-TEST: flow threshold failed and event requirements not met
        '''
        # Set up inputs parameters and pass to test function
        EWR_info = {'min_flow': 5, 'max_flow': 20, 'gap_tolerance': 0, 'min_event':10}
        iteration = 50
        flow = 5
        event = [5]*9
        all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
                        2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
        no_event = 50
        all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                2014:[[400], [2], [25]], 2015:[[450]]}
        gap_track = 0
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)
        total_event = 9
        event, all_events, no_event, all_no_events, gap_track, total_event = evaluate_EWRs.flow_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, gap_track, water_years, total_event)
        # Set up expected results and test
        expected_event = [5]*10
        expected_all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
                                2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
        expected_no_event = 51
        expected_all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                        2014:[[400], [2], [25]], 2015:[[450]]}
        expected_gap_track = 0
        expected_total_event = 10
        assert event == expected_event
        for year in all_events:
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        assert no_event == expected_no_event
        assert all_no_events == expected_all_no_events
        for year in all_no_events:
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event ==expected_all_no_events[year][i]
        assert gap_track == expected_gap_track
        assert total_event == expected_total_event

def test_lowflow_check():
        '''
        1. Test flow passes and event requirement just met
        2. TO-TEST: flow threshold below but event requirement passed
        3. TO-TEST: flow threshold failed and event requirements failed
        '''
        # Set up variables for all tests
        EWR_info = {'min_flow': 5, 'max_flow': 20}
        flow = 5
        water_year = 2015
        event = [5]*9
        iteration = 365+365+365+100
        all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
                        2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
        no_event = 0
        all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                2014:[[400], [2], [25]], 2015:[[450]]}
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)
        event, all_events, no_event, all_no_events = evaluate_EWRs.lowflow_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, water_years)
        # Set up expected output and test
        expected_event = [5]*10
        expected_all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
                                2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
        expected_no_event = 0
        expected_all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                        2014:[[400], [2], [25]], 2015:[[450]]}
        assert event == expected_event

        for year in all_events:
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]

        assert no_event == expected_no_event
        assert all_no_events == expected_all_no_events
        
        for year in all_no_events:
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]

def test_ctf_check():
        '''
        1. flow threshold fails but event meets requirements
        2. TO-TEST: flow threshold passed
        3. TO-TEST: flow threshold failed but no event recorded
        '''
        # Set up input variables and pass to test function
        EWR_info = {'min_flow': 0, 'max_flow': 1}
        flow = 2
        event = [0]*9
        iteration = 365+365+365+100
        all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
                        2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
        no_event = 10
        all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                2014:[[400], [2], [25]], 2015:[[450]]}
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)

        event, all_events, no_event, all_no_events = evaluate_EWRs.ctf_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, water_years)
        # Set up expected outputs and test
        expected_event = []
        expected_all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50],
                                2014:[[10]*10, [15]*15, [10]*20], 2015:[[0]*9]}
        expected_no_event = 1
        expected_all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                        2014:[[400], [2], [25]], 2015:[[450], [10]]}
        assert event ==expected_event
        for year in all_events:
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
                assert no_event == expected_no_event
        for year in all_no_events:
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]

def test_level_check():
        '''
        1. Test level threshold fails but event requirement passed
        2. TO-TEST: test level threshold passes and event requirement met
        3. TO-TEST: level threshold fails and event requirement failed
        '''
        # Set up input variables
        EWR_info = {'min_level': 10, 'max_level':20, 'duration': 5, 'drawdown_rate':0.04}
        level = 5
        level_change = 0.04
        water_year = 2015
        event = [10]*5
        iteration = 365+365+365+100
        all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
                        2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
        no_event = 15
        all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                2014:[[400], [2], [25]], 2015:[[450]]}
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)

        event, all_events, no_event, all_no_events = evaluate_EWRs.level_check(EWR_info, iteration, level, level_change, event, all_events, no_event, all_no_events, water_years)\
        # Expected results - TEST 1: #
        expected_event = []
        expected_all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
                                2014:[[10]*10, [15]*15, [10]*20], 2015:[[10]*5]}
        expected_no_event = 1
        expected_all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                        2014:[[400], [2], [25]], 2015:[[450], [10]]}
        assert event == expected_event

        for year in all_events:
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        assert no_event == expected_no_event
        assert all_no_events == expected_all_no_events
        for year in all_no_events:
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]
        
def test_flow_check_sim():
        '''
        1. flow threshold below for both sites but event requirement passed
        2. TO-TEST: Test flow threshold passes for both sites and event requirements just met
        3. TO-TEST: event requirements failed but there is some gap track remaining
        4. TO-TEST: flow threshold failed and event requirements not met
        5. TO-TEST: flow threshold above for one site and below for another, event requirements not met
        '''
        # Set up input parameters and send to test function
        iteration = 370
        EWR_info1 = {'min_flow': 10, 'max_flow': 20, 'min_event': 5, 'gap_tolerance': 0}
        EWR_info2 = {'min_flow': 10, 'max_flow': 20, 'min_event': 5, 'gap_tolerance': 0}
        flow1 = 5
        flow2 = 7
        event = [10]*5
        total_event = 5
        all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
                        2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
        no_event = 25
        all_no_events = {2012:[[25], [2]], 2013:[[250]],
                                2014:[[400], [2], [25]], 2015:[[450]]}
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)
        gap_track = 0
        event, all_events, no_event, all_no_events, gap_track, total_event = evaluate_EWRs.flow_check_sim(iteration, EWR_info1, EWR_info2, water_years, flow1, flow2, event, all_events, no_event, all_no_events, gap_track, total_event)
        # Set up expected results and test
        expected_event = []
        expected_all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50, [10]*5],
                                2014:[[10]*10, [15]*15, [10]*20], 2015:[[10]*5]}
        expected_no_event = 1
        expected_all_no_events = {2012:[[25], [2]], 2013:[[250], [20]],
                                        2014:[[400], [2], [25]], 2015:[[450]]}
        expected_gap_track = 0
        expected_total_event = 0
        assert gap_track == expected_gap_track
        assert event == expected_event
        #         print(all_events)
        #         print(expected_all_events)
        for year in all_events:
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        assert no_event == expected_no_event
        assert all_no_events == expected_all_no_events

        for year in all_no_events:
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]
                
def test_flow_calc():
        '''
        1. Test functions ability to identify and save all events and event gaps for series of flows (TO-TEST: flows overlapping water year edge)
        2. TO-TEST: constrain timing window
        '''
        # Test 1
        # Set up input data
        EWR_info = {'min_flow': 5, 'max_flow': 20, 'gap_tolerance': 0, 'min_event':10, 'duration': 10}
        flows = np.array([0]*355+[10]*10 + [0]*355+[10]*10 + [0]*355+[10]*10 + [0]*355+[10]*10+[10]*1)
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
        dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        # Set up expected output data        
        expected_all_events = {2012: [[10]*10], 2013: [[10]*10], 2014: [[10]*10], 2015: [[10]*11]}
        expected_all_no_events = {2012: [[355]], 2013: [[355]], 2014: [[355]], 2015: [[355]]}
        expected_durations = [10]*4
        expected_min_events = [10]*4
        # Send inputs to test function and test
        all_events, all_no_events, durations, min_events = evaluate_EWRs.flow_calc(EWR_info, flows, water_years, dates, masked_dates)
        for year in all_events:
                assert len(all_events[year]) == len(expected_all_events[year])
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        for year in all_no_events:
                assert len(all_no_events[year]) == len(expected_all_no_events[year])
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]
        assert durations == expected_durations
        assert min_events == expected_min_events

def test_lowflow_calc():
        '''
        1. Test functions ability to identify and save all events and event gaps for series of flows
        2. Constrain timing window and test functions ability to identify and save all events and event gaps for series of flows
        '''
        # Test 1
        # set up input data 
        EWR_info = {'min_flow': 5, 'max_flow': 20, 'min_event':1, 'duration': 300, 'duration_VD': 10}
        flows = np.array([5]*295+[0]*25+[10]*45 + [0]*355+[5000]*10 + [0]*355+[10]*10 + [5]*295+[0]*25+[10]*45+[10]*1)
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
        climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
        dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        # Set up expected output data
        expected_all_events = {2012: [[5]*295, [10]*45], 2013: [], 2014: [[10]*10], 2015: [[5]*295, [10]*46]}
        expected_all_no_events = {2012: [[25]], 2013: [], 2014: [[720]], 2015: [[25]]}
        expected_durations = [300,300,10,300] # adding in a very dry year climate year
        expected_min_events = [1,1,1,1]
        # Send inputs to test function and test
        all_events, all_no_events, durations, min_events = evaluate_EWRs.lowflow_calc(EWR_info, flows, water_years, climates, dates, masked_dates)
        for year in all_events:
                assert len(all_events[year]) == len(expected_all_events[year])
                
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        for year in all_no_events:
                assert len(all_no_events[year]) ==len(expected_all_no_events[year])
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]
        assert durations == expected_durations
        assert min_events == expected_min_events
        #------------------------------------------------
        # Test 2
        # Set up input data
        EWR_info = {'min_flow': 5, 'max_flow': 20, 'min_event':1, 'duration': 10, 
                        'duration_VD': 5, 'start_month': 7, 'end_month': 12, 'start_day': None, 'end_day': None}
        flows = np.array([10]*5+[0]*35+[5]*5+[0]*295+[0]*25 + [0]*355+[5]*10 + [10]*10+[0]*355 + [5]*295+[0]*25+[10]*45+[10]*1)
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
        climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
        dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        masked_dates = masked_dates[((masked_dates.month >= 7) & (masked_dates.month <= 12))] # Just want the dates in the date range
        # Set up expected output data
        expected_all_events = {2012: [[10]*5, [5]*5], 2013: [], 2014: [[10]*10], 2015: [[5]*184]}
        expected_all_no_events = {2012: [[35]], 2013: [[685]], 2014: [[355]], 2015: [[181]]}
        expected_durations = [10,10,5,10] # adding in a very dry year climate year
        expected_min_events = [1,1,1,1]
        # Send to test function and test
        all_events, all_no_events, durations, min_events = evaluate_EWRs.lowflow_calc(EWR_info, flows, water_years, climates, dates, masked_dates)
        for year in all_events:
                assert len(all_events[year]) == len(expected_all_events[year])
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        for year in all_no_events:
                assert len(all_no_events[year]) == len(expected_all_no_events[year])
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]
        assert durations == expected_durations
        assert min_events == expected_min_events

def test_ctf_calc():
        '''
        1. Test functions ability to identify and save all events and event gaps for series of flows, 
                ensuring events are cut off at the end of the water year even though dates are not constrained
        2. Constrain timing window and test functions ability to identify and save all events and event gaps for series of flows
        '''
        # Test 1
        # Set up input data
        EWR_info = {'min_flow': 0, 'max_flow': 1, 'min_event':5, 'duration': 20, 'duration_VD': 10}
        flows = np.array([5]*295+[0]*25+[10]*45 + [20]*355+[5000]*5+[0]*5 + [0]*355+[10]*10 + [1]*295+[20]*25+[0]*45+[0]*1)
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
        climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
        dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        # Set up expected output data
        expected_all_events = {2012: [[0]*25], 2013: [[0]*5], 2014: [[0]*355], 2015: [[1]*295, [0]*46]}
        expected_all_no_events = {2012: [[295]], 2013: [[405]], 2014: [], 2015: [[10], [25]]}
        expected_durations = [20,20,10,20] # adding in a very dry year climate year
        expected_min_events = [5,5,5,5]
        # Send to test function and then test
        all_events, all_no_events, durations, min_events = evaluate_EWRs.ctf_calc(EWR_info, flows, water_years, climates, dates, masked_dates)
        for year in all_events:
                assert len(all_events[year]) == len(expected_all_events[year])
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        for year in all_no_events:
                assert len(all_no_events[year]) == len(expected_all_no_events[year])
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]
        assert durations == expected_durations
        assert min_events == expected_min_events
        #--------------------------------------------------
        # Test 2
        # Set up input data
        EWR_info = {'min_flow': 5, 'max_flow': 20, 'min_event':1, 'duration': 10,
                        'duration_VD': 5, 'start_month': 7, 'end_month': 12, 'start_day': None, 'end_day': None}
        flows = np.array([10]*5+[0]*35+[5]*5+[0]*295+[0]*25 + [0]*355+[5]*10 + [10]*10+[0]*355 + [5]*295+[0]*25+[10]*45+[10]*1)
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
        climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
        dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        masked_dates = masked_dates[((masked_dates.month >= 7) & (masked_dates.month <= 12))] # Just want the dates in the date range
        # Set up expected output data
        expected_all_events = {2012: [[10]*5, [5]*5], 2013: [], 2014: [[10]*10], 2015: [[5]*184]}
        expected_all_no_events = {2012: [[35]], 2013: [[685]], 2014: [[355]], 2015: [[181]]}
        expected_durations = [10,10,5,10] # adding in a very dry year climate year
        expected_min_events = [1,1,1,1]
        # Send to test function and then test
        all_events, all_no_events, durations, min_events = evaluate_EWRs.lowflow_calc(EWR_info, flows, water_years, climates, dates, masked_dates)
        for year in all_events:
                assert len(all_events[year]) ==len(expected_all_events[year])
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        for year in all_no_events:
                assert len(all_no_events[year]) == len(expected_all_no_events[year])
                for i, no_event in enumerate(all_no_events[year]):
                        assert no_event == expected_all_no_events[year][i]
        assert durations == expected_durations
        assert min_events == expected_min_events

def test_ctf_calc_anytime():
        '''
        1. Test functions ability to identify and save all events and event gaps for series of flows, ensure events overlapping water year edges are registered
        '''
        # Set up input data
        EWR_info = {'min_flow': 0, 'max_flow': 1, 'min_event':5, 'duration': 20, 'duration_VD': 10}
        flows = np.array([5]*295+[0]*25+[10]*45 + [20]*355+[5000]*5+[0]*5 + [0]*355+[10]*10 + [1]*295+[20]*25+[0]*45+[0]*1)
        water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
        climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
        dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
        # Set up expected output data
        expected_all_events = {2012: [[0]*25], 2013: [], 2014: [[0]*360], 2015: [[1]*295, [0]*46]}
        expected_all_no_events = {2012: [[295]], 2013: [[405]], 2014: [], 2015: [[10], [25]]}
        expected_durations = [20,20,10,20] # adding in a very dry year climate year
        expected_min_events = [5,5,5,5]
        # Send to test function and then test
        all_events, all_no_events, durations, min_events = evaluate_EWRs.ctf_calc_anytime(EWR_info, flows, water_years, climates)
        for year in all_events:
                assert len(all_events[year]) == len(expected_all_events[year])
                for i, event in enumerate(all_events[year]):
                        assert event == expected_all_events[year][i]
        for year in all_no_events:
                assert len(all_no_events[year]) == len(expected_all_no_events[year])
                for i, no_event in enumerate(all_no_events[year]):
                     assert no_event == expected_all_no_events[year][i]
        assert durations == expected_durations
        assert min_events == expected_min_events
        
def test_flow_calc_anytime():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows, ensure events overlapping water year edges are registered
	'''
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'gap_tolerance': 0, 'min_event':10, 'duration': 10}
	flows = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*10 + [10]*5+[0]*350+[10]*10+[10]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	# Set up expected output data
	expected_all_events = {2012: [[10]*10], 2013: [], 2014: [[10]*20, [10]*15], 2015: [[10]*11]}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [[345]], 2015: [[350]]}
	expected_durations = [10]*4
	expected_min_events = [10]*4
	# Send to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.flow_calc_anytime(EWR_info, flows, water_years)
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]
	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
	
def test_lake_calc():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of lake levels, 
		ensuring events are cut off at the end of the water year even though dates are not constrained
	'''
	# Set up input data
	EWR_info = {'min_level': 50, 'max_level': 60, 'duration': 10, 'min_event': 10, 'drawdown_rate': 0.04}
	levels = np.array([0]*350+[50]*10+[0]*5 + [0]*355+[61]*10 + [50]*10+[0]*345+[50]*10 + [50]*5+[0]*350+[50]*10+[50]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events = {2012: [[50]*10], 2013: [], 2014: [[50]*10], 2015: [[50]*11]}
	expected_all_no_events = {2012: [[350]], 2013: [], 2014: [[725]], 2015: [[355]]}
	expected_durations = [10]*4
	expected_min_events = [10]*4
	# Send to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.lake_calc(EWR_info, levels, water_years, dates, masked_dates)
#         print(all_events)
#         print(expected_all_events)
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events

def test_cumulative_calc():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows, 
		ensuring events are cut off at the end of the water year even though dates are not constrained
	'''
	# Set up input data
	EWR_info = {'min_volume': 100, 'min_flow': 50, 'min_event': 2, 'duration': 2}
	flows = np.array([0]*350+[10]*10+[20]*5 + [0]*360+[100]*5 + [75]*1+[25]*1+[0]*353+[50]*10 + [50]*2+[0]*362+[49]*1+[100]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events = {2012: [], 
							2013: [[100], [100]*2, [100]*2], 
							2014: [[50]*2, [50]*2, [50]*2, [50]*2, [50]*2], 
							2015: [[50]*2, [100]*1]}
	expected_all_no_events = {2012: [], 2013: [[724]], 2014: [[355]], 2015: [[362]]}
	expected_durations = [2]*4
	expected_min_events = [2]*4
	all_events, all_no_events, durations, min_events = evaluate_EWRs.cumulative_calc(EWR_info, flows, water_years, dates, masked_dates)
#         print(all_events)
#         print(expected_all_events)
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]
	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
        
        
def test_cumulative_calc_anytime():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows, 
		ensuring events crossing water years are identified and registered
			- Test event crossing water years
			- Test event on final day of series
			- TO-TEST: event on first day of series
	'''
	# Set up input data
	EWR_info = {'min_volume': 100, 'min_flow': 50, 'min_event': 2, 'duration': 2}
	flows = np.array([0]*350+[10]*14+[50]*1 + [50]*1+[0]*358+[100]*6 + [75]*1+[25]*1+[0]*353+[50]*10 + [50]*2+[0]*362+[49]*1+[100]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	# Set up expected outputs
	expected_all_events = {2012: [], 
							2013: [[50]*2, [100]*1, [100]*2, [100]*2], 
							2014: [[100,75], [50]*2, [50]*2, [50]*2, [50]*2, [50]*2], 
							2015: [[50]*2, [100]*1]}
	expected_all_no_events = {2012: [], 2013: [[364], [357]], 2014: [[354]], 2015: [[362]]}
	expected_durations = [2]*4
	expected_min_events = [2]*4
	# Send inputs to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.cumulative_calc_anytime(EWR_info, flows, water_years)       
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]
	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
			
def test_nest_calc_weirpool():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows and levels, ensure events cannot overlap water years. Other tests:
		- check if event exluded when flow requirement is passed but the level requirement is not passed
		- TO-TEST: check if event exluded when flow requirement is not passed but the level requirement is passed
		- TO-TEST: check if event is excluded when flow and level requirements are passed but the drawdown rate is exceeded
	'''
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10}
	flows = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*10 + [10]*5+[0]*351+[10]*10)
	levels = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*9+[1]*1 + [10]*5+[0]*351+[10]*10)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events = {2012: [[10]*10], 2013: [[10]*10], 2014: [[10]*10], 2015: [[10]*10]}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[711]]}
	expected_durations = [10]*4
	expected_min_events = [10]*4
	# Send to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.nest_calc_weirpool(EWR_info, flows, levels, water_years, dates, masked_dates)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
        
def test_nest_calc_percent():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows, ensure events cannot overlap water years. Other tests:
		- check if event exluded when flow requirement is passed but the drawdown rate is exceeded
	'''
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'drawdown_rate': '10%', 'min_event': 10, 'duration': 10}
	flows = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*9+[8]*1 + [10]*9+[9]*1+[0]*346+[10]*10)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events = {2012: [[10]*10], 2013: [[10]*10], 2014: [[10]*10], 2015: [[10]*9+[9]*1, [10]*10]}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[355], [346]]}
	expected_durations = [10]*4
	expected_min_events = [10]*4
	# Send to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.nest_calc_percent(EWR_info, flows, water_years, dates, masked_dates)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
	
def test_nest_calc_percent_trigger():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows, ensure events cannot overlap water years. Other tests:
		- check if event exluded when flow requirement is passed but the drawdown rate is exceeded
	'''
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'drawdown_rate': '10%', 'min_event': 10, 'duration': 10, 'trigger_day': 15, 'trigger_month': 10}
	flows = np.array([0]*106+[11]*1+[10]*9+[0]*249 + [0]*106+[9]*1+[10]*9+[0]*249 + [0]*106+[10]*9+[9]*1+[0]*249 + [0]*106+[10]*9+[8]*1+[0]*250)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events = {2012: [[11]*1+[10]*9], 2013: [[9]*1+[10]*9], 2014: [[10]*9+[9]*1], 2015: []}
	expected_all_no_events = {2012: [[106]], 2013: [[355]], 2014: [[355]], 2015: [[615]]}
	expected_durations = [10]*4
	expected_min_events = [10]*4
	# Send input data to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.nest_calc_percent_trigger(EWR_info, flows, water_years, dates)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
	
def test_weirpool_calc():
	'''
	1. Test weirpool drawdown
	2. Test weirpool raising
	
	For the above two tests: Test functions ability to identify and save all events and event gaps for series of flows and levels, ensure events cannot overlap water years. Other tests:
		- check if event exluded when flow requirement is passed but the level requirement is not passed
		- check if event is excluded when flow and level requirements are passed but the drawdown rate is exceeded
		- TO-TEST: check if event exluded when flow requirement is not passed but the level requirement is passed
	'''
	# Test 1
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10}
	flows = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*10 + [10]*5+[8]*5+[0]*346+[10]*10)
	levels = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*9+[1]*1 + [11]*5+[10]*5+[0]*346+[11]*1+[10]*9)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events = {2012: [[10]*10], 2013: [[10]*10], 2014: [[10]*10], 2015: []}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[721]]}
	expected_durations = [10]*4
	expected_min_events = [10]*4
	weirpool_type = 'falling'
	# Send to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.weirpool_calc(EWR_info, flows, levels, water_years, weirpool_type, dates, masked_dates)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
	#--------------------------------------------------
	# Test 2
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'min_level': 10, 'max_level': 20, 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10}
	flows = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*10 + [10]*5+[8]*5+[0]*346+[10]*10)
	levels = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*9+[1]*1 + [11]*5+[10.96]*5+[0]*346+[10]*9+[9.96]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events = {2012: [[10]*10], 2013: [[10]*10], 2014: [[10]*10], 2015: [[10]*5+[8]*5]}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[355], [356]]}
	expected_durations = [10]*4
	expected_min_event = [10]*4
	weirpool_type = 'raising'
	# Send to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.weirpool_calc(EWR_info, flows, levels, water_years, weirpool_type, dates, masked_dates)
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]
	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event ==expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
	
def test_flow_calc_anytime_sim():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows
		- Check events can span multiple water years
		- Check event not registered when only one site meets flow requirement
	'''
	# Set up input data
	EWR_info1 = {'min_flow': 10, 'max_flow': 20, 'gap_tolerance': 0, 'min_event':10, 'duration': 10}
	EWR_info2 = {'min_flow': 20, 'max_flow': 30, 'gap_tolerance': 0, 'min_event':10, 'duration': 10}
	flows1 = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*10 + [10]*5+[0]*351+[10]*10)
	flows2 = np.array([0]*350+[30]*10+[0]*5 + [0]*355+[30]*10 + [30]*10+[0]*345+[10]*10 + [10]*10+[0]*346+[30]*10)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	# Set up expected output data
	expected_all_events = {2012: [[10]*10], 2013: [], 2014: [[10]*20], 2015: [[10]*10]}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[711]]}
	expected_durations = [10]*4
	expected_min_events = [10]*4
	# Send to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.flow_calc_anytime_sim(EWR_info1, EWR_info2, flows1, flows2, water_years)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events

def test_flow_calc_sim():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows
		- Check events cannot span multiple water years
		- Check event not registered when only one site meets flow requirement
		- TO-TEST: constrain months of water year and repeat test
	'''
	# Set up input data
	EWR_info1 = {'min_flow': 10, 'max_flow': 20, 'gap_tolerance': 0, 'min_event':10, 'duration': 10}
	EWR_info2 = {'min_flow': 20, 'max_flow': 30, 'gap_tolerance': 0, 'min_event':10, 'duration': 10}
	flows1 = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*10 + [10]*5+[0]*351+[10]*10)
	flows2 = np.array([0]*350+[30]*10+[0]*5 + [0]*355+[30]*10 + [30]*10+[0]*345+[10]*10 + [10]*10+[0]*346+[30]*10)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events = {2012: [[10]*10], 2013: [[10]*10], 2014: [[10]*10], 2015: [[10]*10]}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[711]]}
	expected_durations = [10]*4
	expected_min_events = [10]*4
	# Send to test function and then test
	all_events, all_no_events, durations, min_events = evaluate_EWRs.flow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, dates, masked_dates)
	for year in all_events:
		assert len(all_events[year]) ==  len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]
	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
	
def test_lowflow_calc_sim():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows
		- Test to ensure it does not matter event sequencing at each site, as long as minimum day duration is met for each year, event should be registered
	'''
	# Set up input data
	EWR_info1 = {'min_flow': 10, 'max_flow': 20, 'min_event': 1, 'duration': 10, 'duration_VD': 5}
	EWR_info2 = {'min_flow': 20, 'max_flow': 30, 'min_event': 1, 'duration': 10, 'duration_VD': 5}
	flows1 = np.array([10]*1+[0]*350+[10]*9+[0]*5 + [0]*360+[10]*5 + [10]*10+[0]*345+[10]*10 + [8]*5+[0]*351+[10]*10)
	flows2 = np.array([25]*1+[0]*350+[30]*9+[0]*5 + [0]*360+[30]*5 + [30]*10+[0]*345+[10]*10 + [18]*10+[0]*346+[30]*10)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	climates = np.array(['Wet']*365 + ['Very Dry']*365 +['Very Wet']*365 + ['Dry']*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected output data
	expected_all_events1 = {2012: [[10]*1, [10]*9], 2013: [[10]*5], 2014: [[10]*10, [10]*10], 2015: [[10]*10]}
	expected_all_events2 = {2012: [[25]*1, [30]*9], 2013: [[30]*5], 2014: [[30]*10], 2015: [[30]*10]}
	expected_all_no_events1 = {2012: [[350]], 2013: [[365]], 2014: [[345]], 2015: [[356]]}
	expected_all_no_events2 = {2012: [[350]], 2013: [[365]], 2014: [], 2015: [[711]]}
	expected_durations = [10,5,10,10]
	expected_min_events = [1,1,1,1]
	# Send inputs to function and then test:
	all_events1, all_events2, all_no_events1, all_no_events2, durations, min_events = evaluate_EWRs.lowflow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates, dates, masked_dates)
#         print(all_events1)
#         print(expected_all_events1)
	for year in all_events1:
		assert len(all_events1[year]) == len(expected_all_events1[year])
		for i, event in enumerate(all_events1[year]):
			assert event == expected_all_events1[year][i]
	for year in all_events2:
		assert len(all_events2[year]) == len(expected_all_events2[year])
		for i, event in enumerate(all_events2[year]):
			assert event ==expected_all_events2[year][i]

	for year in all_no_events1:
		assert len(all_no_events1[year]) == len(expected_all_no_events1[year])
		for i, no_event in enumerate(all_no_events1[year]):
			assert no_event == expected_all_no_events1[year][i]

	for year in all_no_events2:
		assert len(all_no_events2[year]) == len(expected_all_no_events2[year])
		for i, no_event in enumerate(all_no_events2[year]):
			assert no_event == expected_all_no_events2[year][i]
	assert durations == expected_durations
	assert min_events == expected_min_events
	
	
def test_ctf_calc_sim():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows
	'''
	# Set up input data
	EWR_info1 = {'min_flow': 0, 'max_flow': 1, 'min_event': 10, 'duration': 10, 'duration_VD': 5}
	EWR_info2 = {'min_flow': 0, 'max_flow': 1, 'min_event': 10, 'duration': 10, 'duration_VD': 5}
	flows1 = np.array([10]*1+[0]*350+[10]*9+[1]*5 + [0]*360+[10]*5 + [10]*10+[0]*345+[10]*1+[1]*9 + [8]*5+[10]*351+[0]*10)
	flows2 = np.array([10]*1+[0]*350+[10]*9+[1]*5 + [0]*360+[10]*5 + [10]*10+[0]*345+[10]*1+[1]*9 + [8]*5+[10]*351+[0]*10)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	climates = np.array(['Wet']*365 + ['Very Dry']*365 +['Very Wet']*365 + ['Dry']*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	# Set up expected outputs
	expected_all_events1 = {2012: [[0]*350, [1]*5], 2013: [[0]*360], 2014: [[0]*345, [1]*9], 2015: [[0]*10]}
	expected_all_events2 = {2012: [[0]*350, [1]*5], 2013: [[0]*360], 2014: [[0]*345, [1]*9], 2015: [[0]*10]}
	expected_all_no_events1 = {2012: [[1], [9]], 2013: [], 2014: [[15], [1]], 2015: [[356]]}
	expected_all_no_events2 = {2012: [[1], [9]], 2013: [], 2014: [[15], [1]], 2015: [[356]]}
	expected_durations = [10,5,10,10]
	min_events = [10,10,10,10]
	# Send inputs to function and then test
	all_events1, all_events2, all_no_events1, all_no_events2, durations, min_events = evaluate_EWRs.ctf_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates, dates, masked_dates)

	for year in all_events1:
		assert len(all_events1[year]) == len(expected_all_events1[year])
		for i, event in enumerate(all_events1[year]):
			assert event == expected_all_events1[year][i]
#         print(all_events2)
#         print(expected_all_events2)
	for year in all_events2:
		assert len(all_events2[year]) == len(expected_all_events2[year])
		for i, event in enumerate(all_events2[year]):
			assert event == expected_all_events2[year][i]

	for year in all_no_events1:
		assert len(all_no_events1[year]) == len(expected_all_no_events1[year])
		for i, no_event in enumerate(all_no_events1[year]):
			assert no_event == expected_all_no_events1[year][i]

	for year in all_no_events2:
		assert len(all_no_events2[year]) == len(expected_all_no_events2[year])
		for i, no_event in enumerate(all_no_events2[year]):
			assert no_event == expected_all_no_events2[year][i]
	assert durations == expected_durations
