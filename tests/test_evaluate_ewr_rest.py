from cmath import exp
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np

from py_ewr import evaluate_EWRs, data_inputs
import pytest

def test_component_pull():
	'''
	1. Test correct value is pulled from EWR dataset
	'''
	EWR_table, bad_EWRs = data_inputs.get_EWR_table()
	gauge = '409025'
	PU = 'PU_0000253'
	EWR = 'SF1_P'
	component = 'Duration'
	assert  evaluate_EWRs.component_pull(EWR_table, gauge, PU, EWR, component) == '10'

def test_get_EWRs():
	'''
	1. Ensure requested parts of EWR are returned
	'''
	EWR_table, bad_EWRs = data_inputs.get_EWR_table()
	PU = 'PU_0000283'
	gauge = '410007'
	EWR = 'SF1_P'
	components = ['SM', 'EM']

	expected = {'gauge': '410007', 'planning_unit': 'PU_0000283', 'EWR_code': 'SF1_P', 'start_day': None, 'start_month': 10, 'end_day': None, 'end_month':4}
	assert evaluate_EWRs.get_EWRs(PU, gauge, EWR, EWR_table, components) == expected

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
	index = pd.date_range(start=datetime.strptime('2019-01-01', '%Y-%m-%d'), end=datetime.strptime('2019-12-31', '%Y-%m-%d'))
	df = pd.DataFrame(data = data, index = index)

	#----------------------------------------------------------------------------#
	# Test 1
	EWR_info = {'start_day': None, 'end_day': None, 'start_month': 7, 'end_month': 6}
	masked_7to6 = set(pd.date_range(start=datetime.strptime('2019-01-01', '%Y-%m-%d'), end=datetime.strptime('2019-12-31', '%Y-%m-%d')))
	assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_7to6
	#-----------------------------------------------------------------------
	# Test 2
	EWR_info = {'start_day': None, 'end_day': None, 'start_month': 7, 'end_month': 9}
	masked_7to9 = set(pd.date_range(start=datetime.strptime('2019-07-01', '%Y-%m-%d'), end=datetime.strptime('2019-09-30', '%Y-%m-%d')))
	assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_7to9
	#-----------------------------------------------------------------------
	# Test 3
	EWR_info = {'start_day': None, 'end_day': None, 'start_month': 6, 'end_month': 8}
	masked_6to8 = set(pd.date_range(start=datetime.strptime('2019-06-01', '%Y-%m-%d'), end= datetime.strptime('2019-08-31', '%Y-%m-%d')))
	assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_6to8
	#-----------------------------------------------------------------------
	# Test 4
	EWR_info = {'start_day': 12, 'end_day': 28, 'start_month': 6, 'end_month': 6}
	masked_612to628 = set(pd.date_range(start=datetime.strptime('2019-06-12', '%Y-%m-%d'), end=datetime.strptime('2019-06-28', '%Y-%m-%d')))
	assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_612to628
	#-----------------------------------------------------------------------
	# Test 5
	EWR_info = {'start_day': 12, 'end_day': 18, 'start_month': 8, 'end_month': 9}
	masked_812to918 = set(pd.date_range(start=datetime.strptime('2019-08-12', '%Y-%m-%d'), end=datetime.strptime('2019-09-18', '%Y-%m-%d')))
	assert evaluate_EWRs.mask_dates(EWR_info, df) == masked_812to918


def test_wateryear_daily():
	'''
	1. Testing non standard water year part of function
	2. Testing standard water year part of function
	'''
	# Test 1
	EWR_info = {'start_day': None, 'end_day': None, 'start_month': 5, 'end_month': 8}
	data = {'409102': list(range(0,3650,10)), '425012': list(range(0,3650,10))}
	index = pd.date_range(start=datetime.strptime('2019-01-01', '%Y-%m-%d'), end=datetime.strptime('2019-12-31', '%Y-%m-%d'))
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
	index = pd.date_range(start=datetime.strptime('2019-01-01', '%Y-%m-%d'), end=datetime.strptime('2019-12-31', '%Y-%m-%d'))
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


@pytest.mark.parametrize("event_info,min_duration,expected_year",[
	((date(2012, 6, 25), date(2012, 7, 6), 12, [2011, 2012]), 7, 2012),
	((date(2012, 6, 24), date(2012, 7, 6), 13, [2011, 2012]), 7, 2011),
	((date(2012, 6, 23), date(2012, 7, 6), 14, [2011, 2012]), 7, 2011),
	((date(2012, 6, 23), date(2012, 7, 7), 15, [2011, 2012]), 7, 2012),
],)
def test_which_year_lake_event(event_info, min_duration, expected_year):
	result = evaluate_EWRs.which_year_lake_event(event_info, min_duration)
	assert result == expected_year
        
def test_construct_event_dict():
	'''
	1. Test event dictionary with correct keys and values are returned
	'''
	water_years = [2018]*365+[2019]*365
	all_events = evaluate_EWRs.construct_event_dict(water_years)
	expected_result = {2018:[], 2019:[]}
	assert all_events == expected_result

def test_get_event_years():
	'''
	Year 1: check 1 is returned when there are 3 events with 2 required per year
	Year 2: check 0 is returned when there is 1 event with 2 required per year
	Year 3: check 1 is returned when there are 4 events with 2 required per year
	Year 4: check 0 is returned when there are 0 events with 2 required per year
	'''
	EWR_info = {'events_per_year': 2,'min_event': 5}
	events = {2012: [[5]*5, [10]*5, [20*8]], 2013: [[50]*20],
					2014: [[5]*5, [10]*5, [20*8], [20*8]], 2015: []}
	unique_water_years = [2012, 2013, 2014, 2015]
	durations = [5,5,5,5]
	event_years = evaluate_EWRs.get_event_years(EWR_info, events, unique_water_years, durations)
	expected_event_years = [1,0,1,0]
	assert event_years == expected_event_years


@pytest.mark.parametrize("events,unique_water_years,expected_event_years", [
					 ( 
					 {2012:[ [(date(2012, 11, 1) + timedelta(days=i), 0) for i in range(5)], 
                			 [(date(2013, 6, 26) + timedelta(days=i), 0) for i in range(10)]], 
            		  2013:[[(date(2013, 11, 2) + timedelta(days=i), 0) for i in range(3)], 
                  			[(date(2014, 6, 26) + timedelta(days=i), 0) for i in range(3)]], 
            		  2014:[[(date(2014, 11, 1) + timedelta(days=i), 0) for i in range(5)]], 
            		  2015:[]},
					  [2012, 2013, 2014, 2015],
					  [1,1,1,0]),
					  ]
)
def test_get_event_years_max_rolling_days(events, unique_water_years, expected_event_years):
	'''
	0: when there is evaluation of event years based on ANY duration achieved
	   then : then check achievement against max rolling days. e.g. "CF1_a"

	'''
	event_years = evaluate_EWRs.get_event_years_max_rolling_days(events, unique_water_years)
	assert event_years == expected_event_years

@pytest.mark.parametrize("events,expected_results",[
	({ 2012: [[5]*5, [10]*5, [20]*8], 
	   2013: [[50]*20],
	   2014: [[5]*5, [10]*5, [20]*8, [20]*8], 
	   2015: []},
	   [3,1,4,0]),
	({ 2015: [[5]*5, [10]*5, [20]*8], 
	   2014: [[50]*20],
	   2013: [[5]*5, [10]*5, [20]*8, [20]*8], 
	   2012: []},
	   [0,4,1,3]),
],)
def test_get_all_events(events, expected_results):
	result = evaluate_EWRs.get_all_events(events)
	assert result == expected_results

@pytest.mark.parametrize("events,expected_results",[
	({ 2012: [[5]*5, [10]*5, [20]*8], 
	   2013: [[50]*20],
	   2014: [[5]*5, [10]*5, [20]*8, [20]*8], 
	   2015: []},
	   [18,20,26,0]),
	({ 2015: [[5]*5, [10]*5, [20]*8], 
	   2014: [[50]*20],
	   2013: [[5]*5, [10]*5, [20]*8, [20]*8], 
	   2012: []},
	   [0,26,20,18]),
],)
def test_get_all_event_days(events, expected_results):
	result = evaluate_EWRs.get_all_event_days(events)
	assert result == expected_results


@pytest.mark.parametrize("EWR_info,events,expected_results",[
	(	{'min_event':6},
		{ 2012: [[5]*5, [10]*5, [20]*8], 
	   2013: [[50]*20],
	   2014: [[5]*5, [10]*5, [20]*8, [20]*8], 
	   2015: []},
	   [8,20,16,0]),
	(	{'min_event':6},
		{ 2015: [[5]*5, [10]*5, [20]*8], 
	   2014: [[50]*20],
	   2013: [[5]*5, [10]*5, [20]*8, [20]*8], 
	   2012: []},
	   [0,16,20,8]),
],)
def test_get_achieved_event_days(EWR_info, events, expected_results):
	result = evaluate_EWRs.get_achieved_event_days(EWR_info, events)
	assert result == expected_results


@pytest.mark.parametrize("EWR_info,events,expected_results",[
	(	{'min_event':6},
		{ 2012: [[5]*5, [10]*5, [20]*10, [20]*20], 
	   2013: [[50]*20],
	   2014: [[5]*5, [10]*5, [20]*8, [20]*10], 
	   2015: []},
	   [15.,20.,9.,0]),
	(	{'min_event':6},
		{ 2015: [[5]*5, [10]*5, [20]*10, [20]*20], 
	   2014: [[50]*20],
	   2013: [[5]*5, [10]*5, [20]*8, [20]*10], 
	   2012: []},
	   [0,9.,20.,15.]),

],)
def test_get_average_event_length_achieved(EWR_info, events, expected_results):
	result = evaluate_EWRs.get_average_event_length_achieved(EWR_info, events)
	assert result == expected_results
	

def test_get_achievements():
	'''
	1. Testing 1 event per year requirement with four unique events per year ranges
	2. Testing 2 events per year requirement with four unique events per year ranges
	'''
	EWR_info = {'events_per_year': 1, 'min_event':5}
	events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
					2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
	unique_water_years = [2012, 2013, 2014, 2015]
	durations = [5,5,5,5]
	num_events = evaluate_EWRs.get_achievements(EWR_info, events, unique_water_years, durations)
	expected_num_events = [3,1,4,0]
	assert num_events == expected_num_events
	#-------------------------------------------------
	# Test 2
	EWR_info = {'events_per_year': 2, 'min_event':5}
	events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
					2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
	unique_water_years = [2012, 2013, 2014, 2015]
	durations = [5,5,5,5]
	num_events = evaluate_EWRs.get_achievements(EWR_info, events, unique_water_years, durations)
	expected_num_events = [1,0,2,0]
	assert num_events == expected_num_events

def test_get_number_events():
	'''
	1. Testing 1 event per year requirement with four unique events per year ranges
	2. Testing 2 events per year requirement with four unique events per year ranges
	'''
	# Test 1
	EWR_info = {'events_per_year': 1, 'min_event':5}
	events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
					2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
	unique_water_years = [2012, 2013, 2014, 2015]
	durations = [5,5,5,5]
	num_events = evaluate_EWRs.get_number_events(EWR_info, events, unique_water_years, durations)
	expected_num_events = [3,1,4,0]
	assert num_events == expected_num_events
	#--------------------------------------------------
	# Test 2
	EWR_info = {'events_per_year': 2, 'min_event':5}
	events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
					2014: [[5]*5, [10]*5, [20]*8, [20]*8], 2015: []}
	unique_water_years = [2012, 2013, 2014, 2015]
	durations = [5,5,5,5]
	num_events = evaluate_EWRs.get_number_events(EWR_info, events, unique_water_years, durations)
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

def test_get_max_event_days():
	'''
	1. Given events in a year test return the max event days for each year
	'''
	events = {2012: [[5]*5, [10]*5, [20]*8], 2013: [[50]*20],
					2014: [], 2015: [[5]*5,[5]*5]}
	unique_water_years = [2012, 2013, 2014, 2015]
	total_days = evaluate_EWRs.get_max_event_days(events, unique_water_years)
	expected_total_days = [8,20,0,5]
	assert total_days == expected_total_days

def test_get_data_gap():
	'''
	1. Check event gaps are accurate
	'''
	data = {'409102': list(range(0,3650,10)), '425012': list(range(0,3650,10))}
	index = pd.date_range(start=datetime.strptime('2019-01-01', '%Y-%m-%d'), end=datetime.strptime('2019-12-31', '%Y-%m-%d'))
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
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	event_start = flow_date.date() - timedelta(days=9)
	event = [(event_start + timedelta(days=i), 5) for i in range(9)]
	all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
					2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
	gap_track = 0
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)
	total_event = 9
	event, all_events, gap_track, total_event = evaluate_EWRs.flow_check(EWR_info, iteration, flow, event, all_events, gap_track, water_years, total_event, flow_date)
	# Set up expected results and test
	expected_event =  [(event_start + timedelta(days=i), 5) for i in range(10)]
	expected_all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
							2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
	expected_gap_track = 0
	expected_total_event = 10
	assert event == expected_event
	for year in all_events:
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]

	assert gap_track == expected_gap_track
	assert total_event == expected_total_event

def test_lowflow_check():
	'''
	1. Test flow passes and event requirement just met
	2. TO-TEST: flow threshold below but event requirement passed
	3. TO-TEST: flow threshold failed and event requirements failed
	'''
	# Set up variables for all tests
	EWR_info = {'min_flow': 10, 'max_flow': 20}
	flow = 5
	water_year = 2015
	flow_date = date(2012,1,17)
	event = [(date(2015, 10, 9)+timedelta(days=i),5) for i in range(9)]
	iteration = 365+365+365+100
	all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
					2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)
	event, all_events = evaluate_EWRs.lowflow_check(EWR_info, iteration, flow, event, all_events, water_years, flow_date)
	# Set up expected output and test
	expected_event = []
	expected_all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
							2014:[[10]*10, [15]*15, [10]*20], 2015:[[(date(2015, 10, 9)+timedelta(days=i),5) for i in range(9)]]}
	expected_no_event = 1
	expected_all_no_events = {2012:[[25], [2]], 2013:[[250]],
									2014:[[400], [2], [25]], 2015:[[450]]}
	assert event == expected_event

	for year in all_events:
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]


def test_ctf_check():
	'''
	1. flow threshold fails but event meets requirements
	2. TO-TEST: flow threshold passed
	3. TO-TEST: flow threshold failed but no event recorded
	'''
	# Set up input variables and pass to test function
	EWR_info = {'min_flow': 0, 'max_flow': 1}
	flow = 2
	iteration = 365+365+365+100
	event = [0]*9
	flow_date = date(2012,1,17)
	all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
					2014:[[10]*10, [15]*15, [10]*20], 2015:[]}

	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)

	event, all_events = evaluate_EWRs.ctf_check(EWR_info, iteration, flow, event, all_events, water_years, flow_date)
	# Set up expected outputs and test
	expected_event = []
	expected_all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50],
							2014:[[10]*10, [15]*15, [10]*20], 2015:[[0]*9]}
	
	assert event ==expected_event
	for year in all_events:
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]
                
@pytest.mark.parametrize("flows,expected_all_events,expected_all_no_events",
						 [ 
				            (np.array([0]*350+[10]*15 + 
	                                   [10]*11+ [0]*354 + 
									   [0]*365 +
									   [0]*366),
							{2012: [[(date(2013, 6, 16)+timedelta(days=i),10) for i in range(15)]], 
								2013: [[(date(2013, 7, 1)+timedelta(days=i),10) for i in range(11)]], 
								2014: [ ], 
								2015: [] },

							{2012: [[350]], 2013: [], 2014: [], 2015: [[1085]]}
							 ),
							  (np.array([0]*356+[10]*9 + 
	                                   [10]*11+ [0]*354 + 
									   [0]*365 +
									   [0]*366),
							{  2012: [[(date(2013, 6, 22)+timedelta(days=i),10) for i in range(9)]],
								2013: [[(date(2013, 7, 1)+timedelta(days=i),10) for i in range(11)]], 
								2014: [ ], 
								2015: [] },

							{2012: [], 2013: [[365]], 2014: [], 2015: [[1085]]}
							 ),
							  (np.array([0]*356+[10]*9 + 
	                                   [10]*9+ [0]*356 + 
									   [0]*365 +
									   [0]*366),
							{  2012: [[(date(2013, 6, 22)+timedelta(days=i),10) for i in range(9)]], 
								2013: [[(date(2013, 7, 1)+timedelta(days=i),10) for i in range(9)]], 
								2014: [], 
								2015: [] },

							{2012: [], 2013: [], 2014: [], 2015: [[1461]]}
							 ),
							  (np.array([10]*365 + 
	                                    [10]*365 + 
									    [10]*365 +
									    [10]*366),
							{  2012:  [[(date(2012, 7, 1)+timedelta(days=i),10) for i in range(365)]], 
								2013: [[(date(2013, 7, 1)+timedelta(days=i),10) for i in range(365)]], 
								2014: [[(date(2014, 7, 1)+timedelta(days=i),10) for i in range(365)]], 
								2015: [[(date(2015, 7, 1)+timedelta(days=i),10) for i in range(366)]] },

							{2012: [], 2013: [], 2014: [], 2015: []}
							 ),
							  (np.array([10]*100 + [0]*1 + [10]*264 + 
	                                    [10]*100 + [0]*1 + [10]*264 +
									    [10]*100 + [0]*1 + [10]*264 +
									    [10]*100 + [0]*1 + [10]*265),
							{  2012:  [[(date(2012, 7, 1)+timedelta(days=i),10) for i in range(100)],
							            [(date(2012, 10, 10)+timedelta(days=i),10) for i in range(264)]], 
								2013: [[(date(2013, 7, 1)+timedelta(days=i),10) for i in range(100)],
								       [(date(2013, 10, 10)+timedelta(days=i),10) for i in range(264)] ], 
								2014: [[(date(2014, 7, 1)+timedelta(days=i),10) for i in range(100)],
										[(date(2014, 10, 10)+timedelta(days=i),10) for i in range(264)]], 
								2015: [[(date(2015, 7, 1)+timedelta(days=i),10) for i in range(100)],
								        [(date(2015, 10, 10)+timedelta(days=i),10) for i in range(265)]] },

							{2012: [[1]], 2013: [[1]], 2014: [[1]], 2015: [[1]]}
							 )])
def test_flow_calc(flows,expected_all_events,expected_all_no_events):
	"""
	0: when event start and finish goes beyond boundary of 2 water years and there are sufficient days in both years
	   then : each year gets the part of the event as a separate event
	1: when event start and finish goes beyond boundary of 2 water years and there are sufficient days only second year
	   then : first year get no event, second year gets the part of the event as a separate event
	2: when event start and finish goes beyond boundary of 2 water years and there not sufficient days in both years and total event meets sufficient days
	   then : none of the years get the event
	3: when event start and finish goes beyond boundary of 4 water years and there are sufficient days for all years
	   then : all years get 1 event each with all days as event days
	4: when 2 events start and finish within the boundary of the water year and both meets the sufficient days, however the second event of each year finishes at the last
	 day of the year continuing into the next water year.
	   then : all years get 2 event each with all days as event days excluding the event gaps.
	"""
	# Test 1
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'gap_tolerance': 0, 'min_event':10, 'duration': 10}
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Set up expected output data        
	expected_durations = [10]*4
	# Send inputs to test function and test
	all_events, durations = evaluate_EWRs.flow_calc(EWR_info, flows, water_years, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]
	assert durations == expected_durations

def test_lowflow_calc():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows
	2. Constrain timing window and test functions ability to identify and save all events and event gaps for series of flows
	'''
	# Test 1
	# set up input data 
	EWR_info = {'min_flow': 10, 'max_flow': 20, 'min_event':1, 'duration': 300, 'duration_VD': 10}
	flows = np.array([5]*295+[0]*25+[10]*45 + [0]*355+[5000]*10 + [0]*355+[10]*10 + [5]*295+[0]*25+[10]*45+[10]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Set up expected output data
	expected_all_events = {2012: [[(date(2013, 5, 17), 10), (date(2013, 5, 18), 10), (date(2013, 5, 19), 10), 
	(date(2013, 5, 20), 10), (date(2013, 5, 21), 10), (date(2013, 5, 22), 10), (date(2013, 5, 23), 10), 
	(date(2013, 5, 24), 10), (date(2013, 5, 25), 10), (date(2013, 5, 26), 10), (date(2013, 5, 27), 10), 
	(date(2013, 5, 28), 10), (date(2013, 5, 29), 10), (date(2013, 5, 30), 10), (date(2013, 5, 31), 10), 
	(date(2013, 6, 1), 10), (date(2013, 6, 2), 10), (date(2013, 6, 3), 10), (date(2013, 6, 4), 10), 
	(date(2013, 6, 5), 10), (date(2013, 6, 6), 10), (date(2013, 6, 7), 10), (date(2013, 6, 8), 10), 
	(date(2013, 6, 9), 10), (date(2013, 6, 10), 10), (date(2013, 6, 11), 10), (date(2013, 6, 12), 10), 
	(date(2013, 6, 13), 10), (date(2013, 6, 14), 10), (date(2013, 6, 15), 10), (date(2013, 6, 16), 10), 
	(date(2013, 6, 17), 10), (date(2013, 6, 18), 10), (date(2013, 6, 19), 10), (date(2013, 6, 20), 10), 
	(date(2013, 6, 21), 10), (date(2013, 6, 22), 10), (date(2013, 6, 23), 10), (date(2013, 6, 24), 10), 
	(date(2013, 6, 25), 10), (date(2013, 6, 26), 10), (date(2013, 6, 27), 10), (date(2013, 6, 28), 10), 
	(date(2013, 6, 29), 10), (date(2013, 6, 30), 10)]], 
						   2013: [], 
						   2014: [[(date(2015, 6, 21), 10), (date(2015, 6, 22), 10), 
	(date(2015, 6, 23), 10), (date(2015, 6, 24), 10), (date(2015, 6, 25), 10), (date(2015, 6, 26), 10), 
	(date(2015, 6, 27), 10), (date(2015, 6, 28), 10), (date(2015, 6, 29), 10), (date(2015, 6, 30), 10)]], 
						   2015: [[(date(2016, 5, 16), 10), (date(2016, 5, 17), 10), (date(2016, 5, 18), 10), (date(2016, 5, 19), 10), 
	(date(2016, 5, 20), 10), (date(2016, 5, 21), 10), (date(2016, 5, 22), 10), (date(2016, 5, 23), 10), 
	(date(2016, 5, 24), 10), (date(2016, 5, 25), 10), (date(2016, 5, 26), 10), (date(2016, 5, 27), 10), 
	(date(2016, 5, 28), 10), (date(2016, 5, 29), 10), (date(2016, 5, 30), 10), (date(2016, 5, 31), 10), 
	(date(2016, 6, 1), 10), (date(2016, 6, 2), 10), (date(2016, 6, 3), 10), (date(2016, 6, 4), 10), 
	(date(2016, 6, 5), 10), (date(2016, 6, 6), 10), (date(2016, 6, 7), 10), (date(2016, 6, 8), 10), 
	(date(2016, 6, 9), 10), (date(2016, 6, 10), 10), (date(2016, 6, 11), 10), (date(2016, 6, 12), 10), 
	(date(2016, 6, 13), 10), (date(2016, 6, 14), 10), (date(2016, 6, 15), 10), (date(2016, 6, 16), 10), 
	(date(2016, 6, 17), 10), (date(2016, 6, 18), 10), (date(2016, 6, 19), 10), (date(2016, 6, 20), 10), 
	(date(2016, 6, 21), 10), (date(2016, 6, 22), 10), (date(2016, 6, 23), 10), (date(2016, 6, 24), 10), 
	(date(2016, 6, 25), 10), (date(2016, 6, 26), 10), (date(2016, 6, 27), 10), (date(2016, 6, 28), 10), 
	(date(2016, 6, 29), 10), (date(2016, 6, 30), 10)]]}
	expected_all_no_events = {2012: [[320]], 2013: [], 2014: [[720]], 2015: [[320]]}
	expected_durations = [300,300,10,300]
	# Send inputs to test function and test
	all_events, durations = evaluate_EWRs.lowflow_calc(EWR_info, flows, water_years, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]             

	#------------------------------------------------
	# Test 2
	# Set up input data
	EWR_info = {'min_flow': 10, 'max_flow': 20, 'min_event':1, 'duration': 10, 
					'duration_VD': 5, 'start_month': 7, 'end_month': 12, 'start_day': None, 'end_day': None}
	flows = np.array([10]*5+[0]*35+[5]*5+[0]*295+[0]*25 + [0]*355+[5]*10 + [10]*10+[0]*355 + [5]*295+[0]*25+[10]*45+[10]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = masked_dates[((masked_dates.month >= 7) & (masked_dates.month <= 12))] # Just want the dates in the date range
	# Set up expected output data
	expected_all_events = {2012: [[(date(2012, 7, 1), 10), (date(2012, 7, 2), 10), (date(2012, 7, 3), 10), 
	(date(2012, 7, 4), 10), (date(2012, 7, 5), 10)]], 2013: [], 2014: [[(date(2014, 7, 1), 10), 
	(date(2014, 7, 2), 10), (date(2014, 7, 3), 10), (date(2014, 7, 4), 10), (date(2014, 7, 5), 10), 
	(date(2014, 7, 6), 10), (date(2014, 7, 7), 10), (date(2014, 7, 8), 10), (date(2014, 7, 9), 10), 
	(date(2014, 7, 10), 10)]], 2015: []}
	# Send to test function and test
	all_events, durations = evaluate_EWRs.lowflow_calc(EWR_info, flows, water_years, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]


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
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Set up expected output data
	expected_all_events = {2012: [[(date(2013, 4, 22)+timedelta(days=i), 0) for i in range(25)]],
	  					   2013: [[(date(2014, 6, 26)+timedelta(days=i), 0) for i in range(5)]], 
						   2014: [[(date(2014, 7, 1)+timedelta(days=i), 0) for i in range(355)]],
						   2015: [[(date(2015, 7, 1)+timedelta(days=i), 1) for i in range(295)], 
						   [(date(2016, 5, 16)+timedelta(days=i), 0) for i in range(46)]]}
	# Send to test function and then test
	all_events, _ = evaluate_EWRs.ctf_calc(EWR_info, flows, water_years, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]

	#--------------------------------------------------
	# Test 2
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'min_event':1, 'duration': 10,
					'duration_VD': 5, 'start_month': 7, 'end_month': 12, 'start_day': None, 'end_day': None}
	flows = np.array([10]*5+[0]*35+[5]*5+[0]*295+[0]*25 + 
	[0]*355+[5]*10 + 
	[10]*10+[0]*355 +
	[5]*295+[0]*25+[10]*45+[10]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = masked_dates[((masked_dates.month >= 7) & (masked_dates.month <= 12))] # Just want the dates in the date range
	# Set up expected output data
	expected_all_events = {2012: [[(date(2012, 7, 1)+timedelta(days=i), 10) for i in range(5)],
	[(date(2012, 8, 10)+timedelta(days=i), 5) for i in range(5)]], 
	2013: [], 
	2014: [[(date(2014, 7, 1)+timedelta(days=i), 10) for i in range(10)]], 
	2015: [[(date(2015, 7, 1)+timedelta(days=i), 5) for i in range(184)]] 
	}
	# Send to test function and then test
	all_events, _ = evaluate_EWRs.ctf_calc(EWR_info, flows, water_years, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) ==len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]


@pytest.mark.parametrize("flows,expected_all_events,expected_all_no_events", [
					 (np.array([20]*350 + [0]*15 +
					 [0]*10 + [20]*355+ 
					 [20]*365+ 
					 [20]*366), 
					 {2012: [], 
						   2013: [[(date(2013, 6, 16)+timedelta(days=i), 0) for i in range(25)]], 
						   2014: [], 
						   2015: []},
					  {2012: [[350]], 2013: [], 2014: [], 2015: [[1086]]}),
					  (np.array([20]*355 + [0]*10 +
					 [0]*15 + [20]*350+ 
					 [20]*365+ 
					 [20]*366), 
					 {2012: [], 
						   2013: [[(date(2013, 6, 21)+timedelta(days=i), 0) for i in range(25)]], 
						   2014: [], 
						   2015: []},
					  {2012: [[355]], 2013: [], 2014: [], 2015: [[1081]]}),
					  (np.array([20]*355 + [0]*10 +
					 [0]*365+ 
					 [0]*15 + [20]*350+ 
					 [20]*366), 
					 {2012: [], 
						   2013: [], 
						   2014: [[(date(2013, 6, 21)+timedelta(days=i), 0) for i in range(390)]],
						   2015: []},
					  {2012: [[355]], 2013: [], 2014: [], 2015: [[716]]}),
					  (np.array([20]*340 + [0]*21 + [20]*4 +
					 [20]*365+ 
					 [20]*365+
					 [20]*366), 
					 {2012: [[(date(2013, 6, 6)+timedelta(days=i), 0) for i in range(21)]], 
						   2013: [], 
						   2014: [],
						   2015: []},
					  {2012: [[340]], 2013: [], 2014: [], 2015: [[1100]]}),
					  (np.array([20]*345 + [0]*20 +
					 [0]*1 + [20]*364+ 
					 [20]*365+ 
					 [20]*366), 
					 {2012: [], 
						   2013: [[(date(2013, 6, 11)+timedelta(days=i), 0) for i in range(21)]], 
						   2014: [], 
						   2015: []},
					  {2012: [[345]], 2013: [], 2014: [], 2015: [[1095]]}),
					  (np.array([20]*344 + [0]*21 +
					 [20]*365+ 
					 [20]*365+ 
					 [20]*366), 
					 {2012: [[(date(2013, 6, 10)+timedelta(days=i), 0) for i in range(21)]], 
						   2013: [], 
						   2014: [], 
						   2015: []},
					  {2012: [[344]], 2013: [], 2014: [], 2015: [[1096]]}),			   
														])
def test_ctf_calc_anytime(flows, expected_all_events, expected_all_no_events):
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows, 
	ensure events overlapping water year edges are registered AT THE YEAR it ends

	0: when event start and finish goes beyond boundary of 2 water years more days on the first year
	   then : whole event gets allocated to year it ends i.e. SECOND
	1: when event start and finish goes beyond boundary of 2 water years more days on the second year
	   then : whole event gets allocated to year it ends i.e. SECOND
	2: when event start and finish goes beyond boundary of 3 water years
	   then : whole event gets allocated to year it ends i.e. THIRD
	3: when event start and finish same water year
	   then : whole event gets allocated to year it ends 
	4: when event start and finish goes beyond boundary of 2 water years and finish first day of second year
	   then : whole event gets allocated to year it ends i.e. SECOND
	5: when event start and finish same water year and finished on the last day of the water year 30/June
	   then : whole event gets allocated to year it ends i.e. FIRST
	'''
	# Set up input data
	EWR_info = {'min_flow': 0, 'max_flow': 1, 'min_event':5, 'duration': 20, 'duration_VD': 10}
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Send to test function and then test
	all_events, durations = evaluate_EWRs.ctf_calc_anytime(EWR_info, flows, water_years, dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]


@pytest.mark.parametrize("flows,expected_all_events,expected_all_no_events",
						 [ 
							 (np.array([0]*350+[10]*15 + 
	                                   [10]*11+ [0]*354 + 
									   [0]*365 +
									   [0]*366),
							{2012: [[(date(2013, 6, 16)+timedelta(days=i),10) for i in range(15+11)]], 
								2013: [], 
								2014: [ ], 
								2015: [] },

							{2012: [[350]], 2013: [], 2014: [], 2015: [[1085]]}
							 ),
							  (np.array([0]*344+[10]*21 + 
	                                   [10]*28+ [0]*337 + 
									   [0]*365 +
									   [0]*366),
							{2012: [], 
								2013: [[(date(2013, 6, 10) + timedelta(days=i), 10) for i in range(21+28)]], 
								2014: [], 
								2015: [] },

							{2012: [[344]], 2013: [], 2014: [], 2015: [[1068]]}
							 ),
							  (np.array([0]*344+[10]*21 + 
	                                   [10]*21+ [0]*344 + 
									   [0]*365 +
									   [0]*366),
							{2012: [], 
								2013: [[(date(2013, 6, 10) + timedelta(days=i), 10) for i in range(21+21)]], 
								2014: [], 
								2015: [] },

							{2012: [[344]], 2013: [], 2014: [], 2015: [[1075]]}
							 )]
							 )        
def test_flow_calc_anytime(flows, expected_all_events, expected_all_no_events):
	"""
	0: when event start and finish goes beyond boundary of 2 water years and there are more days in the first year
	   then : whole event gets allocated to FIRST year
	1: when event start and finish goes beyond boundary of 2 water years and there are more days in the second year
	   then : whole event gets allocated to SECOND year
	2: when event start and finish goes beyond boundary of 2 water years and there are same number of days in both years
	   then : whole event gets allocated to SECOND year
	"""
	# Set up input data
	EWR_info = {'min_flow': 5, 'max_flow': 20, 'gap_tolerance': 0, 'min_event':10, 'duration': 10}
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	# Set up expected output data
	expected_durations = [10]*4
	# Send to test function and then test
	dates = 1
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	all_events, durations = evaluate_EWRs.flow_calc_anytime(EWR_info, flows, water_years, dates)

	for year in all_events:
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]
	assert durations == expected_durations


def test_get_index_date(period_date, stamp_date):
	assert evaluate_EWRs.get_index_date(period_date) == evaluate_EWRs.get_index_date(stamp_date)


@pytest.mark.parametrize("EWR_info,flows,expected_all_events,expected_all_no_events",[
	( {'min_volume': 120, 'min_flow': 0, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':7, 'end_month':6 ,'gap_tolerance':0},
	   np.array([20]*10+[0]*355   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{2012: [[(date(2012, 7, 6), 120),
			(date(2012, 7, 7), 140),
			(date(2012, 7, 8), 160),
			(date(2012, 7, 9), 180),
			(date(2012, 7, 10), 200),
			(date(2012, 7, 11), 180),
			(date(2012, 7, 12), 160),
			(date(2012, 7, 13), 140),
			(date(2012, 7, 14), 120)]],
		2013: [],
		2014: [],
		2015: []},
	{2012: [[5]], 2013: [], 2014: [], 2015: [[1447]]}),
 ( {'min_volume': 120, 'min_flow': 0, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':7, 'end_month':6 ,'gap_tolerance':0},
	   np.array( [0]*345 +[20]*20  + 
                 [20]*10 + [0]*355 + 
                    [0]*365 + 
                    [0]*366),
	  {2012: [[(date(2013, 6, 16), 120),
    (date(2013, 6, 17), 140),
    (date(2013, 6, 18), 160),
    (date(2013, 6, 19), 180),
    (date(2013, 6, 20), 200),
    (date(2013, 6, 21), 200),
    (date(2013, 6, 22), 200),
    (date(2013, 6, 23), 200),
    (date(2013, 6, 24), 200),
    (date(2013, 6, 25), 200),
    (date(2013, 6, 26), 200),
    (date(2013, 6, 27), 200),
    (date(2013, 6, 28), 200),
    (date(2013, 6, 29), 200),
    (date(2013, 6, 30), 200)]],
  2013: [[(date(2013, 7, 6), 120),
    (date(2013, 7, 7), 140),
    (date(2013, 7, 8), 160),
    (date(2013, 7, 9), 180),
    (date(2013, 7, 10), 200),
    (date(2013, 7, 11), 180),
    (date(2013, 7, 12), 160),
    (date(2013, 7, 13), 140),
    (date(2013, 7, 14), 120)]],
	2014: [],
	2015: []},
	{2012: [[350]], 2013: [[5]], 2014: [], 2015: [[1082]]}),
],
)
def test_cumulative_calc(EWR_info, flows, expected_all_events, expected_all_no_events):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	all_events, durations = evaluate_EWRs.cumulative_calc(EWR_info, flows, water_years, dates, masked_dates)

	assert all_events == expected_all_events

@pytest.mark.parametrize("EWR_info,flows,expected_all_events",[
	( {'min_volume': 120, 'min_flow': 0, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':7, 'end_month':6 ,'gap_tolerance':0},
	   np.array([20]*10+[0]*355   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{2012: [[(date(2012, 7, 6), 120),
			(date(2012, 7, 7), 140),
			(date(2012, 7, 8), 160),
			(date(2012, 7, 9), 180),
			(date(2012, 7, 10), 200),
			(date(2012, 7, 11), 180),
			(date(2012, 7, 12), 160),
			(date(2012, 7, 13), 140),
			(date(2012, 7, 14), 120)]],
		2013: [],
		2014: [],
		2015: []}),
	( {'min_volume': 120, 'min_flow': 0, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':7, 'end_month':6 ,'gap_tolerance':0},
	   np.array([10]*20+[0]*345   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{   2012: [],
		2013: [],
		2014: [],
		2015: []}),

	( {'min_volume': 120, 'min_flow': 0, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':7, 'end_month':6 ,'gap_tolerance':0},
	   np.array(   [0]*358+[20]*7 + 
                    [20]*3 +[0]*360 + 
                    [0]*365 + 
                    [0]*366),
	{   2012: [],
		2013: [[(date(2013, 6, 29), 120),
			(date(2013, 6, 30), 140),
			(date(2013, 7, 1), 160),
			(date(2013, 7, 2), 180),
			(date(2013, 7, 3), 200),
			(date(2013, 7, 4), 180),
			(date(2013, 7, 5), 160),
			(date(2013, 7, 6), 140),
			(date(2013, 7, 7), 120)]],
		2014: [],
		2015: []}),
 
],)
def test_cumulative_calc_qld(EWR_info, flows, expected_all_events):
	""" 1. reaches volume from day 6 to day 14 of the flows within the accumulation period and records the event
	    2. reaches volume from day beyond accumulation period and does not record the event
	    3. record a volume event across year boundary and does not split the event
	"""
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	all_events, _ = evaluate_EWRs.cumulative_calc_qld(EWR_info, flows, water_years, dates, masked_dates)

	assert all_events == expected_all_events
	
@pytest.mark.parametrize("EWR_info,iteration,flows,event,all_events,all_no_events,expected_all_events,expected_event",
[
	({'min_volume': 100, 'min_flow': 0, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':10, 'end_month':4 ,'gap_tolerance':0},
     5,	
	 np.array([20]*10+[0]*355   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 6)+timedelta(days=i),120) for i in range(1)],	
	 ),
	({'min_volume': 100, 'min_flow': 25, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':10, 'end_month':4 ,'gap_tolerance':0},
     5,	
	 np.array([20]*10+[0]*355   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),	 
],)
def test_volume_check(EWR_info,iteration,flows,event,all_events,all_no_events,expected_all_events,expected_event):
	flow = 20
	roller = 5
	max_roller = EWR_info['accumulation_period']
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	total_event = 9
	gap_track = 0
	event, all_events,  gap_track, total_event, roller = evaluate_EWRs.volume_check(EWR_info, iteration, flow, event,
	 all_events, gap_track, water_years, total_event, flow_date, roller, max_roller, flows)

	assert event == expected_event
	assert expected_all_events != None


@pytest.mark.parametrize("EWR_info,iteration,flows,event,all_events,all_no_events,expected_all_events,expected_event",
[
	({'min_volume': 100, 'min_flow': 0, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':10, 'end_month':4 ,'gap_tolerance':0},
     5,	
	 np.array([20]*10+[0]*355   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 6)+timedelta(days=i),120) for i in range(1)],	
	 ),
	({'min_volume': 130, 'min_flow': 25, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':10, 'end_month':4 ,'gap_tolerance':0},
     5,	
	 np.array([20]*10+[0]*355   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),	 
],)
def test_volume_check_qld(EWR_info,iteration,flows,event,all_events,all_no_events,expected_all_events,expected_event):
	"""
	1. achieve volume and record the event on iteration 5.
	2. do not achieve volume and do not record the event on iteration 5.
	"""
	roller = 5
	max_roller = EWR_info['accumulation_period']
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	total_event = 0
	event, all_events,  total_event, roller = evaluate_EWRs.volume_check_qld(EWR_info, iteration, event,
	 all_events, water_years, total_event, flow_date, roller, max_roller, flows)

	assert event == expected_event
	assert expected_all_events != None
																						

@pytest.mark.parametrize("roller,flow_date,EWR_info,expected_roller",[
	(0,date(2000,1,1),{'start_month': 10},0),
	(5,date(2000,1,1),{'start_month': 10},5),
	(5,date(2000,7,1),{'start_month': 7},0),
	(5,date(2000,10,1),{'start_month': 10},0),
],)
def test_check_roller_reset_points(roller, flow_date, EWR_info,expected_roller):
	result = evaluate_EWRs.check_roller_reset_points(roller, flow_date, EWR_info)
	assert result == expected_roller

@pytest.mark.parametrize("events,unique_water_years,expected_max_vols", [
					 ( 
					 {2012:[ [(date(2012, 11, 1) + timedelta(days=i), 5) for i in range(5)], 
                			 [(date(2013, 6, 26) + timedelta(days=i), 10) for i in range(10)]], 
            		  2013:[[(date(2013, 11, 2) + timedelta(days=i), 3) for i in range(3)], 
                  			[(date(2014, 6, 26) + timedelta(days=i), 3) for i in range(3)]], 
            		  2014:[[(date(2014, 11, 1) + timedelta(days=i), 5) for i in range(5)]], 
            		  2015:[]},
					  [2012, 2013, 2014, 2015],
					  [10,3,5,0]),
					  ]
)
def test_get_max_volume(events,unique_water_years,expected_max_vols):
	max_vols = evaluate_EWRs.get_max_volume(events,unique_water_years)
	assert max_vols == expected_max_vols


@pytest.mark.parametrize("events,unique_water_years,expected_years_achieved", [
					 ( 
					 {2012:[ [(date(2012, 11, 1) + timedelta(days=i), 5) for i in range(5)], 
                			 [(date(2013, 6, 26) + timedelta(days=i), 10) for i in range(10)]], 
            		  2013:[[(date(2013, 11, 2) + timedelta(days=i), 3) for i in range(3)], 
                  			[(date(2014, 6, 26) + timedelta(days=i), 3) for i in range(3)]], 
            		  2014:[[(date(2014, 11, 1) + timedelta(days=i), 5) for i in range(5)]], 
            		  2015:[]},
					  [2012, 2013, 2014, 2015],
					  [1,1,1,0]),
					  ]
)
def test_get_event_years_volume_achieved(events,unique_water_years,expected_years_achieved):
	max_vols = evaluate_EWRs.get_event_years_volume_achieved(events,unique_water_years)
	assert max_vols == expected_years_achieved


@pytest.mark.parametrize("all_no_events,unique_water_years,expected_result",[
	({2012: [[92], [75]],
	2013: [[112], [59]],
	2014: [[187]],
	2015: []},
	[2012, 2013, 2014, 2015],
	[92,112,187,0])
])
def test_get_max_inter_event_days(all_no_events,unique_water_years,expected_result):
	result = evaluate_EWRs.get_max_inter_event_days(all_no_events,unique_water_years)
	assert result == expected_result


@pytest.mark.parametrize("EWR_info,no_events,unique_water_years,expected_results",[
		({'max_inter-event':0.5},
			
		{2012: [[92], [75]],
		2013: [[112], [59]],
		2014: [[187]],
		2015: []},
		[2012, 2013, 2014, 2015],
		[1,1,0,1])
])
def test_get_event_max_inter_event_achieved(EWR_info,no_events,unique_water_years,expected_results):
	result = evaluate_EWRs.get_event_max_inter_event_achieved(EWR_info,no_events,unique_water_years)
	assert result == expected_results

@pytest.mark.parametrize("gauge,ewr,pu,expected_result",[
	("421004", "CF" , "PU_0000129", False),
	("421090", "CF" , "PU_0000130", True),
	("11111", "XX" , "DD", False),
],)
def test_is_multigauge(parameter_sheet, gauge, ewr, pu, expected_result):
	result = evaluate_EWRs.is_multigauge(parameter_sheet, gauge, ewr, pu)
	assert result == expected_result


@pytest.mark.parametrize("gauge,ewr,pu,expected_result",[
	("414203", "VF" , "PU_0000260", True),
	("414203", "WP2" , "PU_0000260", True),
	("11111", "XX" , "DD", False),
],)
def test_is_weirpool_gauge(parameter_sheet, gauge, ewr, pu, expected_result):
	result = evaluate_EWRs.is_weirpool_gauge(parameter_sheet, gauge, ewr, pu)
	assert result == expected_result


@pytest.mark.parametrize("gauge,ewr,pu,expected_result",[
	("421090", "CF" , "PU_0000130", "421088"),
],)
def test_get_second_multigauge(parameter_sheet, gauge, ewr, pu, expected_result):
	result = evaluate_EWRs.get_second_multigauge(parameter_sheet, gauge, ewr, pu)
	assert result == expected_result


@pytest.mark.parametrize("weirpool_type,level,EWR_info,expected_result",[
	("raising", 5 , {'min_level': 5, 'max_level': 10}, True),
	("raising", 4 , {'min_level': 5, 'max_level': 10}, False),
	("falling", 4 , {'min_level': 5, 'max_level': 10}, True),
	("falling", 11 , {'min_level': 5, 'max_level': 10}, False),
],)
def test_check_wp_level(weirpool_type, level, EWR_info,expected_result):
	result = evaluate_EWRs.check_wp_level(weirpool_type, level, EWR_info)
	assert result == expected_result


@pytest.mark.parametrize("level_change,EWR_info,expected_result",[
	(0.04, {'drawdown_rate': 0.04}, True),
	(0.05, {'drawdown_rate': 0.04}, False),
	(-1000, {'drawdown_rate': 0.04}, True),
	(0.03, {'drawdown_rate': 0.04}, True),
	(0.03, {'drawdown_rate': 0.0}, True),
	(-1, {'drawdown_rate': 0.0}, True),
	(1000000, {'drawdown_rate': 0.0}, True),
],)
def test_check_draw_down(level_change, EWR_info, expected_result):
	result = evaluate_EWRs.check_draw_down(level_change, EWR_info)
	assert result == expected_result


@pytest.mark.parametrize("EWR_info,iteration,flow,level,event,all_events,all_no_events,weirpool_type,level_change,total_event,expected_all_events,expected_event",
[
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     0,	
	 50,
	 5,
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 "raising",
	 0.04,
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 1), 50)],	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     1,	
	 50,
	 5,
	[(date(2012, 7, 1), 50)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 "raising",
	 0.04,
	 1,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 1), 50),(date(2012, 7, 2), 50)],	
	 ),
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     2,	
	 4,
	 5,
	[(date(2012, 7, 1), 50),(date(2012, 7, 2), 50)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 "raising",
	 0.04,
	 2,
	{ 2012: [[(date(2012, 7, 1), 50), (date(2012, 7, 2), 50)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
],)
def test_weirpool_check(EWR_info, iteration, flow, level, event, all_events, all_no_events, weirpool_type, level_change,total_event,
	expected_all_events, expected_event):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	gap_track = 0
	
	event, all_events, gap_track, total_event = evaluate_EWRs.weirpool_check(EWR_info, iteration, flow, level, event, all_events, gap_track, 
               water_years, total_event, flow_date, weirpool_type, level_change)
	assert event == expected_event
	assert all_events == expected_all_events

@pytest.mark.parametrize("EWR_info,flows,levels,weirpool_type,expected_all_events,expected_all_no_events", [
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10},
	 np.array([5]*2+[0]*363 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([8]*2+[0]*363 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'falling',
	 {2012: [[(date(2012, 7, 1), 5), (date(2012, 7, 2), 5)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1459]]}
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10},
	 np.array([5]*4+[0]*361 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([8]*3+[11]*1 +[0]*361 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'falling',
	 {2012: [[(date(2012, 7, 1), 5), (date(2012, 7, 2), 5), (date(2012, 7, 3), 5)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1458]]}
	 ),
	  ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10},
	 np.array([5]*3+[4]*1+[0]*361 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([8]*4 +[0]*361 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'falling',
	 {2012: [[(date(2012, 7, 1), 5), (date(2012, 7, 2), 5), (date(2012, 7, 3), 5)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1458]]}
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10},
	 np.array([5]*4+[0]*361 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([8] + [7.98] + [7.90] + [7.80] +[0]*361 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'falling',
	 {2012: [[(date(2012, 7, 1), 5), (date(2012, 7, 2), 5)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1459]]}
	 ),
	  ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10},
	 np.array(  [0]*350 + [5]*15 + 
	 			[5]*15 + [0]*350 + 
				[0]*365 + 
				[0]*366),
	np.array(   [0]*350 + [8]*15 + 
	 			[8]*15 + [0]*350 + 
				[0]*365 + 
				[0]*366),
		'falling',
	 {2012: [[(date(2013, 6, 16) + timedelta(days=i), 5) for i in range(15)]], 
	  2013: [[(date(2013, 7, 1) + timedelta(days=i), 5) for i in range(15)]], 
	  2014: [], 
	  2015: []},
	 {2012: [[350]], 2013: [], 2014: [], 2015: [[1081]]}
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10},
	 np.array(  [0]*345 + [5]*15 + [0]*5 + 
	 			[0]*345 + [5]*15 + [0]*5 +
				[0]*345 + [5]*15 + [0]*5 + 
				[0]*345 + [5]*15 + [0]*6),
	np.array(  [0]*345 + [11]*15 + [0]*5 + 
	 			[0]*345 + [11]*15 + [0]*5 +
				[0]*345 + [11]*15 + [0]*5 + 
				[0]*345 + [11]*15 + [0]*6),
		'raising',
	 {2012: [[(date(2013, 6, 11) + timedelta(days=i), 5) for i in range(15)]], 
	  2013: [[(date(2014, 6, 11) + timedelta(days=i), 5) for i in range(15)]], 
	  2014: [[(date(2015, 6, 11) + timedelta(days=i), 5) for i in range(15)]], 
	  2015: [[(date(2016, 6, 10) + timedelta(days=i), 5) for i in range(15)]]},
	 {2012: [[345]], 2013: [[350]], 2014: [[350]], 2015: [[350],[6]]}
	 ),
],)
def test_weirpool_calc(EWR_info, flows, levels, weirpool_type, expected_all_events, expected_all_no_events):
	"""
	1. test flow and level outside requirements at the same time
	2. test flow meeting and level not meeting requirements at the same time
	3. test flow not meeting and level meeting requirements at the same time 
	4. test drawdown not meeting requirements and event ending 
	5. test event across water year boundary
	6. test multiple events in a year in more than a year with raising type
	"""
	# non changing parameters
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	
	all_events, _ = evaluate_EWRs.weirpool_calc(EWR_info, flows, levels, water_years, weirpool_type, dates, masked_dates)


	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]




@pytest.mark.parametrize("EWR_info,iteration,level,level_change,event,all_events,all_no_events,total_event,expected_all_events,expected_event",
[
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     0,	
	 50,
	 0.04,
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     0,	
	 6,
	 0.04,
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 1), 6)],	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'max_duration': 11 ,'gap_tolerance':0},
     12,	
	 4,
	 0.04,
	[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(12)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'max_duration': 13 ,'gap_tolerance':0},
     12,	
	 4,
	 0.04,
	[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(12)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(12)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 14, 'duration': 14, 'max_duration': 300 ,'gap_tolerance':0},
     12,	
	 4,
	 0.04,
	[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(12)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	  ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'max_duration': 300 ,'gap_tolerance':0},
     11,	
	 6,
	 0.06,
	[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(10)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(10)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	   ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 7, 'duration': 7, 'max_duration': 300 ,'gap_tolerance':0},
     371,	
	 4,
	 0.04,
	[(date(2013, 6, 25) + timedelta(days=i), 6) for i in range(12)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [[(date(2013, 6, 25) + timedelta(days=i), 6) for i in range(12)]], 
		2014: [], 
		2015: []},
	[],	
	 ),
	    ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 7, 'duration': 7, 'max_duration': 300 ,'gap_tolerance':0},
     371,	
	 4,
	 0.04,
	[(date(2013, 6, 24) + timedelta(days=i), 6) for i in range(13)],
	{2012:[[(date(2013, 6, 24) + timedelta(days=i), 6) for i in range(6)]], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [[(date(2013, 6, 24) + timedelta(days=i), 6) for i in range(6)],
			[(date(2013, 6, 24) + timedelta(days=i), 6) for i in range(13)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	 	    ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 7, 'duration': 7, 'max_duration': 300 ,'gap_tolerance':0},
     372,	
	 4,
	 0.04,
	[(date(2013, 6, 24) + timedelta(days=i), 6) for i in range(14)],
	{2012:[[(date(2013, 6, 24) + timedelta(days=i), 6) for i in range(6)]], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [[(date(2013, 6, 24) + timedelta(days=i), 6) for i in range(6)]], 
		2013: [[(date(2013, 6, 24) + timedelta(days=i), 6) for i in range(14)]], 
		2014: [], 
		2015: []},
	[],	
	 ),
])
def test_level_check(EWR_info, iteration, level, level_change, event, all_events, all_no_events, total_event,
	expected_all_events, expected_event):
	'''
	1. Test level below threshold min
	2. Test level within min and max
	3. Test event length above max_duration
	4. Test event length within duration and max_duration
	5. Test event length below duration
	6. Test level drop above drawdown
	7. test last event part on last year below min duration and previous event year as well i.e. less than min duration
	8. test last event part on last year below min duration
	9. test last event part on last year equals or greater min duration
	'''
	# non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	level_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	gap_track = 0



	event, all_events, gap_track, total_event = evaluate_EWRs.level_check(EWR_info, iteration, level, level_change, 
																				event, all_events, 
																				gap_track ,water_years, total_event, level_date)

	assert event == expected_event

	for year in all_events:
		for i, event in enumerate(all_events[year]):
				assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,levels,expected_all_events,expected_all_no_events",[
	(
		{'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 14, 'duration': 14, 'max_duration': 100 ,'gap_tolerance':0},
	np.array(   [0]*350 + [6]*15 + 
	 			[6]*15 + [0]*350 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2013, 6, 16) + timedelta(days=i), 6) for i in range(15)]], 
	  2013: [[(date(2013, 6, 16) + timedelta(days=i), 6) for i in range(30)]], 
	  2014: [], 
	  2015: []},
	 {2012: [[350]], 2013: [], 2014: [], 2015: [[1081]]}
	),
	(
		{'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 14, 'duration': 14, 'max_duration': 400 ,'gap_tolerance':0},
	np.array(   [0]*350 + [6]*15 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2013, 6, 16) + timedelta(days=i), 6) for i in range(15)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[350]], 2013: [], 2014: [], 2015: [[1096]]}
	),
	(
		{'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 14, 'duration': 14, 'max_duration': 800 ,'gap_tolerance':0},
	np.array(   [6]*365 + 
	 			[6]*365 + 
				[6]*365 + 
				[6]*100 + [0]*266),
	 {2012: [[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(365)]], 
	  2013: [[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(365+365)]], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[731]]}
	),
	(
		{'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 14, 'duration': 14, 'max_duration': 800 ,'gap_tolerance':0},
	np.array( [6]*100+[0]*1+[6]*264  
			+ [6]*100+[0]*1+[6]*264 	
			+ [6]*100+[0]*1+[6]*264 
			+ [6]*100+[0]*1+[6]*265),
	 {  2012: [[(date(2012, 7, 1)+timedelta(days=i), 6) for i in range(100)],
		       [(date(2012, 10, 10)+timedelta(days=i), 6) for i in range(264)]], 
		2013: [[(date(2012, 10, 10)+timedelta(days=i), 6) for i in range(264+100)],
			   [(date(2013, 10, 10)+timedelta(days=i), 6) for i in range(264)]	
				], 
		2014: [[(date(2013, 10, 10)+timedelta(days=i), 6) for i in range(264+100)],
			   [(date(2014, 10, 10)+timedelta(days=i), 6) for i in range(264)]], 
		2015: [[(date(2014, 10, 10)+timedelta(days=i), 6) for i in range(264+100)],
			   [(date(2015, 10, 10)+timedelta(days=i), 6) for i in range(265)]]},
	 {2012: [[1]], 2013: [[1]], 2014: [[1]], 2015: [[1]]}
	),
	(
		{'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 5, 'duration': 5, 'max_duration': 100 ,'gap_tolerance':0},
	np.array(   [6]*8 + [5.98]*1 + [5.1]*1+ [0]*355 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012, 7, 1) + timedelta(days=i), 6) for i in range(8)] + [(date(2012, 7, 9) , 5.98)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1452]]}
	),
	(
		{'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 5, 'duration': 5, 'max_duration': 100 ,'gap_tolerance':0},
	np.array(   [0]*361 + [6]*4 +
	 			[6]*4 + [0]*361 + 
				[0]*365 + 
				[0]*366),
	 {2012: [], 
	  2013: [[(date(2013, 6, 27) + timedelta(days=i), 6) for i in range(8)]], 
	  2014: [], 
	  2015: []},
	 {2012: [[361]], 2013: [], 2014: [], 2015: [[1092]]}
	),
	(
		{'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 5, 'duration': 5, 'max_duration': 100 ,'gap_tolerance':0},
	np.array(   [0]*360 + [6]*5 +
	 			[6]*4 + [0]*361 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2013, 6, 26) + timedelta(days=i), 6) for i in range(5)],
		   [(date(2013, 6, 26) + timedelta(days=i), 6) for i in range(9)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[360]], 2013: [], 2014: [], 2015: [[1092]]}
	),
	(
		{'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 5, 'duration': 5, 'max_duration': 100 ,'gap_tolerance':0},
	np.array(   [0]*360 + [6]*5 +
	 			[6]*5 + [0]*360 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2013, 6, 26) + timedelta(days=i), 6) for i in range(5)]], 
	  2013: [[(date(2013, 6, 26) + timedelta(days=i), 6) for i in range(10)]], 
	  2014: [], 
	  2015: []},
	 {2012: [[360]], 2013: [], 2014: [], 2015: [[1091]]}
	),
],)
def test_lake_calc(EWR_info, levels, expected_all_events, expected_all_no_events):
	"""
	0: when event start and finish goes beyond boundary of 2 water years 
	   then : first year records the event up to the last of day of the water year
	   		  second year records the whole event
	1: when event start and finish the same water year on the last day of the water year 30-Jun 
	   then : first year records the event up to the last of day of the water year
	2: when event start and finish goes beyond boundary of 4 water years and finished within the 4th year. Event last for the whole
		duration od 1st, 2nd and 3rd year and partially on the 4th year. total days 1195 (365+365+365+100)
	   then : first year records the event up to the last of day of the water year (365 days)
	   		  second year records the event up to the last of day of the water year (365+365 days)
			  third year record nothing because total event up to end of year is greater than maximum duration (365+365+365 > 800)
			  forth year record nothing because total event up to end event is greater than maximum duration (365+365+365+100 > 800)
	3: having multiple events starting and finishing throughout all 4 water years and each year the first event drops below threshold for 1 day after 100 days.
		example flows:
			                    year 1: 100 days above threshold + 1 day below + 264 days above threshold
			                    year 2: 100 days above threshold + 1 day below + 264 days above threshold
			                    year 3: 100 days above threshold + 1 day below + 264 days above threshold
			                    year 4: 100 days above threshold + 1 day below + 265 days above threshold
	   then : first year records 1 event of 100 days and 1 event of 264 days
	   		  second year records 1 event of 264+100 days and 1 event of 264 days	
	   		  second year records 1 event of 264+100 days and 1 event of 264 days	
	   		  second year records 1 event of 264+100 days and 1 event of 264 days	
	4: test drawdown drop above the drawdown rate
	5: test last event part on last year below min duration and previous event year as well i.e. less than min duration
	6: test last event part on last year below min duration
	7: test last event part on last year equals or greater min duration
	"""
	
	# non changing parameters
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	
	all_events, _  = evaluate_EWRs.lake_calc(EWR_info, levels, water_years, dates, masked_dates)


	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize('gauge,PU,EWR,component,expected_result',[
	('409025','PU_0000253','NestS1','TriggerDay', '15'),
	('409025','PU_0000253','NestS1','TriggerMonth', '9'),
	('414203','PU_0000260','NestS1a','DrawDownRateWeek', '30%'),
],)
def test_component_pull_nest(gauge, PU, EWR, component, expected_result):
	'''
	1. Test pulling TriggerDay
	2. Test pulling TriggerMonth
	'''
	EWR_table, bad_EWRs = data_inputs.get_EWR_table('./unit_testing_files/MURRAY_MDBA_update_nest.csv')

	assert  evaluate_EWRs.component_pull(EWR_table, gauge, PU, EWR, component) == expected_result

@pytest.mark.parametrize("EWR_info,iteration,flow,flow_percent_change,event,all_events,all_no_events,total_event,expected_all_events,expected_event",
[
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': "10%", 
           'min_event': 30, 'duration': 30, 'gap_tolerance':0, "trigger_month": 9,
           "trigger_day": 15, 'start_month': 9, 'end_month': 12},
     0,	
	 4,
	 0,
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': "10%", 
           'min_event': 30, 'duration': 30, 'gap_tolerance':0, "trigger_month": 9,
           "trigger_day": 15, 'start_month': 9, 'end_month': 12},
     0,	
	 6,
	 0,
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 1), 6)],	
	 ),
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': "10%", 
           'min_event': 30, 'duration': 30 ,'gap_tolerance':0, "trigger_month": 9,
           "trigger_day": 15, 'start_month': 9, 'end_month': 12},
     100,	
	 4,
	 0,
	[(date(2012, 9, 9) + timedelta(days=i), 6) for i in range(30)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [[(date(2012, 9, 9) + timedelta(days=i), 6) for i in range(30)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': "10%", 
           'min_event': 30, 'duration': 30, 'gap_tolerance':0, "trigger_month": 9,
           "trigger_day": 15, 'start_month': 9, 'end_month': 12},
     96,	
	 4,
	 0,
	[(date(2012, 9, 9) + timedelta(days=i), 6) for i in range(26)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [[(date(2012, 9, 9) + timedelta(days=i), 6) for i in range(26)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	 	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': "10%", 
           'min_event': 30, 'duration': 30, 'gap_tolerance':0, "trigger_month": 9,
           "trigger_day": 15, 'start_month': 9, 'end_month': 12},
     98,	
	 6,
	 -40,
	[(date(2012, 9, 9) + timedelta(days=i), 6) for i in range(26)] + [(date(2012, 10, 5) , 10)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [[(date(2012, 9, 9) + timedelta(days=i), 6) for i in range(26)] + [(date(2012, 10, 5) , 10)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	 	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': "10%", 
           'min_event': 30, 'duration': 30, 'gap_tolerance':0, "trigger_month": 9,
           "trigger_day": 15, 'start_month': 9, 'end_month': 12},
     97,	
	 21,
	 -40,
	[(date(2012, 9, 9) + timedelta(days=i), 6) for i in range(26)] + [(date(2012, 10, 5) , 10)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 9, 9) + timedelta(days=i), 6) for i in range(26)] + [(date(2012, 10, 5) , 10)]+ [(date(2012, 10, 6) , 21)],	
	 ),

])
def test_nest_flow_check(EWR_info, iteration, flow, flow_percent_change, event, all_events,
							all_no_events, total_event, expected_all_events, expected_event):
	'''
	0. Test flow below threshold min
	1. Test flow above threshold min
	2. Test event length greater or equal min_event
	3. Test event length less than min_event
	4. Test level percent drop above max drawdown while flow is within range
	5. Test level percent drop above max drawdown while flow is above max flow
	'''
	# non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	gap_track = 0
	iteration_no_event = 0

	event, all_events, gap_track, total_event, iteration_no_event = evaluate_EWRs.nest_flow_check(EWR_info, iteration, flow, 
																		event, all_events, gap_track, 
                        													water_years, total_event, flow_date, flow_percent_change, iteration_no_event)
	
	assert event == expected_event

	for year in all_events:
		for i, event in enumerate(all_events[year]):
				assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,flows,expected_all_events,expected_all_no_events",[
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*76 + [10]*30 +[0]*259 +
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012, 9, 15) + timedelta(days=i), 10) for i in range(30)]],
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[76]], 2013: [], 2014: [], 2015: [[1355]]}
	),
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*76 + [10]*29 +[0]*260 +
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,15) + timedelta(days=i), 10) for i in range(29)]],
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1461]]}
	),
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*91 + [10]*31 +[0]*243 +
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1461]]} 
	),
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*76 + [10]*120 +[0]*169 +
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,15) + timedelta(days=i), 10) for i in range(108)]],
	  2013: [], 
	  2014: [], 
	  2015: []},
	 	 {2012: [[76]], 2013: [], 2014: [], 2015: [[1277]]}
	),
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*76 + [10]*30 + [8] +[0]*258 + # 20% Drawdown
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,15) + timedelta(days=i), 10) for i in range(30)] ],
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[76]], 2013: [], 2014: [], 2015: [[1355]]}
	),

	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*76 + [10]*30 + [9.5] +[0]*258 + # 5% Drawdown
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,15) + timedelta(days=i), 10) for i in range(30)] + [(date(2012,10,15) , 9.5)]],
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[76]], 2013: [], 2014: [], 2015: [[1354]]}
	),
	({'min_flow': 5, 'max_flow': 11, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*76 + [10]*30 + [15] + [12] + [0]*257 + # 26% Drawdown above max flow
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,15) + timedelta(days=i), 10) for i in range(30)] + [(date(2012,10,15) , 15)]+[(date(2012,10,16) , 12)]],
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[76]], 2013: [], 2014: [], 2015: [[1353]]}
	),
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*365 +
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1461]]} 
	),
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*76 + [10]*5 +[0] + [10]*5 +[0]*278 +
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,15) + timedelta(days=i), 10) for i in range(5)],[(date(2012,9,21) + timedelta(days=i), 10) for i in range(5)]],
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1461]]}
	),
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30, 
	   'gap_tolerance':0, "trigger_month": 9,"trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	np.array(   [0]*76 + [10]*5 +[0] +[10]*35 +[0]*248 +
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,15) + timedelta(days=i), 10) for i in range(5)], [(date(2012,9,21) + timedelta(days=i), 10) for i in range(35)]],
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[82]], 2013: [], 2014: [], 2015: [[1344]]}
	),
	({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': "10%", 'min_event': 30, 'duration': 30,
	  'gap_tolerance': 0, "trigger_month": 9, "trigger_day": 15, 'start_month': 9, 'end_month': 12, 'end_day': None},
	 np.array([0] * 71 + [10]*35 + [0]*259 +   # event begins before trigger day
			  [0] * 365 +
			  [0] * 365 +
			  [0] * 366),
	 {2012: [[(date(2012, 9, 15) + timedelta(days=i), 10) for i in range(30)]],
	  2013: [],
	  2014: [],
	  2015: []},
	 {2012: [[76]], 2013: [], 2014: [], 2015: [[1355]]}
	 ),
	
],)  
def test_nest_calc_percent_trigger(EWR_info, flows, expected_all_events, expected_all_no_events):
	"""
	0: Event triggered and above min_event 
	1: Event triggered and below min_event
	2: Event miss the trigger - Failing
	3: Event triggered and extend to the cut date
	4: Event triggered drawdown is above drawdown rate and flow within the range
	5: Event triggered drawdown is below drawdown rate and flow within the range
	6: Event triggered drawdown is above drawdown rate and flow is above the range
	7: Flow never reaches min flow threshold - Failing
	8: Event Start-Finish and Start-Finish within the trigger window - Failing
	9: Event Start-Finish and Fail length and another Start within the trigger window and Succeed 
	
	"""
	# non changing parameters
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	
	all_events, _  = evaluate_EWRs.nest_calc_percent_trigger(EWR_info, flows, water_years, dates)
								
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

@pytest.mark.parametrize("EWR_info,iteration,flow,level,event,all_events,all_no_events,weirpool_type,levels,total_event,expected_all_events,expected_event",
[
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'gap_tolerance':0,"drawdown_rate_week" : "0.3"},
     2,	
	 10,
	 5,
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 "raising",
	 np.array(  [5] + [5] + [5] + [0]*365 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 3), 10)],	
	 ),
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'gap_tolerance':0,"drawdown_rate_week" : "0.3"},
     2,	
	 3,
	 5,
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 "raising",
	 np.array(  [5] + [5] + [5] + [0]*365 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'drawdown_rate': 0.04, 
	'min_event': 10, 'duration': 10, 'gap_tolerance':0,"drawdown_rate_week" : "0.3"},
     6,	
	 10,
	 5,
	[(date(2012,9,1) + timedelta(days=i), 5) for i in range(6)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 "raising",
	 np.array(  [5] + [5] + [5] + [5] + [5] + [5] + [4.5] + [0]*358 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 0,
	{ 2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(6)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ),
],)
def test_nest_weirpool_check(EWR_info, iteration, flow, level, event, all_events, all_no_events, weirpool_type, levels,total_event,
	expected_all_events, expected_event):
	"""
	0: level, flow, and drawdown meets requirement
	1: level and drawdown meets requirement and flow does not
	2: level and flow meets requirement and drawdown  does not
	"""

	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	gap_track = 0
	
	event, all_events, gap_track, total_event = evaluate_EWRs.nest_weirpool_check(EWR_info, iteration, flow, level, 
									event, all_events, gap_track, 
               						water_years, total_event, flow_date, weirpool_type, levels)
	assert event == expected_event
	assert all_events == expected_all_events

#TODO: delete the references to the minimum level requirements - they are no longer used.
@pytest.mark.parametrize("EWR_info,flows,levels,weirpool_type,expected_all_events,expected_all_no_events", [
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10, "drawdown_rate_week" : "0.3",
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([5]*2+[0]*363 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([8]*2+[0]*363 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'raising',
	 {2012: [], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1461]]}
	 ),
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10, "drawdown_rate_week" : "0.3",
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([0]*62+ [5]*30+ [0]*273 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([0]*62+ [8]*30+ [0]*273 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'raising',
	 {2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(30)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[62]], 2013: [], 2014: [], 2015: [[1369]]}
	 ),
	({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10, "drawdown_rate_week" : "0.3",
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([0]*62+ [4]*30+ [0]*273 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([0]*62+ [8]*30+ [0]*273 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'raising',
	 {2012: [], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1461]]}
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10, "drawdown_rate_week" : "0.3",
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([0]*62+ [5]*30+ [0]*273 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([0]*62+ [3]*30+ [0]*273 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'raising',
	 {2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(30)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[62]], 2013: [], 2014: [], 2015: [[1369]]}
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10, "drawdown_rate_week" : "0.3",
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([0]*62+ [5]*30+ [5]*7+ [0]*266 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([0]*62+ [8]*30+ [8]*5+ [7.9] + [7.6] + [0]*266 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
		'raising',
	 {2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(36)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[62]], 2013: [], 2014: [], 2015: [[1363]]}
	 ),
	  ({'min_flow': 5, 'max_flow': 20, 'min_level': 5, 'max_level': 10, 'gap_tolerance':0,
	 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10, "drawdown_rate_week" : "0.3",
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array(  [5]*365 + 
	 			[5]*365 + 
				[5]*365 + 
				[5]*366),
	np.array(   [8]*365 +
	 			[8]*365 + 
				[8]*365 + 
				[8]*366),
		'raising',
	 {2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(122)]], 
	  2013: [[(date(2013,9,1) + timedelta(days=i), 5) for i in range(122)]], 
	  2014: [[(date(2014,9,1) + timedelta(days=i), 5) for i in range(122)]], 
	  2015: [[(date(2015,9,1) + timedelta(days=i), 5) for i in range(122)]]},
	 {2012: [[62]], 2013: [[243]], 2014: [[243]], 2015: [[243], [182]]}
	 ),
],)
def test_nest_calc_weirpool(EWR_info, flows, levels, weirpool_type, expected_all_events, expected_all_no_events):
	"""
	0: test event meeting requirements outside time window
	1: test event meeting requirements inside time window
	2: test event meeting requirements inside time window flow not meeting requirements
	3: test event meeting requirements inside time window level not meeting requirements - this should still pass as level no longer considered
	4: test event meeting requirements inside time window drawdown not meeting requirements
	5. test meeting requirements all the time over water years. save any events at year boundary.
	"""
	# non changing parameters
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	df_F = pd.DataFrame(index=dates)
	masked_dates = evaluate_EWRs.mask_dates(EWR_info, df_F)
	
	all_events, _  = evaluate_EWRs.nest_calc_weirpool(EWR_info, flows, levels, water_years, dates, masked_dates, weirpool_type )

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,events,expected_result",[
	({'min_event': 10},
	{2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(10)]], 
	  2013: [[(date(2013,9,1) + timedelta(days=i), 5) for i in range(5)]], 
	  2014: [[(date(2014,9,1) + timedelta(days=i), 5) for i in range(10)],[(date(2014,10,1) + timedelta(days=i), 5) for i in range(5)]], 
	  2015: [[(date(2015,9,1) + timedelta(days=i), 5) for i in range(15)]]},

	  {2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(10)]], 
	  2013: [], 
	  2014: [[(date(2014,9,1) + timedelta(days=i), 5) for i in range(10)]], 
	  2015: [[(date(2015,9,1) + timedelta(days=i), 5) for i in range(15)]]}
	  ),
],)
def test_filter_min_events(EWR_info,events,expected_result):
	result = evaluate_EWRs.filter_min_events(EWR_info, events)
	assert result == expected_result

# TODO

@pytest.mark.parametrize("multigauge,expected_result",[(
	True, "multigauge"
	),
	(False, "single"
	),
	(True, "multigauge"
	),
])
def test_get_gauge_calc_type(multigauge,expected_result):
	calc_type = evaluate_EWRs.get_gauge_calc_type(multigauge)
	assert calc_type == expected_result

@pytest.mark.parametrize("ewr_code,prefixes,expected_result",[
	("CF_P", ["CF","LF"],"CF"),
	("LF_S", ["CF","LF"],"LF"),
	("Gluble_Ubble", ["CF","LF"],"unknown"),
])
def test_get_ewr_prefix(ewr_code, prefixes, expected_result):
	result = evaluate_EWRs.get_ewr_prefix(ewr_code, prefixes)
	assert result == expected_result

@pytest.mark.parametrize("function_name, expected_result",[
	('ctf_handle','ctf_handle'),
	('ctf_handle_multi','ctf_handle_multi'),
])
def test_get_handle_function(function_name, expected_result):
	result = evaluate_EWRs.get_handle_function(function_name)
	assert result.__name__ == expected_result

@pytest.mark.parametrize("args,function_name,expected_result",[
	({"PU": "PU" , 
	"gauge": "gauge", 
	"EWR": "EWR", 
	"EWR_table": "EWR_table", 
	"df_F": "df_F", 
	"df_L": "df_L",
	"PU_df": "PU_df", 
	},
		'ctf_handle', 
	{"PU": "PU" , 
	"gauge": "gauge", 
	"EWR": "EWR", 
	"EWR_table": "EWR_table", 
	"df_F": "df_F", 
	"PU_df": "PU_df", 
	}),
	({"PU": "PU" , 
	"gauge": "gauge", 
	"EWR": "EWR", 
	"EWR_table": "EWR_table", 
	"df_F": "df_F", 
	"df_L": "df_L",
	"PU_df": "PU_df", 
	},
		'level_handle', 
	{"PU": "PU" , 
	"gauge": "gauge", 
	"EWR": "EWR", 
	"EWR_table": "EWR_table", 
	"df_L": "df_L", 
	"PU_df": "PU_df", 
	}),
])
def test_build_args(args, function_name, expected_result):
	function = evaluate_EWRs.HANDLING_FUNCTIONS[function_name]
	result = evaluate_EWRs.build_args(args, function)
	assert result == expected_result


@pytest.mark.parametrize("level_change,EWR_info,expected_result",[
	(0.1, {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, True),
	(0.11, {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, False),
	(0.2, {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, False),
	(-0.45, {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, True),
	(-0.5, {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, True),
	(-0.6, {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, False),
])
def test_check_daily_level_change(level_change,EWR_info,expected_result):
	result = evaluate_EWRs.check_daily_level_change(level_change,EWR_info)
	assert result == expected_result

@pytest.mark.parametrize("levels,EWR_info,iteration,event_length,expected_result",[
	([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, 5, 6, True),
	([0.1,0.2,0.3,0.4,0.5,0.8,0.7,0.8,0.9,1.0], {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, 5, 6, True),
	([0.1,0.2,0.3,0.4,0.5,0.9,0.7,0.8,0.9,1.0], {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, 5, 6, False),
	([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, 9, 10, True),
	([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.2], {'drawdown_rate': 0.5 ,'max_level_raise': 0.1}, 9, 10, False),
	([1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1], {'drawdown_rate': 0.1 ,'max_level_raise': 0.1}, 9, 10, True),
	([1.0,0.9,0.8,0.9,0.6,0.5,0.4,0.3,0.2,0.1], {'drawdown_rate': 0.1 ,'max_level_raise': 0.1}, 9, 10, False),
])
def test_check_weekly_level_change(levels, EWR_info, iteration, event_length, expected_result):
	result = evaluate_EWRs.check_weekly_level_change(levels, EWR_info, iteration, event_length)
	assert result == expected_result



@pytest.mark.parametrize("EWR_info,iteration,flow,level,event,all_events,all_no_events,level_change,levels,total_event,expected_all_events,expected_event",
[
	 ({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': 0.1, 'max_level_raise': 0.1,
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     6,	
	 5,
	 5,
	[(date(2012,7,1) + timedelta(days=i), 5) for i in range(6)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0.4,
	 np.array(  [5] + [5] + [5] + [5] + [5] + [5] + [4.5] + [0]*358 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012,7,1) + timedelta(days=i), 5) for i in range(7)] ,	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': 0.1, 'max_level_raise': 0.1,
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     6,	
	 5,
	 5,
	[(date(2012,7,1) + timedelta(days=i), 5) for i in range(6)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 1,
	 np.array(  [5] + [5] + [5] + [5] + [5] + [5] + [6] + [0]*358 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 0,
	{ 2012: [[(date(2012,7,1) + timedelta(days=i), 5) for i in range(6)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[] ,	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': 0.1, 'max_level_raise': 0.1,
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     6,	
	 5,
	 5,
	[(date(2012,7,1) + timedelta(days=i), 5) for i in range(6)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 -1,
	 np.array(  [5] + [5] + [5] + [5] + [5] + [5] + [4] + [0]*358 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 0,
	{ 2012: [[(date(2012,7,1) + timedelta(days=i), 5) for i in range(6)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[] ,	
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'drawdown_rate': 0.1, 'max_level_raise': 0.1,
	'min_event': 10, 'duration': 10, 'gap_tolerance':0},
     6,	
	 4,
	 5,
	[(date(2012,7,1) + timedelta(days=i), 5) for i in range(6)],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 -1,
	 np.array(  [5] + [5] + [5] + [5] + [5] + [5] + [4] + [0]*358 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 0,
	{ 2012: [[(date(2012,7,1) + timedelta(days=i), 5) for i in range(6)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[] ,	
	 ),
],)
def test_flow_level_check(EWR_info, iteration, flow, level, event, all_events, all_no_events, level_change, levels,total_event,
	expected_all_events, expected_event):
	""" Test the flow level check function.
	1. Test happy path level change and flow min is above threshold
	2. Test Level change increase is above max allowed
	3. Test Level change decrease is above max allowed
	4. Test flow below min threshold
	"""
	
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	gap_track = 0
	
	event, all_events, gap_track, total_event = evaluate_EWRs.flow_level_check(EWR_info, iteration, flow, level, 
									event, all_events, gap_track, 
               						water_years, total_event, flow_date, level_change ,levels)
	assert event == expected_event
	assert all_events == expected_all_events


@pytest.mark.parametrize("EWR_info,flows,levels,expected_all_events,expected_all_no_events", [
	({'min_flow': 5, 'max_flow': 20, 'gap_tolerance':0,
	 'drawdown_rate': 0.1, 'max_level_raise': 0.1, 'min_event': 10, 'duration': 10,
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([0]*31+ [0]*31 + [5]*7 + [0]*296 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([0]*31+ [0]*31 + [1]*7 + [0]*296 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(7)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[62]], 2013: [], 2014: [], 2015: [[1391]]}
	 ),
	 ({'min_flow': 5, 'max_flow': 20, 'gap_tolerance':0,
	 'drawdown_rate': 0.1, 'max_level_raise': 0.1, 'min_event': 10, 'duration': 10,
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([0]*31+ [0]*31 + [4]*7 + [0]*296 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([0]*31+ [0]*31 + [1]*7 + [0]*296 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [], 2013: [], 2014: [], 2015: [[1460]]}
	 ),
	  ({'min_flow': 5, 'max_flow': 20, 'gap_tolerance':0,
	 'drawdown_rate': 0.1, 'max_level_raise': 0.1, 'min_event': 10, 'duration': 10,
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([0]*31+ [0]*31 + [5]*7 + [0]*296 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([0]*31+ [0]*31 + [1]*6+ [2] + [0]*296 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(6)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[62]], 2013: [], 2014: [], 2015: [[1392]]}
	 ),
	   ({'min_flow': 5, 'max_flow': 20, 'gap_tolerance':0,
	 'drawdown_rate': 0.1, 'max_level_raise': 0.1, 'min_event': 10, 'duration': 10,
	 'start_month': 9, 'end_month': 12, 'start_day': None, 'end_day': None},
	 np.array([0]*31+ [0]*31 + [5]*7 + [0]*296 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	np.array([0]*31+ [0]*31 + [1]*6+ [0] + [0]*296 + 
	 			[0]*365 + 
				[0]*365 + 
				[0]*366),
	 {2012: [[(date(2012,9,1) + timedelta(days=i), 5) for i in range(6)]], 
	  2013: [], 
	  2014: [], 
	  2015: []},
	 {2012: [[62]], 2013: [], 2014: [], 2015: [[1392]]}
	 ),
])
def test_flow_level_calc(EWR_info, flows, levels, expected_all_events, expected_all_no_events):
	""" Test the flow level calc function.
	1. Test happy path level change and flow min is above threshold
	2. Test flow below min threshold	
	3. Test Level change increase is above max allowed
	4. Test Level change decrease is above max allowed
	"""
	
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	df_F = pd.DataFrame(index=dates)
	masked_dates = evaluate_EWRs.mask_dates(EWR_info, df_F)
	
	all_events, _ = evaluate_EWRs.flow_level_calc(EWR_info, flows, levels, water_years, dates, masked_dates )


	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("year,expected_result",[
	(2000,True),
	(2004,True),
	(2005,False),
	(1900,False),
])
def test_is_leap_year(year,expected_result):
	result = evaluate_EWRs.is_leap_year(year)
	assert result == expected_result

@pytest.mark.parametrize("month,year,expected_result",[
	(1,2000,2000),
	(7,2000,1999),
	(6,2000,2000),
])
def test_get_water_year(month,year,expected_result):
	result = evaluate_EWRs.get_water_year(month, year)
	assert result == expected_result

@pytest.mark.parametrize("month,year,expected_result",[
	(1,2000,31),
	(2,2023,28),
	(2,2024,29),
])
def test_get_days_in_month(month, year, expected_result):
	result = evaluate_EWRs.get_days_in_month(month, year)
	assert result == expected_result


@pytest.mark.parametrize("iteration_date,data_length",[
	(date(2013,6,30), 365),
	(date(2016,6,30), 366),
])
def test_filter_last_year_flows(iteration_date,data_length):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_data = np.array([0]*365 + [0]*365 + [0]*365 + [0]*366)
	flows_series = pd.Series(flows_data, index=dates)
	flows = evaluate_EWRs.filter_last_year_flows(flows_series, iteration_date)
	assert len(flows) == data_length
	
@pytest.mark.parametrize("iteration_date,data_length",[
	(date(2013,6,30), 365),
	(date(2014,6,30), 2*365),
	(date(2015,6,30), 3*365),
	(date(2016,6,30), 2*365+366),
])
def test_filter_last_three_years_flows(iteration_date, data_length):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_data = np.array([0]*365 + [0]*365 + [0]*365 + [0]*366)
	flows_series = pd.Series(flows_data, index=dates)
	flows = evaluate_EWRs.filter_last_three_years_flows(flows_series, iteration_date)
	assert len(flows) == data_length


@pytest.mark.parametrize("flows,start,end,flow_date,expected_start,expected_end",[
	(np.array([0]*365 + 
			  [0]*365 + 
			  [0]*365 + 
			  [0]*366),
			  9,
			  9, 
			  date(2013,6,30),
			  f'{2012}-09-01',
			  f'{2012}-09-30'
			  ),
	(np.array([0]*365 + 
			  [0]*365 + 
			  [0]*365 + 
			  [0]*366),
			  9,
			  12, 
			  date(2013,6,30),
			  f'{2012}-09-01',
			  f'{2012}-12-31'
			  ),
	(np.array([0]*365 + 
			  [0]*365 + 
			  [0]*365 + 
			  [0]*366),
			  9,
			  1, 
			  date(2013,6,30),
			  f'{2012}-09-01',
			  f'{2013}-01-31'
			  ),
])
def test_filter_timing_window_std(flows, start, end, flow_date, expected_start, expected_end):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_series = pd.Series(flows, index=dates)
	result = evaluate_EWRs.filter_timing_window_std(flows_series, flow_date, start, end)
	assert result.index[0].strftime('%Y-%m-%d') == expected_start
	assert result.index[-1].strftime('%Y-%m-%d') == expected_end


@pytest.mark.parametrize("flows,start,end,flow_date,expected_start,expected_end",[
	(np.array([0]*365 + 
			  [0]*365 + 
			  [0]*365 + 
			  [0]*366),
			  1,
			  8, 
			  date(2013,6,30),
			   f'{2013}-01-01',
			  f'{2012}-08-31'
			  ),
	(np.array([0]*365 + 
			  [0]*365 + 
			  [0]*365 + 
			  [0]*366),
			  2,
			  8, 
			  date(2013,6,30),
			   f'{2013}-02-01',
			  f'{2012}-08-31'
			  ),
])
def test_filter_timing_window_non_std(flows, start, end, flow_date, expected_start, expected_end):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_series = pd.Series(flows, index=dates)
	result = evaluate_EWRs.filter_timing_window_non_std(flows_series, flow_date, start, end)
	assert result.index[0].strftime('%Y-%m-%d') == expected_start
	assert result.index[-1].strftime('%Y-%m-%d') == expected_end


@pytest.mark.parametrize("EWR_info,flows,event,all_events,all_no_events,expected_all_events,expected_event,flow_date",
[
	({"low_release_window_start":1, 
	  "low_release_window_end":8,
	  "high_release_window_start":9, 
	  "high_release_window_end":12,
	  'EWR_code': "CLLMM1a",
	  'annual_barrage_flow': 2000000,
	  'three_years_barrage_flow': 6000000
	  },
     np.array([5000]*62 + [16500]*122 + [5000]*181 + 
			  [5000]*62 + [16500]*122 + [5000]*181 +
			  [5000]*62 + [16500]*122 + [5000]*181 +
			  [5000]*62 + [16500]*122 + [5000]*182 ),	
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	{ 2012: [[(date(2013,6,30) , 3228000)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2013,6,30) , 3228000)],
	date(2013,6,30)	
	 ),
	 ({"low_release_window_start":1, 
	  "low_release_window_end":8,
	  "high_release_window_start":9, 
	  "high_release_window_end":12,
	  'EWR_code': "CLLMM1a",
	  'annual_barrage_flow': 2000000,
	  'three_years_barrage_flow': 6000000
	  },
     np.array([5000]*62 + [5000]*122 + [5000]*181 + 
			  [5000]*62 + [5000]*122 + [5000]*181 +
			  [5000]*62 + [5000]*122 + [5000]*181 +
			  [5000]*62 + [5000]*122 + [5000]*182 ),	
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],
	date(2013,6,30)	
	 ),
	({"low_release_window_start":1, 
	  "low_release_window_end":8,
	  "high_release_window_start":9, 
	  "high_release_window_end":12,
	  'EWR_code': "CLLMM1b",
	  'annual_barrage_flow': 2000000,
	  'three_years_barrage_flow': 6000000
	  },
     np.array([5000]*62 + [16500]*122 + [5000]*181 + 
			  [5000]*62 + [16500]*122 + [5000]*181 +
			  [5000]*62 + [16500]*122 + [5000]*181 +
			  [5000]*62 + [16500]*122 + [5000]*182 ),	
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [[(date(2015,6,30) , 9684000)]], 
		2015: []},
	[(date(2015,6,30) , 9684000)],
	date(2015,6,30)	
	 ),
])
def test_barrage_flow_check(EWR_info,flows,event,all_events,all_no_events,expected_all_events,expected_event,flow_date):

	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_series = pd.Series(flows, index=dates)

	event, all_events = evaluate_EWRs.barrage_flow_check(EWR_info, flows_series, event, all_events, flow_date)
	
	assert event == expected_event

	for year in all_events:
		for i, event in enumerate(all_events[year]):
				assert event == expected_all_events[year][i]



@pytest.mark.parametrize("EWR_info,flows,expected_all_events",[
	({"low_release_window_start":1, 
	  "low_release_window_end":8,
	  "high_release_window_start":9, 
	  "high_release_window_end":12,
	  'EWR_code': "CLLMM1_a",
	  'annual_barrage_flow': 2000000,
	  'three_years_barrage_flow': 6000000,
	  'duration': 1
	  },
     np.array([5000]*62 + [16500]*122 + [5000]*181 + 
			  [5000]*62 + [16500]*122 + [5000]*181 +
			  [5000]*62 + [16500]*122 + [5000]*181 +
			  [5000]*62 + [16500]*122 + [5000]*182 ),
			  { 2012:[[(date(2013,6,30) , 3228000)]], 
				2013:[[(date(2014,6,30) , 3228000)]], 
				2014:[[(date(2015,6,30) , 3228000)]], 
				2015:[[(date(2016,6,30) , 3233000)]]}),
	({"low_release_window_start":1, 
	  "low_release_window_end":8,
	  "high_release_window_start":9, 
	  "high_release_window_end":12,
	  'EWR_code': "CLLMM1_b",
	  'annual_barrage_flow': 2000000,
	  'three_years_barrage_flow': 6000000,
	  'duration': 1
	  },
     np.array([5000]*62 + [16500]*122 + [5000]*181 + 
			  [5000]*62 + [16500]*122 + [5000]*181 +
			  [5000]*62 + [16500]*122 + [5000]*181 +
			  [5000]*62 + [16500]*122 + [5000]*182 ),
			  { 2012:[], 
				2013:[], 
				2014:[[(date(2015,6,30) , 9684000)]], 
				2015:[[(date(2016,6,30) , 9689000)]]}),
])
def test_barrage_flow_calc(EWR_info,flows,expected_all_events):

	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_series = pd.Series(flows, index=dates)
	all_events, durations = evaluate_EWRs.barrage_flow_calc(EWR_info, flows_series, water_years, dates)

	for year in all_events:
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("values,expected_result",[
	(
	[1,2,3,4,5,6],
	[1, 1, 1, 1, 1]
	),
	(
	[6,5,4,3,2,1],
	[-1, -1, -1, -1, -1]
	),
])
def test_calculate_change(values, expected_result):
	result = evaluate_EWRs.calculate_change(values)
	assert result == expected_result

@pytest.mark.parametrize("values,expected_result",[
	(
	[1,2,3,4,5,6],
	[2.0, 3.0, 4.0, 5.0]
	),
	(
	[-1,-2,-3,-4,-5,-6],
	[-2.0, -3.0, -4.0, -5.0]
	),
])
def test_rolling_average(values, expected_result):
	result = evaluate_EWRs.rolling_average(values, 3)
	assert result == expected_result

@pytest.mark.parametrize("flows,EWR_info,interation,mode,period,expected_result",[
	( [3000]*80 + [3000+460] + [3000+460+460] + [3000+460+460+460] + [3000+460+460+460]*80,
	  {"max_level_raise": 450,
       "drawdown_rate": 200},
	  90,
	  "backwards",
	  3,
	  False
	),
	( [3000]*80 + [3000+450] + [3000+450+450] + [3000+450+450+450] + [3000+450+450+450]*80,
	  {"max_level_raise": 450,
       "drawdown_rate": 200},
	  90,
	  "backwards",
	  3,
	  True
	),
	( [3000]*95 + [3000-200] + [3000-200-200] + [3000-200-200-200] + [3000-200-200-200-200]*80,
	  {"max_level_raise": 450,
       "drawdown_rate": 200},
	  90,
	  "forwards",
	  3,
	  True
	),
	( [3000]*90 + [3000-210] + [3000-210-210] + [3000-210-210-210] + [3000-210-210-210-210]*80,
	  {"max_level_raise": 450,
       "drawdown_rate": 200},
	  90,
	  "forwards",
	  3,
	  False
	),
	( [3000]*90 + [3000+210] + [3000+210+210] + [3000+210+210+210] + [3000+210+210+210+210]*80,
	  {"max_level_raise": 450,
       "drawdown_rate": 200},
	  90,
	  "forwards",
	  3,
	  True
	),
])
def test_check_period_flow_change(flows, EWR_info, interation, mode, period, expected_result):
	result = evaluate_EWRs.check_period_flow_change(flows, EWR_info, interation, mode, period)
	assert result == expected_result


def test_get_last_year_peak():
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_data = np.array([0]*300 + [400] + [0]*64  + [0]*365 + [0]*365 + [0]*366)
	flows_series = pd.Series(flows_data, index=dates)
	result = evaluate_EWRs.get_last_year_peak(flows_series, date(2013,6,30))
	assert result == 400

def test_get_last_year_low():
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_data = np.array([400]*300 + [100] + [400]*64  + [0]*365 + [0]*365 + [0]*366)
	flows_series = pd.Series(flows_data, index=dates)
	result = evaluate_EWRs.get_last_year_low(flows_series, date(2013,6,30))
	assert result == 100


@pytest.mark.parametrize("EWR_info,levels,last_year_peak,expected_result",[
	( {'peak_level_window_start':9,
	     		'peak_level_window_end':12},
		np.array([100]*70 + [500] + [100]*294),
		500.,
		True
	),
	(  {'peak_level_window_start':9,
	     		'peak_level_window_end':12},
		np.array([100]*40 + [500] + [100]*324),
		500.,
		False
	),
])
def test_last_year_peak_within_window(EWR_info, levels, last_year_peak, expected_result):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2013-06-30', '%Y-%m-%d'))
	flows_series = pd.Series(levels, index=dates)
	result = evaluate_EWRs.last_year_peak_within_window(last_year_peak, flows_series,  EWR_info)
	assert result == expected_result

@pytest.mark.parametrize("EWR_info,levels,last_year_peak,expected_result",[
	( 			{'low_level_window_start':1,
	     		'low_level_window_end':5},
		np.array([100]*70 + [500] + [100]*294),
		100.,
		True
	),
	( 			{'low_level_window_start':1,
	     		'low_level_window_end':5},
		np.array([300]*180 + [400]*155 + [100]*30),
		100.,
		False
	),
])
def test_last_year_low_within_window(EWR_info, levels, last_year_peak, expected_result):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2013-06-30', '%Y-%m-%d'))
	flows_series = pd.Series(levels, index=dates)
	result = evaluate_EWRs.last_year_low_within_window(last_year_peak, flows_series,  EWR_info)
	assert result == expected_result


@pytest.mark.parametrize("EWR_info,iteration,levels_data,event,all_events,all_no_events,total_event,expected_all_events,expected_event", [
	(
	 {'max_level': 5,
     'min_level': 3 },
     0,	
	 np.array([4]*1 +[0]*364 + [0]*365 + [0]*365 + [0]*366),
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012,7,1) , 4)],
	),
	(
	 {'max_level': 5,
	 'min_level': 5 },
     0,	
	 np.array([4]*2 +[0]*363 + [0]*365 + [0]*365 + [0]*366),
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],
	),
])
def test_coorong_check(EWR_info, iteration, levels_data, event, all_events,
			     all_no_events, total_event, expected_all_events, expected_event):
	'''
	1. Meet the level threshold
	2. Do not meet the level threshold
	'''
	# non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	levels_series = pd.Series(levels_data, index=dates)
	level_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	
	event, all_events = evaluate_EWRs.coorong_check(EWR_info, levels_series, event, all_events, 
										 level_date, water_years, iteration, total_event)
	
	assert event == expected_event

	for year in all_events:
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,levels_data,expected_all_events", [
	(
	{'min_level': 5 ,'max_level': 5 ,'duration': 10},
	np.array([5]*2 +[0]*363 + [0]*365 + [0]*365 + [0]*366),
	{2012: [[(date(2012,7,1) + timedelta(days=i), 5) for i in range(2)]], 
	  2013: [], 
	  2014: [], 
	  2015: []}

	)
])
def test_coorong_level_calc(EWR_info, levels_data, expected_all_events):
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	levels = pd.Series(levels_data, index=dates)

	all_events, _ = evaluate_EWRs.coorong_level_calc(EWR_info, levels, water_years, dates, masked_dates)


	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
				assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,iteration,levels_data,event,all_events,all_no_events, expected_all_events, expected_event",[
	(
	{'max_level': 500,
		'min_level': 200,
		'peak_level_window_start': 9,
		'peak_level_window_end':12,
		'low_level_window_start':1,
		'low_level_window_end':5 },
     365,	
	 np.array( [400]*70 + [600] + [400]*115 + [220]*179 + 
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	{ 2012: [[(date(2013, 7, 1), 600)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2013, 7, 1), 600)],
	)
])
def test_lower_lakes_level_check(EWR_info, iteration, levels_data, event, all_events,
			     all_no_events, expected_all_events, expected_event):

	# non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	levels_series = pd.Series(levels_data, index=dates)
	level_date = dates[iteration]

	event, all_events = evaluate_EWRs.lower_lakes_level_check(EWR_info, levels_series, event, all_events, level_date)
	
	assert event == expected_event

	for year in all_events:
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,levels_data,expected_all_events",[
	(
		{'max_level': 500,
		'min_level': 200,
		'peak_level_window_start': 9,
		'peak_level_window_end':12,
		'low_level_window_start':1,
		'low_level_window_end':5,
		 'duration':10 },
	np.array([400]*70 + [600] + [400]*115 + [220]*179 + 
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012: [[(date(2013, 6, 30), 600)]], 
	  2013: [], 
	  2014: [], 
	  2015: []}

	)
])
def test_lower_lakes_level_calc(EWR_info, levels_data, expected_all_events):

	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	levels = pd.Series(levels_data, index=dates)

	all_events, _ = evaluate_EWRs.lower_lakes_level_calc(EWR_info, levels, water_years, dates, masked_dates)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,iteration,flows_data,event,all_events,all_no_events,total_event,expected_all_events,expected_event",[
	(
	 {'gap_tolerance': 0 ,
      'min_flow' : 70,
      "max_level_raise": 100,
      'drawdown_rate': 50,
	   'duration':10 
    },
     0,	
	 np.array([6]*2 +[0]*363 + [0]*365 + [0]*365 + [0]*366),
	[],
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{2012:[],
	 2014:[],
	 2013: [], 
	 2015:[]},
	 0,
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],
	)
])
def test_flow_check_rise_fall(EWR_info, iteration, flows_data, event, all_events,
			     all_no_events, total_event, expected_all_events, expected_event):

	# non changing variable
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_series = pd.Series(flows_data, index=dates)
	flow_date = dates[iteration]
	flow = flow_series[iteration]
	gap_track = 0
	
	event, all_events, _, _ = evaluate_EWRs.flow_check_rise_fall(EWR_info, iteration, flow, event, all_events, gap_track, 
               water_years, total_event, flow_date, flow_series)
	
	assert event == expected_event

	for year in all_events:
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,flows_data,expected_all_events",[
	(
	   {'gap_tolerance': 0 ,
      'min_flow' : 70,
      "max_level_raise": 100,
      'drawdown_rate': 50,
	   'duration':10 
    },
	np.array([60]*60 + [71]*2 + [40]*303 +
			 [0]*365 + 
			 [0]*365 + 
			 [0]*366),
	{2012:[[(date(2012,8,30) + timedelta(days=i), 71) for i in range(2)]], 
	 2013:[], 
	 2014:[], 
	 2015:[]}
	
	),
	(
	   {'gap_tolerance': 0 ,
      'min_flow' : 150,
      "max_level_raise": 10,
      'drawdown_rate': 5,
	   'duration':10 
    },
	np.array([60]*55 + [60+11] + [60+11+11] + [60+11+11+11] + [60+11+11+11+11] + [60+11+11+11+11+11] + [150]*2 + [40]*303 +
			 [0]*365 + 
			 [0]*365 + 
			 [0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]}
	
	),
	(
	   {'gap_tolerance': 0 ,
      'min_flow' : 150,
      "max_level_raise": 10,
      'drawdown_rate': 5,
	   'duration':10 
    },
	np.array([60]*55 + [60+9] + [60+9+9] + [60+9+9+9] + [60+9+9+9+9] + [60+9+9+9+9+9] +
	  					[150]*2 + 
						[60-6] + [60-6-6] + [60-6-6-6] + [60-6-6-6-6] + [60-6-6-6-6-6] + 
						[60+10+10+10+10+10]*298 +
			 [0]*365 + 
			 [0]*365 + 
			 [0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]}
	
	),
])
def test_flow_calc_sa(EWR_info, flows_data, expected_all_events):
	"""
	1. pre event pass and and post event pass
	2. pre event fail
	3. pre event pass and post event fail
	"""
	
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	flow_series = pd.Series(flows_data, index=dates)

	all_events, _ = evaluate_EWRs.flow_calc_sa(EWR_info, flow_series, water_years, dates, masked_dates)
	
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info, iteration, flows, all_events, ctf_state, expected_all_events, expected_ctf_state",[
	(
	 {
      'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
     0,	
	 np.array(  [0]*15 + # first dry spell
				[5]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{
	  'events': [], 
		'in_event': False
	},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(1)]], 
		'in_event': True
	},
	),
	(
	 {
      'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
     1,	
	 np.array(  [0]*15 + # first dry spell
				[5]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(1)]], 
		'in_event': True
	},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(2)]], 
		'in_event': True
	},
	),
	(
	 {
      'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
     15,	
	 np.array(  [0]*15 + # first dry spell
				[5]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': True
	},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': False
	},
	),
	(
	 {
      'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 16,
	   'ctf_threshold': 1
    },
     15,	
	 np.array(  [0]*15 + # first dry spell
				[5]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': True
	},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	{
	  'events': [], 
		'in_event': False
	},
	),
	(
	 {
      'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
     25,	
	 np.array(  [0]*15 + # first dry spell
				[5]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': False
	},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)],
			     [(date(2012,7,26) + timedelta(days=i), 0) for i in range(1)]], 
		'in_event': True
	},
	),
	(
	 {
       'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
     40,	
	 np.array(  [0]*15 + # first dry spell
				[5]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)],
			     [(date(2012,7,26) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': True
	},
	{ 2012: [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)] +
		   	 [(date(2012,7,16) + timedelta(days=i), 5) for i in range(10)] +
			 [(date(2012,7,26) + timedelta(days=i), 0) for i in range(15)]
	        ], 
		2013: [], 
		2014: [], 
		2015: []},
	{
	  'events': [[(date(2012,7,26) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': False
	},
	),
	(
	 {
       'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
     40,	
	 np.array(  [0]*15 + # first dry spell
				[20]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)],
			     [(date(2012,7,26) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': True
	},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	{
	  'events': [[(date(2012,7,26) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': False
	},
	),
	(
	 {
       'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
     39,	
	 np.array(  [0]*15 + # first dry spell
				[20]*10 + # in between
				[0]*14 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)],
			     [(date(2012,7,26) + timedelta(days=i), 0) for i in range(14)]], 
		'in_event': True
	},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	{
	  'events': [[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)]], 
		'in_event': False
	},
	),
])
def test_flow_check_ctf(EWR_info, iteration, flows, all_events, ctf_state, expected_all_events, expected_ctf_state):
	"""
		1. Test starting a first dry spell event
		2. Test appending to a current dry spell event
		3. Test ending the first dry spell event with a dry spell meeting non_flow_spell duration
		4. Test ending the first dry spell event with a dry spell not meeting non_flow_spell duration
		5. Test starting the second dry spell event with a dry spell meeting non_flow_spell duration
		6. Test ending the second dry spell event evaluating an unsuccessful event Fish Dispersal
		7. Test ending the second dry spell event evaluating an successful event Fish Dispersal
		8. Test ending the second dry spell event second dry spell not meeting non_flow_spell duration
	"""
	# non changing variable
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	
	all_events, ctf_state = evaluate_EWRs.flow_check_ctf(EWR_info, iteration, flows, all_events, water_years, flow_date, ctf_state)

	assert ctf_state == expected_ctf_state

	for year in all_events:
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

@pytest.mark.parametrize("EWR_info,flows_data,expected_all_events",[
	(
 	{
      'min_flow' : 20,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
		 np.array(  [0]*15 + # first dry spell
				[20]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]}
	
	),
	(
 	{
      'min_flow' : 21,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
		 np.array(  [0]*15 + # first dry spell
				[20]*10 + # in between
				[0]*15 + # second dry spell
				[6]*325 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[
			[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)] +
		   	 [(date(2012,7,16) + timedelta(days=i), 20) for i in range(10)] +
			 [(date(2012,7,26) + timedelta(days=i), 0) for i in range(15)]
	], 
	 2013:[], 
	 2014:[], 
	 2015:[]}
	
	),
	(
 	{
      'min_flow' : 21,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
		 np.array(  [0]*15 + # first dry spell
				[20]*10 + # in between
				[0]*15 + # second dry spell
				[20]*10 + # in between
				[0]*15 + # third dry spell
				[6]*300 +		      
				[0]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[
			[(date(2012,7,1) + timedelta(days=i), 0) for i in range(15)] +
		   	 [(date(2012,7,16) + timedelta(days=i), 20) for i in range(10)] +
			 [(date(2012,7,26) + timedelta(days=i), 0) for i in range(15)],
			 [(date(2012,7,26) + timedelta(days=i), 0) for i in range(15)] +
		   	 [(date(2012,8,10) + timedelta(days=i), 20) for i in range(10)] +
			 [(date(2012,8,20) + timedelta(days=i), 0) for i in range(15)]

	], 
	 2013:[], 
	 2014:[], 
	 2015:[]}
	
	),
		(
 	{
      'min_flow' : 21,
	   'duration':5,
	   'min_event': 5,
	   'non_flow_spell': 15,
	   'ctf_threshold': 1
    },
		 np.array( 
				[6]*325 +
				[0]*15 + # first dry spell
				[20]*10 + # in between
				[0]*15 + # second dry spell		      
				[6]*365 + 
				[0]*365 + 
				[0]*366),
	{2012:[], 
	 2013:[
		     [(date(2013,5,22) + timedelta(days=i), 0) for i in range(15)] +
		   	 [(date(2013,6,6) + timedelta(days=i), 20) for i in range(10)] +
			 [(date(2013,6,16) + timedelta(days=i), 0) for i in range(15)]
	 ], 
	 2014:[], 
	 2015:[]}
	
	),
])
def test_flow_calc_check_ctf(EWR_info,flows_data,expected_all_events):
	'''
		1. Test a successful fish dispersal (DO NOT RECORD)
		2. Test a unsuccessful fish dispersal (RECORD)
		3. Test a consecutive unsuccessful fish dispersal (RECORD 2 overlapping events)
		4. Test a unsuccessful fish dispersal (RECORD) over a year boundary
	'''
	
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	flows = pd.Series(flows_data, index=dates)

	all_events, _ = evaluate_EWRs.flow_calc_check_ctf(EWR_info, flows, water_years, dates, masked_dates)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info,iteration,flow,total_event,levels,event,event_state_in,event_state_out,all_events,expected_all_events,expected_event",
[
	({'min_volume': 80, 'min_flow': 15, 'max_flow': 1000000, 'min_event': 0,
      'accumulation_period': 10,'max_level': 5},
     5,
     20,
     0,	
	 np.array(		[4]*6 + [6]*1 + [5]*358 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[],
    {"level_crossed_down": False ,
	 "level_crossed_up" : False},
    {"level_crossed_down": False ,
	 "level_crossed_up" : False},
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(1)],	
	 ), 
	({'min_volume': 80, 'min_flow': 15, 'max_flow': 1000000, 'min_event': 0,
      'accumulation_period': 10,'max_level': 5},
     6,
     20,
     1,	
	 np.array(		[4]*6 + [6]*1 + [5]*358 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(1)],
    {"level_crossed_down": False ,
	 "level_crossed_up" : False},
    {"level_crossed_down": False ,
	 "level_crossed_up" : True},
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(2)],	
	 ), 
	({'min_volume': 80, 'min_flow': 15, 'max_flow': 1000000, 'min_event': 0,
      'accumulation_period': 10,'max_level': 5},
     7,
     20,
     1,	
	 np.array(		[4]*6 + [6]*2 + [5]*357 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(2)],
    {"level_crossed_down": False ,
	 "level_crossed_up" : True},
    {"level_crossed_down": False ,
	 "level_crossed_up" : True},
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(3)],	
	 ), 
	({'min_volume': 80, 'min_flow': 15, 'max_flow': 1000000, 'min_event': 0,
      'accumulation_period': 10,'max_level': 5},
     8,
     20,
     1,	
	 np.array(		[4]*6 + [6]*2 + [5]*357 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(3)],
    {"level_crossed_down": False ,
	 "level_crossed_up" : True},
    {"level_crossed_down": False ,
	 "level_crossed_up" : False},
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{ 2012: [[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(4)]], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ), 
	({'min_volume': 100, 'min_flow': 15, 'max_flow': 1000000, 'min_event': 0,
      'accumulation_period': 10,'max_level': 5},
     8,
     20,
     1,	
	 np.array(		[4]*6 + [6]*2 + [4]*357 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(3)],
    {"level_crossed_down": False ,
	 "level_crossed_up" : True},
    {"level_crossed_down": False ,
	 "level_crossed_up" : False},
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ), 
	({'min_volume': 60, 'min_flow': 15, 'max_flow': 1000000, 'min_event': 0,
      'accumulation_period': 10,'max_level': 5},
     8,
     0,
     1,	
	 np.array(		[4]*6 + [4]*2 + [4]*357 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	[(date(2012, 7, 6)+timedelta(days=i), 20) for i in range(3)],
    {"level_crossed_down": False ,
	 "level_crossed_up" : False},
    {"level_crossed_down": False ,
	 "level_crossed_up" : False},
	{2012:[], 
	 2013:[], 
	 2014:[], 
	 2015:[]},
	{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
	[],	
	 ), 
],)
def test_volume_level_check_bbr(EWR_info, iteration, flow, total_event, levels,event, event_state_in, event_state_out, all_events, expected_all_events, expected_event):
	"""Test Cases
	1. First iteration of an event i.e. meet the flow condition and at the iteration level still below the level at back lake.
	2. Second iteration of an ongoing event and the level crosses above the level threshold.
	3. Third iteration of an ongoing event and the still the level threshold event keep going
	4. Fourth iteration of an ongoing event and the level crosses below the level threshold and register the event.
	5. Fourth iteration of an ongoing event and the level crosses below the level threshold and not register the event because volume is below the target.
	6. Fourth iteration of an ongoing event and the level still below the level threshold but flow reach ctf zone. DO NOT register the event despite volume is on target level never achieved.
	"""
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_date = dates[iteration]
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	event, all_events, total_event, event_state = evaluate_EWRs.volume_level_check_bbr(EWR_info, iteration, flow, event,
	 all_events, water_years, total_event, flow_date, event_state_in, levels)

	assert event == expected_event

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	assert event_state == event_state_out


@pytest.mark.parametrize("EWR_info,flows,levels,expected_all_events,expected_all_no_events",[
	( {'min_volume': 120, 'min_flow': 30, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':7, 'end_month':6 ,'max_level': 120.746},
	   np.array(    [20]*10+[0]*355   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	   np.array([20]*10+[0]*355   + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{   2012: [],
		2013: [],
		2014: [],
		2015: []},
	{2012: [], 2013: [], 2014: [], 2015: []}),
	
	( {'min_volume': 120, 'min_flow': 30, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 10, 'start_month':7, 'end_month':6 ,'max_level': 50.7},
	   np.array(    [20]*5+[40]*6 +[0]*354 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	   np.array(    [20]*5+[20]*3+[60]*2+[20]*5+[0]*340 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{   2012: [[(date(2012, 7, 6)+timedelta(days=i), 40) for i in range(6)]],
		2013: [],
		2014: [],
		2015: []},
	{2012: [], 2013: [], 2014: [], 2015: []}),

	
	( {'min_volume': 120, 'min_flow': 2, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 5, 'start_month':7, 'end_month':6 ,'gap_tolerance':0,'max_level': 50.7},
	   np.array(    [20]*7 +[0]*358 +
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	   np.array(    [20]*7+[60]*3+[20]*355 +
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{   2012: [],
		2013: [],
		2014: [],
		2015: []},
	{2012: [], 2013: [], 2014: [], 2015: []}),

	( {'min_volume': 120, 'min_flow': 2, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 5, 'start_month':7, 'end_month':6 ,'gap_tolerance':0,'max_level': 50.7},
	   np.array(    [20]*7 +[0]*358 +
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	   np.array(    [20]*7+[20]*3+[20]*355 +
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{   2012: [],
		2013: [],
		2014: [],
		2015: []},
	{2012: [], 2013: [], 2014: [], 2015: []}),

	( {'min_volume': 100, 'min_flow': 2, 'max_flow': 1000000, 'min_event': 0, 'duration': 0
            , 'accumulation_period': 5, 'start_month':7, 'end_month':6 ,'gap_tolerance':0,'max_level': 50.7},
	   np.array(    [20]*7 +[0]*358 +
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	   np.array(    [20]*7+[20]*3+[20]*355 +
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{   2012: [],
		2013: [],
		2014: [],
		2015: []},
	{2012: [], 2013: [], 2014: [], 2015: []}),

],)
def test_cumulative_calc_bbr(EWR_info, flows, levels, expected_all_events, expected_all_no_events):
	"""
	1. Not reaching the min flow condition
	2. Textbook - entering on flow and exiting on level drop (cross up and then then down)
	3. Achieving volume but not within the max accumulation period
	4. Level condition is not met and exit on ctf, meeting volume condition
	5. Level condition is not met and exit on ctf, NOT meeting volume condition

	"""
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	all_events, _ = evaluate_EWRs.cumulative_calc_bbr(EWR_info, flows, levels, water_years, dates, masked_dates)

	assert all_events == expected_all_events

@pytest.mark.parametrize("event, EWR_info, expected_result",[
	(
	[(1,1),(1,1),(1,1),(1,10),(1,10),(1,10)],
	{"accumulation_period":4, "min_volume":10},
	True
	),
	(
	[(1,1),(1,1),(1,1),(1,10),(1,10),(1,10)],
	{"accumulation_period":3, "min_volume":10},
	False
	),
	(
	[(1,10),(1,10),(1,1),(1,1),(1,1),(1,1)],
	{"accumulation_period":3, "min_volume":10},
	True
	),
	(
	[],
	{"accumulation_period":3, "min_volume":10},
	False
	),
])
def test_achieved_min_volume(event, EWR_info, expected_result):
	result = evaluate_EWRs.achieved_min_volume(event, EWR_info)
	assert result == expected_result

	
@pytest.mark.parametrize("flow_date, flows, iteration, EWR_info, expected_results", [
	(
	pd.Period('2023-05-24', freq='D'),
	[1,1,3,4,5,6,7,1,1,1],
	6,
	{"larvae_days_spell":1,"eggs_days_spell":2},
	[(date(2023, 5, 24), 7), 
	(date(2023, 5, 25), 1),
	(date(2023, 5, 26), 1)
	],
	),
])
def test_create_water_stability_event(flow_date, flows, iteration, EWR_info, expected_results):

	result = evaluate_EWRs.create_water_stability_event(flow_date, flows, iteration, EWR_info)
	assert result == expected_results


@pytest.mark.parametrize("levels, iteration, EWR_info, expected_result",[
	(
	[1,1,1,1,1,1],
	0,
	{"max_level_raise": 0.05, "drawdown_rate": 0.05, "eggs_days_spell":3,"larvae_days_spell":3},
	True
	),
	(
	[1,1.05,1.05,1.1,1,1],
	0,
	{"max_level_raise": 0.05, "drawdown_rate": 0.05, "eggs_days_spell":3,"larvae_days_spell":3},
	False
	),
	(
	[1,1.05,1,1,1,1],
	0,
	{"max_level_raise": 0.05, "drawdown_rate": 0.05, "eggs_days_spell":3,"larvae_days_spell":3},
	False
	),
])
def test_check_water_stability_level(levels, iteration, EWR_info, expected_result):
	
	result = evaluate_EWRs.check_water_stability_level(levels, iteration, EWR_info)
	assert result == expected_result

@pytest.mark.parametrize("flows, iteration, EWR_info, expected_result",[
	(
	[31,31,31,50,70,40],
	1,
	{'max_flow': 80, 'min_flow': 30, "eggs_days_spell":2,"larvae_days_spell":3},
	True
	),
	(
	[31,31,31,80,70,40],
	1,
	{'max_flow': 80, 'min_flow': 30, "eggs_days_spell":2,"larvae_days_spell":3},
	False
	),
])
def test_check_water_stability_flow(flows, iteration, EWR_info, expected_result):
	
	result = evaluate_EWRs.check_water_stability_flow(flows, iteration, EWR_info)
	assert result == expected_result

@pytest.mark.parametrize("EWR_info, iteration, flows, all_events, levels, expected_all_events",[
	(
	{
      'min_flow': 70,
	  'max_flow' : 120,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'end_month': 11
	},
	39,
	 np.array(      [71]*365 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
	{   2012: [],
		2013: [],
		2014: [],
		2015: []},
	np.array(      [1]*365 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 8, 9)+timedelta(days=i), 71) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}, 
	),
])
def test_water_stability_check(EWR_info, iteration, flows, all_events, levels, expected_all_events):
	
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	flow_date = dates[iteration]
	
	all_events = evaluate_EWRs.water_stability_check(EWR_info, iteration, flows, all_events, water_years, flow_date, levels)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info, flows, levels, expected_all_events",[
	(   
	{
      'min_flow': 70,
	  'max_flow' : 120,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'min_event': 1,
	  'duration': 0,	
	  'start_month':8, 
	  'end_month' : 12 
	},
	   np.array(    [0]*31 + [71]*10 + [0]*324 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
		    
	 np.array(      [1]*365 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 8, 1)+timedelta(days=i), 71) for i in range(9)], 
	    [(date(2012, 8, 2)+timedelta(days=i), 71) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
	{
      'min_flow': 70,
	  'max_flow' : 120,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'min_event': 1,
	  'duration': 0,	
	  'start_month':8, 
	  'end_month' : 12 
	},
	   np.array(    [0]*31 + [71]*9 + [69]*1 + [0]*324 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
		    
	 np.array(      [1]*365 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 8, 1)+timedelta(days=i), 71) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
	{
      'min_flow': 70,
	  'max_flow' : 120,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'min_event': 1,
	  'duration': 1,	
	  'start_month':8, 
	  'end_month' : 12 
	},
	   np.array(    [0]*31 + [71]*10 + [0]*324 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
		    
	 np.array(      [0]*31 + [1]*9 + [2]*1 + [0]*324 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 8, 1)+timedelta(days=i), 71) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
	{
      'min_flow': 70,
	  'max_flow' : 120,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'min_event': 1,
	  'duration': 0,	
	  'start_month':8, 
	  'end_month' : 12 
	},
	   np.array(    [0]*31 + [71]*10 + [0]*2 + [71]*9 + [0]*313 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
		    
	 np.array(      [1]*365 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 8, 1)+timedelta(days=i), 71) for i in range(9)],
	    		[(date(2012, 8, 2)+timedelta(days=i), 71) for i in range(9)],
				[(date(2012, 8, 13)+timedelta(days=i), 71) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
	{
      'min_flow': 70,
	  'max_flow' : 120,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'min_event': 1,
	  'duration': 0,	
	  'start_month':8, 
	  'end_month' : 9 
	},
	   np.array(    [0]*83 + [71]*10 + [0]*272 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366),
		    
	 np.array(      [1]*365 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 9, 22)+timedelta(days=i), 71) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}
	),
])
def test_water_stability_calc(EWR_info, flows, levels, expected_all_events):
	"""
	1. meeting 2 opportunity flow and level met and events are overlapping
	2. breaking flow the second event and fail
	3. breaking level the second event and fail
	4. meeting 3 opportunities flow and level met with a gap of 2 days
	5. meeting 2 opportunity but second one the last day is outside window
	"""

	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	
	all_events, _ = evaluate_EWRs.water_stability_calc(EWR_info, flows, levels,water_years, dates, masked_dates)

	assert all_events == expected_all_events

@pytest.mark.parametrize("current_date, window_end, days_forward, expected_result", [
    (datetime(2023, 7, 4),  datetime(2023, 7, 7), 3, True),
    (datetime(2023, 7, 4),  datetime(2023, 7, 10), 2, True),
    (datetime(2023, 7, 4),  datetime(2023, 7, 7), 10, False),
    (datetime(2023, 7, 4),  datetime(2023, 7, 4), 0, True),
    (datetime(2023, 7, 4),  datetime(2023, 7, 4), 1, True),
])
def test_is_date_in_window(current_date, window_end, days_forward, expected_result):
	result = evaluate_EWRs.is_date_in_window(current_date, window_end, days_forward)
	assert result == expected_result

@pytest.mark.parametrize("iteration_date, month_window_end, expected_result",[
	(
	date(2023,12,23),
	2,
	date(2024,2,29),
	),
	(
	date(2023,1,23),
	2,
	date(2023,2,28),
	),
	(
	date(2023,1,23),
	12,
	date(2023,12,31),
	),
	(
	date(2012,9,23),
	9,
	date(2012,9,30),
	),
])
def test_get_last_day_of_window(iteration_date, month_window_end, expected_result):
	result = evaluate_EWRs.get_last_day_of_window(iteration_date, month_window_end)
	assert result == expected_result

@pytest.mark.parametrize('levels, EWR_info, expected_results',[
	( [1,1,1],
      {"max_level_raise": 0.05},
      True
	),
	( [1,1.1,1],
      {"max_level_raise": 0.05},
      False
	),
	( [1,1.04,1.04],
      {"max_level_raise": 0.05},
      True
	),
	( [1,1.01,1.02,1.01,1.02,1.01],
      {"max_level_raise": 0.05},
      True
	),
])
def test_is_egg_phase_stable(levels, EWR_info, expected_results):
	result = evaluate_EWRs.is_egg_phase_stable(levels, EWR_info)
	assert result == expected_results

@pytest.mark.parametrize('levels, EWR_info, expected_results',[
	( [1,1,1],
      {"max_level_raise": 0.05},
      True
	),
	( [1,1.1,1],
      {"max_level_raise": 0.05},
      False
	),
	( [1,1.04,1.08,1.12],
      {"max_level_raise": 0.05},
      True
	),
	( [1,1.01,1.02,1.08,1.02,1.01],
      {"max_level_raise": 0.05},
      False
	),
])
def test_is_larva_phase_stable(levels, EWR_info, expected_results):
	result = evaluate_EWRs.is_larva_phase_stable(levels, EWR_info)
	assert result == expected_results


@pytest.mark.parametrize("levels, iteration, EWR_info, expected_result",[
	(
	[1,1,1,1,1.5,1.64],
	0,
	{'max_level': 1.65, 'min_level': 0, "eggs_days_spell":3,"larvae_days_spell":3},
	True
	),
	(
	[1,1,1,1,1,1.7],
	0,
	{'max_level': 1.65, 'min_level': 0, "eggs_days_spell":3,"larvae_days_spell":3},
	False
	),
])
def test_check_water_stability_height(levels, iteration, EWR_info, expected_result):
	
	result = evaluate_EWRs.check_water_stability_height(levels, iteration, EWR_info)
	assert result == expected_result


@pytest.mark.parametrize("EWR_info, iteration, all_events, levels, expected_all_events",[
	(
	{
      'min_level': 0,
	  'max_level' : 1.65,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'end_month': 11
	},
	39,
	{   2012: [],
		2013: [],
		2014: [],
		2015: []},
	np.array(      [1]*365 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 8, 9)+timedelta(days=i), 1) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}, 
	),
])
def test_water_stability_level_check(EWR_info, iteration, all_events, levels, expected_all_events):
	
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	flow_date = dates[iteration]
	
	all_events = evaluate_EWRs.water_stability_level_check(EWR_info, iteration, all_events, water_years, flow_date, levels)

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]


@pytest.mark.parametrize("EWR_info, levels, expected_all_events",[
	(   
	{
      'min_level': .5,
	  'max_level' : 1.65,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'min_event': 1,
	  'duration': 0,	
	  'start_month':8, 
	  'end_month' : 12 
	},	    
	 np.array(      [0]*31 + [1]*10 + [0]*324 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 8, 1)+timedelta(days=i), 1) for i in range(9)], 
	    [(date(2012, 8, 2)+timedelta(days=i), 1) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
	{
      'min_level': .5,
	  'max_level' : 1.65,
	  "eggs_days_spell": 3,
	  "larvae_days_spell": 6,
	  "max_level_raise" : 0.05,
	  "drawdown_rate" : 0.05,
	  'min_event': 1,
	  'duration': 0,	
	  'start_month':8, 
	  'end_month' : 9 
	},	    
	 np.array(      [0]*83 + [1]*10 + [0]*272 + 
                    [0]*365 + 
                    [0]*365 + 
                    [0]*366), 
	{   2012: [[(date(2012, 9, 22)+timedelta(days=i), 1) for i in range(9)]],
		2013: [],
		2014: [],
		2015: []}
	),
])
def test_water_stability_level_calc(EWR_info, levels, expected_all_events):
	"""
	1. meeting 2 opportunity flow and level met and events are overlapping
	2. meeting 2 opportunity but second one the last day is outside window
	"""

	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	
	all_events, _ = evaluate_EWRs.water_stability_level_calc(EWR_info, levels, water_years, dates, masked_dates)

	assert all_events == expected_all_events

@pytest.mark.parametrize("flows_data, iteration_date, years_shift, max_flow",[
	(
	np.array([0]*364+[4]*1 + [0]*364+[3]*1 + [0]*364+[2]*1 + [0]*365+[1]*1),
	date(2016,6,30), 
	1,
	1
	),
	(
	np.array([0]*364+[4]*1 + [0]*364+[3]*1 + [0]*364+[2]*1 + [0]*365+[1]*1),
	date(2016,6,30), 
	2,
	2
	),
	(
	np.array([0]*364+[4]*1 + [0]*364+[3]*1 + [0]*364+[2]*1 + [0]*365+[1]*1),
	date(2016,6,30), 
	3,
	3
	),
	(
	np.array([0]*364+[4]*1 + [0]*364+[3]*1 + [0]*364+[2]*1 + [0]*365+[1]*1),
	date(2016,6,30), 
	4,
	4
	),
])
def test_filter_n_year_shift_flows(flows_data, iteration_date, years_shift, max_flow):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_series = pd.Series(flows_data, index=dates)
	flows = evaluate_EWRs.filter_n_year_shift_flows(flows_series, iteration_date, years_shift)
	assert max(flows) == max_flow



@pytest.mark.parametrize("flows_data, iteration_date, expected_min_flow",[
	(
	np.array([0]*364+[300]*1 + [0]*364+[400]*1 + [0]*364+[400]*1 + [0]*365+[300]*1),
	date(2016,6,30),
	300
	),
	(
	np.array([0]*364+[300]*1 + [0]*364+[400]*1 + [0]*364+[400]*1 + [0]*365+[300]*1),
	date(2013,6,30),
	0
	),
])
def test_get_min_each_last_three_years_volume(flows_data, iteration_date, expected_min_flow):
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flows_series = pd.Series(flows_data, index=dates)
	min_flow = evaluate_EWRs.get_min_each_last_three_years_volume(flows_series, iteration_date)
	assert min_flow == expected_min_flow



def test_calculate_n_day_moving_average():

	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2012-07-16', '%Y-%m-%d'))
	df = pd.DataFrame(data={
		'44444':[5,10,5,10,5,10,5,10,5,10,5,10,5,10,5,10],
		'44443':[50,100,50,100,50,100,50,100,50,100,50,100,50,100,50,100]},
		 index=dates)
	
	expected_result = pd.DataFrame(data={
		'44444':[5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5],
		'44443':[50,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75]},
		 index=dates)
	
	result = evaluate_EWRs.calculate_n_day_moving_average(df,2)

	assert result.to_dict() == expected_result.to_dict()


@pytest.mark.parametrize("EWR_info, expected_type",[
	(
	{ 'EWR_code': "CLLMM1c_P"},
	'c'
	),
	(
	{ 'EWR_code': "CLLMM1d"},
	'd'
	),
	(
	{ 'EWR_code': "CLLMM1a_S"},
	'a'
	),
])
def test_what_cllmm_type(EWR_info, expected_type):
	result = evaluate_EWRs.what_cllmm_type(EWR_info)
	assert result == expected_type
	
@pytest.mark.parametrize("flows, thresholds, expected_result",[
	(
	[1,1,1,3,3,3,4,4,4,4,
     5,5,6,6,7,7,
	 8,8,
	 7,7,7,6,6,6,5,5,5,
	 4,4,4,
	 5,5,5,5,6,6,6,7,7,7,
	 8,8,9,10,10,11,12,14,13,10,8,
	 7,6],
	 (5,8),
	 {'first_threshold': [[5, 5, 6, 6, 7, 7], 
		       			  [7, 7, 7, 6, 6, 6, 5, 5, 5], 
						  [5, 5, 5, 5, 6, 6, 6, 7, 7, 7], 
						  [7, 6]], 
      'second_threshold': [[8, 8], 
			              [8, 8, 9, 10, 10, 11, 12, 14, 13, 10, 8]]}

	),
])
def test_flow_intervals(flows, thresholds, expected_result):
	result = evaluate_EWRs.flow_intervals_stepped(flows, thresholds)
	assert result == expected_result

def test_calculate_change_previous_day():
	flows = [1,1,1,3,3,3,6]
	result = evaluate_EWRs.calculate_change_previous_day(flows)
	assert result == [1.0, 1.0, 3.0, 1.0, 1.0, 2.0]


@pytest.mark.parametrize("flows, EWR_info, iteration, mode, expected_result",[
	(
		[500]*10 + [1000]*10 + [3000]*5 + [6000]*15,
		{
		  'rate_of_rise_max1': 2,
		  'rate_of_rise_max2': 2.7
		},
		30,
		'backwards_stepped',
		False
	),
	(
		[500]*10 + [1000]*10 + [2000]*5 + [5000]*1 + [13501]*5,
		{
		  'rate_of_rise_max1': 2,
		  'rate_of_rise_max2': 2.7
		},
		30,
		'backwards_stepped',
		False
	),
	(
		[500]*5 + [1000]*5 + [1800]*5 + [3000]*5 + [4000]*5 + [5000]*5 + [6000]*5,
		{
		  'rate_of_rise_max1': 2,
		  'rate_of_rise_max2': 2.7
		},
		30,
		'backwards_stepped',
		True
	),
	(
		 [1000]*10 + [1800]*5 + [3000]*5 + [4000]*5 + [5000]*5 + [6000]*5,
		{'rate_of_rise_stage': 2},
		30,
		'backwards',
		True
	),
	(
		 [6000]*10 + [5000]*5 + [4000]*5 + [4000]*5 + [3500]*5 + [3000]*5,
		{'rate_of_fall_stage': .8},
		1,
		'forwards',
		True
	),
	(
		[1000]*2 + [700]*1 + [1000]*10 + [1800]*5 + [3000]*5 + [4000]*5 + [5000]*5 ,
		{'rate_of_fall_stage': .8},
		1,
		'forwards',
		False
	),
])
def test_check_period_flow_change_stepped(flows, EWR_info, iteration, mode, expected_result):
	result = evaluate_EWRs.check_period_flow_change_stepped(flows, EWR_info, iteration, mode)
	assert result == expected_result 

@pytest.mark.parametrize("EWR_info, flows, expected_all_events",[
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array( [1, 2.1, 4.45] + [1000,2001, 3000, 4000, 5000, 15000] + [1000]*356 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366), 
	{   2012: [[(date(2012, 7, 4), 1000.0), (date(2012, 7, 5), 2001.0)],
				[(date(2012, 7, 9), 15000.0)]],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array( [1, 2.1, 4.45] + [900,1500, 3000, 4000, 5000] + [1000]*357 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366), 
	{   2012: [],
		2013: [],
		2014: [],
		2015: []}
	),
])
def test_rate_rise_flow_calc(EWR_info, flows, expected_all_events):
    # non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()

	all_events, _ = evaluate_EWRs.rate_rise_flow_calc(EWR_info, flows, water_years, dates, masked_dates)

	assert all_events == expected_all_events

@pytest.mark.parametrize("EWR_info, iteration, event, all_events, total_event, flows_data, expected_all_events,expected_event",[
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'gap_tolerance': 0},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1000,2001,4002,8004,6000] + [5000] *360 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		[(date(2012, 7, 2), 2001)],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'gap_tolerance': 0},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1000,2000,4000,8000,6000] + [5000] *360 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		[],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'gap_tolerance': 0},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [5000,13501,4000,8000,6000] + [5000] *360 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		[(date(2012, 7, 2), 13501)],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'gap_tolerance': 0},
		3,
		[(date(2012, 7, 2), 2001),(date(2012, 7, 3), 4002)],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1000,2001,4002,8000,6000] + [5000] *360 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [[(date(2012, 7, 2), 2001),(date(2012, 7, 3), 4002)]], 
		2013: [], 
		2014: [], 
		2015: []},
		[],
	),
])
def test_rate_rise_flow_check(EWR_info, iteration, event, all_events, total_event, flows_data, expected_all_events,expected_event):

	# non changing variable
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_series = pd.Series(flows_data, index=dates)
	flow_date = dates[iteration]
	gap_track = 0

	event, all_events, _, _ = evaluate_EWRs.rate_rise_flow_check(EWR_info, iteration, event, all_events, gap_track, water_years, total_event, flow_date, flow_series)

	assert event == expected_event
	assert all_events == expected_all_events


@pytest.mark.parametrize("EWR_info, flows, expected_all_events",[
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array(    [35,30] + [30]*363 + 
				  [30]*365 +
				  [30]*365 + 
				  [30]*366), 
	{   2012: [],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array(    [40,30,41,30] + [30]*361 + 
				  [30]*365 +
				  [30]*365 + 
				  [30]*366), 
	{   2012: [[(date(2012, 7, 2), 30)],[(date(2012, 7, 4), 30)]],
		2013: [],
		2014: [],
		2015: []}
	),
])
def test_rate_fall_flow_calc(EWR_info, flows, expected_all_events):
	  # non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()

	all_events, _ = evaluate_EWRs.rate_fall_flow_calc(EWR_info, flows, water_years, dates, masked_dates)

	assert all_events == expected_all_events

@pytest.mark.parametrize("EWR_info, iteration, event, all_events, total_event, flows_data, expected_all_events,expected_event",[
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1000, 700] + [700]*363 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[(date(2012, 7, 2), 700)],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1000, 800] + [700]*363 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0},
		2,
		[(date(2012, 7, 2), 700)],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1000, 700, 700] + [700]*362 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [[(date(2012, 7, 2), 700)]], 
			2013: [], 
			2014: [], 
			2015: []},
		[],
	),
])
def test_rate_fall_flow_check(EWR_info, iteration, event, all_events, total_event, flows_data, expected_all_events,expected_event):
	# non changing variable
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	flow_series = pd.Series(flows_data, index=dates)
	flow_date = dates[iteration]
	gap_track = 0

	event, all_events, _, _ = evaluate_EWRs.rate_fall_flow_check(EWR_info, iteration, event, all_events, gap_track, water_years, total_event, flow_date, flow_series)

	assert event == expected_event
	assert all_events == expected_all_events


@pytest.mark.parametrize("EWR_info, levels, expected_all_events",[
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array( [1, 1, 1 ] + [0]*362 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366), 
	{   2012: [],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array( [1, 1.39, 1.8 ] + [0]*362 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366), 
	{   2012: [[(date(2012, 7, 2), 1.39), (date(2012, 7, 3), 1.8)]],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array( [1, 1.39, 1.8, 1, 1.39, 1.8 ] + [0]*359 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366), 
	{   2012: [[(date(2012, 7, 2), 1.39), (date(2012, 7, 3), 1.8)], [(date(2012, 7, 5), 1.39), (date(2012, 7, 6), 1.8)]],
		2013: [],
		2014: [],
		2015: []}
	),
])
def test_rate_rise_level_calc(EWR_info, levels, expected_all_events):
	
	# non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()

	all_events, _ = evaluate_EWRs.rate_rise_level_calc(EWR_info, levels, water_years, dates, masked_dates)

	assert all_events == expected_all_events

@pytest.mark.parametrize("EWR_info, iteration, event, all_events, total_event, levels_data, expected_all_events,expected_event",[
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1, 1.39] + [0]*363 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[(date(2012, 7, 2), 1.39)],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38},
		2,
		[(date(2012, 7, 2), 1.39)],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1, 1.3] + [0]*363 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [[(date(2012, 7, 2), 1.39)]], 
			2013: [], 
			2014: [], 
			2015: []},
		[],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1, 1.3] + [0]*363 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[],
	),
])
def test_rate_rise_level_check(EWR_info, iteration, event, all_events, total_event, levels_data, expected_all_events,expected_event):
	
	# non changing variable
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	level_series = pd.Series(levels_data, index=dates)
	flow_date = dates[iteration]
	gap_track = 0

	event, all_events, _, _ = evaluate_EWRs.rate_rise_level_check(EWR_info, iteration, event, all_events, gap_track, water_years, total_event, flow_date, level_series)

	assert event == expected_event
	assert all_events == expected_all_events

@pytest.mark.parametrize("EWR_info, levels, expected_all_events",[
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array( [1.22, 1, .77 ] + [0]*362 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366), 
	{   2012: [[(date(2012, 7, 2), 1.0), (date(2012, 7, 3), 0.77), (date(2012, 7, 4), 0.0)]],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array(    [0]*365 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366), 
	{   2012: [],
		2013: [],
		2014: [],
		2015: []}
	),
	(   
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21,
		  'duration': 1},	    
	 np.array(    [1,1] +[1]*363 + 
				  [1]*365 +
				  [1]*365 + 
				  [1]*366), 
	{   2012: [],
		2013: [],
		2014: [],
		2015: []}
	),
])
def test_rate_fall_level_calc(EWR_info, levels, expected_all_events):

	# non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()

	all_events, _ = evaluate_EWRs.rate_fall_level_calc(EWR_info, levels, water_years, dates, masked_dates)

	assert all_events == expected_all_events



	
@pytest.mark.parametrize("EWR_info, iteration, event, all_events, total_event, levels_data, expected_all_events,expected_event",[
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1.22, 1] + [0]*363 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[(date(2012, 7, 2), 1)],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21},
		1,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1.20, 1] + [0]*363 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[],
	),
	(
		{'rate_of_rise_threshold2':5000,
          'rate_of_rise_max2':2.7,
		  'rate_of_rise_max1':2.,
		  'rate_of_rise_threshold1':1000,
		  'rate_of_fall_min': 0.8,
		  'gap_tolerance': 0,
		  'rate_of_rise_river_level': 0.38,
		  'rate_of_fall_river_level': 0.21},
		2,
		[(date(2012, 7, 2), 1)],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1.22, 1] + [1]*363 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [[(date(2012, 7, 2), 1)]], 
			2013: [], 
			2014: [], 
			2015: []},
		[],
	),
])
def test_rate_fall_level_check(EWR_info, iteration, event, all_events, total_event, levels_data, expected_all_events,expected_event):
	# non changing variable
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	level_series = pd.Series(levels_data, index=dates)
	flow_date = dates[iteration]
	gap_track = 0

	event, all_events, _, _ = evaluate_EWRs.rate_fall_level_check(EWR_info, iteration, event, all_events, gap_track, water_years, total_event, flow_date, level_series)

	assert event == expected_event
	assert all_events == expected_all_events


@pytest.mark.parametrize("EWR_info, levels, iteration, expected", [
    (
		{'min_level_rise': .5},
		[1, 2, 3, 4, 5, 6, 7], 
		6,
		 True
		 ),
    (
		{'min_level_rise': .5},
		[1.0, 1.2, 1.3, 1.4, 1.3, 1.4, 1.3], 
		6,
		 False
		 ),
])
def test_evaluate_level_change(EWR_info, levels, iteration, expected):
	
	result = evaluate_EWRs.evaluate_level_change(EWR_info, levels, iteration)
	
	assert result == expected

@pytest.mark.parametrize("EWR_info, iteration, event, all_events, total_event, levels_data, expected_all_events,expected_event",[
	(
		{
		  'gap_tolerance': 0,
		  'min_level_rise': .5,
		  'min_event':7,
		  'start_month':7
		  },
		7,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1.0, 1.2, 1.3, 1.4, 1.3, 1.4, 1.3, 1.2, 1.4] + [0]*356 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[],
	),
	(
		{
		  'gap_tolerance': 0,
		  'min_level_rise': .5,
		  'min_event':7,
		  'start_month':7},
		7,
		[],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1.0, 1.0, 1.3, 1.4, 1.3, 1.4, 1.3, 1.6] + [0]*357 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[(date(2012, 7, 2), 1.0), (date(2012, 7, 3), 1.3), (date(2012, 7, 4), 1.4), 
			(date(2012, 7, 5), 1.3), (date(2012, 7, 6), 1.4), 
			(date(2012, 7, 7), 1.3), (date(2012, 7, 8), 1.6)],
	),
	(
		{
		  'gap_tolerance': 0,
		  'min_level_rise': .5,
		  'min_event':7,
		  'start_month':7},
		8,
		[(date(2012, 7, 2), 1.0), (date(2012, 7, 3), 1.3), (date(2012, 7, 4), 1.4), 
			(date(2012, 7, 5), 1.3), (date(2012, 7, 6), 1.4), 
			(date(2012, 7, 7), 1.3), (date(2012, 7, 8), 1.6)],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1.0, 1.0, 1.3, 1.4, 1.3, 1.4, 1.3, 1.6, 1.9] + [0]*356 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [], 
			2013: [], 
			2014: [], 
			2015: []},
		[(date(2012, 7, 2), 1.0), (date(2012, 7, 3), 1.3), (date(2012, 7, 4), 1.4), 
			(date(2012, 7, 5), 1.3), (date(2012, 7, 6), 1.4), 
			(date(2012, 7, 7), 1.3), (date(2012, 7, 8), 1.6),(date(2012, 7, 9), 1.9)],
	),
	(
		{
		  'gap_tolerance': 0,
		  'min_level_rise': .5,
		  'min_event':7,
		  'start_month':7
		  },
		9,
		[(date(2012, 7, 2), 1.0), (date(2012, 7, 3), 1.3), (date(2012, 7, 4), 1.4), 
			(date(2012, 7, 5), 1.3), (date(2012, 7, 6), 1.4), 
			(date(2012, 7, 7), 1.3), (date(2012, 7, 8), 1.6),(date(2012, 7, 9), 1.9)],
		{ 2012: [], 
		2013: [], 
		2014: [], 
		2015: []},
		0,
		np.array( [1.0, 1.0, 1.3, 1.4, 1.3, 1.4, 1.3, 1.6, 1.9, 1.5] + [0]*355 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366),
		{ 2012: [[(date(2012, 7, 2), 1.0), (date(2012, 7, 3), 1.3), (date(2012, 7, 4), 1.4), 
			(date(2012, 7, 5), 1.3), (date(2012, 7, 6), 1.4), 
			(date(2012, 7, 7), 1.3), (date(2012, 7, 8), 1.6),(date(2012, 7, 9), 1.9)]], 
			2013: [], 
			2014: [], 
			2015: []},
		[],
	),
])
def test_level_change_check(EWR_info, iteration, event, all_events, total_event, levels_data, expected_all_events,expected_event):

	# non changing variable
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
	level_series = pd.Series(levels_data, index=dates)
	level_date = dates[iteration]
	gap_track = 0

	event, all_events, _, _ = evaluate_EWRs.level_change_check(EWR_info, iteration, level_series, event, all_events, gap_track, 
															water_years, total_event, level_date)

	assert event == expected_event
	assert all_events == expected_all_events



@pytest.mark.parametrize("EWR_info, levels, expected_all_events",[
	(   
		{ 'gap_tolerance': 0,
		  'min_level_rise': .5,
		  'duration': 1,
		  'min_event': 7,
		  'start_month': 7,},	    
	 np.array( [1.0, 1.0, 1.3, 1.4, 1.3, 1.4, 1.3, 1.6, 1.9, 1.5] + [0]*355 + 
				  [0]*365 +
				  [0]*365 + 
				  [0]*366), 
	{   2012: [[(date(2012, 7, 2), 1.0), (date(2012, 7, 3), 1.3), (date(2012, 7, 4), 1.4), (date(2012, 7, 5), 1.3), 
			 (date(2012, 7, 6), 1.4), (date(2012, 7, 7), 1.3), (date(2012, 7, 8), 1.6), (date(2012, 7, 9), 1.9)]],
		2013: [],
		2014: [],
		2015: []}
	),
])
def test_level_change_calc(EWR_info, levels, expected_all_events):
	
	# non changing variable
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()

	all_events, _ = evaluate_EWRs.level_change_calc(EWR_info, levels, water_years, dates, masked_dates)

	assert all_events == expected_all_events

@pytest.mark.parametrize("ewr_key, expected_result", [
	( 
		'IC2_S-single-F', 'flow_handle'
	),
	( 
		'FLR-single-L', 'level_change_handle'
	),
	( 
		'XXXXXX-single-L', 'unknown'
	),
])
def test_find_function(ewr_key, expected_result, ewr_calc_config):
	result = evaluate_EWRs.find_function(ewr_key, ewr_calc_config)
	assert result == expected_result


@pytest.mark.parametrize("flows, iteration, ctf_state,expected_flows",[
	(
		[0]*15 + # first dry spell
		[5]*10 + # in between
		[0]*15 + # second dry spell
		[6]*30,
		40,
		{
			'events':[
						[(date(2020,1,1) + timedelta(days=i), 0)  for i in range(15)], # first dry spell
						[(date(2020,1,26) + timedelta(days=i), 0)  for i in range(15)]  # second dry spell
					 ], 
		'in_event': False
		},
		[5]*10
	)
])
def test_get_flows_in_between_dry_spells(flows, iteration, ctf_state,expected_flows):
	result = evaluate_EWRs.get_flows_in_between_dry_spells(flows, iteration, ctf_state)
	assert result == expected_flows

@pytest.mark.parametrize("flows, iteration, ctf_state, expected_event",[
	(
	   flows := [0]*15 + # first dry spell
				[5]*10 + # in between
				[0]*15 + # second dry spell
				[6]*30,
		40,
		{
			'events':[
						[(date(2020,1,1) + timedelta(days=i), 0)  for i in range(15)], # first dry spell
						[(date(2020,1,26) + timedelta(days=i), 0)  for i in range(15)]  # second dry spell
					 ], 
		'in_event': False
		},
		[(date(2020,1,1) + timedelta(days=i), flow)  for i, flow in zip(range(40), flows[:40]) ]
	)
])
def test_get_full_failed_event(flows, iteration, ctf_state, expected_event):
	result = evaluate_EWRs.get_full_failed_event(flows, iteration, ctf_state)
	assert result == expected_event


@pytest.mark.parametrize("EWR_info, flows, expected_events",[
	(
		{
		 'min_flow': 6.
		},
		[2,2,2,8,8,8,8,6,6,5,5,4,4,4,3,3,5,5,6,6,6,7,7,7,7,8,8,8,9],
		[[8, 8, 8, 8, 6, 6], [6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9]]
	),
	(
		{
		 'min_flow': 20.
		},
		[2,2,2,8,8,8,8,6,6,5,5,4,4,4,3,3,5,5,6,6,6,7,7,7,7,8,8,8,9],
		[]
	),
])
def test_get_threshold_events(EWR_info, flows, expected_events):
	result = evaluate_EWRs.get_threshold_events(EWR_info, flows)
	assert result == expected_events