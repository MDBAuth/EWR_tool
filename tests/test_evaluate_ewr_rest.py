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
	no_event = 50
	all_no_events = {2012:[[25], [2]], 2013:[[250]],
							2014:[[400], [2], [25]], 2015:[[450]]}
	gap_track = 0
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)
	total_event = 9
	event, all_events, no_event, all_no_events, gap_track, total_event = evaluate_EWRs.flow_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, gap_track, water_years, total_event, flow_date)
	# Set up expected results and test
	expected_event =  [(event_start + timedelta(days=i), 5) for i in range(10)]
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
	EWR_info = {'min_flow': 10, 'max_flow': 20}
	flow = 5
	water_year = 2015
	flow_date = date(2012,1,17)
	event = [(date(2015, 10, 9)+timedelta(days=i),5) for i in range(9)]
	iteration = 365+365+365+100
	all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
					2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
	no_event = 0
	all_no_events = {2012:[[25], [2]], 2013:[[250]],
							2014:[[400], [2], [25]], 2015:[[450]]}
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)
	event, all_events, no_event, all_no_events = evaluate_EWRs.lowflow_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, water_years, flow_date)
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
	iteration = 365+365+365+100
	event = [0]*9
	flow_date = date(2012,1,17)
	all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
					2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
	no_event = 10
	all_no_events = {2012:[[25], [2]], 2013:[[250]],
							2014:[[400], [2], [25]], 2015:[[450]]}
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)

	event, all_events, no_event, all_no_events = evaluate_EWRs.ctf_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, water_years, flow_date)
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
	flow_date = 1
	flow_date = date(2012,1,17)
	event = [10]*5
	total_event = 5
	all_events = {2012:[[10]*10, [15]*12], 2013:[[10]*50], 
					2014:[[10]*10, [15]*15, [10]*20], 2015:[]}
	no_event = 25
	all_no_events = {2012:[[25], [2]], 2013:[[250]],
							2014:[[400], [2], [25]], 2015:[[450]]}
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*365)
	gap_track = 0
	event, all_events, no_event, all_no_events, gap_track, total_event = evaluate_EWRs.flow_check_sim(iteration, EWR_info1, EWR_info2, water_years, flow1, flow2, event, all_events, no_event, all_no_events, gap_track, total_event, flow_date)
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
	for year in all_events:
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]
	assert no_event == expected_no_event
	assert all_no_events == expected_all_no_events

	for year in all_no_events:
			for i, no_event in enumerate(all_no_events[year]):
					assert no_event == expected_all_no_events[year][i]
                
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
	all_events, all_no_events, durations = evaluate_EWRs.flow_calc(EWR_info, flows, water_years, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]
	for year in all_no_events:
			assert len(all_no_events[year]) == len(expected_all_no_events[year])
			for i, no_event in enumerate(all_no_events[year]):
					assert no_event == expected_all_no_events[year][i]
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
	climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
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
	expected_durations = [300,300,10,300] # adding in a very dry year climate year
	# Send inputs to test function and test
	all_events, all_no_events, durations = evaluate_EWRs.lowflow_calc(EWR_info, flows, water_years, climates, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]             
	for year in all_no_events:
			assert len(all_no_events[year]) ==len(expected_all_no_events[year])
			for i, no_event in enumerate(all_no_events[year]):
					assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	#------------------------------------------------
	# Test 2
	# Set up input data
	EWR_info = {'min_flow': 10, 'max_flow': 20, 'min_event':1, 'duration': 10, 
					'duration_VD': 5, 'start_month': 7, 'end_month': 12, 'start_day': None, 'end_day': None}
	flows = np.array([10]*5+[0]*35+[5]*5+[0]*295+[0]*25 + [0]*355+[5]*10 + [10]*10+[0]*355 + [5]*295+[0]*25+[10]*45+[10]*1)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = masked_dates[((masked_dates.month >= 7) & (masked_dates.month <= 12))] # Just want the dates in the date range
	# Set up expected output data
	expected_all_events = {2012: [[(date(2012, 7, 1), 10), (date(2012, 7, 2), 10), (date(2012, 7, 3), 10), 
	(date(2012, 7, 4), 10), (date(2012, 7, 5), 10)]], 2013: [], 2014: [[(date(2014, 7, 1), 10), 
	(date(2014, 7, 2), 10), (date(2014, 7, 3), 10), (date(2014, 7, 4), 10), (date(2014, 7, 5), 10), 
	(date(2014, 7, 6), 10), (date(2014, 7, 7), 10), (date(2014, 7, 8), 10), (date(2014, 7, 9), 10), 
	(date(2014, 7, 10), 10)]], 2015: []}
	expected_all_no_events = {2012: [], 2013: [[725]], 2014: [], 2015: [[720]]}
	expected_durations = [10,10,5,10] # adding in a very dry year climate year
	# Send to test function and test
	all_events, all_no_events, durations = evaluate_EWRs.lowflow_calc(EWR_info, flows, water_years, climates, dates, masked_dates)
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
	climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Set up expected output data
	expected_all_events = {2012: [[(date(2013, 4, 22)+timedelta(days=i), 0) for i in range(25)]],
	  					   2013: [[(date(2014, 6, 26)+timedelta(days=i), 0) for i in range(5)]], 
						   2014: [[(date(2014, 7, 1)+timedelta(days=i), 0) for i in range(355)]],
						   2015: [[(date(2015, 7, 1)+timedelta(days=i), 1) for i in range(295)], 
						   [(date(2016, 5, 16)+timedelta(days=i), 0) for i in range(46)]]}
	expected_all_no_events = {2012: [[295]], 2013: [[405]], 2014: [], 2015: [[10], [25]]}
	expected_durations = [20,20,10,20] # adding in a very dry year climate year
	# Send to test function and then test
	all_events, all_no_events, durations = evaluate_EWRs.ctf_calc(EWR_info, flows, water_years, climates, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]
	for year in all_no_events:
			assert len(all_no_events[year]) == len(expected_all_no_events[year])
			for i, no_event in enumerate(all_no_events[year]):
					assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
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
	climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Very Dry']*365 + ['Dry']*366)
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
	expected_all_no_events = {2012: [[35]], 2013: [], 2014: [[685]], 2015: [[536]]}
	expected_durations = [10,10,5,10] # adding in a very dry year climate year
	# Send to test function and then test
	all_events, all_no_events, durations = evaluate_EWRs.ctf_calc(EWR_info, flows, water_years, climates, dates, masked_dates)
	for year in all_events:
			assert len(all_events[year]) ==len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]
	for year in all_no_events:
			assert len(all_no_events[year]) == len(expected_all_no_events[year])
			for i, no_event in enumerate(all_no_events[year]):
					assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations

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
	climates = np.array(['Wet']*365 + ['Very Wet']*365 + ['Wet']*365 + ['Dry']*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# expected_all_no_events = {2012: [[295]], 2013: [[405]], 2014: [], 2015: [[10], [25]]}
	expected_durations = [20,20,20,20] # adding in a very dry year climate year
	# Send to test function and then test
	all_events, all_no_events, durations = evaluate_EWRs.ctf_calc_anytime(EWR_info, flows, water_years, climates, dates)
	for year in all_events:
			assert len(all_events[year]) == len(expected_all_events[year])
			for i, event in enumerate(all_events[year]):
					assert event == expected_all_events[year][i]
	for year in all_no_events:
			assert len(all_no_events[year]) == len(expected_all_no_events[year])
			for i, no_event in enumerate(all_no_events[year]):
					assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations

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
	all_events, all_no_events, durations = evaluate_EWRs.flow_calc_anytime(EWR_info, flows, water_years, dates)

	for year in all_events:
		# assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]
	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations


# def test_cumulative_calc_anytime():
# 	'''
# 	1. Test functions ability to identify and save all events and event gaps for series of flows, 
# 		ensuring events crossing water years are identified and registered
# 			- Test event crossing water years
# 			- Test event on final day of series
# 			- TO-TEST: event on first day of series
# 	'''
# 	# Set up input data
# 	EWR_info = {'min_volume': 100, 'min_flow': 50, 'min_event': 2, 'duration': 2}
# 	flows = np.array([0]*350+[10]*14+[50]*1 + [50]*1+[0]*358+[100]*6 + [75]*1+[25]*1+[0]*353+[50]*10 + [50]*2+[0]*362+[49]*1+[100]*1)
# 	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
# 	# Set up expected outputs
# 	expected_all_events = {2012: [], 
# 							2013: [[50]*2, [100]*1, [100]*2, [100]*2], 
# 							2014: [[100,75], [50]*2, [50]*2, [50]*2, [50]*2, [50]*2], 
# 							2015: [[50]*2, [100]*1]}
# 	expected_all_no_events = {2012: [], 2013: [[364], [357]], 2014: [[354]], 2015: [[362]]}
# 	expected_durations = [2]*4
# 	# Send inputs to test function and then test
# 	all_events, all_no_events, durations = evaluate_EWRs.cumulative_calc_anytime(EWR_info, flows, water_years)       
# 	for year in all_events:
# 		assert len(all_events[year]) == len(expected_all_events[year])
# 		for i, event in enumerate(all_events[year]):
# 			assert event == expected_all_events[year][i]
# 	for year in all_no_events:
# 		assert len(all_no_events[year]) == len(expected_all_no_events[year])
# 		for i, no_event in enumerate(all_no_events[year]):
# 			assert no_event == expected_all_no_events[year][i]
# 	assert durations == expected_durations
			
# def test_nest_calc_weirpool():
# 	'''
# 	1. Test functions ability to identify and save all events and event gaps for series of flows and levels, ensure events cannot overlap water years. Other tests:
# 		- check if event exluded when flow requirement is passed but the level requirement is not passed
# 		- TO-TEST: check if event exluded when flow requirement is not passed but the level requirement is passed
# 		- TO-TEST: check if event is excluded when flow and level requirements are passed but the drawdown rate is exceeded
# 	'''
# 	# Set up input data
# 	EWR_info = {'min_flow': 5, 'max_flow': 20, 'drawdown_rate': 0.04, 'min_event': 10, 'duration': 10}
# 	flows = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*10 + [10]*5+[0]*351+[10]*10)
# 	levels = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*9+[1]*1 + [10]*5+[0]*351+[10]*10)
# 	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
# 	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
# 	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))
# 	# Set up expected output data
# 	expected_all_events = {2012: [[10]*10], 2013: [[10]*10], 2014: [[10]*10], 2015: [[10]*10]}
# 	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[711]]}
# 	expected_durations = [10]*4
# 	# Send to test function and then test
# 	all_events, all_no_events, durations = evaluate_EWRs.nest_calc_weirpool(EWR_info, flows, levels, water_years, dates, masked_dates)

# 	for year in all_events:
# 		assert len(all_events[year]) == len(expected_all_events[year])
# 		for i, event in enumerate(all_events[year]):
# 			assert event == expected_all_events[year][i]

# 	for year in all_no_events:
# 		assert len(all_no_events[year]) == len(expected_all_no_events[year])
# 		for i, no_event in enumerate(all_no_events[year]):
# 			assert no_event == expected_all_no_events[year][i]
# 	assert durations == expected_durations
        
# def test_nest_calc_percent():
# 	'''
# 	1. Test functions ability to identify and save all events and event gaps for series of flows, ensure events cannot overlap water years. Other tests:
# 		- check if event exluded when flow requirement is passed but the drawdown rate is exceeded
# 	'''
# 	# Set up input data
# 	EWR_info = {'min_flow': 5, 'max_flow': 20, 'drawdown_rate': '10%', 'min_event': 10, 'duration': 10}
# 	flows = np.array([0]*350+[10]*10+[0]*5 + [0]*355+[10]*10 + [10]*10+[0]*345+[10]*9+[8]*1 + [10]*9+[9]*1+[0]*346+[10]*10)
# 	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
# 	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
# 	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
# 	# Set up expected output data
# 	expected_all_events = {2012: [[10]*10], 2013: [[10]*10], 2014: [[10]*10], 2015: [[10]*9+[9]*1, [10]*10]}
# 	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[355], [346]]}
# 	expected_durations = [10]*4
# 	# Send to test function and then test
# 	all_events, all_no_events, durations = evaluate_EWRs.nest_calc_percent(EWR_info, flows, water_years, dates, masked_dates)

# 	for year in all_events:
# 		assert len(all_events[year]) == len(expected_all_events[year])
# 		for i, event in enumerate(all_events[year]):
# 			assert event == expected_all_events[year][i]

# 	for year in all_no_events:
# 		assert len(all_no_events[year]) == len(expected_all_no_events[year])
# 		for i, no_event in enumerate(all_no_events[year]):
# 			assert no_event == expected_all_no_events[year][i]
# 	assert durations == expected_durations
	
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
	expected_all_events = {2012: [[(date(2013, 6, 16), 10), (date(2013, 6, 17), 10), (date(2013, 6, 18), 10), 
	(date(2013, 6, 19), 10), (date(2013, 6, 20), 10), (date(2013, 6, 21), 10), (date(2013, 6, 22), 10), 
	(date(2013, 6, 23), 10), (date(2013, 6, 24), 10), (date(2013, 6, 25), 10)]], 
	2013: [], 
	2014: [[(date(2014, 6, 21), 10), (date(2014, 6, 22), 10), (date(2014, 6, 23), 10), 
	(date(2014, 6, 24), 10), (date(2014, 6, 25), 10), (date(2014, 6, 26), 10), 
	(date(2014, 6, 27), 10), (date(2014, 6, 28), 10), (date(2014, 6, 29), 10), (date(2014, 6, 30), 10), 
	(date(2014, 7, 1), 10), (date(2014, 7, 2), 10), (date(2014, 7, 3), 10), (date(2014, 7, 4), 10), 
	(date(2014, 7, 5), 10), (date(2014, 7, 6), 10), (date(2014, 7, 7), 10), (date(2014, 7, 8), 10), 
	(date(2014, 7, 9), 10), (date(2014, 7, 10), 10)]], 
	2015: [[(date(2016, 6, 21), 10), (date(2016, 6, 22), 10), 
	(date(2016, 6, 23), 10), (date(2016, 6, 24), 10), (date(2016, 6, 25), 10), (date(2016, 6, 26), 10),
	(date(2016, 6, 27), 10), (date(2016, 6, 28), 10), (date(2016, 6, 29), 10), (date(2016, 6, 30), 10)]]}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[711]]}
	expected_durations = [10]*4

	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Send to test function and then test
	all_events, all_no_events, durations = evaluate_EWRs.flow_calc_anytime_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, dates)
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations

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
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Set up expected output data
	expected_all_events = {2012: [[(date(2013, 6, 16), 10), (date(2013, 6, 17), 10), (date(2013, 6, 18), 10), 
	(date(2013, 6, 19), 10), (date(2013, 6, 20), 10), (date(2013, 6, 21), 10), (date(2013, 6, 22), 10), 
	(date(2013, 6, 23), 10), (date(2013, 6, 24), 10), (date(2013, 6, 25), 10)]], 
	2013: [[(date(2014, 6, 21), 10), 
	(date(2014, 6, 22), 10), (date(2014, 6, 23), 10), (date(2014, 6, 24), 10), (date(2014, 6, 25), 10), 
	(date(2014, 6, 26), 10), (date(2014, 6, 27), 10), (date(2014, 6, 28), 10), (date(2014, 6, 29), 10), 
	(date(2014, 6, 30), 10)]], 
	2014: [[(date(2014, 7, 1), 10), (date(2014, 7, 2), 10), (date(2014, 7, 3), 10),
	 (date(2014, 7, 4), 10), (date(2014, 7, 5), 10), (date(2014, 7, 6), 10), (date(2014, 7, 7), 10), 
	 (date(2014, 7, 8), 10), (date(2014, 7, 9), 10), (date(2014, 7, 10), 10)]], 
	 2015: [[(date(2016, 6, 21), 10), 
	 (date(2016, 6, 22), 10), (date(2016, 6, 23), 10), (date(2016, 6, 24), 10), (date(2016, 6, 25), 10), 
	 (date(2016, 6, 26), 10), (date(2016, 6, 27), 10), (date(2016, 6, 28), 10), (date(2016, 6, 29), 10), 
	 (date(2016, 6, 30), 10)]]}
	expected_all_no_events = {2012: [[350]], 2013: [[360]], 2014: [], 2015: [[711]]}
	expected_durations = [10]*4
	# Send to test function and then test
	all_events, all_no_events, durations = evaluate_EWRs.flow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, dates, masked_dates)
	for year in all_events:
		assert len(all_events[year]) ==  len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]
	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]
	assert durations == expected_durations
	
def test_lowflow_calc_sim():
	'''
	1. Test functions ability to identify and save all events and event gaps for series of flows
		- Test to ensure it does not matter event sequencing at each site, as long as minimum day duration is met for each year, event should be registered
	'''
	# Set up input data
	EWR_info1 = {'min_flow': 15, 'max_flow': 20, 'min_event': 1, 'duration': 10, 'duration_VD': 5}
	EWR_info2 = {'min_flow': 20, 'max_flow': 30, 'min_event': 1, 'duration': 10, 'duration_VD': 5}
	flows1 = np.array([10]*1+[0]*350+[10]*9+[0]*5 + [0]*360+[10]*5 + [10]*10+[0]*345+[10]*10 + [8]*5+[0]*351+[10]*10)
	flows2 = np.array([25]*1+[0]*350+[30]*9+[0]*5 + [0]*360+[30]*5 + [30]*10+[0]*345+[10]*10 + [18]*10+[0]*346+[30]*10)
	water_years = np.array([2012]*365 + [2013]*365 + [2014]*365 + [2015]*366)
	climates = np.array(['Wet']*365 + ['Very Dry']*365 +['Very Wet']*365 + ['Dry']*366)
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Set up expected output data
	expected_all_events1 = {2012: [], 2013: [], 2014: [], 2015: []}
	expected_all_events2 = {2012: [[(date(2012, 7, 1), 25)], [(date(2013, 6, 17), 30), (date(2013, 6, 18), 30), 
	(date(2013, 6, 19), 30), (date(2013, 6, 20), 30), (date(2013, 6, 21), 30), (date(2013, 6, 22), 30), 
	(date(2013, 6, 23), 30), (date(2013, 6, 24), 30), (date(2013, 6, 25), 30)]], 
	2013: [[(date(2014, 6, 26), 30), 
	(date(2014, 6, 27), 30), (date(2014, 6, 28), 30), (date(2014, 6, 29), 30), (date(2014, 6, 30), 30)]], 
	2014: [[(date(2014, 7, 1), 30), (date(2014, 7, 2), 30), (date(2014, 7, 3), 30), (date(2014, 7, 4), 30), 
	(date(2014, 7, 5), 30), (date(2014, 7, 6), 30), (date(2014, 7, 7), 30), (date(2014, 7, 8), 30), 
	(date(2014, 7, 9), 30), (date(2014, 7, 10), 30)]], 
	2015: [[(date(2016, 6, 21), 30), (date(2016, 6, 22), 30),
	 (date(2016, 6, 23), 30), (date(2016, 6, 24), 30), (date(2016, 6, 25), 30), (date(2016, 6, 26), 30), 
	 (date(2016, 6, 27), 30), (date(2016, 6, 28), 30), (date(2016, 6, 29), 30), (date(2016, 6, 30), 30)]]}
	expected_all_no_events1 = {2012: [], 2013: [], 2014: [], 2015: [[1461]]}
	expected_all_no_events2 = {2012: [[350]], 2013: [[365]], 2014: [], 2015: [[711]]}
	expected_durations = [10,5,10,10]
	# Send inputs to function and then test:
	all_events1, all_events2, all_no_events1, all_no_events2, durations = evaluate_EWRs.lowflow_calc_sim(EWR_info1, EWR_info2, flows1,
         flows2, water_years, climates, dates, masked_dates)
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
	dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	masked_dates = pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()
	# Set up expected outputs
	expected_all_events1 = {2012: [[(date(2012, 7, 2)+timedelta(days=i),0) for i in range(350)], 
									[(date(2013, 6, 26)+timedelta(days=i), 1) for i in range(5)]], 
							2013: [[(date(2013, 7, 1)+timedelta(days=i),0) for i in range(360)]], 
							2014: [[(date(2014, 7, 11)+timedelta(days=i),0) for i in range(345)], 
							[(date(2015, 6, 22)+timedelta(days=i),1) for i in range(9)]], 
							2015: [[(date(2016, 6, 21)+timedelta(days=i),0) for i in range(10)]]}
	expected_all_events2 = {2012: [[(date(2012, 7, 2)+timedelta(days=i),0) for i in range(350)], 
									[(date(2013, 6, 26)+timedelta(days=i), 1) for i in range(5)]], 
							2013: [[(date(2013, 7, 1)+timedelta(days=i),0) for i in range(360)]], 
							2014: [[(date(2014, 7, 11)+timedelta(days=i),0) for i in range(345)], 
							[(date(2015, 6, 22)+timedelta(days=i),1) for i in range(9)]], 
							2015: [[(date(2016, 6, 21)+timedelta(days=i),0) for i in range(10)]]}
	expected_all_no_events1 = {2012: [[1], [9]], 2013: [], 2014: [[15], [1]], 2015: [[356]]}
	expected_all_no_events2 = {2012: [[1], [9]], 2013: [], 2014: [[15], [1]], 2015: [[356]]}
	expected_durations = [10,5,10,10]
	# Send inputs to function and then test
	all_events1, all_events2, all_no_events1, all_no_events2, durations = evaluate_EWRs.ctf_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates, dates, masked_dates)
	for year in all_events1:
		assert len(all_events1[year]) == len(expected_all_events1[year])
		for i, event in enumerate(all_events1[year]):
			assert event == expected_all_events1[year][i]
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
	all_events, all_no_events, durations = evaluate_EWRs.cumulative_calc(EWR_info, flows, water_years, dates, masked_dates)

	assert all_events == expected_all_events
	assert all_no_events == expected_all_no_events
	
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
	no_event = 50
	total_event = 9
	gap_track = 0
	event, all_events, no_event, all_no_events, gap_track, total_event, roller = evaluate_EWRs.volume_check(EWR_info, iteration, flow, event,
	 all_events, no_event, all_no_events, gap_track, water_years, total_event, flow_date, roller, max_roller, flows)

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
	# ("423001", "WL2" , "PU_0000251", "423002"), # c423001,423002 fix this ena put separate in a column...
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
	no_event = 0
	gap_track = 0
	
	event, all_events, no_event, all_no_events, gap_track, total_event = evaluate_EWRs.weirpool_check(EWR_info, iteration, flow, level, event, all_events, no_event, all_no_events, gap_track, 
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
	
	all_events, all_no_events, _ = evaluate_EWRs.weirpool_calc(EWR_info, flows, levels, water_years, weirpool_type, dates, masked_dates)


	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]



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
	no_event = 0
	gap_track = 0



	event, all_events, no_event, all_no_events, gap_track, total_event = evaluate_EWRs.level_check(EWR_info, iteration, level, level_change, 
																				event, all_events, no_event, all_no_events, 
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
	
	all_events, all_no_events, _ = evaluate_EWRs.lake_calc(EWR_info, levels, water_years, dates, masked_dates)


	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]


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
	no_event = 0
	gap_track = 0
	iteration_no_event = 0

	event, all_events, no_event, all_no_events, gap_track, total_event, iteration_no_event = evaluate_EWRs.nest_flow_check(EWR_info, iteration, flow, 
																		event, all_events, no_event, all_no_events, gap_track, 
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
	
	all_events, all_no_events, _ = evaluate_EWRs.nest_calc_percent_trigger(EWR_info, flows, water_years, dates)
								
	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]

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
	no_event = 0
	gap_track = 0
	
	event, all_events, no_event, all_no_events, gap_track, total_event = evaluate_EWRs.nest_weirpool_check(EWR_info, iteration, flow, level, 
									event, all_events, no_event, all_no_events, gap_track, 
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
	
	all_events, all_no_events, _ = evaluate_EWRs.nest_calc_weirpool(EWR_info, flows, levels, water_years, dates, masked_dates, weirpool_type )

	for year in all_events:
		assert len(all_events[year]) == len(expected_all_events[year])
		for i, event in enumerate(all_events[year]):
			assert event == expected_all_events[year][i]

	for year in all_no_events:
		assert len(all_no_events[year]) == len(expected_all_no_events[year])
		for i, no_event in enumerate(all_no_events[year]):
			assert no_event == expected_all_no_events[year][i]

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