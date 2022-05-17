import pandas as pd
from pandas._testing import assert_frame_equal

from py_ewr import data_inputs, summarise_results

def test_sum_events():
    '''
    Series input to summarise_results function will only be made up of combination of 1's or 0's. 
    1. Test to see if function is counting occurences of 1's
    '''
    # Test 1
    input_series = pd.Series(index=[1895, 1896, 1897, 1898, 1899], data=[0,1,1,1,0])
    s = summarise_results.sum_events(input_series)
    expected_s = 3
    assert s == expected_s

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

def test_get_average():
        '''
        Input will be a series of years and either integers or floats
        1. Test average
        2. Test average if all numbers in series are 0
        '''
        # Test 1
        input_series = pd.Series(index=[1895, 1896, 1897, 1898, 1899], data=[100,100,50,50,0])
        f = summarise_results.get_average(input_series)
        expected_f = 60
        assert f == expected_f
        # ----------------------------
        # Test 2
        input_series = pd.Series(index=[1895, 1896, 1897, 1898, 1899], data=[0,0,0,0,0])
        f = summarise_results.get_average(input_series)
        expected_f = 0
        assert f == expected_f

def test_event_length():
    '''
    Input is a tuple containing dictionaries of events. Typical tuples contain one dictionary, however, for the EWRs that require flows to be met across multiple sites, there will be multiple event dictionaries within the tuple
    1. Test getting the average event length over a multi site EWR
    2. Test getting the average event length over a single site EWR
    '''
    # Test 1
    events1 = {2012: [[50,50,50,100,100,100],[5,5,7.5,10,10]], 2013: [[50,50,100,100],[5,5,7.5,10,10]], 2014: [[50,50,50,100,100,100],[5,5,7.5,10,10]]}
    events2 = {2012: [[50,50,50,100,100,100]], 2013: [[50,50,100,100],[5,5,7.5,10,10]], 2014: []}
    tupled = tuple([events1, events2])
    EWR = {}
    EWR['VL'] = tupled
    PU = {}
    PU['planning unit 1'] = EWR
    gauge = {}
    gauge['425001'] = PU
    scenario = {}
    scenario['scenario 1'] = gauge
    f = summarise_results.get_event_length(scenario['scenario 1']['425001']['planning unit 1']['VL'])
    
    expected_f = 5.111111111111111
    assert expected_f ==  f
    #------------------------------
    # Test 2
    events1 = {2012: [[50,50,50,100,100,100],[5,5,7.5,10,10]], 2013: [[50,50,100,100],[5,5,7.5,10,10]], 2014: [[50,50,50,100,100,100],[5,5,7.5,10,10]]}
    tupled = tuple([events1])
    EWR = {}
    EWR['VL'] = tupled
    PU = {}
    PU['planning unit 1'] = EWR
    gauge = {}
    gauge['425001'] = PU
    scenario = {}
    scenario['scenario 1'] = gauge

    f = summarise_results.get_event_length(scenario['scenario 1']['425001']['planning unit 1']['VL'])
    
    expected_f = 5.166666666666667
    assert expected_f == f

def test_get_threshold_days():
    '''
    Input is a tuple containing dictionaries of events. Typical tuples contain one dictionary, however, for the EWRs that require flows to be met across multiple sites, there will be multiple event dictionaries within the tuple
    1. Test getting the total event length of multi site EWR
    2. Test getting the total event length over a single site EWR
    '''
    # Test 1
    events1 = {2012: [[50,50,50,100,100,100],[5,5,7.5,10,10]], 2013: [[50,50,100,100],[5,5,7.5,10,10]], 2014: [[50,50,50,100,100,100],[5,5,7.5,10,10]]}
    events2 = {2012: [[50,50,50,100,100,100]], 2013: [[50,50,100,100],[5,5,7.5,10,10]], 2014: []}
    tupled = tuple([events1, events2])
    EWR = {}
    EWR['VL'] = tupled
    PU = {}
    PU['planning unit 1'] = EWR
    gauge = {}
    gauge['425001'] = PU
    scenario = {}
    scenario['scenario 1'] = gauge
    f = summarise_results.get_event_length(scenario['scenario 1']['425001']['planning unit 1']['VL'])
    
    expected_f = 5.111111111111111
    assert expected_f == f 
    #------------------------------
    # Test 2
    events1 = {2012: [[50,50,50,100,100,100],[5,5,7.5,10,10]], 2013: [[50,50,100,100],[5,5,7.5,10,10]], 2014: [[50,50,50,100,100,100],[5,5,7.5,10,10]]}
    tupled = tuple([events1])
    EWR = {}
    EWR['VL'] = tupled
    PU = {}
    PU['planning unit 1'] = EWR
    gauge = {}
    gauge['425001'] = PU
    scenario = {}
    scenario['scenario 1'] = gauge

    f = summarise_results.get_event_length(scenario['scenario 1']['425001']['planning unit 1']['VL'])
    
    expected_f = 5.166666666666667
    assert expected_f == f 

def test_count_exceedence():
        '''
        Input is a series with water years along the index, and a list of lists, containing integers for the data.
        1. Test number of exceedences when there is a max interevent to be exceeded
        2. Test the function returns 'N/A' when there is no max interevent to be checked event for this EWR
        '''
        # Test 1
        index = [1895, 1896, 1897, 1898, 1899, 1900]
        data = [[15, 30], [], [], [450, 2], [10,12], [200,10]]
        series = pd.Series(index=index, data=data)

        EWR_info = {'max_inter-event': 2}

        result = summarise_results.count_exceedence(series, EWR_info)
        expected_result = 8 # There are 8 exceedences in the above list
        assert result == expected_result
        # ---------------------------------------
        # Test 2
        index = [1895, 1896, 1897, 1898, 1899, 1900]
        data = [[15, 30], [], [], [450, 2], [10,12], [200,10]]
        series = pd.Series(index=index, data=data)

        EWR_info = {'max_inter-event': None}

        result = summarise_results.count_exceedence(series, EWR_info)
        expected_result = 'N/A' # There is no max interevent period defined so cannot return a result
        assert result == expected_result

def test_initialise_summary_df_columns():
        '''
        Input is a dictionary of scenario EWR results for each gauge and planning unit.
        1. Test proper initialisation of multi index columns
        '''
        # Test 1
        # Input data and send to function
        columns = ['Event years','Frequency','Target frequency','Achievement count', 'Achievements per year', 'Event count','Events per year',
                    'Event length','Threshold days','Inter-event exceedence count', 'Max inter event period (years)', 'No data days',
                    'Total days']
        df = pd.DataFrame()
        df1 = pd.DataFrame()
        input_dictionary = {'Scenario 1': {'Gauge 1': {'Planning unit 1': df}}, 'Scenario 2': {'Gauge 1': {'Planning unit 1': df1}}}

        multi_index = summarise_results.initialise_summary_df_columns(input_dictionary)
        # Define expected output given the above input
        tuples = [('Scenario 1', 'Event years'),
                  ('Scenario 1', 'Frequency'),
                  ('Scenario 1', 'Target frequency'),
                  ('Scenario 1', 'Achievement count'),
                  ('Scenario 1', 'Achievements per year'),
                  ('Scenario 1', 'Event count'),
                  ('Scenario 1', 'Events per year'),
                  ('Scenario 1', 'Event length'),
                  ('Scenario 1', 'Threshold days'),
                  ('Scenario 1', 'Inter-event exceedence count'),
                  ('Scenario 1', 'Max inter event period (years)'),
                  ('Scenario 1', 'No data days'),
                  ('Scenario 1', 'Total days'),
                  ('Scenario 2', 'Event years'),
                  ('Scenario 2', 'Frequency'),
                  ('Scenario 2', 'Target frequency'),
                  ('Scenario 2', 'Achievement count'),
                  ('Scenario 2', 'Achievements per year'),
                  ('Scenario 2', 'Event count'),
                  ('Scenario 2', 'Events per year'),
                  ('Scenario 2', 'Event length'),
                  ('Scenario 2', 'Threshold days'),
                  ('Scenario 2', 'Inter-event exceedence count'),
                  ('Scenario 2', 'Max inter event period (years)'),
                  ('Scenario 2', 'No data days'),
                  ('Scenario 2', 'Total days')]
        expected_multi_index = pd.MultiIndex.from_tuples(tuples, names=['scenario', 'type'])
        
        # test result:
        for i, tup in enumerate(expected_multi_index):
            for index, val in enumerate(tup):
                assert multi_index[i][index] == val

def test_initialise_summary_df_rows():
        '''
        Input is a dictionary of scenario EWR results for each gauge and planning unit.
        1. Test for expected initialisation of three layered multi index
        '''
        # Test 1
        # Setting up input data and sending to function:
        columns = ['CF1_eventYears','CF1_numAchieved','CF1_numEvents','CF1_eventLength','CF1_totalEventDays','CF1_daysBetweenEvents','CF1_missingDays','CF1_totalPossibleDays','VF1_eventYears',
         'VF1_numAchieved','OB-L1_S_missingDays','OB-L1_S_totalPossibleDays','OB-L1_P_eventYears','OB-L1_P_numAchieved','OB-L1_P_numEvents','OB-L1_P_eventLength']
        df = pd.DataFrame(columns=columns)
        input_dictionary = {'Scenario 1': {'Gauge 1': {'Planning unit 1': df}}, 'Scenario 2': {'Gauge 1': {'Planning unit 1': df}}}

        multi_index = summarise_results.initialise_summary_df_rows(input_dictionary)

        # Setting up expected output:
        tuples = [('Gauge 1', 'Planning unit 1', 'CF1'), ('Gauge 1', 'Planning unit 1', 'VF1'), ('Gauge 1', 'Planning unit 1', 'OB-L1_S'), ('Gauge 1', 'Planning unit 1', 'OB-L1_P')]
        expected_multi_index = pd.MultiIndex.from_tuples(tuples, names=['gauge', 'planning unit', 'EWR'])
        
        # Test result:
        for i, tup in enumerate(expected_multi_index):
            for index, val in enumerate(tup):
                assert multi_index[i][index] == val

def test_allocate():
    '''
    Input is a dataframe to save information to, the information to be saved, and the coordinates of the dataframe
    1. Test information is being saved in the expected location in the dataframe
    '''
    # Test 1
    # Define input data and send to function:
    tuples = [('Scenario 1', 'Event years'),
                ('Scenario 1', 'Frequency'),
                ('Scenario 2', 'No data days'),
                ('Scenario 2', 'Total days')]
    multi_column = pd.MultiIndex.from_tuples(tuples, names=['scenario', 'type'])        
    tuples = [('Gauge 1', 'Planning unit 1', 'CF1'), 
                ('Gauge 1', 'Planning unit 1', 'VF1'), 
                ('Gauge 1', 'Planning unit 1', 'OB-L1_S'), 
                ('Gauge 1', 'Planning unit 1', 'OB-L1_P')]
    multi_index = pd.MultiIndex.from_tuples(tuples, names=['gauge', 'planning unit', 'EWR'])
    df = pd.DataFrame(index = multi_index, columns=multi_column)
    add_this = 12
    idx = pd.IndexSlice
    site = 'Gauge 1'
    PU = 'Planning unit 1'
    EWR = 'OB-L1_S'
    scenario = 'Scenario 2'
    category = 'No data days'
    result_df = summarise_results.allocate(df, add_this, idx, site, PU, EWR, scenario, category)
    # Set expected dataframe
    expected_df = pd.DataFrame(index = multi_index, columns=multi_column)
    expected_df.loc[idx[[site], [PU], [EWR]], idx[scenario, category]] = 12
    # Test result:
    assert_frame_equal(result_df, expected_df)

def test_summarise():
    '''
    Inputs are a dictionary containing annualised results for each Scenario>Gauge>Planning Unit>EWR and 
    a dictionary containing individual event information for each Scenario>Gauge>Planning Unit>EWR
    1. Test each part of the function are working correctly and producing an overall expected output
    '''
    
    # Test 1
    # setting up input data and sending to function
    df = pd.read_csv('unit_testing_files/input_for_summarise_data.csv', index_col  =0)
    for col in df:
        if 'daysBetweenEvents' in col:
            for i, val in enumerate(df[col]):
                new = df[col].iloc[i]
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

                df[col].iloc[i] = new_list
    input_dict = {'Scenario 1': {'410007': {'Upper Yanco Creek': df}}}
    input_events = {'Scenario 1': {'410007': {'Upper Yanco Creek': {'VF1': tuple([{1895: [[400]*100, [400]*50],
                                                                            1896: [], 
                                                                            1897: [[500]*50, [400]*30], 
                                                                            1898: [], 
                                                                            1899: [], 
                                                                            1900: [[500]*345]}])}}}}
    result = summarise_results.summarise(input_dict, input_events)
    # Set up expected outputs and test
    tuples = [('Scenario 1', 'Event years'),
                ('Scenario 1', 'Frequency'),
                ('Scenario 1', 'Target frequency'),
                ('Scenario 1', 'Achievement count'),
                ('Scenario 1', 'Achievements per year'),
                ('Scenario 1', 'Event count'),
                ('Scenario 1', 'Events per year'),
                ('Scenario 1', 'Event length'),
                ('Scenario 1', 'Threshold days'),
                ('Scenario 1', 'Inter-event exceedence count'),
                ('Scenario 1', 'Max inter event period (years)'),
                ('Scenario 1', 'No data days'),
                ('Scenario 1', 'Total days')]
    multi_column = pd.MultiIndex.from_tuples(tuples, names=['scenario', 'type'])        
    tuples = [('410007', 'Upper Yanco Creek', 'VF1')]
    multi_index = pd.MultiIndex.from_tuples(tuples, names=['gauge', 'planning unit', 'EWR'])
    expected_df = pd.DataFrame(index = multi_index, columns=multi_column)
    expected_df.iloc[0]=[3, 50, str(100), 3, 0.5, 3, 0.5, 115.0, 575, 0, 0.010958904, 0, 2191]        
    
    assert_frame_equal(result, expected_df)