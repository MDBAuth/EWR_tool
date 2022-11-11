from datetime import date, timedelta, datetime
import re

import pandas as pd
import pytest

from py_ewr import observed_handling, scenario_handling, data_inputs

@pytest.fixture(scope="function")
def pu_df():
    df_data = {'CF1_a_eventYears': {2016: 0, 2017: 0},
                'CF1_a_numAchieved': {2016: 0, 2017: 0},
                'CF1_a_numEvents': {2016: 0, 2017: 0},
                'CF1_a_numEventsAll': {2016: 0, 2017: 0},
                'CF1_a_eventLength': {2016: 0.0, 2017: 0.0},
                'CF1_a_eventLengthAchieved': {2016: 0.0, 2017: 0.0},
                'CF1_a_totalEventDays': {2016: 0, 2017: 0},
                'CF1_a_totalEventDaysAchieved': {2016: 0, 2017: 0},
                'CF1_a_maxInterEventDaysAchieved': {2016: [], 2017: []},
                'CF1_a_missingDays': {2016: 0, 2017: 0},
                'CF1_a_totalPossibleDays': {2016: 365, 2017: 365}}
    return pd.DataFrame.from_dict(df_data)


@pytest.fixture(scope="function")
def detailed_results(pu_df):
    return {"observed": {"419001": {"Keepit to Boggabri": pu_df},
                         "419002": {"Keepit to Boggabri": pu_df}}}


@pytest.fixture(scope="function")
def item_to_process(pu_df):
    return { "scenario" : 'observed',
                         "gauge" : '419001',
                         "pu" : 'Keepit to Boggabri',
                         "pu_df" : pu_df }

@pytest.fixture(scope="function")
def items_to_process(pu_df):

    return [ { "scenario" : 'observed',
                "gauge" : '419001',
                "pu" : 'Keepit to Boggabri',
                "pu_df" : pu_df },
             { "scenario" : 'observed',
                "gauge" : '419002',
                "pu" : 'Keepit to Boggabri',
                "pu_df" : pu_df }
           ]

@pytest.fixture(scope="function")
def gauge_events():
    return  {'observed': {'419001': {'Keepit to Boggabri': {'CF1_a': ({2010: [],
                2011: [],
                2012: [],
                2013: [],
                2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},)}
                                },
                         '419002': {'Keepit to Boggabri': {'CF1_a': ({2010: [],
                2011: [],
                2012: [],
                2013: [],
                2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},)}
                                }
                     },                
            }

@pytest.fixture(scope="function")
def yearly_events():
    return {2010: [],
                2011: [],
                2012: [],
                2013: [],
                2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}

@pytest.fixture(scope="function")
def event_item_to_process():
    return  {  "scenario" : 'observed',
                "gauge" : '419001',
                "pu" : 'Keepit to Boggabri',
                "ewr": 'CF1_a',
                "ewr_events" :  {  2010: [],
                                    2011: [],
                                    2012: [],
                                    2013: [],
                                    2014: [[(date(2020, 11, 30), 0.0),
                                            (date(2020, 12, 1), 0.0),
                                            (date(2020, 12, 2), 0.0),
                                            (date(2020, 12, 3), 0.0),
                                            (date(2020, 12, 4), 0.0),
                                            (date(2020, 12, 5), 0.0),]]}
                                            }

@pytest.fixture(scope="function")
def event_items_to_process():
    return  [{  "scenario" : 'observed',
                "gauge" : '419001',
                "pu" : 'Keepit to Boggabri',
                "ewr": 'CF1_a',
                "ewr_events" :  {  2010: [],
                                    2011: [],
                                    2012: [],
                                    2013: [],
                                    2014: [[(date(2020, 11, 30), 0.0),
                                            (date(2020, 12, 1), 0.0),
                                            (date(2020, 12, 2), 0.0),
                                            (date(2020, 12, 3), 0.0),
                                            (date(2020, 12, 4), 0.0),
                                            (date(2020, 12, 5), 0.0),]]}
                                    },
                {  "scenario" : 'observed',
                "gauge" : '419002',
                "pu" : 'Keepit to Boggabri',
                "ewr": 'CF1_a',
                "ewr_events" : {  2010: [],
                                    2011: [],
                                    2012: [],
                                    2013: [],
                                    2014: [[(date(2020, 11, 30), 0.0),
                                            (date(2020, 12, 1), 0.0),
                                            (date(2020, 12, 2), 0.0),
                                            (date(2020, 12, 3), 0.0),
                                            (date(2020, 12, 4), 0.0),
                                            (date(2020, 12, 5), 0.0),]]}
                                    }
                                    ]

@pytest.fixture(scope="function")
def stamp_index():
    return pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d'))

@pytest.fixture(scope="function")
def period_index():
    return pd.date_range(start= datetime.strptime('2012-07-01', '%Y-%m-%d'), end = datetime.strptime('2016-06-30', '%Y-%m-%d')).to_period()

@pytest.fixture(scope="function")
def stamp_date(stamp_index):
    return stamp_index[0]

@pytest.fixture(scope="function")
def period_date(period_index):
    return period_index[0]


@pytest.fixture(scope="module")
def observed_handler_expected_detail():
    # Load and format expected results
    expected_detailed_results = pd.read_csv('unit_testing_files/detailed_results_observed.csv', index_col = 0)
    expected_detailed_results.index = expected_detailed_results.index.astype('object')
    cols = expected_detailed_results.columns[expected_detailed_results.columns.str.contains('eventLength')]
    expected_detailed_results[cols] = expected_detailed_results[cols].astype('float64')
    for i_col, col in enumerate(expected_detailed_results):
        if 'daysBetweenEvents' in col:
            for i, val in enumerate(expected_detailed_results[col]):
                new = expected_detailed_results[col].iloc[i]
                if new == '[]':
                    new_list = []
                else:
                    new = re.sub(r'\[', '', new)
                    new = re.sub(r'\]', '', new)
                    new = new.split(',')
                    new_list = []
                    for days in new:
                        new_days = days.strip()
                        new_days = int(new_days)
                        new_list.append(new_days)

                expected_detailed_results.iat[i, i_col] = new_list
    
    return expected_detailed_results

@pytest.fixture(scope="module")
def scenario_handler_expected_detail():
    # Expected output params
    expected_detailed_results = pd.read_csv('unit_testing_files/detailed_results_test.csv', index_col=0)
    expected_detailed_results.index = expected_detailed_results.index.astype('object')
    expected_detailed_results.index.astype('object')
    cols = expected_detailed_results.columns[expected_detailed_results.columns.str.contains('eventLength')]
    expected_detailed_results[cols] = expected_detailed_results[cols].astype('float64')
    for i_col, col in enumerate(expected_detailed_results):
        if 'daysBetweenEvents' in col:
            for i, val in enumerate(expected_detailed_results[col]):
                new = expected_detailed_results[col].iloc[i]
                if new == '[]':
                    new_list = []
                else:
                    new = re.sub(r'\[', '', new)
                    new = re.sub(r'\]', '', new)
                    new = new.split(',')
                    new_list = []
                    for days in new:
                        new_days = days.strip()
                        new_days = int(new_days)
                        new_list.append(new_days)

                expected_detailed_results.iat[i, i_col] = new_list

    return expected_detailed_results

@pytest.fixture(scope="module")
def observed_handler_instance():
    # Set up input parameters and pass to test function
    gauges = ['419039']
    dates = {'start_date': date(2020, 7, 1), 'end_date': date(2021, 6, 30)}
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'

    ewr_oh = observed_handling.ObservedHandler(gauges, dates, allowance, climate, parameter_sheet='unit_testing_files/parameter_sheet.csv')

    ewr_oh.process_gauges()

    return ewr_oh

@pytest.fixture(scope="module")
def scenario_handler_instance():
    # Testing the MDBA bigmod format:
    # Input params
    scenarios =  ['unit_testing_files/Low_flow_EWRs_Bidgee_410007.csv']
    model_format = 'Bigmod - MDBA'
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'
    
    # Pass to the class
    
    ewr_sh = scenario_handling.ScenarioHandler(scenarios, model_format, allowance, climate)
    
    ewr_sh.process_scenarios()

    return ewr_sh

@pytest.fixture(scope="function")
def parameter_sheet():
    EWR_table, _ = data_inputs.get_EWR_table("unit_testing_files/parameter_sheet.csv")
    return EWR_table


@pytest.fixture(scope="function")
def wp_df_F_df_L():

    murray_IQQM_df_wp = pd.read_csv("unit_testing_files/murray_IQQM_df_wp.csv", index_col = 'Date')
    df_F, df_L = scenario_handling.cleaner_IQQM_10000yr(murray_IQQM_df_wp)

    return df_F, df_L

@pytest.fixture(scope="function")
def wp_EWR_table(parameter_sheet):

    wp_flow_level_gauges = ['414203', '414209', '425010', '4260501' ]


    return parameter_sheet[(parameter_sheet["Gauge"].isin(wp_flow_level_gauges))&(parameter_sheet["Code"].isin(["WP3","WP4","LF2_WP","SF_WP"]))] 


@pytest.fixture(scope="function")
def PU_df_wp():
    df_data = {
              'WP3_eventYears': {1896: 1, 1897: 0, 1898: 1, 1895: 0}, 
              'WP3_numAchieved': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
              'WP3_numEvents': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
              'WP3_numEventsAll': {1896: 1, 1897: 1, 1898: 1, 1895: 1},  
              
              'WP4_eventYears': {1896: 0, 1897: 1, 1898: 0, 1895: 1}, 
              'WP4_numAchieved': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
              'WP4_numEvents': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
              'WP4_numEventsAll': {1896: 1, 1897: 1, 1898: 1, 1895: 1},  
              
              'SF_WP_eventYears': {1896: 1, 1897: 1, 1898: 0, 1895: 0}, 
              'SF_WP_numAchieved': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
              'SF_WP_numEvents': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
              'SF_WP_numEventsAll': {1896: 1, 1897: 1, 1898: 1, 1895: 1},  
              
              'LF2_WP_eventYears': {1896: 0, 1897: 0, 1898: 1, 1895: 1}, 
              'LF2_WP_numAchieved': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
              'LF2_WP_numEvents': {1896: 1, 1897: 1, 1898: 1, 1895: 1}, 
              'LF2_WP_numEventsAll': {1896: 1, 1897: 1, 1898: 1, 1895: 1}
              } 

    return pd.DataFrame(df_data)

@pytest.fixture(scope="function")
def interEvent_item_to_process():
    return {'scenario': ['example_scenario']*9,
            'gauge': ['409025']*6+['410007']*3, 
            'pu': ['Murray River - Yarrawonga to Barmah']*6+['Upper Yanco Creek']*3, 
            'ewr': ['VF']*3+['LF2']*3+['SF2']*3,
            'waterYear': ['1901', '1901', '1904']*3, 
            'startDate': [date(1901, 8, 1), date(1901, 12, 1), date(1904, 1, 31), date(1901, 8, 5), date(1901, 12, 1), date(1904, 1, 31), date(1901, 8, 10), date(1901, 12, 6), date(1904, 1, 31)], 
            'endDate': [date(1901, 8, 31), date(1901, 12, 15), date(1904, 3, 31), date(1901, 8, 25), date(1901, 12, 10), date(1904, 2, 15), date(1901, 8, 15), date(1901, 12, 8), date(1904, 2, 5)], 
            'eventDuration': [31, 15, 61, 21, 10, 16, 6, 3, 6],
            'eventLength': [31, 15, 61, 21, 10, 16, 6, 3, 6],
            'Multigauge': ['']*9}

@pytest.fixture(scope="function")
def successfulEvent_item_to_process():
    return {'scenario': ['example_scenario']*9,
            'gauge': ['409025']*6+['410007']*3, 
            'pu': ['Murray River - Yarrawonga to Barmah']*6+['Upper Yanco Creek']*3, 
            'ewr': ['VF']*3+['LF2']*3+['SF2']*3,
            'waterYear': ['1901', '1901', '1904']*3, 
            'startDate': [date(1901, 8, 1), date(1901, 12, 1), date(1904, 1, 31), date(1901, 8, 5), date(1901, 12, 1), date(1904, 1, 31), date(1901, 8, 10), date(1901, 12, 6), date(1904, 1, 31)], 
            'endDate': [date(1901, 8, 31), date(1901, 12, 2), date(1904, 3, 31), date(1901, 8, 25), date(1901, 12, 9), date(1904, 2, 15), date(1901, 8, 15), date(1901, 12, 8), date(1904, 3, 5)], 
            'eventDuration': [31, 2, 61, 21, 9, 16, 6, 3, 34],
            'eventLength': [31, 2, 61, 21, 9, 16, 6, 3, 34],
            'Multigauge': ['']*9}