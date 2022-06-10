from datetime import date, timedelta, datetime
import re

import pandas as pd
import pytest

from py_ewr import observed_handling, scenario_handling

@pytest.fixture(scope="function")
def pu_df():
    df_data = {'CF1_a_eventYears': {2016: 0, 2017: 0},
                'CF1_a_numAchieved': {2016: 0, 2017: 0},
                'CF1_a_numEvents': {2016: 0, 2017: 0},
                'CF1_a_eventLength': {2016: 0.0, 2017: 0.0},
                'CF1_a_totalEventDays': {2016: 0, 2017: 0},
                'CF1_a_daysBetweenEvents': {2016: [], 2017: []},
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
    for col in expected_detailed_results:
        if 'daysBetweenEvents' in col:
            for i, val in enumerate(expected_detailed_results[col]):
                new = expected_detailed_results[col].iloc[i]
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

                expected_detailed_results[col].iloc[i] = new_list
    
    return expected_detailed_results

@pytest.fixture(scope="module")
def scenario_handler_expected_detail():
    # Expected output params
    expected_detailed_results = pd.read_csv('unit_testing_files/detailed_results_test.csv', index_col=0)
    expected_detailed_results.index = expected_detailed_results.index.astype('object')
    expected_detailed_results.index.astype('object')
    cols = expected_detailed_results.columns[expected_detailed_results.columns.str.contains('eventLength')]
    expected_detailed_results[cols] = expected_detailed_results[cols].astype('float64')
    for col in expected_detailed_results:
        if 'daysBetweenEvents' in col:
            for i, val in enumerate(expected_detailed_results[col]):
                new = expected_detailed_results[col].iloc[i]
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

                expected_detailed_results[col].iloc[i] = new_list

    return expected_detailed_results

@pytest.fixture(scope="module")
def observed_handler_instance():
    # Set up input parameters and pass to test function
    gauges = ['419039']
    dates = {'start_date': date(2020, 7, 1), 'end_date': date(2021, 6, 30)}
    allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
    climate = 'Standard - 1911 to 2018 climate categorisation'

    ewr_oh = observed_handling.ObservedHandler(gauges, dates, allowance, climate)

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
