from datetime import date, timedelta, datetime

import pandas as pd
import pytest


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
                "ewr_events" : ( {  2010: [],
                                    2011: [],
                                    2012: [],
                                    2013: [],
                                    2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},)}

@pytest.fixture(scope="function")
def event_items_to_process():
    return  [{  "scenario" : 'observed',
                "gauge" : '419001',
                "pu" : 'Keepit to Boggabri',
                "ewr": 'CF1_a',
                "ewr_events" : ( {  2010: [],
                                    2011: [],
                                    2012: [],
                                    2013: [],
                                    2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},)},
                {  "scenario" : 'observed',
                "gauge" : '419002',
                "pu" : 'Keepit to Boggabri',
                "ewr": 'CF1_a',
                "ewr_events" : ( {  2010: [],
                                    2011: [],
                                    2012: [],
                                    2013: [],
                                    2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},)}
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


