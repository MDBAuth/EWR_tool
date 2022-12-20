from pathlib import Path
import io

import pandas as pd
from pandas._testing import assert_frame_equal
import numpy as np
import requests
from datetime import datetime
import os

from py_ewr import data_inputs

BASE_PATH = Path(__file__).resolve().parents[1]
    
    
def test_gauge_to_catchment():
    '''
    1. Test gauge in Gwydir returns Gwydir
    2. Test gauge in Namoi returns Namoi
    '''
    # Test 1
    expected_catchment = 'Gwydir'
    input_gauge = '418026'
    catchment = data_inputs.gauge_to_catchment(input_gauge)
    assert catchment == expected_catchment
    # Test 2
    expected_catchment = 'Namoi'
    input_gauge = '419049'
    catchment = data_inputs.gauge_to_catchment(input_gauge)
    assert catchment == expected_catchment    
    
def test_wy_to_climate():
    '''
    1. Test an array of climate categorisations are returned for each day of the input data
    '''
    # Setting up input data 
    data = {'418026': list(range(0,730,1)), '425012': list(range(0,1460,2))}
    index = pd.date_range(start=datetime.strptime('2017-01-01', '%Y-%m-%d'), end=datetime.strptime('2018-12-31', '%Y-%m-%d'))
    df = pd.DataFrame(data = data, index = index)
    water_years = np.array([2017]*365+[2018]*365)
    climate_daily = data_inputs.wy_to_climate(water_years, 'Gwydir', 'Standard - 1911 to 2018 climate categorisation')
    # Setting up expected output data and the test
    expected_climate_daily = np.array(['Dry']*365 + ['Very Dry']*365)
    assert np.array_equal(climate_daily, expected_climate_daily)
    
    
def test_get_multi_gauges():
    '''
    1. Test for returning planning units and gauges where there are multi gauge EWR requirements
    2. Test for returning the unique gauge to gauge dictionaries where there are multi gauge EWR requirements
    '''
    # Test 1
    expected_multi_gauges = {'PU_0000130': {'421090': '421088', '421088': '421090'},
              'PU_0000131': {'421090': '421088', '421088': '421090'},
              'PU_0000132': {'421090': '421088', '421088': '421090'},
              'PU_0000133': {'421090': '421088', '421088': '421090'},
              'PU_0000251': {'423001': '423002'},
              'PU_0000280': {'1AS': '1ES'}
             }
    multi_gauges = data_inputs.get_multi_gauges('all')
    assert expected_multi_gauges == multi_gauges
    #--------------------------------------------------------
    # Test 2
    expected_multi_gauges = {'421090': '421088', '421088': '421090', '423001': '423002', '1AS': '1ES'}
    multi_gauges = data_inputs.get_multi_gauges('gauges')
    assert expected_multi_gauges == multi_gauges
    
    
def test_get_simultaneous_gauges():
    '''
    1. Test for returning planning units and gauges where there are simultaneous gauge EWR requirements
    2. Test for returning the unique gauge to gauge dictionaries where there are simultaneous gauge EWR requirements
    '''
    # Testing the return all request
    expected_sim_gauges = {'PU_0000131': {'421090': '421022', '421022': '421090'},
                            'PU_0000132': {'421090': '421022', '421022': '421090'},
                            'PU_0000133': {'421090': '421022', '421022': '421090'}}
    sim_gauges = data_inputs.get_simultaneous_gauges('all')
    assert expected_sim_gauges == sim_gauges
    #----------------------------------------------------
    # Testing the return just gauges request
    expected_sim_gauges = {'421090': '421022', '421022': '421090'}
    sim_gauges = data_inputs.get_simultaneous_gauges('gauges')
    assert expected_sim_gauges == sim_gauges
    
    
def test_get_climate_cats():
    '''
    - 1. Loading the correct climate cat csv
    '''
    # Test 1
    climate_file = 'Standard - 1911 to 2018 climate categorisation'
    result = data_inputs.get_climate_cats(climate_file)
    expected_result = pd.read_csv( BASE_PATH / 'py_ewr'/'climate_data/climate_cats.csv', index_col = 0)
    assert_frame_equal(result, expected_result)
    

def test_get_EWR_table():
    '''
    - 1. Test for equal entries (no lost EWRs)
    - 2. Test to ensure no bad EWRs make it through using a subset of EWRs
    '''
    # Test 1
    proxies={} # Populate with your proxy settings
    my_url = os.path.join(BASE_PATH, "py_ewr/parameter_metadata/NSWEWR.csv")
    #s = requests.get(my_url, proxies=proxies).text
    df = pd.read_csv(my_url,#io.StringIO(s),
                        usecols=['PlanningUnitID', 'PlanningUnitName',  'LTWPShortName', 'CompliancePoint/Node', 'Gauge', 'Code', 'StartMonth',
                              'EndMonth', 'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax', 'EventsPerYear', 'Duration', 'MinSpell', 
                              'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                              'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'AccumulationPeriod',
                              'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek'],
                     dtype='str', encoding='cp1252'
                    )
    
    # Get the cleaned dataset:
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    
    total_len = len(EWR_table)+len(bad_EWRs)
    assert len(df), total_len
    
def test_map_gauge_to_catchment():
    '''
    1. Run test data (stored on MDBA public data repository) through to see if gauges are mapping correctly
    '''
    
    EWR_table = os.path.join(BASE_PATH, "py_ewr/parameter_metadata/NSWEWR.csv")
    
    result = data_inputs.map_gauge_to_catchment(EWR_table)
    expected_result = {'419007': 'Namoi River downstream of Keepit Dam ',
                                 '419001': 'Namoi River at Gunnedah', '419012': 'Namoi River at Boggabri ', 
                                 '419039': 'Namoi River at Mollee ', '419021': 'Namoi River at Bugilbone ', 
                                 '419026': 'Namoi River at Goangra ', '419091': 'Namoi River upstream of Walgett ',
                                  '419049': 'Pian Creek at Waminda ', '419020': 'Manilla River at Brabri', '419045':
                                  'Peel River downstream Chaffey Dam', '419015': 'Peel River at Piallamore ', '419006': 
                                  'Peel River at Carroll ', '419016': 'Cockburn River at Mulla Crossing ', '419028':
                                   'Macdonald River at Retreat ', '419022': 'Namoi River at Manilla ', '419027': 
                                   'Mooki River at Breeza ', '419032': 'Coxs Creek at Boggabri '}
    
    assert result["Namoi"] == expected_result      
