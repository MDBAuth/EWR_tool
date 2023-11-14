from pathlib import Path
import io

import pandas as pd
from pandas._testing import assert_frame_equal
import numpy as np
import requests
from datetime import datetime
import os

from py_ewr import data_inputs
import pytest

BASE_PATH = Path(__file__).resolve().parents[1]   
    
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

def test_get_EWR_table():
    '''
    - 1. Test for equal entries (no lost EWRs)
    - 2. Test to ensure no bad EWRs make it through using a subset of EWRs
    '''
    # Test 1
    proxies={} # Populate with your proxy settings
    my_url = os.path.join(BASE_PATH, "py_ewr/parameter_metadata/parameter_sheet.csv")
    df = pd.read_csv(my_url,
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

def test_get_ewr_calc_config():
    '''
    1. Test for correct return of EWR calculation config
    assert it returns a dictionary
    '''

    ewr_calc_config = data_inputs.get_ewr_calc_config()

    assert isinstance(ewr_calc_config, dict)
    assert "flow_handle" in ewr_calc_config.keys()


def test_get_barrage_flow_gauges():
    result = data_inputs.get_barrage_flow_gauges()
    assert isinstance(result, dict)
    # there is only one key in the dictionary
    assert len(result) == 1
    ## all key main gauge belong to the list of gauges
    for k, v in result.items():
        assert isinstance(v, list)
        assert k in v


def test_get_barrage_level_gauges():
    result = data_inputs.get_barrage_level_gauges()
    assert isinstance(result, dict)
    # there is only one key in the dictionary
    ## all key main gauge belong to the list of gauges
    for k, v in result.items():
        assert isinstance(v, list)
        assert k in v


def test_get_cllmm_gauges():
    result = data_inputs.get_cllmm_gauges()
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

@pytest.mark.parametrize('expected_results', 
[
    (
    ['A4260527', 'A4260633', 'A4260634', 'A4260635', 'A4260637', 'A4261002']
    )
])
def test_get_scenario_gauges(gauge_results, expected_results):
    result = data_inputs.get_scenario_gauges(gauge_results)
    assert sorted(result) == expected_results
    