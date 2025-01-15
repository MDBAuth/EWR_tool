from pathlib import Path
import io
import pandas as pd
from pandas._testing import assert_frame_equal
import numpy as np
import requests
from datetime import datetime
import os
from unittest.mock import mock_open, patch
from pathlib import Path
import json
from py_ewr import data_inputs
import pytest
import re

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
    url = os.path.join(BASE_PATH, "unit_testing_files/parameter_sheet.csv")
    df = pd.read_csv(url,
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

# def test_get_ewr_calc_config():
#     # Test with a valid file_path
#     # mock_config = {"Flow_type": ["EWR_code1", "EWR_code2"]}
#     # mock_file_path = "EWR_tool/unit_testing_files/mock_ewr_calc_config.json"
    
#     # with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
#     #     result = data_inputs.get_ewr_calc_config(mock_file_path)
#     #     assert result == mock_config

#     # Test with the default path
#     default_mock_config = {"flow_handle": ["EWR_code1", "EWR_code2"]}
#     default_path = os.path.join(BASE_PATH, "parameter_metadata/ewr_calc_config.json")
    
#     with patch("builtins.open", mock_open(read_data=json.dumps(default_mock_config))):
#         ewr_calc_config = data_inputs.get_ewr_calc_config()
#         assert isinstance(ewr_calc_config, dict)
#         assert "flow_handle" in ewr_calc_config.keys()
#     def find_unusual_characters(s):
#         # Define a regex pattern for unusual characters
#         pattern = r'[^a-zA-Z0-9\s.,!?;:()\'"-]'
        
#         # Find all unusual characters
#         unusual_chars = set(re.findall(pattern, s))
    
#         return unusual_chars
#     # Test for rogue characters
#     #rogue_chars = {'@', '$', '#', "*",''}
#     # Test for rogue characters
#     test_string = "This is a test string with some unusual characters: @, $, #, *, ©, €, ™, ±"
#     rogue_chars =find_unusual_characters(test_string)
#     unique_chars = set()
#     for k, v in ewr_calc_config.items():
#         for char in k:
#             unique_chars.add(char)
#         for char in v:
#             unique_chars.add(char)
    
#     assert not (unique_chars & rogue_chars), f"Rogue characters found: {unique_chars & rogue_chars}"

#     # Test with a nonexistent file
#     mock_file_path = "/mock/path/to/nonexistent_config.json"
    
#     with patch("builtins.open", mock_open()) as mock_file:
#         mock_file.side_effect = FileNotFoundError
#         with pytest.raises(FileNotFoundError):
#             data_inputs.get_ewr_calc_config(mock_file_path)



def test_get_barrage_flow_gauges():
    '''
    1. Test to check if a dictionary of the right length of keys is returned by the 
        function and that this dictionary contains a list of 1 gauge.
    '''
    result = data_inputs.get_barrage_flow_gauges()
    assert isinstance(result, dict)
    # there is only one key in the dictionary
    assert len(result) == 1
    ## all key main gauge belong to the list of gauges
    for k, v in result.items():
        assert isinstance(v, list)
        assert k in v


def test_get_barrage_level_gauges():
    '''
    1. Test to check if a dictionary is returned by the 
        function and that this dictionary contains list of gauges
    '''
    result = data_inputs.get_barrage_level_gauges()
    assert isinstance(result, dict)
    assert len(result) == 2
    ## all key main gauge belong to the list of gauges
    for k, v in result.items():
        assert isinstance(v, list)
        assert k in v

# cllmm_gauges
def test_get_cllmm_gauges():
    '''
    1. Test to check if a list is returned by the function, 
        that this list is the correct length, and that the elements are strings 
    '''
    result = data_inputs.get_cllmm_gauges()
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

# @pytest.mark.parametrize('expected_results', 
# [
#     (
#     ['A4260527', 'A4260633', 'A4260634', 'A4260635', 'A4260637', 'A4261002']
#     )
# ])
# def test_get_scenario_gauges(gauge_results, expected_results):
#     result = data_inputs.get_scenario_gauges(gauge_results)
#     assert sorted(result) == expected_results
@pytest.mark.parametrize(
    "gauge_results, expected_results",
    [
        # Test Case 1: Basic scenario
        (
            {
                "scenario1": {"A4260527": {"data": [1, 2, 3]}, "A4260633": {"data": [4, 5, 6]}},
                "scenario2": {"A4260527": {"data": [7, 8, 9]}, "A4260634": {"data": [10, 11, 12]}},
                "scenario3": {"A4260527": {"data": [13, 14, 15]}, "A4260635": {"data": [16, 17, 18]}},
            },
            ['A4260527', 'A4260633', 'A4260634', 'A4260635']
        ),
        # Test Case 2: Empty scenario
        (
            {},
            []
        ),
        # Test Case 3: Single scenario with single gauge
        (
            {"scenario1": {"A4260527": {"data": [1, 2, 3]}}},
            ['A4260527']
        ),
        # Test Case 4: Multiple scenarios with overlapping gauges
        (
            {
                "scenario1": {"A4260527": {"data": [1, 2, 3]}},
                "scenario2": {"A4260527": {"data": [4, 5, 6]}, "A4260633": {"data": [7, 8, 9]}},
            },
            ["A4260527", "A4260633"]
        ),
    ]
)
def test_get_scenario_gauges(gauge_results, expected_results):
    result = data_inputs.get_scenario_gauges(gauge_results)
    assert sorted(result) == expected_results

def test_get_level_gauges():
    level_result, weirpool_result = data_inputs.get_level_gauges()
    menindee_len = 3
    lachlan_len = 1
    weirpool_len = 4
    assert isinstance(level_result, list)
    assert isinstance(weirpool_result, dict)
    assert len(level_result) == menindee_len+lachlan_len
    assert len(weirpool_result) == weirpool_len


def test_get_gauges():
    '''
    1. test that the value error gets raised.
    '''
    # Test 1
    url = os.path.join(BASE_PATH, "unit_testing_files/parameter_sheet.csv")
    with pytest.raises(ValueError):
        data_inputs.get_gauges(category = 'not_a_category', 
                   ewr_table_path = url)
    with pytest.raises(ValueError):
        data_inputs.get_gauges(category = '', ewr_table_path=url)
    
def test_get_iqqm_codes():
    result = data_inputs.get_iqqm_codes()
    stations = {
        '229': '421023',
        '42': '421001',
        '464': '421011',
        '240': '421019',
        '266': '421146',
        '951': '421090',
        '487': '421022',
        '130': '421012',
        '171': '421004',
    }
    assert stations == result


# def generate_years(start, end, leap = False):
#     if leap == True:
#         leaps =  [year for year in range(start, end + 1) \
#                 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]
#         rand_year = random.choice(leaps)
#     else:
#         non_leaps = [year for year in range(start, end + 1) \
#                 if (year % 4 != 0 and year % 100 == 0) or (year % 400 != 0)]
#         rand_year = random.choice(non_leaps)
#     return rand_year

# def check_EWR_logic(df: pd.DataFrame, year: int):
#     '''
#     Checks the logic between columns that specify aspects of the flow regime related to timing
#     1. DURATION CHECK: 
#         Checks that duration <= days between start and end month
#     2. EVENT NUMBER CHECK: 
#         Checks that the sum of events*event number per year <= 365 days.
#     3. minSpell is not greater than duration
#     4. FLOW THRESHOLD CHECK: 
#         Checks: minimum flow threshold  =< maximum flow threshold
#     5. TARGET FREQUENCY CHECK:
#         Checks minimum target frequenc =< target_frequency >= maximum target frequency
#     6. LEVEL THRESHOLD CHECK
#         Check minimum level threshold =< maximum level threshold. 
#     7. DUPLICATE ROW CHECK
#         Check unique combinations of planning unit, gauge occur only once in the dataset
#     8. SPECIAL CHARACTER CHECK
#         Checks if the dataframe is free of special characters.
    

#     args: df: an EWR table as a dataframe
#          year: an integer year
    
#     return: 
#         number of rows per dataset that violate the conditions described above.
#     '''
#     df = df.copy()

#     # Create a regex pattern that matches strings composed only of the allowed characters
#     allowed_chars = '.' + '_' + '-'+''+' '+string.ascii_letters + string.digits
#     pattern = f'^[{re.escape(allowed_chars)}]*$'

#     #Apply the pattern to filter the DataFrame, keeping rows where all cells match the pattern
#     spec_char = df[~df.apply(lambda x: x.astype(str).str.match(pattern) | x.isna()).all(axis=1)]
#     okay_df = df[df.apply(lambda x: x.astype(str).str.match(pattern)| x.isna()).all(axis=1)]

#     columns = ['TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax',
#            'EventsPerYear', 'Duration', 'MinSpell', 'FlowThresholdMin',
#            'FlowThresholdMax', 'LevelThresholdMin', 'LevelThresholdMax']
#     okay_df[columns] = okay_df[columns].replace({' ': np.nan, '': np.nan}, regex=True)
#     okay_df[columns] = okay_df[columns].astype(float)
    
#     dur_filter = okay_df.copy()
#     dur_filter[['StartDay', 'EndDay']] = ''
    
    
#     #check_duration

#     dur_filter[['StartMonth', 'StartDay']] = dur_filter['StartMonth'].str.split(
#         '.', expand=True).fillna(1).astype(int)

#     dur_filter[['EndMonth', 'EndDay']] = dur_filter['EndMonth'].str.split(
#          '.', expand=True).fillna(1).astype(int)

#     date_cols = ['StartMonth', 'StartDay', 'EndMonth', 'EndDay']

#     dur_filter[date_cols] = dur_filter[date_cols].replace(0, 1)
    
#     dur_filter = dur_filter[~(dur_filter['StartMonth'] == 7) & (dur_filter["EndMonth"] == 6)]

#     dur_filter['StartDate'] = pd.to_datetime(dur_filter.apply(
#     lambda row: f"{year}-{row['StartMonth']:02d}-{row['StartDay']:02d}", axis=1))

#     dur_filter['EndDate'] = pd.to_datetime(dur_filter.apply(
#     lambda row: f"{year+1 if row['StartMonth'] > row['EndMonth'] else year}-{row['EndMonth']: 02d}-{row['EndDay']: 02d}", axis=1))+pd.offsets.MonthEnd(1)

#     dur_filter['DaysBetween'] = (dur_filter['EndDate'] - dur_filter['StartDate']).dt.days + 1
#     duration_violation = dur_filter[dur_filter['Duration'] > dur_filter['DaysBetween']]

#     # Calculate MaxEventDays
#     okay_df['MaxEventDays'] = okay_df['EventsPerYear'] * okay_df['Duration']

#     # Event Number Check
#     event_number_violation = okay_df[okay_df['MaxEventDays'] > 365] 

#     # MinSpell Check
#     min_spell_violation = okay_df[okay_df['MinSpell'] > okay_df['Duration']]

#     # # Flow Threshold Check
#     flow_threshold_violation = okay_df[(okay_df['FlowThresholdMax'] > 0) & (okay_df['FlowThresholdMin'] > okay_df['FlowThresholdMax'])]

#     # level Threshold Check
#     level_threshold_violation = okay_df[(okay_df['LevelThresholdMax'] > 0) & (
#         okay_df['LevelThresholdMin'] > okay_df['LevelThresholdMax'])]

#     # Target Frequency Check
#     target_frequency_violation = okay_df[
#         (okay_df['TargetFrequencyMax'] > 0) & 
#         (
#             (okay_df['TargetFrequencyMin'] >= okay_df['TargetFrequency']) | 
#             (okay_df['TargetFrequency'] >= okay_df['TargetFrequencyMax']) |
#             (okay_df['TargetFrequencyMin'] >= okay_df['TargetFrequencyMax'])
#         )
#     ]

#     # duplicate_EWR planning units and gauges 
#     okay_df['unique_ID'] = okay_df['Gauge']+'_'+okay_df['PlanningUnitID']+'_'+okay_df['Code']
#     duplicates = okay_df[okay_df.duplicated('unique_ID', keep=False)]
#     dup_set = set(duplicates['unique_ID'])


#     # Collect indices for each type of violation
#     duration_violation_indices = duration_violation.index.tolist()
#     event_number_violation_indices = event_number_violation.index.tolist()
#     min_spell_violation_indices = min_spell_violation.index.tolist()
#     flow_threshold_violation_indices = flow_threshold_violation.index.tolist()
#     level_threshold_violation_indices = level_threshold_violation.index.tolist()
#     target_frequency_violation_indices = target_frequency_violation.index.tolist()
#     duplicate_indices = duplicates.index.tolist()
#     special_char_cols = spec_char.index.tolist()

#     # Print indices where violations occur
#     print("Duration Violation at rows:", duration_violation_indices)
#     print("Event Number Violation at rows:", event_number_violation_indices)
#     print("MinSpell Violation at rows:", min_spell_violation_indices)
#     print("Flow Threshold Violation at rows:",
#           flow_threshold_violation_indices)
#     print("Level Threshold Violation at rows:",
#           level_threshold_violation_indices)
#     print("Target Frequency Violation at rows:",
#           target_frequency_violation_indices)
#     print("Duplicate rows:", duplicate_indices),
#     print("special characters in the following rows:", special_char_cols)

#     # Check if there are no violations
#     no_violations = all(len(v) == 0 for v in [
#          duration_violation,
#         event_number_violation,
#         min_spell_violation,
#         flow_threshold_violation,
#         level_threshold_violation,
#         target_frequency_violation,
#         duplicates,
#         special_char_cols
#     ])
#     assert no_violations, "Errors were found with the logic in the EWR table"

#     # run on EWR_table
    

# def test_check_EWR_logics():
#     non_leap = generate_years(1910, 2024)
#     leap = generate_years(1910, 2024, True)
#     url = os.path.join(BASE_PATH, "py_ewr/parameter_metadata/parameter_sheet.csv")
#     EWR_table, bad_EWRs = data_inputs.get_EWR_table(url)
#     check_EWR_logic(EWR_table, non_leap)
#     check_EWR_logic(EWR_table, leap)


def test_get_bad_QA_codes():
        expected_result = [151, 152, 153, 155, 180, 201, 202, 204, 205, 207, 223, 255]
        assert data_inputs.get_bad_QA_codes() == expected_result

def test_ewr_parameter_grabber():
    mock_data={
            'Gauge': ['G1', 'G2', 'G1', 'G3'],
            'Code': ['EWR1', 'EWR2', 'EWR1', 'EWR3'],
            'PlanningUnitName': ['PU1', 'PU2', 'PU1', 'PU3'],
            'Parameter1': ['Value1', 'Value2', 'Value3', 'Value4'],
            'Parameter2': ['Value5', 'Value6', 'Value7', 'Value8']
        }
    ewr_table_mock=pd.DataFrame(mock_data)
    result=data_inputs.ewr_parameter_grabber(ewr_table_mock,'G1', 'PU1', 'EWR1', 'Parameter1')
    #testing basic functionality
    assert result == 'Value1'
    #testing non matching data
    with pytest.raises(IndexError):
        data_inputs.ewr_parameter_grabber(ewr_table_mock, 'G4', 'PU1', 'EWR1', 'Parameter1')
    with pytest.raises(KeyError):
        data_inputs.ewr_parameter_grabber(ewr_table_mock, 'G4', 'PU1', 'EWR1', 'NonExistingParameter')
    with pytest.raises(IndexError):
        empty_df = pd.DataFrame(columns=mock_data.keys())
        data_inputs.ewr_parameter_grabber(empty_df, 'G4', 'PU1', 'EWR1', 'Parameter1')

   
  
def test_weirpool_type():
    assert data_inputs.weirpool_type('WP1') == 'falling'
    assert data_inputs.weirpool_type('WP2') == 'raising'
    assert data_inputs.weirpool_type('WP3') == 'falling'
    assert data_inputs.weirpool_type('WP4') == 'falling'
def test_get_causal_ewr():

    causal_ewr = data_inputs.get_causal_ewr()
    assert isinstance(causal_ewr, dict)

    assert len(causal_ewr) > 0
    
    expected_keys = {"ewr2obj", "obj2target", "obj2yrtarget"}
    assert set(causal_ewr.keys()) == expected_keys
    
    for value in causal_ewr.values():
        assert isinstance(value, pd.DataFrame)

