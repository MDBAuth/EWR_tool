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
    #my_url = os.path.join(BASE_PATH, "py_ewr/parameter_metadata/parameter_sheet.csv")
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
    # Test 2 
    # setting the conditions for each column or combination of 
    # columns we want to check against:
    drop_conditions = {
        ("FlowThresholdMin", "FlowThresholdMax"): '',
        ("VolumeThreshold",): '',
        ("LevelThresholdMin", "LevelThresholdMax"): '', 
        ('Duration',):'',
        ('StartMonth',): 'See note', 
        ('EndMonth',): 'See note',
        ('Code',): 'DSF', 
        ('Code',): 'DSF1'
    }

    conditions = pd.Series(False, index=EWR_table.index)
    # EWR_table
    for columns, condition in drop_conditions.items():
        current_condition = pd.Series(True, index=EWR_table.index)
        for column in columns:
            current_condition &= (EWR_table[column] == condition)
        conditions |= current_condition

    assert not conditions.all(), "Some values that should be filtered out are still in the final EWR_table"

    # bad_EWRs
    conditions = pd.Series(False, index=bad_EWRs.index)

    for columns, condition in drop_conditions.items():
        current_condition = pd.Series(True, index=bad_EWRs.index)
        for column in columns:
            current_condition |= (bad_EWRs[column] == condition)
        conditions |= current_condition

    assert conditions.any() 

def test_get_EWR_table_errors():
    # - 1. Test if file not found error is raised
    # - 2. Test if columns not found error is raised
    
    # file not found
    url = os.path.join(BASE_PATH, "unit_testing_files/non_exist_parameter_sheet.csv")
    with pytest.raises(FileNotFoundError):
        data_inputs.get_EWR_table(file_path= url)
    
    # correct column names 
    url = os.path.join(BASE_PATH, "unit_testing_files/parameter_sheet_alt_colnames.csv")
    with pytest.raises(ValueError):
        data_inputs.get_EWR_table(file_path= url)



def test_get_ewr_calc_config():
    '''
    1. Test for correct return of EWR calculation config
    assert it returns a dictionary
    '''
    ewr_calc_config = data_inputs.get_ewr_calc_config()

    assert isinstance(ewr_calc_config, dict)
    assert "flow_handle" in ewr_calc_config.keys()


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
    assert len(result) == 3
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

def test_get_vic_level_gauges():
    '''
    1. Test to check if a list is returned by the function, 
        that this list is the correct length, and that the elements are strings
    '''
    result = data_inputs.get_vic_level_gauges()
    assert len(result) == 3
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

def test_get_qld_flow_gauges():
    ''''
    1. Test to check if a list is returned by the function, 
        that this list is the correct length, and that the elements are strings
    '''
    result = data_inputs.get_qld_flow_gauges()
    assert len(result) == 7
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

def test_get_qld_level_gauges():
    ''''
    1. Test to check if a list is returned by the function, 
        that this list is the correct length, and that the elements are strings
    '''
    result = data_inputs.get_qld_level_gauges()
    assert len(result) == 5
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

# scenario gauges
@pytest.mark.parametrize('expected_results', 
[
    (
    ['A4260527', 'A4260633', 'A4260634', 'A4260635', 'A4260637', 'A4261002']
    )
])


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
    1. test that the value error gets raised
    '''
    # Test 1
    url = os.path.join(BASE_PATH, "unit_testing_files/parameter_sheet.csv")
    with pytest.raises(ValueError):
        data_inputs.get_gauges(category = 'not_a_category', 
                   ewr_table_path = url)
    with pytest.raises(ValueError):
        data_inputs.get_gauges(category = '', ewr_table_path=url)


    

# parameter sheet logic test
    
def generate_years(start, end, leap=False):
    if leap == True:
        leaps = [year for year in range(start, end + 1)
                 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]
        rand_year = random.choice(leaps)
    else:
        non_leaps = [year for year in range(start, end + 1)
                     if (year % 4 != 0 and year % 100 == 0) or (year % 400 != 0)]
        rand_year = random.choice(non_leaps)
    return rand_year


def check_EWR_logic(df: pd.DataFrame, year: int) -> str:
    '''
    Checks the logic between columns that specify aspects of the flow regime related to timing
    1. DURATION CHECK: 
        Checks that duration <= days between start and end month
    2. EVENT NUMBER CHECK: 
        Checks that the sum of events*event number per year <= 365 days.
    3. minSpell is not greater than duration
    4. FLOW THRESHOLD CHECK: 
        Checks: minimum flow threshold  =< flow threshold >= maximum flow threshold
    5. TARGET FREQUENCY CHECK:
        Checks minimum target frequenc =< target_frequency >= maximum target frequency
    6. DUPLICATE ROW CHECK
        Check unique combinations of planning unit, gauge occur only once in the dataset
    7. SPECIAL CHARACTER CHECK
        Checks if the dataframe is free of special characters.
    

    args: df: an EWR table as a dataframe
         year: an integer year
    
    return: 
        number of rows per dataset that violate the conditions described above.
    '''
    # Handle StartMonth and EndMonth parsing
    df[['StartMonth', 'StartDay']] = df['StartMonth'].str.split(
        '.', expand=True).fillna(1).astype(int)
    df[['EndMonth', 'EndDay']] = df['EndMonth'].str.split(
        '.', expand=True).fillna(1).astype(int)

    date_cols = ['StartMonth', 'StartDay', 'EndMonth', 'EndDay']
    df[date_cols] = df[date_cols].replace(0, 1)

    df['StartDate'] = pd.to_datetime(df.apply(
        lambda row: f"{year}-{row['StartMonth']:02d}-{row['StartDay']:02d}", axis=1))

    df['EndDate'] = pd.to_datetime(df.apply(lambda row: f"{year+1 if row['StartMonth'] >= row['StartDay'] else year} \
                                            -{row['EndMonth']+1:02d}-{row['EndDay']:02d}", axis=1)) - pd.Timedelta(days=1)

    df['DaysBetween'] = (df['EndDate'] - df['StartDate']).dt.days + 1

    # prep columns for calculations

    columns = ['TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax',
           'EventsPerYear', 'Duration', 'MinSpell', 'FlowThresholdMin',
           'FlowThresholdMax']

    df[columns] = df[columns].replace({' ': 0, '': 0}, regex=True).apply(lambda x: x.astype(float))

    # Calculate MaxEventDays
    df['MaxEventDays'] = df['EventsPerYear'] * df['Duration']

   # Duration Check
    duration_violation = df[df['Duration'] > df['DaysBetween']]

    # Event Number Check
    event_number_violation = df[df['MaxEventDays'] > 365]

    # MinSpell Check
    min_spell_violation = df[df['MinSpell'] > df['Duration']]

    # Flow Threshold Check
    flow_threshold_violation = df[(df['FlowThresholdMax'] > 0) & ~(
        df['FlowThresholdMin'] <= df['FlowThresholdMax'])]

    # Target Frequency Check

    target_frequency_violation = df[(df['TargetFrequencyMax'] > 0) & ~((df['TargetFrequencyMin'] <= df['TargetFrequency'])
                                                                       & (df['TargetFrequency'] <= df['TargetFrequencyMax']))]

    # duplicate_EWR planning units and gauges 
    df['unique_ID'] = df['Gauge']+'_'+df['PlanningUnitID']+'_'+df['Code']
    duplicates = df[df.duplicated('unique_ID', keep=False)]
    dup_set = set(duplicates['unique_ID'])

   # special characters not allowed to be in the dataframe
    allowed = list('.') + list('_') + list('-')
    punc_and_spaces = string.whitespace + string.punctuation
    not_allowed = ''.join([re.escape(c)
                          for c in punc_and_spaces if c not in allowed])
    special_char_bool = df.apply(lambda x: x.astype(
        str).str.contains(not_allowed, regex=True)).any(axis=1)

    # Collect indices for each type of violation
    duration_violation_indices = duration_violation.index.tolist()
    event_number_violation_indices = event_number_violation.index.tolist()
    min_spell_violation_indices = min_spell_violation.index.tolist()
    flow_threshold_violation_indices = flow_threshold_violation.index.tolist()
    target_frequency_violation_indices = target_frequency_violation.index.tolist()
    duplicate_indices = duplicates.index.tolist()
    special_char_cols = special_char_bool[special_char_bool].index.tolist()

    # Print indices where violations occur
    print("Duration Violation at rows:", duration_violation_indices)
    print("Event Number Violation at rows:", event_number_violation_indices)
    print("MinSpell Violation at rows:", min_spell_violation_indices)
    print("Flow Threshold Violation at rows:",
          flow_threshold_violation_indices)
    print("Target Frequency Violation at rows:",
          target_frequency_violation_indices)
    print("Duplicate rows:", duplicate_indices),
    print("special characters in the following rows:", special_char_cols)

    # Check if there are no violations
    no_violations = all(len(v) == 0 for v in [
        duration_violation,
        event_number_violation,
        min_spell_violation,
        flow_threshold_violation,
        target_frequency_violation,
        duplicates,
        special_char_cols
    ])
    assert no_violations, "Errors were found with the logic in the EWR table"
