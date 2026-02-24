from pathlib import Path
import io

import pandas as pd
from pandas._testing import assert_frame_equal
import numpy as np
import requests
from datetime import datetime
import os
import random
import string
import pytest
import re
import json


from unittest.mock import mock_open, patch
from py_ewr import data_inputs


BASE_PATH = Path(__file__).resolve().parents[1]   

parameter_sheet_path = os.path.join(BASE_PATH, "py_ewr/parameter_metadata/parameter_sheet.csv")
    
def test_get_multi_gauges():
    '''
    1. Test for returning planning units and gauges where there are multi gauge ewr requirements
    2. Test for returning the unique gauge to gauge dictionaries where there are multi gauge ewr requirements
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
    - 3. Test EWR table columns match the expected columns
    -4. Test error thrown when startMonth and endMonth are not found in the table columns
    '''
    # Test 1
    proxies={} # Populate with your proxy settings

    cols=('PlanningUnitID', 'PlanningUnitName',  'LTWPShortName', 'CompliancePoint/Node', 'Gauge', 'Code',
                                'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax', 'EventsPerYear', 'Duration', 'MinSpell', 
                                'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                                'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'AccumulationPeriod',
                                'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek', 'CtfThreshold','NonFlowSpell', 'MaxLevelChange')
    essential_cols = ('StartMonth', 'EndMonth')

    comb_cols = cols + essential_cols

    df = pd.read_csv(parameter_sheet_path,
                    usecols = comb_cols,
                     dtype='str', encoding='cp1252'
                    )
    
    # Get the cleaned dataset:
    # testing 1 and 2 
    EWR_table = data_inputs.get_EWR_table()

    assert len(df), len(EWR_table)

    #test 3 
    EWR_table = data_inputs.get_EWR_table(file_path = parameter_sheet_path, columns_to_keep = comb_cols)
    assert sorted(EWR_table.columns.tolist()) == sorted(comb_cols+('StartDay', 'EndDay'))

    # test 4 if start month and end month are not in the parmaeter sheet raise error
    
    with pytest.raises(KeyError):
        EWR_table = data_inputs.get_EWR_table(file_path = parameter_sheet_path, columns_to_keep = cols)

def test_get_ewr_calc_config():
    '''
    1. Test for correct return of ewr calculation config
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


def generate_years(start, end, leap = False):
    if leap == True:
        leaps =  [year for year in range(start, end + 1) \
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]
        rand_year = random.choice(leaps)
    else:
        non_leaps = [year for year in range(start, end + 1) \
                if (year % 4 != 0 and year % 100 == 0) or (year % 400 != 0)]
        rand_year = random.choice(non_leaps)
    return rand_year

def check_EWR_logic(df: pd.DataFrame, year: int):
    '''
    Checks the logic between columns that specify aspects of the flow regime related to timing
    1. DURATION CHECK: 
        Checks that duration <= days between start and end month
    2. EVENT NUMBER CHECK: 
        Checks that the sum of events*event number per year <= 365 days.
    3. minSpell is not greater than duration
    4. FLOW THRESHOLD CHECK: 
        Checks: minimum flow threshold  =< maximum flow threshold
    5. TARGET FREQUENCY CHECK:
        Checks minimum target frequenc =< target_frequency >= maximum target frequency
    6. LEVEL THRESHOLD CHECK
        Check minimum level threshold =< maximum level threshold. 
    7. DUPLICATE ROW CHECK
        Check unique combinations of planning unit, gauge occur only once in the dataset
    8. SPECIAL CHARACTER CHECK
        Checks if the dataframe is free of special characters.
    9. MISSING PARAMETER CHECK
        Check that potential missing parameters are resolved
    

    args: df: an EWR table as a dataframe
         year: an integer year
    
    return: 
        number of rows per dataset that violate the conditions described above.
    '''
    checking_df = df.copy()

    # Convert numeric columns to float
    columns = ['TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax',
           'EventsPerYear', 'Duration', 'MinSpell', 'FlowThresholdMin',
           'FlowThresholdMax', 'LevelThresholdMin', 'LevelThresholdMax']
    checking_df[columns] = checking_df[columns].replace({' ': np.nan, '': np.nan}, regex=True)
    checking_df[columns] = checking_df[columns].astype(float)
    
    # Duration check
    dur_filter = checking_df.copy()
    
    date_cols = ['StartMonth', 'StartDay', 'EndMonth', 'EndDay']

    # Convert all date columns: handle NaN and zeros
    for col in date_cols:
        dur_filter[col] = pd.to_numeric(dur_filter[col], errors='coerce').fillna(1).replace(0, 1).astype(int)
    
    # # Filter out July-June water years
    # dur_filter = dur_filter[~((dur_filter['StartMonth'] == 7) & (dur_filter["EndMonth"] == 6))]

    # Create date columns
    dur_filter['StartDate'] = pd.to_datetime(dur_filter.apply(
        lambda row: f"{year}-{row['StartMonth']:02d}-{row['StartDay']:02d}", axis=1))

    dur_filter['EndDate'] = pd.to_datetime(dur_filter.apply(
        lambda row: f"{year+1 if row['StartMonth'] > row['EndMonth'] else year}-{row['EndMonth']:02d}-{row['EndDay']:02d}", axis=1)) + pd.offsets.MonthEnd(1)

    dur_filter['DaysBetween'] = (dur_filter['EndDate'] - dur_filter['StartDate']).dt.days + 1


    # remove cease to flows since they don't follow the same rules
    dur_filter_cf_removed = dur_filter[~dur_filter['Code'].str.contains('CF|CTF')]
    
    # Duration violation check with NaN handling
    duration_violation = dur_filter_cf_removed[
        (dur_filter_cf_removed['Duration'] > dur_filter_cf_removed['DaysBetween']) &
        dur_filter_cf_removed['Duration'].notna() &
        dur_filter_cf_removed['DaysBetween'].notna()
    ]

    checking_df = checking_df[~checking_df['Code'].str.contains('CF|CTF')]

    # Calculate MaxEventDays
    checking_df['MaxEventDays'] = checking_df['EventsPerYear'] * checking_df['Duration']

    # Event Number Check with NaN handling
    event_number_violation = checking_df[
        (checking_df['MaxEventDays'] > 365) &
        checking_df['MaxEventDays'].notna()
    ]

    # MinSpell Check with NaN handling
    min_spell_violation = checking_df[
        (checking_df['MinSpell'] > checking_df['Duration']) &
        checking_df['MinSpell'].notna() &
        checking_df['Duration'].notna()
    ]

    # Flow Threshold Check with NaN handling
    flow_threshold_violation = checking_df[
        (checking_df['FlowThresholdMax'] > 0) & 
        (checking_df['FlowThresholdMin'] > checking_df['FlowThresholdMax']) &
        checking_df['FlowThresholdMin'].notna() &
        checking_df['FlowThresholdMax'].notna()
    ]

    # Level Threshold Check with NaN handling
    level_threshold_violation = checking_df[
        (checking_df['LevelThresholdMax'] > 0) & 
        (checking_df['LevelThresholdMin'] > checking_df['LevelThresholdMax']) &
        checking_df['LevelThresholdMin'].notna() &
        checking_df['LevelThresholdMax'].notna()
    ]

    # Target Frequency Check with NaN handling
    target_frequency_violation = checking_df[
        (checking_df['TargetFrequencyMax'] > 0) & 
        (
            (checking_df['TargetFrequencyMin'] >= checking_df['TargetFrequency']) &
            (checking_df['TargetFrequency'] >= checking_df['TargetFrequencyMax']) &
            (checking_df['TargetFrequencyMin'] >= checking_df['TargetFrequencyMax'])
        ) &
        checking_df['TargetFrequency'].notna() &
        checking_df['TargetFrequencyMin'].notna() &
        checking_df['TargetFrequencyMax'].notna()
    ]

    # Duplicate EWR planning units and gauges 
    checking_df['unique_ID'] = checking_df['Gauge'] + '_' + checking_df['PlanningUnitID'] + '_' + checking_df['Code']
    duplicates = checking_df[checking_df.duplicated('unique_ID', keep=False)]


    # additional conditions
    #  EWRs with 'See notes' in start and month
    see_notes = checking_df[(checking_df["StartMonth"] == 'See note') & (checking_df["EndMonth"] == 'See note')]

    # # Filtering those with no flow/level/volume thresholds
    no_thresh = checking_df[(checking_df["FlowThresholdMin"] == '') & \
                    (checking_df["FlowThresholdMax"] == '') &\
                    (checking_df["VolumeThreshold"] == '') &\
                    (checking_df["LevelThresholdMin"] == '') &\
                    (checking_df["LevelThresholdMax"] == '')]

    # # Filtering those with no durations
    no_duration = checking_df[checking_df["Duration"] == '']

    # # Filtering DSF EWRs
    DSF_EWRs = checking_df[checking_df['Code'].str.startswith('DSF')]
   

     # Special Character Check
    allowed_chars = string.ascii_letters + string.digits+',()+-_./:%@ '
    pattern = f'^[{re.escape(allowed_chars)}]*$'
    
    # Apply the pattern to filter the DataFrame
    spec_char = df[~df.apply(lambda x: x.astype(str).str.match(pattern) | x.isna()).all(axis=1)]

    # print the issues

    if not duration_violation.empty:
        print('#------------- Duration Violation -------------#')
        print(duration_violation[['Gauge', 'Code', 'LTWPShortName', 'Duration', 'DaysBetween']])
    if not no_duration.empty:
        print('#------------- Empty duration -------------#')
        print(no_duration[['Gauge', 'Code', 'LTWPShortName', 'Duration']])
    if not event_number_violation.empty:
        print('#------------- Event number over length of seasonal window -------------#')
        print(event_number_violation[['Gauge', 'Code', 'LTWPShortName', 'MaxEventDays', 'EventsPerYear', 'Duration']])
    if not min_spell_violation.empty:
        print('#------------- Minimum Spell > Duration -------------#')
        print(min_spell_violation[['Gauge', 'Code', 'LTWPShortName', 'MinSpell', 'Duration']])
    if not flow_threshold_violation.empty:
        print('#------------- Flow Threshold min max relationships not correct -------------#')
        print(flow_threshold_violation[['Gauge', 'Code', 'LTWPShortName', 'FlowThresholdMin', 'FlowThresholdMax']])
    if not level_threshold_violation.empty:
        print('#------------- Level Threshold min max relationships not correct -------------#')
        print(level_threshold_violation[['Gauge', 'Code', 'LTWPShortName', 'LevelThresholdMin', 'LevelThresholdMax']])
    if not target_frequency_violation.empty:
        print('#-------------  Target Frequency relationships not correct -------------#')
        print(target_frequency_violation[['Gauge', 'Code', 'LTWPShortName', 'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax']])
    if not duplicates.empty:
        print('#------------- Duplicate rows -------------#')
        print(duplicates[['Gauge', 'PlanningUnitID', 'LTWPShortName', 'Code']])
    if not see_notes.empty:
        print("#------------- EWRs with 'see notes' in start or end month columns and therefore will not be calculated -------------#")
        print(see_notes[['Gauge', 'PlanningUnitID', 'LTWPShortName', 'Code']])
    if not no_thresh.empty:
        print('#------------- EWRs with no flow or volume thresholds -------------#')
        print(no_thresh[['Gauge', 
                         'PlanningUnitID', 
                         'LTWPShortName', 
                         'Code', 
                         'FlowThresholdMin', 
                         'FlowThresholdMax',
                         'VolumeThreshold', 
                         'LevelThresholdMin',
                         'LevelThresholdMax"']])
    if not DSF_EWRs.empty:
        print(DSF_EWRs[['Gauge', 'PlanningUnitID', 'LTWPShortName', 'Code']])
    if not spec_char.empty:
        print('#------------- Special characters in the following rows -------------#')
        print(spec_char)

     # Check if there are no violations
    no_violations = all(len(v) == 0 for v in [
        duration_violation,
        no_duration,
        event_number_violation,
        min_spell_violation,
        flow_threshold_violation,
        level_threshold_violation,
        target_frequency_violation,
        duplicates,
        see_notes, 
        no_thresh, 
        DSF_EWRs,
        spec_char
    ])
    assert no_violations, "Errors were found with the logic in the EWR table"
    

def test_check_EWR_logic():
    non_leap = generate_years(1910, 2024)
    leap = generate_years(1910, 2024, True)
    parameter_sheet_path = os.path.join(BASE_PATH, "py_ewr/parameter_metadata/parameter_sheet.csv")
    columns_to_keep = ('PlanningUnitID', 'LTWPShortName', 'PlanningUnitName', 'Gauge', 'Code', 'StartMonth', 'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax', 'State', 'SWSDLName',
                          'EndMonth', 'EventsPerYear', 'Duration', 'MinSpell', 
                          'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                          'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'MaxLevelChange', 'AccumulationPeriod',
                          'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek','AnnualBarrageFlow',
                          'ThreeYearsBarrageFlow', 'HighReleaseWindowStart', 'HighReleaseWindowEnd', 'LowReleaseWindowStart', 'LowReleaseWindowEnd',
                          'PeakLevelWindowStart', 'PeakLevelWindowEnd', 'LowLevelWindowStart', 'LowLevelWindowEnd', 'NonFlowSpell','EggsDaysSpell',
                          'LarvaeDaysSpell', 'RateOfRiseMax1','RateOfRiseMax2','RateOfFallMin','RateOfRiseThreshold1',
                          'RateOfRiseThreshold2','RateOfRiseRiverLevel','RateOfFallRiverLevel', 'CtfThreshold', 'GaugeType')
    EWR_table = data_inputs.get_EWR_table(parameter_sheet_path, columns_to_keep)
    check_EWR_logic(EWR_table, non_leap)
    check_EWR_logic(EWR_table, leap)

@pytest.mark.parametrize('test_id,  test_data, expected_start_month, expected_start_day, expected_end_month,expected_end_day', 
    [ 
        (
           'test case 1', 
            {
                'Gauge': ['409025'],
                'Code': ['nestS1'],
                'StartMonth': ['9.15'],  # First has day, second doesn't
                'EndMonth': ['11.15']
            },
            9, 15, 11, 15
        )
        ,

        (
           'test case 2', 
            {
                'Gauge': ['409025'],
                'Code': ['BF3'],
                'StartMonth': ['7'],  # second doesn't have a specific day
                'EndMonth': ['6']
            },
            7, None, 6, None
        )
    ] 
)

def test_modify_EWR_table(
    test_id, 
    test_data,
    expected_start_month,
    expected_start_day,
    expected_end_month,
    expected_end_day 
):
    '''
    Test that datatypes are coerced to the correct datatype
    1. Test a float is separated into StartDay and StartMonth columns (e.g., "9.15" -> month=9, day=15)
    2. Test integer remains integer (e.g., "7" -> month=7, day=None)
    '''
    # Get the EWR table
    
    test_df = pd.DataFrame(test_data)
    
    # Apply the modify function
    result = data_inputs.modify_EWR_table(test_df)

    row = result.iloc[0]
    
    # Test StartMonth
    assert row['StartMonth'] == expected_start_month, \
        f"StartMonth mismatch for {test_id}: expected {expected_start_month}, got {row['StartMonth']}"
    
    # Test StartDay
    if expected_start_day is None:
        assert pd.isna(row['StartDay']) or row['StartDay'] is None, \
            f"StartDay should be None/NaN for {test_id}, got {row['StartDay']}"
    else:
        assert row['StartDay'] == expected_start_day, \
            f"StartDay mismatch for {test_id}: expected {expected_start_day}, got {row['StartDay']}"
    
    # Test EndMonth
    assert row['EndMonth'] == expected_end_month, \
        f"EndMonth mismatch for {test_id}: expected {expected_end_month}, got {row['EndMonth']}"
    
    # Test EndDay
    if expected_end_day is None:
        assert pd.isna(row['EndDay']) or row['EndDay'] is None, \
            f"EndDay should be None/NaN for {test_id}, got {row['EndDay']}"
    else:
        assert row['EndDay'] == expected_end_day, \
            f"EndDay mismatch for {test_id}: expected {expected_end_day}, got {row['EndDay']}"

def test_modify_EWR_table_datatypes():
    # Get the EWR table
    test_df = pd.read_csv(parameter_sheet_path, dtype = 'str')
    
    # Apply the modify function
    result = data_inputs.modify_EWR_table(test_df)

    expected_dtypes = {
        # Integer columns
        'AccumulationPeriod': 'Int64',
        'AnnualBarrageFlow': 'Int64',
        'Duration': 'Int64',
        'EggsDaysSpell': 'Int64',
        'EndDay': 'Int64',
        'EndMonth': 'Int64',
        'EventsPerYear': 'Int64',
        'FlowThresholdMax': 'Int64',
        'FlowThresholdMin': 'Int64',
        'HighReleaseWindowEnd': 'Int64',
        'HighReleaseWindowStart': 'Int64',
        'LarvaeDaysSpell': 'Int64',
        'LowLevelWindowEnd': 'Int64',
        'LowLevelWindowStart': 'Int64',
        'LowReleaseWindowEnd': 'Int64',
        'LowReleaseWindowStart': 'Int64',
        'MaxSpell': 'Int64',
        'MinSpell': 'Int64',
        'NonFlowSpell': 'Int64',
        'PeakLevelWindowEnd': 'Int64',
        'PeakLevelWindowStart': 'Int64',
        'StartDay': 'Int64',
        'StartMonth': 'Int64',
        'ThreeYearsBarrageFlow': 'Int64',
        'TriggerDay': 'Int64',
        'TriggerMonth': 'Int64',
        'VolumeThreshold': 'Int64',
        'WithinEventGapTolerance': 'Int64',
        
        # Float columns
        'CtfThreshold': 'Float64',
        'DrawDownRateWeek': 'Float64',
        'LevelThresholdMax': 'Float64',
        'LevelThresholdMin': 'Float64',
        'MaxInter-event': 'Float64',
        'MaxLevelChange': 'Float64',
        'RateOfFallMin': 'Float64',
        'RateOfFallRiverLevel': 'Float64',
        'RateOfRiseMax1': 'Float64',
        'RateOfRiseMax2': 'Float64',
        'RateOfRiseRiverLevel': 'Float64',
        'RateOfRiseThreshold1': 'Float64',
        'RateOfRiseThreshold2': 'Float64'
    }

    keys = expected_dtypes.keys()

    result_key_cols = result[keys]

    assert result_key_cols.dtypes.astype(str).to_dict() == expected_dtypes, f"expected datatypes not matching output datatype"
    




    