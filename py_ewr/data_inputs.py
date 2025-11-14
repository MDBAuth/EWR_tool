import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging

from cachetools import cached, TTLCache

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

BASE_PATH = Path(__file__).resolve().parent

@cached(cache=TTLCache(maxsize=1024, ttl=1800))
def get_ewr_calc_config(file_path:str = None) -> dict:
    '''Loads the ewr calculation configuration file from repository or local file
    system
    
    Args:
        file_path (str): Location of the ewr calculation configuration file

    Returns:
        dict: Returns a dictionary of the ewr calculation configuration file
    '''
    
    if file_path:
        with open(file_path, 'r') as fp:
            ewr_calc_config = json.load(fp)
    
    if not file_path:
        repo_path = os.path.join(BASE_PATH, "parameter_metadata/ewr_calc_config.json")
        with open(repo_path, 'r') as fp:
            ewr_calc_config = json.load(fp)

    return ewr_calc_config

def modify_EWR_table(EWR_table:pd.DataFrame) -> pd.DataFrame:
  
    ''' Does all miscellaneous changes to the ewr table to get in the right format for all the handling functions. i.e. datatype changing, splitting day/month data, handling %
    '''

    int_components = ['FlowThresholdMin', 'FlowThresholdMax', 'VolumeThreshold', 'Duration', 'WithinEventGapTolerance', 'EventsPerYear', 'MinSpell', 'AccumulationPeriod', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'AnnualBarrageFlow', 'ThreeYearsBarrageFlow', 'HighReleaseWindowStart', 'HighReleaseWindowEnd', 'LowReleaseWindowStart', 'LowReleaseWindowEnd', 'PeakLevelWindowStart', 'PeakLevelWindowEnd', 'LowLevelWindowStart', 'LowLevelWindowEnd', 'NonFlowSpell', 'EggsDaysSpell', 'LarvaeDaysSpell', 'StartDay', 'EndDay', 'StartMonth', 'EndMonth']
    float_components = ['RateOfRiseMax1', 'RateOfRiseMax2', 'RateOfFallMin', 'RateOfRiseThreshold1', 'RateOfRiseThreshold2', 'RateOfRiseRiverLevel', 'RateOfFallRiverLevel', 'CtfThreshold', 'MaxLevelChange', 'LevelThresholdMin', 'LevelThresholdMax', 'DrawDownRateWeek', 'MaxInter-event']

    # Modify startmonth/endmonth
    col_names = ['StartMonth', 'EndMonth']
    for col_name in col_names:
      rows = EWR_table[col_name].copy().items()
      day_col_name = col_name[:-5]+"Day"
      for r_idx, val in rows:
        if "." in val:
          month, day = val.split('.')
        else:
          month = val
          day = None
        EWR_table.loc[r_idx, col_name] = month
        EWR_table.loc[r_idx, day_col_name] = day # the datatype conversion all takes place in # Modify integers #

    # I actually think the drawdown rate modifications were doing nothing and the handling of percentage / float values is done in all functions that use drawdown_rate.

    # Modify floats
    for col_name in float_components:
      col = pd.to_numeric(EWR_table[col_name], errors='coerce')
      EWR_table[col_name] = pd.Series(col, dtype='Float64')

    # Modify integers
    for col_name in int_components:
      col = pd.to_numeric(EWR_table[col_name], errors='coerce')
      EWR_table[col_name] = pd.Series(col, dtype='Int64')

    return EWR_table

@cached(cache=TTLCache(maxsize=1024, ttl=1800))
def get_EWR_table(file_path:str = None) -> dict:
    
    ''' Loads ewr table from blob storage, separates out the readable ewrs from the 
    ewrs with 'see notes' exceptions, those with no threshold, and those with undefined names,
    does some cleaning, including swapping out '?' in the frequency column with 0
    
    Args:
        file_path (str): Location of the ewr dataset
    Returns:
        tuple(pd.DataFrame, pd.DataFrame): EWRs that meet the minimum requirements; EWRs that dont meet the minimum requirements
    '''
    
    if file_path:
      my_url = file_path
    else:
      my_url = os.path.join(BASE_PATH, "parameter_metadata/parameter_sheet.csv")

    df = pd.read_csv(my_url,
        usecols=['PlanningUnitID', 'PlanningUnitName', 'Gauge', 'Code', 'StartMonth', 'TargetFrequency', 'State', 'SWSDLName',
                          'EndMonth', 'EventsPerYear', 'Duration', 'MinSpell', 
                          'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                          'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'MaxLevelChange', 'AccumulationPeriod',
                          'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek','AnnualBarrageFlow',
                          'ThreeYearsBarrageFlow', 'HighReleaseWindowStart', 'HighReleaseWindowEnd', 'LowReleaseWindowStart', 'LowReleaseWindowEnd',
                          'PeakLevelWindowStart', 'PeakLevelWindowEnd', 'LowLevelWindowStart', 'LowLevelWindowEnd', 'NonFlowSpell','EggsDaysSpell',
                          'LarvaeDaysSpell', 'RateOfRiseMax1','RateOfRiseMax2','RateOfFallMin','RateOfRiseThreshold1',
                          'RateOfRiseThreshold2','RateOfRiseRiverLevel','RateOfFallRiverLevel', 'CtfThreshold', 'GaugeType'],
                    dtype='str', encoding='cp1252'
                    )

    df = df.replace('?','')
    df = df.fillna('')

    # removing the 'See notes'
    see_notes_idx = (df["StartMonth"] == 'See note') & (df["EndMonth"] == 'See note')

    # Filtering those with no flow/level/volume thresholds
    no_thresh_idx = (df["FlowThresholdMin"] == '') & \
                    (df["FlowThresholdMax"] == '') &\
                    (df["VolumeThreshold"] == '') &\
                    (df["LevelThresholdMin"] == '') &\
                    (df["LevelThresholdMax"] == '')

    # Filtering those with no durations
    no_duration_idx = (df["Duration"] == '')

    # Filtering DSF EWRs
    DSF_idx = df['Code'].str.startswith('DSF')

    # Combine the filters and get the okay and bad EWRs
    bad_EWRs_idx = see_notes_idx | no_thresh_idx | no_duration_idx | DSF_idx
    
    okay_EWRs = df[~bad_EWRs_idx].copy(deep=True)
    bad_EWRs = df[bad_EWRs_idx].copy(deep=True)

    # Here are all the prior assumptions of what to fill in to the parameter sheet if the value is missing.
    # The aim is to remove all of these and have the parameter sheet be correct, the tool should not run
    # the calculation of an ewr with missing (or extra?) values.
    okay_EWRs.loc[:, 'FlowThresholdMax'] = (okay_EWRs['FlowThresholdMax'].replace('', '1000000'))
    okay_EWRs.loc[:, 'LevelThresholdMax'] = (okay_EWRs['LevelThresholdMax'].replace('', '1000000'))
    okay_EWRs.loc[:, 'FlowThresholdMin'] = (okay_EWRs['FlowThresholdMin'].replace('', '0'))
    okay_EWRs.loc[:, 'LevelThresholdMin'] = (okay_EWRs['LevelThresholdMin'].replace('', '0'))
    okay_EWRs.loc[:, 'MaxInter-event'] = (okay_EWRs['MaxInter-event'].replace('', '0'))
    okay_EWRs.loc[:, 'WithinEventGapTolerance'] = (okay_EWRs['WithinEventGapTolerance'].replace('', '0'))

    okay_EWRs.loc[:, 'CtfThreshold'] = (okay_EWRs['CtfThreshold'].replace('', '5'))
    okay_EWRs.loc[:, 'NonFlowSpell'] = (okay_EWRs['NonFlowSpell'].replace('', '0'))
    okay_EWRs.loc[:, 'DrawDownRateWeek'] = (okay_EWRs['DrawDownRateWeek'].replace('30', '0.03'))
    okay_EWRs.loc[:, 'DrawDownRateWeek'] = (okay_EWRs['DrawDownRateWeek'].replace('30%', '0.03'))#just for test, change the PS in that test to reflect this
    okay_EWRs.loc[:, 'DrawdownRate'] = (okay_EWRs['DrawdownRate'].replace('', '1000000'))
    okay_EWRs.loc[:, 'MaxSpell'] = (okay_EWRs['MaxSpell'].replace('', '1000000'))
    okay_EWRs.loc[:, 'MaxLevelChange'] = (okay_EWRs['MaxLevelChange'].replace('', '1000000'))

    okay_EWRs = modify_EWR_table(okay_EWRs)
    
    return okay_EWRs, bad_EWRs

def get_components_map() -> dict:
    components_map = {
        'PlanningUnitID': 'PlanningUnit',
        'Gauge': 'Gauge',
        'Code': 'Code',
        'FlowThresholdMin': 'min_flow',
        'FlowThresholdMax': 'max_flow',
        'VolumeThreshold': 'min_volume',
        'Duration': 'duration',
        'WithinEventGapTolerance': 'gap_tolerance',
        'EventsPerYear': 'events_per_year',
        'MinSpell': 'min_event',
        'AccumulationPeriod': 'accumulation_period',
        'MaxSpell': 'max_duration',
        'TriggerDay': 'trigger_day',
        'TriggerMonth': 'trigger_month',
        'AnnualBarrageFlow': 'annual_barrage_flow',
        'ThreeYearsBarrageFlow': 'three_years_barrage_flow',
        'HighReleaseWindowStart': 'high_release_window_start',
        'HighReleaseWindowEnd': 'high_release_window_end',
        'LowReleaseWindowStart': 'low_release_window_start',
        'LowReleaseWindowEnd': 'low_release_window_end',
        'PeakLevelWindowStart': 'peak_level_window_start',
        'PeakLevelWindowEnd': 'peak_level_window_end',
        'LowLevelWindowStart': 'low_level_window_start',
        'LowLevelWindowEnd': 'low_level_window_end',
        'NonFlowSpell': 'non_flow_spell',
        'EggsDaysSpell': 'eggs_days_spell',
        'LarvaeDaysSpell': 'larvae_days_spell',
        'RateOfRiseMax1': 'rate_of_rise_max1',
        'RateOfRiseMax2': 'rate_of_rise_max2',
        'RateOfFallMin': 'rate_of_fall_min',
        'RateOfRiseThreshold1': 'rate_of_rise_threshold1',
        'RateOfRiseThreshold2': 'rate_of_rise_threshold2',
        'RateOfRiseRiverLevel': 'rate_of_rise_river_level',
        'RateOfFallRiverLevel': 'rate_of_fall_river_level',
        'CtfThreshold': 'ctf_threshold',
        'MaxLevelChange': 'max_level_change',
        'LevelThresholdMin': 'min_level',
        'LevelThresholdMax': 'max_level',
        'StartMonth': 'start_month',
        'EndMonth': 'end_month',
        'StartDay': 'start_day',
        'EndDay': 'end_day',
        'DrawDownRateWeek': 'drawdown_rate_week',
        'MaxInter-event': 'max_inter-event',
        'Multigauge': 'second_gauge',
        'DrawdownRate': 'drawdown_rate',
        'FlowLevelVolume': 'flow_level_volume',
        'WeirpoolGauge': 'weirpool_gauge',
        
        # these are required at the end after scenarios are processed
        # TODO: Clarify/check if we require more here for interevents, etc. other 4 outputs of the tool
        'TargetFrequency': 'TargetFrequency',
        'PlanningUnitName': 'PlanningUnitName',
        'State': 'State', 
        'SWSDLName': 'SWSDLName',
        'GaugeType': 'GaugeType',
        }
    return components_map

def get_MDBA_codes(model_type: str) -> pd.DataFrame:
    '''
    Load MDBA model metadata file containing model nodes
    and gauges they correspond to

    Returns:
        pd.DataFrame: dataframe for linking MDBA model nodes to gauges

    '''
    if model_type == 'Bigmod - MDBA':
        metadata = pd.read_csv( BASE_PATH / 'model_metadata/SiteID_MDBA.csv', engine = 'python', dtype=str)#, encoding='windows-1252')
    if model_type == 'FIRM - MDBA':
        metadata = pd.read_csv( BASE_PATH / 'model_metadata/EWR_Sitelist_FIRM_20250718.csv', engine = 'python', dtype=str)

    return metadata
  
def get_NSW_codes() -> pd.DataFrame:
    '''
    Load NSW model metadata file containing model nodes
    and gauges they correspond to

    Returns:
        pd.DataFrame: dataframe for linking NSW model nodes to gauges

    '''
    metadata = pd.read_csv( BASE_PATH / 'model_metadata/SiteID_NSW.csv', engine = 'python', dtype=str)
    
    return metadata

def get_iqqm_codes() -> dict:
    '''
    Load metadata file for Macquarie containing model nodes
    and gauges they correspond to

    Returns:
        dict: dict for linking model nodes to gauges
    '''

    metadf = pd.read_csv( BASE_PATH / 'model_metadata/iqqm_stations.csv', dtype=str)
    metadata = metadf.set_index(metadf.columns[0]).to_dict()[metadf.columns[1]]
    return metadata

def get_level_gauges() -> tuple:
    '''Returning level gauges with EWRs

    Returns:
        tuple[list, dict]: list of lake level gauges, dictionary of weirpool gauges
    
    '''
    
    menindeeGauges = ['425020', '425022', '425023']

    lachlanGauges = ['412107']

    levelGauges = menindeeGauges + lachlanGauges 
    
    weirpoolGauges = {'414203': '414209', 
                      '425010': 'A4260501',
                      'A4260507': 'A4260508',
                      'A4260505': 'A4260506'}
    
    return levelGauges, weirpoolGauges


def get_multi_gauges(dataType: str) -> dict:
    '''
    Multi gauges are for EWRs that require the flow of two gauges to be added together

    Args:
        dataType (str): Pass 'all' to get multi gauges and their planning units, pass 'gauges' to get only the gauges.
    Returns:
        dict: if 'all' nested dict returned with a level for planning units
    '''
    
    all = {'PU_0000130': {'421090': '421088', '421088': '421090'},
              'PU_0000131': {'421090': '421088', '421088': '421090'},
              'PU_0000132': {'421090': '421088', '421088': '421090'},
              'PU_0000133': {'421090': '421088', '421088': '421090'},
              'PU_0000251': {'423001': '423002'},
              'PU_0000280': {'1AS': '1ES'}
             }
    returnData = {}
    if dataType == 'all':
        returnData = all
    if dataType == 'gauges':
        for i in all:
            returnData = {**returnData, **all[i]}
    
    return returnData

def get_EWR_components(category):
    '''
    Ingests ewr category, returns the components required to analyse this type of ewr. 
    Each code represents a unique component in the ewr dataset.

    Args:
        category (str): options =   'flow', 'low flow', 'cease to flow', 'cumulative', 'level', 'weirpool-raising', 'weirpool-falling', 'nest-level', 'nest-percent',
                                    'multi-gauge-flow', 'multi-gauge-low flow', 'multi-gauge-cease to flow', 'multi-gauge-cease to flow', 'multi-gauge-cumulative', 
                                    'simul-gauge-flow', 'simul-gauge-low flow', 'simul-gauge-cease to flow', 'complex'

    Returns:
        list: Components needing to be pulled from the ewr dataset
    '''

    if category == 'flow':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell', 'WithinEventGapTolerance', 'EventsPerYear', 'MaxInter-event', 'FlowLevelVolume']
    elif category == 'low flow':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration',  'MinSpell', 'EventsPerYear', 'MaxInter-event', 'FlowLevelVolume']
    elif category == 'cease to flow':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell', 'EventsPerYear', 'MaxInter-event', 'FlowLevelVolume']
    elif category == 'cumulative':
        pull =  ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'VolumeThreshold', 'Duration', 'MinSpell', 'EventsPerYear', 'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event','AccumulationPeriod','WithinEventGapTolerance', 'FlowLevelVolume']
    elif category == 'cumulative_bbr':
        pull =  ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'VolumeThreshold', 'Duration', 'MinSpell', 'EventsPerYear', 'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event','AccumulationPeriod','WithinEventGapTolerance', 'FlowLevelVolume','LevelThresholdMax','WeirpoolGauge']
    elif category == 'water_stability':
        pull =  ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'Duration', 'MinSpell', 'EventsPerYear', 'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event','AccumulationPeriod','WithinEventGapTolerance', 'FlowLevelVolume','LevelThresholdMax','WeirpoolGauge', 'EggsDaysSpell', 'LarvaeDaysSpell', 'MaxLevelChange', 'DrawdownRate']
    elif category == 'water_stability_level':
        pull =  ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'Duration', 'MinSpell', 'EventsPerYear', 'FlowThresholdMin', 'MaxInter-event','AccumulationPeriod','WithinEventGapTolerance', 'FlowLevelVolume','LevelThresholdMax', 'LevelThresholdMin', 'WeirpoolGauge', 'EggsDaysSpell', 'LarvaeDaysSpell', 'MaxLevelChange', 'DrawdownRate']
    elif category == 'level':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'LevelThresholdMin', 'LevelThresholdMax', 'Duration', 'MinSpell', 'EventsPerYear', 'DrawdownRate', 'MaxInter-event', 'FlowLevelVolume', 'MaxSpell','WithinEventGapTolerance']
    elif category == 'weirpool-raising':
        pull=['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'LevelThresholdMin', 'Duration', 'MinSpell',  'DrawdownRate', 'EventsPerYear','WeirpoolGauge', 'MaxInter-event', 'FlowLevelVolume', 'WithinEventGapTolerance']
    elif category == 'weirpool-falling':
        pull=['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'LevelThresholdMax', 'Duration', 'MinSpell',  'DrawdownRate', 'EventsPerYear','WeirpoolGauge', 'MaxInter-event', 'FlowLevelVolume', 'WithinEventGapTolerance']
    elif category == 'nest-level':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell', 'EventsPerYear', 'WeirpoolGauge', 'MaxInter-event', 'FlowLevelVolume','DrawDownRateWeek','WithinEventGapTolerance']
    elif category == 'nest-percent':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell',  'DrawdownRate', 'EventsPerYear', 'MaxInter-event', 'FlowLevelVolume','TriggerDay','TriggerMonth','WithinEventGapTolerance']
    elif category == 'multi-gauge-flow':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell',  'WithinEventGapTolerance', 'EventsPerYear', 'Multigauge', 'MaxInter-event', 'FlowLevelVolume']
    elif category == 'multi-gauge-low flow':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell', 'EventsPerYear', 'Multigauge', 'MaxInter-event', 'FlowLevelVolume']
    elif category == 'multi-gauge-cease to flow':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell', 'EventsPerYear', 'Multigauge', 'MaxInter-event', 'FlowLevelVolume']
    elif category == 'multi-gauge-cumulative':
        pull =  ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'VolumeThreshold', 'Duration', 'MinSpell', 'EventsPerYear', 'FlowThresholdMin', 'FlowThresholdMax','Multigauge', 'MaxInter-event','AccumulationPeriod','WithinEventGapTolerance', 'FlowLevelVolume']
    elif category == 'flood-plains':
        pull=['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'LevelThresholdMax', 'Duration', 'MinSpell',  'DrawdownRate', 'MaxLevelChange','EventsPerYear','WeirpoolGauge', 'MaxInter-event', 'FlowLevelVolume', 'WithinEventGapTolerance']
    elif category == 'barrage-flow':
        pull=['StartMonth', 'EndMonth', 'StartDay', 'EndDay','Duration', 'MinSpell','EventsPerYear','MaxInter-event','FlowLevelVolume','AnnualBarrageFlow','ThreeYearsBarrageFlow','HighReleaseWindowStart', 'HighReleaseWindowEnd', 'LowReleaseWindowStart', 'LowReleaseWindowEnd']
    elif category == 'barrage-level':
        pull=['StartMonth', 'EndMonth', 'StartDay', 'EndDay','Duration', 'MinSpell','EventsPerYear','MaxInter-event','FlowLevelVolume','HighReleaseWindowStart', 'HighReleaseWindowEnd', 'LowReleaseWindowStart', 'LowReleaseWindowEnd','PeakLevelWindowStart', 'PeakLevelWindowEnd', 'LowLevelWindowStart', 'LowLevelWindowEnd','LevelThresholdMin','LevelThresholdMax']
    elif category == 'flow-ctf':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell', 'WithinEventGapTolerance', 'EventsPerYear', 'MaxInter-event', 'FlowLevelVolume', 'NonFlowSpell', 'CtfThreshold']
    elif category == 'rise_fall':
        pull = ['StartMonth', 'EndMonth', 'StartDay', 'EndDay', 'FlowThresholdMin', 'FlowThresholdMax', 'Duration', 'MinSpell', 'WithinEventGapTolerance', 'EventsPerYear', 'MaxInter-event', 'FlowLevelVolume', 'NonFlowSpell', 'RateOfRiseMax1', 'RateOfRiseMax2', 'RateOfFallMin', 'RateOfRiseThreshold1', 'RateOfRiseThreshold2', 'RateOfRiseRiverLevel', 'RateOfFallRiverLevel' ]
    return pull

def get_bad_QA_codes() -> list:
    '''NSW codes representing poor quality data.
    
    Returns:
        list: quality codes needing to be filtered out.

    '''
    return [151, 152, 153, 155, 180, 201, 202, 204, 205, 207, 223, 255]

def weirpool_type(ewr: str) -> str:
    '''Returns the type of Weirpool ewr. Currently only WP2 EWRs are classified as weirpool raisings
    
    Args:
        ewr (str): WP2 is considered raising, the remaining WP EWRs are considered falling

    Returns:
        str: either 'raising' or 'falling'
    
    '''

    return 'raising' if ewr == 'WP2' else 'falling'

@cached(cache=TTLCache(maxsize=1024, ttl=1800))
def get_planning_unit_info() -> pd.DataFrame:
    '''Run this function to get the planning unit MDBA ID and equivilent planning unit name as specified in the LTWP.
    
    Result:
        pd.DataFrame: dataframe with planning units and their unique planning unit ID.
    
    '''
    EWR_table, bad_EWRs = get_EWR_table()
        
    planningUnits = EWR_table.groupby(['PlanningUnitID', 'PlanningUnitName']).size().reset_index().drop([0], axis=1) 
    
    return planningUnits



# Function to pull out the ewr parameter information
def ewr_parameter_grabber(EWR_TABLE: pd.DataFrame, gauge: str, pu: str, ewr: str, PARAMETER: str) -> str:
    '''
    Input an ewr table to pull data from, a gauge, planning unit, and ewr for the unique value, and a requested parameter

    Args:
        EWR_TABLE (pd.DataFrame): dataset of EWRs
        gauge (str): Gauge string
        pu (str): Planning unit name
        ewr (str): ewr string
        PARAMETER (str): which parameter of the ewr to access
    Results:
        str: requested ewr component
    
    '''
    component = (EWR_TABLE[((EWR_TABLE['Gauge'] == gauge) & 
                           (EWR_TABLE['Code'] == ewr) &
                           (EWR_TABLE['PlanningUnitName'] == pu)
                          )][PARAMETER]).to_list()[0]
    return component if component else 0

def get_barrage_flow_gauges()-> dict:
    """Returns a dictionary of the flow gauges associated with each barrage.
    Results:
        dict: dictionary of flow gauges associated with each barrage.       	
    """

    flow_barrage_gauges = {'A4261002': ['A4261002']}

    return flow_barrage_gauges

def get_barrage_level_gauges()-> dict:
    """Returns a dictionary of the level gauges associated with each barrage.
    Results:
        dict: dictionary of level gauges associated with each barrage.
    """

    level_barrage_gauges = {'A4260527': ['A4260527','A4261133', 'A4260524', 'A4260574', 'A4260575' ],
                            'A4260633' : ['A4260633','A4261209', 'A4261165']}
    
    return level_barrage_gauges

def get_qld_level_gauges()-> list:
    """Returns a dictionary of the level gauges associated with each barrage.
    Results:
        dict: dictionary of level gauges associated with each barrage.
    """
    return ['422015','422030', '422034', '416011', '416048']

def get_qld_flow_gauges()-> list:
    """Returns a dictionary of the flow gauges associated with each barrage.
    Results:
        dict: dictionary of flow gauges associated with each barrage.
    """
    return ['422030','422207A',
            '422209A',
            '422211A',
            '422502A',
            '424201A',
            '422034' ]

def get_vic_level_gauges()-> list:
    """Returns a list of the level gauges for VIC.
    Results:
        dict: dictionary of flow gauges associated with each barrage.
    """
    return ['405201', '405202', '405200']


def get_cllmm_gauges()->list:
    return ["A4261002", "A4260527", "A4260633"]


def get_gauges(category: str, ewr_table_path: str = None) -> set:
    '''
    Gathers a list of either all gauges that have EWRs associated with them,
    a list of all flow type gauges that have EWRs associated with them,
    or a list of all level type gauges that have EWRs associated with them

    Args:
        category(str): options = 'all gauges', 'flow gauges', or 'level gauges'
    Returns:
        list: list of gauges in selected category.

    '''
    EWR_table, bad_EWRs = get_EWR_table(file_path=ewr_table_path)
    menindee_gauges, wp_gauges = get_level_gauges()
    wp_gauges = list(wp_gauges.values())
    flow_barrage_gauges = [ val for sublist in get_barrage_flow_gauges().values() for val in sublist]
    level_barrage_gauges = [ val for sublist in get_barrage_level_gauges().values() for val in sublist]
    qld_level_gauges = get_qld_level_gauges()
    qld_flow_gauges = get_qld_flow_gauges()
    vic_level_gauges = get_vic_level_gauges()
    
    multi_gauges = get_multi_gauges('gauges')
    multi_gauges = list(multi_gauges.values())
    if category == 'all gauges':
        return set(EWR_table['Gauge'].to_list()+menindee_gauges+wp_gauges+multi_gauges+flow_barrage_gauges+level_barrage_gauges+qld_flow_gauges+qld_level_gauges+vic_level_gauges)
    elif category == 'flow gauges':
        return set(EWR_table['Gauge'].to_list() + multi_gauges + flow_barrage_gauges + qld_flow_gauges) 
    elif category == 'level gauges':
        level_gauges = EWR_table[EWR_table['FlowLevelVolume']=='L']['Gauge'].to_list()
        return set(menindee_gauges + wp_gauges + level_barrage_gauges + qld_level_gauges + level_gauges + vic_level_gauges)
    else:
        raise ValueError('''No gauge category sent to the "get_gauges" function''')
    
def get_scenario_gauges(gauge_results: dict) -> list:
    """return a list of gauges process for the scenatios(s)

    Args:
        gauge_results (dict): the dictionary of gauge results either for
        observed or scenarios handlers

    Returns:
        list: list of gauges process for the scenatios(s)
    """
    scenario_gauges = []
    for scenario in gauge_results.values():
        for gauge in scenario.keys():
            scenario_gauges.append(gauge)
    return list(set(scenario_gauges))


def gauge_groups(parameter_sheet: pd.DataFrame) -> dict:
    '''
    Returns a dictionary of flow, level, and lake level gauges based on the parameter sheet and some hard coding of other EWRs

    Args:
        parameter_sheet (pd.DataFrame): input parameter sheet
    
    Returns:
        dict: keys as flow, level, and lake level gauges, values as the list of gauges
    '''
    
    # Hard coded gauges for the CLLMM EWRs
    hard_code_levels = ['A4260527', 'A4260524', 'A4260633', 'A4261209', 'A4261165']
    hard_code_lake_levels = ['A4261133', 'A4260574', 'A4260575']

    flow_gauges = set(parameter_sheet[parameter_sheet['GaugeType'] == 'F']['Gauge']) + set(parameter_sheet['Multigauge'])
    level_gauges = set(parameter_sheet[parameter_sheet['GaugeType'] == 'L']['Gauge']) + set(parameter_sheet['WeirpoolGauge']) + set(hard_code_levels)
    lake_level_gauges = set(parameter_sheet[parameter_sheet['GaugeType'] == 'LL']['Gauge'])+set(hard_code_lake_levels)

    return flow_gauges, level_gauges, lake_level_gauges

# def gauges_to_measurand()


def get_obj_mapping(
    parameter_sheet_path=None,
    objective_reference_path=None
) -> pd.DataFrame:
    '''
    Retrieves objective mapping and merges with parameter sheet on Gauge, Planning Unit, Code, and LTWPShortName
    to create a longform table that lists the associations between each EWR and ecological objectives.
    Returns a DataFrame with the merged result.
    Args:
        param_sheet_path (str) = file path to parameter sheet. If None, default parameter_sheet.csv inside the tool is selected
        obj_ref_path (str) = file path to objective mapping csv. If Not, default objective_reference.csv inside EWR tool is selected
    '''
    param_sheet_cols = [
        'PlanningUnitName', 'LTWPShortName',  'SWSDLName', 'State', 'Gauge', 'Code', 'EnvObj'
    ]

    if not objective_reference_path:
        objective_reference_path = os.path.join(BASE_PATH, "parameter_metadata/obj_reference.csv")
    

    obj_ref = pd.read_csv(objective_reference_path)
    

    okay_EWRs, _ = get_EWR_table(file_path=parameter_sheet_path)
    okay_EWRs_sub = okay_EWRs[param_sheet_cols]
    
    # Split 'EnvObj' by '+' and explode to long format
    longform_ewr = okay_EWRs_sub.assign(
        EnvObj=okay_EWRs_sub['EnvObj'].str.split('+')
    ).explode('EnvObj').drop_duplicates()

    merged_df = longform_ewr.merge(
        obj_ref,
        left_on= ['LTWPShortName', 'PlanningUnitName', 'Gauge', 'Code', 'EnvObj', 'SWSDLName', 'State'],
        right_on=['LTWPShortName', 'PlanningUnitName', 'Gauge', 'Code', 'EnvObj', 'SWSDLName', 'State'],
        how='left'
    )

    return merged_df
