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
    '''Loads the EWR calculation configuration file from repository or local file
    system
    
    Args:
        file_path (str): Location of the EWR calculation configuration file

    Returns:
        dict: Returns a dictionary of the EWR calculation configuration file
    '''
    
    if file_path:
        with open(file_path, 'r') as fp:
            ewr_calc_config = json.load(fp)
    
    if not file_path:
        repo_path = os.path.join(BASE_PATH, "parameter_metadata/ewr_calc_config.json")
        with open(repo_path, 'r') as fp:
            ewr_calc_config = json.load(fp)

    return ewr_calc_config

@cached(cache=TTLCache(maxsize=1024, ttl=1800))
def get_EWR_table(file_path:str = None) -> dict:
    
    ''' Loads ewr table from blob storage, separates out the readable ewrs from the 
    ewrs with 'see notes' exceptions, those with no threshold, and those with undefined names,
    does some cleaning, including swapping out '?' in the frequency column with 0
    
    Args:
        file_path (str): Location of the EWR dataset
    Returns:
        tuple(pd.DataFrame, pd.DataFrame): EWRs that meet the minimum requirements; EWRs that dont meet the minimum requirements
    '''
    
    if file_path:
        df = pd.read_csv(file_path,
         usecols=['PlanningUnitID', 'PlanningUnitName',  'LTWPShortName', 'CompliancePoint/Node', 'Gauge', 'Code', 'StartMonth',
                              'EndMonth', 'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax', 'EventsPerYear', 'Duration', 'MinSpell', 
                              'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                              'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'MaxLevelRise','AccumulationPeriod',
                              'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek', 'AnnualFlowSum','AnnualBarrageFlow',
                              'ThreeYearsBarrageFlow', 'HighReleaseWindowStart', 'HighReleaseWindowEnd', 'LowReleaseWindowStart', 'LowReleaseWindowEnd',
                              'PeakLevelWindowStart', 'PeakLevelWindowEnd', 'LowLevelWindowStart', 'LowLevelWindowEnd', 'NonFlowSpell', 'EggsDaysSpell',
                              'LarvaeDaysSpell','MinLevelRise', 'RateOfRiseMax1','RateOfRiseMax2','RateOfFallMin','RateOfRiseThreshold1',
                              'RateOfRiseThreshold2','RateOfRiseRiverLevel','RateOfFallRiverLevel', 'CtfThreshold', 'GaugeType'],
                               dtype='str', encoding='cp1252') 	


    if not file_path:
        my_url = os.path.join(BASE_PATH, "parameter_metadata/parameter_sheet.csv")
        proxies={} # Populate with your proxy settings
        df = pd.read_csv(my_url,
            usecols=['PlanningUnitID', 'PlanningUnitName',  'LTWPShortName', 'CompliancePoint/Node', 'Gauge', 'Code', 'StartMonth',
                              'EndMonth', 'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax', 'EventsPerYear', 'Duration', 'MinSpell', 
                              'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                              'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'MaxLevelRise', 'AccumulationPeriod',
                              'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek','AnnualFlowSum','AnnualBarrageFlow',
                              'ThreeYearsBarrageFlow', 'HighReleaseWindowStart', 'HighReleaseWindowEnd', 'LowReleaseWindowStart', 'LowReleaseWindowEnd',
                              'PeakLevelWindowStart', 'PeakLevelWindowEnd', 'LowLevelWindowStart', 'LowLevelWindowEnd', 'NonFlowSpell','EggsDaysSpell',
                              'LarvaeDaysSpell','MinLevelRise', 'RateOfRiseMax1','RateOfRiseMax2','RateOfFallMin','RateOfRiseThreshold1',
                              'RateOfRiseThreshold2','RateOfRiseRiverLevel','RateOfFallRiverLevel', 'CtfThreshold', 'GaugeType'],
                        dtype='str', encoding='cp1252'
                        )

    df = df.replace('?','')
    df = df.fillna('')
    # removing the 'See notes'
    okay_EWRs = df.loc[(df["StartMonth"] != 'See note') & (df["EndMonth"] != 'See note')]
    see_notes = df.loc[(df["StartMonth"] == 'See note') & (df["EndMonth"] == 'See note')]

    # Filtering those with no flow/level/volume thresholds
    noThresh_df = okay_EWRs.loc[(okay_EWRs["FlowThresholdMin"] == '') & (okay_EWRs["FlowThresholdMax"] == '') &\
                             (okay_EWRs["VolumeThreshold"] == '') &\
                             (okay_EWRs["LevelThresholdMin"] == '') & (okay_EWRs["LevelThresholdMax"] == '')]
    okay_EWRs = okay_EWRs.loc[(okay_EWRs["FlowThresholdMin"] != '') | (okay_EWRs["FlowThresholdMax"] != '') |\
                        (okay_EWRs["VolumeThreshold"] != '') |\
                        (okay_EWRs["LevelThresholdMin"] != '') | (okay_EWRs["LevelThresholdMax"] != '')]

    # Filtering those with no durations
    okay_EWRs = okay_EWRs.loc[(okay_EWRs["Duration"] != '')]
    no_duration = df.loc[(df["Duration"] == '')]

    # FIltering DSF EWRs
    condDSF = df['Code'].str.startswith('DSF')
    DSF_ewrs = df[condDSF]
    condDSF_inv = ~(okay_EWRs['Code'].str.startswith('DSF'))
    okay_EWRs = okay_EWRs[condDSF_inv]

    # Combine the unuseable EWRs into a dataframe and drop dups:
    bad_EWRs = pd.concat([see_notes, noThresh_df, no_duration, DSF_ewrs], axis=0)
    bad_EWRs = bad_EWRs.drop_duplicates()

    # Changing the flow and level max threshold to a high value when there is none available:
    okay_EWRs['FlowThresholdMax'].replace({'':'1000000'}, inplace = True)
    okay_EWRs['LevelThresholdMax'].replace({'':'1000000'}, inplace = True)
    
    return okay_EWRs, bad_EWRs

def get_MDBA_codes() -> pd.DataFrame:
    '''
    Load MDBA model metadata file containing model nodes
    and gauges they correspond to

    Returns:
        pd.DataFrame: dataframe for linking MDBA model nodes to gauges

    '''
    metadata = pd.read_csv( BASE_PATH / 'model_metadata/SiteID_MDBA.csv', engine = 'python', dtype=str, encoding='windows-1252')

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
    Ingests EWR category, returns the components required to analyse this type of EWR. 
    Each code represents a unique component in the EWR dataset.

    Args:
        category (str): options =   'flow', 'low flow', 'cease to flow', 'cumulative', 'level', 'weirpool-raising', 'weirpool-falling', 'nest-level', 'nest-percent',
                                    'multi-gauge-flow', 'multi-gauge-low flow', 'multi-gauge-cease to flow', 'multi-gauge-cease to flow', 'multi-gauge-cumulative', 
                                    'simul-gauge-flow', 'simul-gauge-low flow', 'simul-gauge-cease to flow', 'complex'

    Returns:
        list: Components needing to be pulled from the EWR dataset
    '''

    if category == 'flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'MIE', 'FLV']
    elif category == 'low flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR',  'ME', 'EPY', 'MIE', 'FLV']
    elif category == 'cease to flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'MIE', 'FLV']
    elif category == 'cumulative':
        pull =  ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MAXF', 'MIE','AP','GP', 'FLV']
    elif category == 'cumulative_bbr':
        pull =  ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MAXF', 'MIE','AP','GP', 'FLV','MAXL','WPG']
    elif category == 'water_stability':
        pull =  ['SM', 'EM', 'DUR', 'ME', 'EPY', 'MINF', 'MAXF', 'MIE','AP','GP', 'FLV','MAXL','WPG', 'EDS', 'LDS', 'ML', 'MD']
    elif category == 'water_stability_level':
        pull =  ['SM', 'EM', 'DUR', 'ME', 'EPY', 'MINF', 'MIE','AP','GP', 'FLV','MAXL', 'MINL', 'WPG', 'EDS', 'LDS', 'ML', 'MD']
    elif category == 'level':
        pull = ['SM', 'EM', 'MINL', 'MAXL', 'DUR', 'ME', 'EPY', 'MD', 'MIE', 'FLV', 'MAXD','GP','MLR']
    elif category == 'weirpool-raising':
        pull=['SM', 'EM', 'MINF', 'MAXF', 'MINL', 'DUR', 'ME',  'MD', 'EPY','WPG', 'MIE', 'FLV', 'GP']
    elif category == 'weirpool-falling':
        pull=['SM', 'EM', 'MINF', 'MAXF', 'MAXL', 'DUR', 'ME',  'MD', 'EPY','WPG', 'MIE', 'FLV', 'GP']
    elif category == 'nest-level':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME',  'MD', 'EPY', 'WPG', 'MIE', 'FLV','WDD','GP']
    elif category == 'nest-percent':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME',  'MD', 'EPY', 'MIE', 'FLV','TD','TM','GP']
    elif category == 'multi-gauge-flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME',  'GP', 'EPY', 'MG', 'MIE', 'FLV']
    elif category == 'multi-gauge-low flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'MG', 'MIE', 'FLV']
    elif category == 'multi-gauge-cease to flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'MG', 'MIE', 'FLV']
    elif category == 'multi-gauge-cumulative':
        pull =  ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MAXF','MG', 'MIE','AP','GP', 'FLV']
    elif category == 'flood-plains':
        pull=['SM', 'EM', 'MINF', 'MAXF', 'MAXL', 'DUR', 'ME',  'MD', 'ML','EPY','WPG', 'MIE', 'FLV', 'GP']
    elif category == 'barrage-flow':
        pull=['SM', 'EM','DUR', 'ME','EPY','MIE','FLV','ABF','TYBF','HRWS', 'HRWE', 'LRWS', 'LRWE']
    elif category == 'barrage-level':
        pull=['SM', 'EM','DUR', 'ME','EPY','MIE','FLV','HRWS', 'HRWE', 'LRWS', 'LRWE','PLWS', 'PLWE', 'LLWS', 'LLWE','MINL','MAXL']
    elif category == 'flow-ctf':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'MIE', 'FLV', 'NFS', 'CTFT']
    elif category == 'rise_fall':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'MIE', 'FLV', 'NFS', 'MLR', 'RRM1', 'RRM2', 'RFM', 'RRT1', 'RRT2', 'RRL', 'RFL' ]
    return pull

def get_bad_QA_codes() -> list:
    '''NSW codes representing poor quality data.
    
    Returns:
        list: quality codes needing to be filtered out.

    '''
    return [151, 152, 153, 155, 180, 201, 202, 204, 205, 207, 223, 255]

def weirpool_type(EWR: str) -> str:
    '''Returns the type of Weirpool EWR. Currently only WP2 EWRs are classified as weirpool raisings
    
    Args:
        EWR (str): WP2 is considered raising, the remaining WP EWRs are considered falling

    Returns:
        str: either 'raising' or 'falling'
    
    '''

    return 'raising' if EWR == 'WP2' else 'falling'

@cached(cache=TTLCache(maxsize=1024, ttl=1800))
def get_planning_unit_info() -> pd.DataFrame:
    '''Run this function to get the planning unit MDBA ID and equivilent planning unit name as specified in the LTWP.
    
    Result:
        pd.DataFrame: dataframe with planning units and their unique planning unit ID.
    
    '''
    EWR_table, bad_EWRs = get_EWR_table()
        
    planningUnits = EWR_table.groupby(['PlanningUnitID', 'PlanningUnitName']).size().reset_index().drop([0], axis=1) 
    
    return planningUnits



# Function to pull out the EWR parameter information
def ewr_parameter_grabber(EWR_TABLE: pd.DataFrame, GAUGE: str, PU: str, EWR: str, PARAMETER: str) -> str:
    '''
    Input an EWR table to pull data from, a gauge, planning unit, and EWR for the unique value, and a requested parameter

    Args:
        EWR_TABLE (pd.DataFrame): dataset of EWRs
        GAUGE (str): Gauge string
        PU (str): Planning unit name
        EWR (str): EWR string
        PARAMETER (str): which parameter of the EWR to access
    Results:
        str: requested EWR component
    
    '''
    component = (EWR_TABLE[((EWR_TABLE['Gauge'] == GAUGE) & 
                           (EWR_TABLE['Code'] == EWR) &
                           (EWR_TABLE['PlanningUnitName'] == PU)
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
        return set(EWR_table['Gauge'].to_list() + menindee_gauges + wp_gauges + multi_gauges)
    elif category == 'flow gauges':
        return set(EWR_table['Gauge'].to_list() + multi_gauges + flow_barrage_gauges + qld_flow_gauges) 
    elif category == 'level gauges':
        level_gauges = EWR_table[EWR_table['FlowLevelVolume']=='L']['Gauge'].to_list()
        return set(menindee_gauges + wp_gauges + level_barrage_gauges + qld_level_gauges + level_gauges)
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
