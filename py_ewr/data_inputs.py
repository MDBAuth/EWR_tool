import io
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from cachetools import cached, TTLCache

BASE_PATH = Path(__file__).resolve().parent

# Importing the climate cat data - to be replaced by RAS data once available:
   
def get_climate_cats(climate_file:str) -> pd.DataFrame:
    '''Uses standard climate categorisation unless user selects the 10,000 year climate sequence,
    in which case this is used
    
    Args:
        climate_file (str): location of the climate categoration file

    Returns:
        pd.DataFrame: Returns a dataframe showing annual climate categories for catchments
    
    '''
    
    if climate_file == 'Standard - 1911 to 2018 climate categorisation':
        climate_cats = pd.read_csv( BASE_PATH / 'climate_data/climate_cats.csv', index_col = 0)
        
    elif climate_file  == 'NSW 10,000 year climate sequence':
        climate_cats = pd.read_csv(BASE_PATH / 'climate_data/climate_cats_10000year.csv', index_col = 0)
        
    return climate_cats

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
                              'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'AccumulationPeriod',
                              'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek'],
                            dtype='str', encoding='cp1252')

    if not file_path:
        my_url = 'https://az3mdbastg001.blob.core.windows.net/mdba-public-data/NSWEWR_LIVE_DEV.csv'
        proxies={} # Populate with your proxy settings
        s = requests.get(my_url, proxies=proxies).text
        df = pd.read_csv(io.StringIO(s),
            usecols=['PlanningUnitID', 'PlanningUnitName',  'LTWPShortName', 'CompliancePoint/Node', 'Gauge', 'Code', 'StartMonth',
                              'EndMonth', 'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax', 'EventsPerYear', 'Duration', 'MinSpell', 
                              'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                              'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'AccumulationPeriod',
                              'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek'],
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

@cached(cache=TTLCache(maxsize=1024, ttl=1800))
def map_gauge_to_catchment(my_url:str = 'https://az3mdbastg001.blob.core.windows.net/mdba-public-data/NSWEWR_LIVE_DEV.csv') -> dict:
    ''' Allocates all the locations in the ewr table with catchments, as indicated by the
    first three numbers for each gauge 
    
    Args:
        my_url (str): location of the EWR dataset
    Returns:
        dict[dict]: Dictinoary of catchments, for each catchment a dictionary of gauge number and name key value pairs 
    '''
    
    lower_darling_gauges = ['425054', '425010', '425011', '425052', '425013', '425056', '425007', 
                            '425057', '425005', '425050', '425048', '425019', '425014', '425023', 
                            '425012', '425044', '425049', '425001', '425022', '42510037', '42510036',
                            '425034', '425046', '425020', ]
    
    EWR_table, bad_EWRs =  get_EWR_table(my_url)
    
    gauge_number = EWR_table['Gauge'].values
    gauge_name = EWR_table['CompliancePoint/Node'].values
    
    gauge_to_name = dict()
    for iteration, value in enumerate(gauge_number):
        if type(value) == str:
            gauge_to_name[value] = gauge_name[iteration]

    gauge_to_catchment = dict()
    namoi_catchment = dict()
    gwydir_catchment = dict()
    macquarie_catchment = dict()
    lachlan_catchment = dict()
    murray_catchment = dict()
    lower_darling_catchment = dict()
    barwon_darling_catchment = dict()
    murrumbidgee_catchment = dict()
    border_rivers_catchment = dict()
    moonie_catchment = dict()
    condamine_balonne = dict()
    warrego_catchment = dict()
    paroo_catchment = dict()
    
    for k, v in gauge_to_name.items():
        if k.startswith('419'):
            namoi_catchment.update({k: v})
        elif k.startswith('418'):
            gwydir_catchment.update({k: v})
        elif (k.startswith('421') or k.startswith('420')):
            macquarie_catchment.update({k: v})
        elif k.startswith('412'):
            lachlan_catchment.update({k: v})
        elif (k.startswith('401') or k.startswith('409') or k.startswith('426') or k.startswith('414')):
            murray_catchment.update({k: v})
        elif k.startswith('425'):
            if k in lower_darling_gauges:
                lower_darling_catchment.update({k: v})   
            else:
                barwon_darling_catchment.update({k: v}) 
        elif k.startswith('410'):
            murrumbidgee_catchment.update({k: v})
        elif k.startswith('416'):
            border_rivers_catchment.update({k: v})
        elif k.startswith('417'):
            moonie_catchment.update({k: v})
        elif k.startswith('422'):
            condamine_balonne.update({k: v})
        elif k.startswith('423'):
            warrego_catchment.update({k: v})
        elif k.startswith('424'):
            paroo_catchment.update({k: v})            
                
    gauge_to_catchment.update({'Namoi': namoi_catchment, 
                               'Gwydir': gwydir_catchment,
                               'Macquarie-Castlereagh': macquarie_catchment, 
                               'Lachlan': lachlan_catchment,
                               'Lower Darling': lower_darling_catchment, 
                               'Barwon-Darling': barwon_darling_catchment,
                               'Murray': murray_catchment,
                               'Murrumbidgee': murrumbidgee_catchment,
                               'Border Rivers': border_rivers_catchment,
                               'Moonie' : moonie_catchment,
                               'Condamine-Balonne': condamine_balonne,
                               'Warrego': warrego_catchment,
                               'Paroo': paroo_catchment
                              })
    return gauge_to_catchment

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

def gauge_to_catchment(input_gauge:str) -> str:
    '''
    Takes in a gauge, maps it to the catchment
    returns the catchment

    Args:
        input_gauge (str): Gauge string
    
    Returns:
        str: The catchment name that the input gauge is located in

    '''
    catchments_gauges = map_gauge_to_catchment()
    for catchment in catchments_gauges:
        if input_gauge in catchments_gauges[catchment]:
            return catchment
    
def wy_to_climate(water_years: np.array, catchment: str, climate_file: str) -> np.array:
    '''
    The function assigns a climate categorisation for every day, depending on the water year and catchment in the climate file

    Args:
        water_years (np.array): Daily water year array
        catchment (str): The catchment that the gauge is in
        climate file (str) = Which climate data to use

    Returns:
        np.array: Daily climate categorisation
    
    '''
    # Get the climate categorisation:
    climate_cats = get_climate_cats(climate_file)
    
    # Get the unique years covered in the flow dataframe, and how many days are in each year:
    unique_years, count_years = np.unique(water_years, return_counts=True)
    
    # Get the years covered by the climate cats, and filter them to those of interest (using min and max from flow dataframe)
    climateCatchment = climate_cats[catchment]
    climateFiltered = climateCatchment[(climateCatchment.index>=min(unique_years)) & (climateCatchment.index<=max(unique_years))].values
    # Repeating the climate result for that year over the total days in each year 
    def mapper(climate, count):
        return np.repeat(climate, count)

    climateDailyYear = list(map(mapper, climateFiltered, count_years))
    climateDaily = np.concatenate(climateDailyYear)
    
    return climateDaily

def get_level_gauges() -> tuple:
    '''Returning level gauges with EWRs

    Returns:
        tuple[list, dict]: list of lake level gauges, dictionary of weirpool gauges
    
    '''
    
    menindeeGauges = ['425020', '425022', '425023']
    
    weirpoolGauges = {'414203': '414209', 
                      '425010': '4260501', 
                      '4260507': '4260508',
                      '4260505': '4260506'}
    
    return menindeeGauges, weirpoolGauges


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
              'PU_0000133': {'421090': '421088', '421088': '421090'}
             }
    returnData = {}
    if dataType == 'all':
        returnData = all
    if dataType == 'gauges':
        for i in all:
            returnData = {**returnData, **all[i]}
    
    return returnData

def get_simultaneous_gauges(dataType: str) -> dict:
    '''
    Call function to return a dictionary of simultaneous gauges.
    Simultaneous gauges are for EWRs that need to be met simultaneously with EWRs at another location

    Args:
        dataType (str): Pass 'all' to get simultaneous gauges and their planning units, pass 'gauges' to get only gauges.
    Returns:
        dict: if 'all', nested dict returned with a level for planning units.

    '''
    
    all = {'PU_0000131': {'421090': '421022', '421022': '421090'},
              'PU_0000132': {'421090': '421022', '421022': '421090'},
              'PU_0000133': {'421090': '421022', '421022': '421090'}
             }
    returnData = {}
    if dataType == 'all':
        returnData = all
    if dataType == 'gauges':
        for i in all:
            returnData = {**returnData, **all[i]}
        
    return returnData

def get_complex_calcs() -> dict:
    '''
    Returns a dictionary of the complex EWRs, and the type of analysis that needs to be undertaken
    These EWRs cannot be calculated using the standard suite of functions

    Returns:
        dict[dict]
    '''
    complexCalcs = {'409025': {'OB2_S': 'flowDurPostReq', 'OB2_P': 'flowDurPostReq',
                              'OB3_S': 'flowDurOutsideReq', 'OB3_P': 'flowDurOutsideReq'}}
    
    return complexCalcs


def get_gauges(category: str) -> set:
    '''
    Gathers a list of either all gauges that have EWRs associated with them,
    a list of all flow type gauges that have EWRs associated with them,
    or a list of all level type gauges that have EWRs associated with them

    Args:
        category(str): options = 'all gauges', 'flow gauges', or 'level gauges'
    Returns:
        list: list of gauges in selected category.

    '''
    EWR_table, bad_EWRs = get_EWR_table()
    menindee_gauges, wp_gauges = get_level_gauges()
    wp_gauges = list(wp_gauges.values())
    
    multi_gauges = get_multi_gauges('gauges')
    simul_gauges = get_simultaneous_gauges('gauges')
    multi_gauges = list(multi_gauges.values())
    simul_gauges = list(simul_gauges.values())
    if category == 'all gauges':
        return set(EWR_table['Gauge'].to_list() + menindee_gauges + wp_gauges + multi_gauges + simul_gauges)
    elif category == 'flow gauges':
        return set(EWR_table['Gauge'].to_list() + multi_gauges + simul_gauges)
    elif category == 'level gauges':
        return set(menindee_gauges + wp_gauges)
    else:
        raise ValueError('''No gauge category sent to the "get_gauges" function''')

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
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR',  'ME', 'EPY', 'DURVD', 'MIE', 'FLV']
    elif category == 'cease to flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MIE', 'FLV']
    elif category == 'cumulative':
        pull =  ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MAXF', 'MIE','AP','GP', 'FLV']
    elif category == 'level':
        pull = ['SM', 'EM', 'MINL', 'MAXL', 'DUR', 'ME', 'EPY', 'MD', 'MIE', 'FLV', 'MAXD','GP']
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
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MG', 'MIE', 'FLV']
    elif category == 'multi-gauge-cease to flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MG', 'MIE', 'FLV']
    elif category == 'multi-gauge-cumulative':
        pull =  ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MAXF','MG', 'MIE','AP','GP', 'FLV']
    elif category == 'simul-gauge-flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE', 'FLV']
    elif category == 'simul-gauge-low flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE', 'FLV']
    elif category == 'simul-gauge-cease to flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE', 'FLV']
    elif category == 'complex':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME',  'GP', 'EPY', 'MIE', 'FLV']  
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
    Results:
        str: requested EWR component
    
    '''
    component = list(EWR_TABLE[((EWR_TABLE['Gauge'] == GAUGE) & 
                           (EWR_TABLE['Code'] == EWR) &
                           (EWR_TABLE['PlanningUnitName'] == PU)
                          )][PARAMETER])[0]
    return component if component else 0