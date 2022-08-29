import io
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from cachetools import cached, TTLCache

BASE_PATH = Path(__file__).resolve().parent

# Importing the climate cat data - to be replaced by RAS data once available:
   
def get_climate_cats(climate_file):
    '''Uses standard climate categorisation unless user selects the 10,000 year climate sequence,
    in which case this is used'''
    
    if climate_file == 'Standard - 1911 to 2018 climate categorisation':
        climate_cats = pd.read_csv( BASE_PATH / 'climate_data/climate_cats.csv', index_col = 0)
        
    elif climate_file  == 'NSW 10,000 year climate sequence':
        climate_cats = pd.read_csv(BASE_PATH / 'climate_data/climate_cats_10000year.csv', index_col = 0)
        
    return climate_cats

@cached(cache=TTLCache(maxsize=1024, ttl=1800))
def get_EWR_table(file_path = None):
    
    ''' Loads ewr table from blob storage, separates out the readable ewrs from the 
    ewrs with 'see notes' exceptions, those with no threshold, and those with undefined names,
    does some cleaning, including swapping out '?' in the frequency column with 0'''
    
    if file_path:
        df = pd.read_csv(file_path,
         usecols=['PlanningUnitID', 'PlanningUnitName',  'CompliancePoint/Node', 'gauge', 'code', 'start month',
                                    'end month', 'frequency', 'events per year', 'duration', 'min event', 'flow threshold min', 'flow threshold max',
                                    'max inter-event', 'within event gap tolerance', 'weirpool gauge', 'flow level volume', 'level threshold min',
                                    'level threshold max', 'volume threshold', 'drawdown rate', 'Accumulation period (Days)','multigauge',
                                     'max_duration','TriggerDay','TriggerMonth','DrawDownRateWeek'],
                            dtype='str', encoding='cp1252')

    if not file_path:
        my_url = 'https://az3mdbastg001.blob.core.windows.net/mdba-public-data/NSWEWR_LIVE_DEV.csv'
        proxies={} # Populate with your proxy settings
        s = requests.get(my_url, proxies=proxies).text
        df = pd.read_csv(io.StringIO(s),
            usecols=['PlanningUnitID', 'PlanningUnitName',  'CompliancePoint/Node', 'gauge', 'code', 'start month',
                                    'end month', 'frequency', 'events per year', 'duration', 'min event', 'flow threshold min', 'flow threshold max',
                                    'max inter-event', 'within event gap tolerance', 'weirpool gauge', 'flow level volume', 'level threshold min',
                                    'level threshold max', 'volume threshold', 'drawdown rate', 'Accumulation period (Days)','multigauge',
                                     'max_duration','TriggerDay','TriggerMonth','DrawDownRateWeek'],
                        dtype='str', encoding='cp1252'
                        )

    df = df.replace('?','')
    df = df.fillna('')

    # removing the 'See notes'
    okay_EWRs = df.loc[(df["start month"] != 'See note') & (df["end month"] != 'See note')]
    see_notes = df.loc[(df["start month"] == 'See note') & (df["end month"] == 'See note')]

    # Filtering those with no flow/level/volume thresholds
    noThresh_df = okay_EWRs.loc[(okay_EWRs["flow threshold min"] == '') & (okay_EWRs["flow threshold max"] == '') &\
                             (okay_EWRs["volume threshold"] == '') &\
                             (okay_EWRs["level threshold min"] == '') & (okay_EWRs["level threshold max"] == '')]
    okay_EWRs = okay_EWRs.loc[(okay_EWRs["flow threshold min"] != '') | (okay_EWRs["flow threshold max"] != '') |\
                        (okay_EWRs["volume threshold"] != '') |\
                        (okay_EWRs["level threshold min"] != '') | (okay_EWRs["level threshold max"] != '')]

    # Filtering those with no durations
    okay_EWRs = okay_EWRs.loc[(okay_EWRs["duration"] != '')]
    no_duration = df.loc[(df["duration"] == '')]

    # FIltering DSF EWRs
    condDSF = df['code'].str.startswith('DSF')
    DSF_ewrs = df[condDSF]
    condDSF_inv = ~(okay_EWRs['code'].str.startswith('DSF'))
    okay_EWRs = okay_EWRs[condDSF_inv]

    # Combine the unuseable EWRs into a dataframe and drop dups:
    bad_EWRs = pd.concat([see_notes, noThresh_df, no_duration, DSF_ewrs], axis=0)
    bad_EWRs = bad_EWRs.drop_duplicates()

    # Changing the flow and level max threshold to a high value when there is none available:
    okay_EWRs['flow threshold max'].replace({'':'1000000'}, inplace = True)
    okay_EWRs['level threshold max'].replace({'':'1000000'}, inplace = True)
    
    return okay_EWRs, bad_EWRs

@cached(cache=TTLCache(maxsize=1024, ttl=1800))
def map_gauge_to_catchment(my_url = 'https://az3mdbastg001.blob.core.windows.net/mdba-public-data/NSWEWR_LIVE_DEV.csv'):
    ''' Allocates all the locations in the ewr table with catchments, as indicated by the
    first three numbers for each gauge '''
    
    lower_darling_gauges = ['425054', '425010', '425011', '425052', '425013', '425056', '425007', 
                            '425057', '425005', '425050', '425048', '425019', '425014', '425023', 
                            '425012', '425044', '425049', '425001', '425022', '42510037', '42510036',
                            '425034', '425046', '425020', ]
    
    EWR_table, bad_EWRs =  get_EWR_table(my_url)
    
    gauge_number = EWR_table['gauge'].values
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

def get_MDBA_codes():
    '''
    Load MDBA model metadata file containing model nodes
    and gauges they correspond to
    '''
    metadata = pd.read_csv( BASE_PATH / 'model_metadata/SiteID_MDBA.csv', engine = 'python', dtype=str, encoding='windows-1252')
#     metadata = metadata.where(pd.notnull(metadata), None)

    return metadata
  
def get_NSW_codes():
    '''
    Load NSW model metadata file containing model nodes
    and gauges they correspond to
    '''
    metadata = pd.read_csv( BASE_PATH / 'model_metadata/SiteID_NSW.csv', engine = 'python', dtype=str)
    
    return metadata

def gauge_to_catchment(input_gauge):
    '''
    Takes in a gauge, maps it to the catchment
    returns the catchment
    '''
    catchments_gauges = map_gauge_to_catchment()
    for catchment in catchments_gauges:
        if input_gauge in catchments_gauges[catchment]:
            return catchment
    
def wy_to_climate(water_years, catchment, climate_file):
    '''
    water_years = a daily water year value array
    catchment = the catchment that the gauge is in
    climate file = which climate data to use
    
    The function assigns a climate categorisation for every day, depending on the water year and catchment in the climate file
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


def get_level_gauges():
    '''Returning level gauges with EWRs'''
    
    menindeeGauges = ['425020', '425022', '425023']
    
    weirpoolGauges = {'414203': '414209', 
                      '425010': '4260501', 
                      '4260507': '4260508',
                      '4260505': '4260506'}
    
    return menindeeGauges, weirpoolGauges


def get_multi_gauges(dataType):
    '''
    Call function to return a dictionary of multi gauges.
    Multi gauges are for EWRs that require the flow of two gauges to be added together
    '''
    
    gauges = {'PU_0000130': {'421090': '421088', '421088': '421090'},
              'PU_0000131': {'421090': '421088', '421088': '421090'},
              'PU_0000132': {'421090': '421088', '421088': '421090'},
              'PU_0000133': {'421090': '421088', '421088': '421090'}
             }
    returnData = {}
    if dataType == 'all':
        returnData = gauges
    if dataType == 'gauges':
        for i in gauges:
            returnData = {**returnData, **gauges[i]}
    
    return returnData

def get_simultaneous_gauges(dataType):
    '''
    Call function to return a dictionary of simultaneous gauges.
    Simultaneous gauges are for EWRs that need to be met simultaneously with EWRs at another location
    '''
    
    gauges = {'PU_0000131': {'421090': '421022', '421022': '421090'},
              'PU_0000132': {'421090': '421022', '421022': '421090'},
              'PU_0000133': {'421090': '421022', '421022': '421090'}
             }
    returnData = {}
    if dataType == 'all':
        returnData = gauges
    if dataType == 'gauges':
        for i in gauges:
            returnData = {**returnData, **gauges[i]}
        
    return returnData

def get_complex_calcs():
    '''
    Returns a dictionary of the complex EWRs, and the type of analysis that needs to be undertaken
    These EWRs cannot be calculated using the standard suite of functions
    '''
    complexCalcs = {'409025': {'OB2_S': 'flowDurPostReq', 'OB2_P': 'flowDurPostReq',
                              'OB3_S': 'flowDurOutsideReq', 'OB3_P': 'flowDurOutsideReq'}}
    
    return complexCalcs


def get_gauges(category):
    '''
    Gathers a list of either all gauges that have EWRs associated with them,
    a list of all flow type gauges that have EWRs associated with them,
    or a list of all level type gauges that have EWRs associated with them
    '''
    EWR_table, bad_EWRs = get_EWR_table()
    menindee_gauges, wp_gauges = get_level_gauges()
    wp_gauges = list(wp_gauges.values())
    
    multi_gauges = get_multi_gauges('gauges')
    simul_gauges = get_simultaneous_gauges('gauges')
    multi_gauges = list(multi_gauges.values())
    simul_gauges = list(simul_gauges.values())
    if category == 'all gauges':
        return set(EWR_table['gauge'].to_list() + menindee_gauges + wp_gauges + multi_gauges + simul_gauges)
    elif category == 'flow gauges':
        return set(EWR_table['gauge'].to_list() + multi_gauges + simul_gauges)
    elif category == 'level gauges':
        return set(menindee_gauges + wp_gauges)
    else:
        raise ValueError('''No gauge category sent to the "get_gauges" function''')

def get_EWR_components(category):
    '''
    Ingests an EWR type and a gauge, returns the components required to analyse for
    this type of EWR. Each code stands for a component.    
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
    
def get_bad_QA_codes():
    '''These codes are NSW specific'''
    return [151, 152, 153, 155, 180, 201, 202, 204, 205, 207, 223, 255]
    

def additional_nest_pull(EWR_info, gauge, EWR, allowance):
    '''Additional EWR information not yet included in the database has been hard coded'''
    if ((gauge == '409025') and ('NestS1' in EWR)):
        EWR_info['trigger_day'] = 15
        EWR_info['trigger_month'] = 9
    elif ((gauge == '409207') and ('NestS1' in EWR)):
        EWR_info['trigger_day'] = 1
        EWR_info['trigger_month'] = 10
        EWR_info['start_month'] = 10
        EWR_info['start_day'] = None
    elif 'NestS1' in EWR:
        EWR_info['trigger_day'] = None
        EWR_info['trigger_month'] = None
        
    return EWR_info

def analysis():
    '''Returns a list of types of analysis to be shown in the summary table'''
    
    return ['Event years','Frequency','Target frequency','Achievement count', 'Achievements per year', 'Event count','Events per year',
            'Event length','Threshold days','Inter-event exceedence count', 'Max inter event period (years)', 'No data days',
            'Total days']

def weirpool_type(EWR):
    '''Returns the type of Weirpool EWR. Currently only WP2 EWRs are classified as weirpool raisings'''

    return 'raising' if EWR == 'WP2' else 'falling'

def convert_max_interevent(unique_water_years, water_years, EWR_info):
    '''Max interevent is saved in the database as years, we want to convert it to days.'''
    new_max_interevent = 365 * EWR_info['max_inter-event']
    
    return new_max_interevent

def get_planning_unit_info():
    '''Run this function to get the planning unit MDBA ID and equivilent planning unit name as specified in the LTWP'''
    EWR_table, bad_EWRs = get_EWR_table()
        
    planningUnits = EWR_table.groupby(['PlanningUnitID', 'PlanningUnitName']).size().reset_index().drop([0], axis=1) 
    
    return planningUnits