import io
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


# Importing the climate cat data - to be replaced by RAS data once available:

def get_climate_cats(climate_file: str) -> pd.DataFrame:
    '''Uses standard climate categorisation unless user selects the 10,000 year climate sequence,
    in which case this is used'''

    """Using climate data path in the event that the folder is moved or the name is changed,
     you won't need to find/replace all the file paths"""
    climate_data_path = 'New/climate_data'

    """Converted if/elif to dictionary lookup. Makes adding additional climate files easier and cleaner"""
    climate_files = {
        'Standard - 1911 to 2018 climate categorisation': f'{climate_data_path}/climate_cats.csv',
        'NSW 10,000 year climate sequence': f'{climate_data_path}/climate_cats_10000year.csv',
    }

    # if climate_file == 'Standard - 1911 to 2018 climate categorisation':
    #     return pd.read_csv('Climate_data/climate_cats.csv', index_col = 0)
    # elif climate_file  == 'NSW 10,000 year climate sequence':
    #     return pd.read_csv('Climate_data/climate_cats_10000year.csv', index_col = 0)
    try:
        return pd.read_csv(climate_files[climate_file], index_col=0)
    except KeyError:
        raise Exception(f"Unable to parse climate_file '{climate_file}'")


def get_EWR_table(my_url: str = 'https://az3mdbastg001.blob.core.windows.net/mdba-public-data/NSWEWR_LIVE.csv') -> \
Tuple[pd.DataFrame, pd.DataFrame]:
    ''' Loads ewr table from blob storage, seperates out the readable ewrs from the
    ewrs with 'see notes' exceptions, those with no threshold, and those with undefined names,
    does some cleaning, including swapping out '?' in the frequency column with 0'''

    proxies = {}  # Populate with your proxy settings # TODO Don't use this, use system settings!
    s = requests.get(my_url, proxies=proxies).text
    df = pd.read_csv(io.StringIO(s),
                     usecols=['PlanningUnitID', 'PlanningUnitName', 'CompliancePoint/Node', 'gauge', 'code',
                              'start month',
                              'end month', 'frequency', 'events per year', 'duration', 'min event',
                              'flow threshold min', 'flow threshold max',
                              'max inter-event', 'within event gap tolerance', 'weirpool gauge', 'flow level volume',
                              'level threshold min',
                              'level threshold max', 'volume threshold', 'drawdown rate'],
                     dtype='str'
                     )

    df = df.replace('?', '')
    df = df.fillna('')

    # removing the 'See notes'

    # TODO

    # okay_EWRs = df.loc[[(df["start month"] != 'See note'), (df["end month"] != 'See note')]]
    okay_EWRs = df.loc[(df["start month"] != 'See note') & (df["end month"] != 'See note')]
    see_notes = df.loc[(df["start month"] == 'See note') & (df["end month"] == 'See note')]

    # Filtering those with no flow/level/volume thresholds

    # flow_level_volume_thresholds = ["flow threshold min", "flow threshold max", "volume threshold",
    #                                 "level threshold min", "level threshold max"]

    # noThresh_df = okay_EWRs.loc[[(okay_EWRs[threshold] == '') for threshold in flow_level_volume_thresholds]]
    # okay_EWRs = okay_EWRs.loc[[(okay_EWRs[threshold] != '') for threshold in flow_level_volume_thresholds]]

    noThresh_df = okay_EWRs.loc[(okay_EWRs["flow threshold min"] == '') & (okay_EWRs["flow threshold max"] == '') & \
                                (okay_EWRs["volume threshold"] == '') & \
                                (okay_EWRs["level threshold min"] == '') & (okay_EWRs["level threshold max"] == '')]

    okay_EWRs = okay_EWRs.loc[(okay_EWRs["flow threshold min"] != '') | (okay_EWRs["flow threshold max"] != '') | \
                              (okay_EWRs["volume threshold"] != '') | \
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
    okay_EWRs['flow threshold max'].replace({'': '1000000'}, inplace=True)
    okay_EWRs['level threshold max'].replace({'': '1000000'}, inplace=True)

    return okay_EWRs, bad_EWRs


def map_gauge_to_catchment(
        my_url: str = 'https://az3mdbastg001.blob.core.windows.net/mdba-public-data/NSWEWR_LIVE.csv') -> Dict:
    ''' Allocates all the locations in the ewr table with catchments, as indicated by the
    first three numbers for each gauge '''
    lower_darling_gauges = ['425054', '425010', '425011', '425052', '425013', '425056', '425007',
                            '425057', '425005', '425050', '425048', '425019', '425014', '425023',
                            '425012', '425044', '425049', '425001', '425022', '42510037', '42510036',
                            '425034', '425046', '425020', ]

    gauge_to_catchment: dict = {
        'Namoi': dict(),
        'Gwydir': dict(),
        'Macquarie-Castlereagh': dict(),
        'Lachlan': dict(),
        'Lower Darling': dict(),
        'Barwon-Darling': dict(),
        'Murray': dict(),
        'Murrumbidgee': dict(),
        'Border Rivers': dict(),
        'Moonie': dict(),
        'Condamine-Balonne': dict(),
        'Warrego': dict(),
        'Paroo': dict(),
    }
    catchment_map = {
        '419': gauge_to_catchment['Namoi'],
        '418': gauge_to_catchment['Gwydir'],
        '421': gauge_to_catchment['Macquarie-Castlereagh'],
        '420': gauge_to_catchment['Macquarie-Castlereagh'],
        '412': gauge_to_catchment['Lachlan'],
        '401': gauge_to_catchment['Murray'],
        '409': gauge_to_catchment['Murray'],
        '426': gauge_to_catchment['Murray'],
        '414': gauge_to_catchment['Murray'],
        # '425': Deal with this differently because of lower darling / barwon darling
        '410': gauge_to_catchment['Murrumbidgee'],
        '416': gauge_to_catchment['Border Rivers'],
        '417': gauge_to_catchment['Moonie'],
        '422': gauge_to_catchment['Condamine-Balonne'],
        '423': gauge_to_catchment['Warrego'],
        '424': gauge_to_catchment['Paroo'],
    }

    EWR_table, bad_EWRs = get_EWR_table(my_url)
    gauge_to_name = dict()
    for iteration, value in enumerate(EWR_table['gauge'].values):
        if type(value) is str:
            gauge_to_name[value] = EWR_table['CompliancePoint/Node'].values[iteration].strip() # THERE WERE SPACES IN THE NAMES
    for k, v in gauge_to_name.items():
        if k[0:3] in catchment_map:
            catchment_map[k[0:3]].update({k: v})
        elif k.startswith('425'):
            if k in lower_darling_gauges:
                gauge_to_catchment['Lower Darling'].update({k: v})
            else:
                gauge_to_catchment['Barwon-Darling'].update({k: v})
        else:
            # TODO should this warn or explicitly crash?
            # log.warning(f"Unable to parse gauge_to_name key '{k}'")
            raise KeyError(f"Unable to parse gauge_to_name key '{k}'")
    return gauge_to_catchment


def get_MDBA_codes() -> pd.DataFrame:
    '''
    Load MDBA model metadata file containing model nodes
    and gauges they correspond to
    '''
    return pd.read_csv('model_metadata/SiteID_MDBA.csv', engine='python', dtype=str)


def get_NSW_codes() -> pd.DataFrame:
    '''
    Load NSW model metadata file containing model nodes
    and gauges they correspond to
    '''
    return pd.read_csv('model_metadata/SiteID_NSW.csv', engine='python', dtype=str)


def gauge_to_catchment(input_gauge: str) -> str:
    '''
    Takes in a gauge, maps it to the catchment
    returns the catchment
    '''
    for catchment, gauges in map_gauge_to_catchment().items():
        if input_gauge in gauges:
            return catchment


def wy_to_climate(water_years: List, catchment: str, climate_file: str) -> np.array:
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
    climateFiltered = climateCatchment[
        (climateCatchment.index >= min(unique_years)) & (climateCatchment.index <= max(unique_years))].values

    # Repeating the climate result for that year over the total days in each year
    def mapper(climate, count):
        return np.repeat(climate, count)

    climateDailyYear = list(map(mapper, climateFiltered, count_years))
    climateDaily = np.concatenate(climateDailyYear)

    return climateDaily


def get_level_gauges() -> Tuple[List, Dict]:
    '''Returning level gauges with EWRs'''

    menindeeGauges = ['425020', '425022', '425023']

    weirpoolGauges = {'414203': '414209',
                      '425010': '4260501',
                      '4260507': '4260508',
                      '4260505': '4260506'}

    return menindeeGauges, weirpoolGauges


def get_multi_gauges(dataType: str) -> Dict:
    '''
    Call function to return a dictionary of multi gauges.
    Multi gauges are for EWRs that require the flow of two gauges to be added together
    '''

    gauges = {'PU_0000130': {'421090': '421088', '421088': '421090'},
              'PU_0000131': {'421090': '421088', '421088': '421090'},
              'PU_0000132': {'421090': '421088', '421088': '421090'},
              'PU_0000133': {'421090': '421088', '421088': '421090'}
              }
    # TODO this is duplicate business logic of get_simultaneous_gauges
    returnData: dict = {}
    if dataType == 'all':
        return gauges
    if dataType == 'gauges':
        for i in gauges:
            # TODO Not sure what the point of returnData is
            returnData = {**returnData, **gauges[i]}

    return returnData


def get_simultaneous_gauges(dataType: str) -> Dict:
    '''
    Call function to return a dictionary of simultaneous gauges.
    Simultaneous gauges are for EWRs that need to be met simultaneously with EWRs at another location
    '''
    gauges = {'PU_0000131': {'421090': '421022', '421022': '421090'},
              'PU_0000132': {'421090': '421022', '421022': '421090'},
              'PU_0000133': {'421090': '421022', '421022': '421090'}
              }
    # returnData = {}
    if dataType == 'all':
        return gauges
    if dataType == 'gauges':
        for i in gauges:
            # TODO This seems off, output is static, and returns
            # TODO Not sure what the point of returnData is
            # {'421090': '421022', '421022': '421090'}
            # returnData = {**returnData, **gauges[i]}
            return {**gauges[i]}

    # return returnData


def get_complex_calcs() -> dict:
    '''
    Returns a dictionary of the complex EWRs, and the type of analysis that needs to be undertaken
    These EWRs cannot be calculated using the standard suite of functions
    '''
    return {'409025': {'OB2_S': 'flowDurPostReq', 'OB2_P': 'flowDurPostReq',
                       'OB3_S': 'flowDurOutsideReq', 'OB3_P': 'flowDurOutsideReq'}}


def get_gauges(category: str) -> set:
    '''
    Gathers a list of either all gauges that have EWRs associated with them,
    a list of all flow type gauges that have EWRs associated with them,
    or a list of all level type gauges that have EWRs associated with them
    '''

    """Best practices would be to avoid self reassignment of variables."""

    EWR_table, bad_EWRs = get_EWR_table()
    menindee_gauges, wp_gauges = get_level_gauges()
    wp_gauges = list(wp_gauges.values())

    multi_gauges = list(get_multi_gauges('gauges').values())
    simul_gauges = list(get_simultaneous_gauges('gauges').values())

    # multi_gauges = get_multi_gauges('gauges')
    # simul_gauges = get_simultaneous_gauges('gauges')
    # multi_gauges = list(multi_gauges.values())
    # simul_gauges = list(simul_gauges.values())

    #

    """Converted if/else to dictionary lookup. List to Set type conversion done at return to avoid code reuse."""

    categories = {
        'all gauges': EWR_table['gauge'].to_list() + menindee_gauges + wp_gauges + multi_gauges + simul_gauges,
        'flow gauges': EWR_table['gauge'].to_list() + multi_gauges + simul_gauges,
        'level gauges': menindee_gauges + wp_gauges,
        }

    # if category == 'all gauges':
    #     return set(EWR_table['gauge'].to_list() + menindee_gauges + wp_gauges + multi_gauges + simul_gauges)
    # elif category == 'flow gauges':
    #     return set(EWR_table['gauge'].to_list() + multi_gauges + simul_gauges)
    # elif category == 'level gauges':
    #     return set(menindee_gauges + wp_gauges)

    try:
        return set(categories[category])
    except KeyError:
        raise KeyError(f"category sent to the 'get_gauges' function, '{category}' unhandled")


def get_EWR_components(category: str) -> list:
    '''
    Ingests an EWR type and a gauge, returns the components required to analyse for
    this type of EWR. Each code stands for a component.    
    '''

    """Converted if/elif to dictionary lookup"""

    categories = {
        'flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'MIE'],
        'low flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MIE'],
        'cease to flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MIE'],
        'cumulative': ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MIE'],
        'level': ['SM', 'EM', 'MINL', 'MAXL', 'DUR', 'ME', 'EPY', 'MD', 'MIE'],
        'weirpool-raising': ['SM', 'EM', 'MINF', 'MAXF', 'MINL', 'DUR', 'ME', 'MD', 'EPY', 'WPG', 'MIE'],
        'weirpool-falling': ['SM', 'EM', 'MINF', 'MAXF', 'MAXL', 'DUR', 'ME', 'MD', 'EPY', 'WPG', 'MIE'],
        'nest-level': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'MD', 'EPY', 'WPG', 'MIE'],
        'nest-percent': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'MD', 'EPY', 'MIE'],
        'multi-gauge-flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'MG', 'MIE'],
        'multi-gauge-low flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MG', 'MIE'],
        'multi-gauge-cease to flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MG', 'MIE'],
        'multi-gauge-cumulative': ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MG', 'MIE'],
        'simul-gauge-flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE'],
        'simul-gauge-low flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE'],
        'simul-gauge-cease to flow': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE'],
        'complex': ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'MIE']
    }
    # if category == 'flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'MIE']
    # elif category == 'low flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MIE']
    # elif category == 'cease to flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MIE']
    # elif category == 'cumulative':
    #     return   ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MIE']
    # elif category == 'level':
    #     return  ['SM', 'EM', 'MINL', 'MAXL', 'DUR', 'ME', 'EPY', 'MD', 'MIE']
    # elif category == 'weirpool-raising':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'MINL', 'DUR', 'ME',  'MD', 'EPY','WPG', 'MIE']
    # elif category == 'weirpool-falling':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'MAXL', 'DUR', 'ME',  'MD', 'EPY','WPG', 'MIE']
    # elif category == 'nest-level':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME',  'MD', 'EPY', 'WPG', 'MIE']
    # elif category == 'nest-percent':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME',  'MD', 'EPY', 'MIE']
    # elif category == 'multi-gauge-flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME',  'GP', 'EPY', 'MG', 'MIE']
    # elif category == 'multi-gauge-low flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MG', 'MIE']
    # elif category == 'multi-gauge-cease to flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'EPY', 'DURVD', 'MG', 'MIE']
    # elif category == 'multi-gauge-cumulative':
    #     return   ['SM', 'EM', 'MINV', 'DUR', 'ME', 'EPY', 'MINF', 'MG', 'MIE']
    # elif category == 'simul-gauge-flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE']
    # elif category == 'simul-gauge-low flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE']
    # elif category == 'simul-gauge-cease to flow':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME', 'GP', 'EPY', 'DURVD', 'SG', 'MIE']
    # elif category == 'complex':
    #     return  ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'ME',  'GP', 'EPY', 'MIE']

    try:
        return categories[category]
    except KeyError:
        raise Exception(f"Unable to parse category '{category}'")

    # TODO Would previously throw a NameError as the return value was not provided, presumaably this _should_ crash so this is okay?


def get_bad_QA_codes() -> list:
    '''These codes are NSW specific'''
    return [151, 152, 153, 155, 180, 201, 202, 204, 205, 207, 223, 255]


def additional_nest_pull(EWR_info: dict, gauge: str, EWR: list, allowance) -> dict:
    '''Additional EWR information not yet included in the database has been hard coded'''
    if (gauge == '409025') and ('NestS1' in EWR):
        EWR_info['trigger_day'] = 15
        EWR_info['trigger_month'] = 9
    elif (gauge == '409207') and ('NestS1' in EWR):
        EWR_info['trigger_day'] = 1
        EWR_info['trigger_month'] = 10
        EWR_info['start_month'] = 10
        EWR_info['start_day'] = None
    elif 'NestS1' in EWR:
        EWR_info['trigger_day'] = None
        EWR_info['trigger_month'] = None

    return EWR_info


def analysis() -> list:
    '''Returns a list of types of analysis to be shown in the summary table'''

    return ['Event years', 'Frequency', 'Target frequency', 'Achievement count', 'Achievements per year', 'Event count',
            'Events per year',
            'Event length', 'Threshold days', 'Inter-event exceedence count', 'Max inter event period (years)',
            'No data days',
            'Total days']


def weirpool_type(EWR: str) -> list:
    '''Returns the type of Weirpool EWR. Currently only WP2 EWRs are classified as weirpool raisings'''
    if EWR == 'WP2':
        # return 'raising'
        return get_EWR_components('weirpool-raising')
    # return 'falling'
    return get_EWR_components('weirpool-falling')


def convert_max_interevent(unique_water_years, water_years, EWR_info: dict) -> float:
    '''Max interevent is saved in the database as years, we want to convert it to days.'''
    return 365 * EWR_info['max_inter-event']


def get_planning_unit_info() -> pd.DataFrame:
    '''Run this function to get the planning unit MDBA ID and equivilent planning unit name as specified in the LTWP'''
    EWR_table, bad_EWRs = get_EWR_table()
    return EWR_table.groupby(['PlanningUnitID', 'PlanningUnitName']).size().reset_index().drop([0], axis=1)
