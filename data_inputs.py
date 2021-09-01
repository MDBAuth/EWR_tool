import io
import requests
import pandas as pd
import numpy as np

# Importing the climate cat data - to be replaced by RAS data once available:
   
def get_climate_cats(climate_file):
    '''Uses standard climate categorisation unless user selects the 10,000 year climate sequence,
    in which case this is used'''
    
    if climate_file == 'Standard - 1911 to 2018 climate categorisation':
        climate_cats = pd.read_csv('Climate_data/climate_cats.csv', index_col = 0)
        
    elif climate_file  == 'NSW 10,000 year climate sequence':
        climate_cats = pd.read_csv('Climate_data/climate_cats_10000year.csv', index_col = 0)

    return climate_cats

def get_EWR_table():
    ''' Loads ewr table from blob storage, seperates out the readable ewrs from the 
    ewrs with 'see notes' exceptions, those with no threshold, and those with undefined names,
    does some cleaning, including swapping out '?' in the frequency column with 0'''

    proxy_dict={} # Populate with your proxy settings
    my_url = ('https://az3mdbastg001.blob.core.windows.net/mdba-public-data/NSWEWR_LIVE.csv')
    s = requests.get(my_url, proxies=proxy_dict).text
    df = pd.read_csv(io.StringIO(s), dtype={'gauge':'str', 'weirpool gauge': 'str'})
    # removing the 'See notes'
    use_df = df.loc[(df["start month"] != 'See note') & (df["end month"] != 'See note')]
    see_notes = df.loc[(df["start month"] == 'See note') & (df["end month"] == 'See note')]
    
    # Removing the bad codes
    cond = df['code'].str.startswith('???')
    undefined_ewrs = df[cond]
    cond_1 = ~(use_df['code'].str.startswith('???'))
    use_df = use_df[cond_1]
    
    # Swap nan cells for '?'
    use_df['level threshold min'].fillna('?', inplace = True)
    use_df['level threshold max'].fillna('?', inplace = True)
    use_df['volume threshold'].fillna('?', inplace = True)
    use_df['flow threshold max'].fillna('?', inplace = True)
    
    # Removing the rows with no available thresholds
    noThresh_df = use_df.loc[(use_df["flow threshold min"] == '?') & (use_df["flow threshold max"] == '?') &\
                        (use_df["volume threshold"] == '?') &\
                        (use_df["level threshold min"] == '?') & (use_df["level threshold max"] == '?')]
    
    # Keeping the rows with a threshold:
    use_df = use_df.loc[(use_df["flow threshold min"] != '?') | (use_df["flow threshold max"] != '?') |\
                        (use_df["volume threshold"] != '?') |\
                        (use_df["level threshold min"] != '?') | (use_df["level threshold max"] != '?')]

    # Changing the flow and level max threshold to a high value when there is none available:
    use_df['flow threshold max'].replace({'?':1000000}, inplace = True)
    use_df['level threshold max'].replace({'?':1000000}, inplace = True)
    
    # Removing the rows with no duration
    use_df = use_df.loc[(use_df["duration"] != '?')]
    no_duration = df.loc[(df["duration"] == '?')]   
    
    # Removing the DSF EWRs
    condDSF = df['code'].str.startswith('DSF')
    DSF_ewrs = df[condDSF]
    condDSF_inv = ~(use_df['code'].str.startswith('DSF'))
    use_df = use_df[condDSF_inv]
    
    # Changing the flow max threshold to a high value when there is none available:
    use_df.loc[use_df['flow threshold max'] == '?', 'flow threshold max'] = 1000000
    return use_df, see_notes, undefined_ewrs, noThresh_df, no_duration, DSF_ewrs

lower_darling_gauges = ['425054', '425010', '425011', '425052', '425013', '425056', '425007', 
                        '425057', '425005', '425050', '425048', '425019', '425014', '425023', 
                        '425012', '425044', '425049', '425001', '425022', '42510037', '42510036',
                        '425034', '425046']

def catchments_gauges_dict():
    ''' Allocates all the locations in the ewr table with catchments, as indicated by the
    first three numbers for each gauge '''
    
    temp_ewr_table, see_notes, undefined_ewrs, noThresh_df,\
    no_duration, DSF_ewrs = get_EWR_table()
    gauge_number = temp_ewr_table['gauge'].values
    gauge_name = temp_ewr_table['CompliancePoint/Node'].values
    
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
    for k, v in gauge_to_name.items():
        if k.startswith('419'):
            namoi_catchment.update({k: v})
        elif k.startswith('418'):
            gwydir_catchment.update({k: v})
        elif (k.startswith('421')or k.startswith('420')):
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
              
    gauge_to_catchment.update({'Namoi': namoi_catchment, 'Gwydir': gwydir_catchment,
                               'Macquarie-Castlereagh': macquarie_catchment, 'Lachlan': lachlan_catchment,
                               'Lower Darling': lower_darling_catchment, 
                               'Barwon-Darling': barwon_darling_catchment,
                               'Murray': murray_catchment,
                               'Murrumbidgee': murrumbidgee_catchment
                              })
    return gauge_to_catchment


catchments_gauges = catchments_gauges_dict()

def get_MDBA_codes():
    ''''''
    metadata = pd.read_csv('model_metadata/SiteID_MDBA.csv', engine = 'python', dtype=str)
    metadata = metadata.where(pd.notnull(metadata), None)

    return metadata
  
def get_NSW_codes():
    ''''''
    metadata = pd.read_csv('model_metadata/SiteID_NSW.csv', engine = 'python', dtype=str)
    
    return metadata

def gauge_to_catchment(input_gauge):
    ''' Takes in a gauge, maps it to the catchment
    returns the catchment '''
    
    for catchment in catchments_gauges:
        if input_gauge in catchments_gauges[catchment]:
            return catchment
    
def wy_to_climate(input_df, catchment, climate_file):
    '''Takes in '''
    # Get the climate categorisation:
    climate_cats = get_climate_cats(climate_file)
    
    # Get the unique years covered in the flow dataframe, and how many days are in each year:
    years = input_df.index.year.values
    unique_years, count_years = np.unique(years, return_counts=True)
    
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
    '''Call this function to return the menindee lakes gauges, and the weirpool level gauges'''
    
    menindeeGauges = ['425020', '425022', '425023']
    
    weirpoolGauges = {'414203': '414209', 
                      '425010': '4260501', 
                      '4260507': '4260508',
                      '4260505': '4260506'}
    
    return menindeeGauges, weirpoolGauges


def get_multi_gauges(dataType):
    '''Call function to return a dictionary of associated gauges'''
    
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
    '''Call function to return a dictionary of associated gauges'''
    
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
    '''Call function to return a dictionary of '''
    complexCalcs = {'409025': {'OB2_S': 'flowDurPostReq', 'OB2_P': 'flowDurPostReq',
                              'OB3_S': 'flowDurOutsideReq', 'OB3_P': 'flowDurOutsideReq'}}
    
    return complexCalcs


def get_gauges(category):
    ''''''
    EWR_table, seeNotesEwrs, undefinedEwrs, noThresh_df, noDurationEwrs, DSFewrs = get_EWR_table()
    menindee_gauges, wp_gauges = get_level_gauges()
    wp_gauges = list(wp_gauges.values())
    
    multi_gauges = get_multi_gauges('gauges')
    simul_gauges = get_simultaneous_gauges('gauges')
    multi_gauges = list(multi_gauges.values())
    simul_gauges = list(simul_gauges.values())
    if category == 'all gauges':
        return EWR_table['gauge'].to_list() + menindee_gauges + wp_gauges + multi_gauges + simul_gauges
    elif category == 'flow gauges':
        return EWR_table['gauge'].to_list() + multi_gauges + simul_gauges
    elif category == 'level gauges':
        return menindee_gauges + wp_gauges
    else:
        raise ValueError('''No gauge category sent to the "get_gauges" function''')

def get_EWR_components(category):
    '''Ingests an EWR type and a gauge, returns the components required to analyse for
    this type of EWR.
    '''

    if category == 'flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'GP', 'EPY', 'ME', 'MIE']
    elif category == 'low flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'EPY', 'DURVD', 'MIE']
    elif category == 'cease to flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'EPY', 'DURVD', 'MIE']
    elif category == 'cumulative':
        pull =  ['SM', 'EM', 'MINV', 'DUR', 'EPY', 'MINF', 'MIE']
    elif category == 'level':
        pull = ['SM', 'EM', 'MINL', 'MAXL', 'DUR', 'EPY', 'MD', 'ME', 'MIE']
    elif category == 'weirpool-raising':
        pull=['SM', 'EM', 'MINF', 'MAXF', 'MINL', 'DUR', 'MD', 'EPY','WPG', 'MIE']
    elif category == 'weirpool-falling':
        pull=['SM', 'EM', 'MINF', 'MAXF', 'MAXL', 'DUR', 'MD', 'EPY','WPG', 'MIE']
    elif category == 'nest-level':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'MD', 'EPY', 'WPG', 'MIE']
    elif category == 'nest-percent':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'MD', 'EPY', 'MIE']
    elif category == 'multi-gauge-flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'GP', 'EPY', 'ME', 'MG', 'MIE']
    elif category == 'multi-gauge-low flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'EPY', 'DURVD', 'MG', 'MIE']
    elif category == 'multi-gauge-cease to flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'EPY', 'DURVD', 'MG', 'MIE']
    elif category == 'multi-gauge-cumulative':
        pull =  ['SM', 'EM', 'MINV', 'DUR', 'EPY', 'MINF', 'MG', 'MIE']
    elif category == 'simul-gauge-flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'GP', 'EPY', 'ME', 'DURVD', 'SG', 'MIE']
    elif category == 'simul-gauge-low flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'GP', 'EPY', 'ME', 'DURVD', 'SG', 'MIE']
    elif category == 'simul-gauge-cease to flow':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'GP', 'EPY', 'ME', 'DURVD', 'SG', 'MIE']
    elif category == 'complex':
        pull = ['SM', 'EM', 'MINF', 'MAXF', 'DUR', 'EPY', 'MIE']  
    return pull
    
def get_bad_QA_codes():
    
    return [151, 152, 153, 155, 180, 201, 202, 204, 205, 207, 223, 255]
    

def complex_EWR_pull(EWR_info, gauge, EWR, allowance):
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
    if gauge == '409025' and ('OB2_S' in EWR or 'OB2_P' in EWR):
        EWR_info['min_flow_post'] = int(round(9000*toleranceDict['minThreshold'], 0))
        EWR_info['max_flow_post'] = int(round(1000000*toleranceDict['maxThreshold'], 0))
        EWR_info['duration_post'] = int(round(105*toleranceDict['duration'], 0))
        EWR_info['start_month'] = 7
        EWR_info['end_month'] = 6
        EWR_info['gap_tolerance'] = 7
    if gauge == '409025' and (EWR == 'OB3_S' or EWR == 'OB3_P'):
        EWR_info['min_flow_outside'] = int(round(15000*allowance['minThreshold'], 0))
        EWR_info['max_flow_outside'] = int(round(1000000*allowance['maxThreshold'], 0))
        EWR_info['duration_outside'] = int(round(90*allowance['duration'], 0))
        EWR_info['start_month'] = 7
        EWR_info['end_month'] = 6
        EWR_info['gap_tolerance'] = 7
        
    return EWR_info

def analysis():
    '''Returns a list of types of analysis to be shown in the summary table'''
    
    return ['Event years','Frequency','Target frequency','Event count','Events per year',
            'Event length','Threshold days','Inter-event exceedence count','No data days',
            'Total days']

def weirpool_type(EWR):
    '''Returns the type of Weirpool EWR. Currently only WP2 EWRs are classified as weirpool raisings'''
    if EWR == 'WP2':
        weirpool_type = 'raising'
    else:
        weirpool_type = 'falling'

    return weirpool_type

def convert_max_interevent(unique_water_years, water_years, EWR_info):
    '''Max interevent is saved in the database as years, we want to convert it to days.
    '''
    average_days_per_year = len(water_years)/len(unique_water_years)
    new_max_interevent = average_days_per_year * EWR_info['max_inter-event']
    
    return new_max_interevent, average_days_per_year

def get_planning_unit_info():
    '''Run this function to get the planning unit MDBA ID and equivilent planning unit name as specified in the LTWP'''
    ewr_data, see_notes_ewrs, undefined_ewrs, noThresh_df, no_duration, DSF_ewrs = get_EWR_table()
        
    planningUnits = ewr_data.groupby(['PlanningUnitID', 'PlanningUnitName']).size().reset_index().drop([0], axis=1) 
    
    return planningUnits