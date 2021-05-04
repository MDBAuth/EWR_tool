import io
import requests
import pandas as pd
import numpy as np

# Importing the climate cat data - to be replaced by RAS data once available:

def getLevelGauges():
    '''Call this function to return the menindee lakes gauges, and the weirpool level gauges'''
    
    menindeeGauges = ['425020', '425022', '425023']
    
    weirpoolGauges = {'414203': '414209', 
                      '425010': '4260501', 
                      '4260507': '4260508',
                      '4260505': '4260506'}
    
    return menindeeGauges, weirpoolGauges
    
def get_climate_cats(climate_file):
    '''Uses standard climate categorisation unless user selects the 10,000 year climate sequence,
    in which case this is used'''
    
    if climate_file == 'Standard - 1911 to 2018 climate categorisation':
        climate_cats = pd.read_csv('Climate_data/climate_cats.csv', index_col = 0)
        
    elif climate_file  == 'NSW 10,000 year climate sequence':
        climate_cats = pd.read_csv('Climate_data/climate_cats_10000year.csv', index_col = 0)

    return climate_cats

def get_ewr_table():
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
    no_duration, DSF_ewrs = get_ewr_table()
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

def get_bigmod_codes():
    metadata = pd.read_csv('bigmod_metadata/SiteID.csv', engine = 'python', dtype=str)
    metadata = metadata.where(pd.notnull(metadata), None)

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

def getMultiGauges(dataType):
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

def getSimultaneousGauges(dataType):
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

def getComplexCalcs():
    '''Call function to return a dictionary of '''
    complexCalcs = {'409025': {'OB2_S': 'flowDurPostReq', 'OB2_P': 'flowDurPostReq',
                              'OB3_S': 'flowDurOutsideReq', 'OB3_P': 'flowDurOutsideReq'}}
    
    return complexCalcs