import io
import requests
import pandas as pd
import dashboard

# Importing the climate cat data - to be replaced by RAS data once available:

def get_climate_cats():
    '''Uses standard climate categorisation unless user selects the 10,000 year climate sequence,
    in which case this is used'''
    
    if dashboard.climate_type.value == 'Standard - 1911 to 2018 climate categorisation':
        climate_cats = pd.read_csv('Climate_data/climate_cats.csv', index_col = 0)
        
    elif dashboard.climate_type.value  == 'NSW 10,000 year climate sequence':
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
    # Changing the flow max threshold to a high value when there is none available:
    use_df['flow threshold max'].fillna(1000000, inplace = True)
    
    # Removing the rows with no available thresholds
    use_df = use_df.loc[(use_df["flow threshold min"] != '?') | (use_df["flow threshold max"] != '?') |\
                        (use_df["volume threshold"] != '?') |\
                        (use_df["level threshold min"] != '?') | (use_df["level threshold max"] != '?')]
    
    noThresh_df = use_df.loc[(use_df["flow threshold min"] == '?') | (use_df["volume threshold"] == '?') |\
                        (use_df["level threshold min"] == '?') | (use_df["level threshold max"] == '?')]
    
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
    weirpoolGauge = temp_ewr_table['weirpool gauge'].values
    # Update when we have the common names in the database:
    weirpoolGaugeName = temp_ewr_table['weirpool gauge'].values
    
    gauge_to_name = dict()
    for iteration, value in enumerate(gauge_number):
        if type(value) == str:
            gauge_to_name[value] = gauge_name[iteration]
    for iteration, value in enumerate(weirpoolGauge):
        if type(value) == str:
            gauge_to_name[value] = weirpoolGaugeName[iteration]   
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
                               'Macquarie': macquarie_catchment, 'Lachlan': lachlan_catchment,
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
    
def wy_to_climate(input_df, catchment):
    '''Takes in '''
    climate_cat = get_climate_cats()
    df_reindex = input_df.reset_index()
    df_reindex['climate'] = df_reindex['water year'].apply(lambda x: climate_cat[catchment].loc[x])
#     df_reindex['climate'] = df_reindex['Date'].apply(lambda x: climate_cat[catchment].loc[x.year])
    df = df_reindex.set_index('Date')
    
    return df
    
ewr_cats = {'vlf': 'Very low flows', 
            'bf': 'Baseflows', 
            'sf': 'Small freshes', 
            'lf': 'Large freshes',
            'bk': 'Bankfull flows', 
            'ob': 'Overbank flows', 
            'ac': 'Anabranch connection'}