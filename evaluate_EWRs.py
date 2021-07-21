import pandas as pd
import re
import numpy as np
from datetime import date, timedelta
from datetime import time

from tqdm import tqdm

import data_inputs

#----------------------------- Getting EWRs from the database ----------------------#

def component_pull(EWR_table, gauge, PU, EWR, component):
    '''Pass EWR characteristics and the table, 
    this function will pull the characteristic from the table
    '''
    info = list(EWR_table[((EWR_table['gauge'] == gauge) & 
                           (EWR_table['code'] == EWR) &
                           (EWR_table['PlanningUnitID'] == PU)
                          )][component])[0]

    return info

def apply_correction(info, correction):
    '''pass EWR info and the correction,
    returns corrected EWR info
    '''
    corrected = round(int(info)*correction, 0)   
    
    return corrected

def get_EWRs(PU, gauge, EWR, EWR_table, allowance, components):
    '''Pulls the ewr componenets for the gauge, planning unit, 
    ewr requested
    '''
    ewrs = {}
    if 'SM' in components:
        start_date = str(component_pull(EWR_table, gauge, PU, EWR, 'start month'))
        if '.' in start_date:
            ewrs['start_day'] = int(start_date.split('.')[1])
            ewrs['start_month'] = int(start_date.split('.')[0])
        else:
            ewrs['start_day'] = None
            ewrs['start_month'] = int(start_date)
    if 'EM' in components:
        end_date = str(component_pull(EWR_table, gauge, PU, EWR, 'end month'))
        if '.' in end_date:  
            ewrs['end_day'] = int(end_date.split('.')[1])
            ewrs['end_month'] = int(end_date.split('.')[0])
        else:
            ewrs['end_day'] = None    
            ewrs['end_month'] =int(end_date)
    if 'MINF' in components:
        min_flow = component_pull(EWR_table, gauge, PU, EWR, 'flow threshold min')
        corrected = apply_correction(min_flow, allowance['minThreshold'])
        ewrs['min_flow'] = int(corrected)
    if 'MAXF' in components:
        max_flow = component_pull(EWR_table, gauge, PU, EWR, 'flow threshold max')
        corrected = apply_correction(max_flow, allowance['maxThreshold'])
        ewrs['max_flow'] = int(corrected)
    if 'MINL' in components:
        min_level = component_pull(EWR_table, gauge, PU, EWR, 'level threshold min')
        corrected = apply_correction(min_level, allowance['minThreshold'])
        ewrs['min_level'] = int(corrected)
    if 'MAXL' in components:
        max_level = component_pull(EWR_table, gauge, PU, EWR, 'level threshold max')
        corrected = apply_correction(max_level, allowance['maxThreshold'])
        ewrs['max_level'] = int(corrected)
    if 'MINV' in components:
        min_volume = component_pull(EWR_table, gauge, PU, EWR, 'volume threshold')
        corrected = apply_correction(min_volume, allowance['minThreshold'])
        ewrs['min_volume'] = int(corrected)
    if 'DUR' in components:
        duration = component_pull(EWR_table, gauge, PU, EWR, 'duration')
        corrected = apply_correction(duration, allowance['duration'])
        ewrs['duration'] = int(corrected)
    if 'GP' in components:
        gap_tolerance = component_pull(EWR_table, gauge, PU, EWR, 'within event gap tolerance')
        ewrs['gap_tolerance'] =int(gap_tolerance)
    if 'EPY' in components:
        events_per_year = component_pull(EWR_table, gauge, PU, EWR, 'events per year')
        ewrs['events_per_year'] =int(events_per_year)       
    if 'ME' in components:
        min_event = component_pull(EWR_table, gauge, PU, EWR, 'min event')
        ewrs['min_event'] =int(min_event)
    if 'MD' in components:
        try: # There may not be a recommended drawdown rate
            max_drawdown = component_pull(EWR_table, gauge, PU, EWR, 'drawdown rate')
            if '%' in max_drawdown:
                value_only = int(max_drawdown.replace('%', ''))
                corrected = apply_correction(value_only, allowance['drawdown'])
                ewrs['drawdown_rate'] = str(int(corrected))+'%'
            else:
                corrected = apply_correction(max_drawdown, allowance['drawdown'])
                ewrs['drawdown_rate'] = str(int(corrected)/100)
        except ValueError: # In this case set a large number
            ewrs['drawdown_rate'] = str(1000000)          
    if 'DURVD' in components:
        try: # There may not be a very dry duration available for this EWR
            EWR_VD = str(EWR + '_VD')
            duration_vd = component_pull(EWR_table, gauge, PU, EWR_VD, 'duration')
            corrected = apply_correction(duration_vd, allowance['duration'])
            ewrs['duration_VD'] =int(corrected)
        except IndexError: # In this case return None type for this component
            ewrs['duration_VD'] = None
    if 'WPG' in components:
        weirpool_gauge = component_pull(EWR_table, gauge, PU, EWR, 'weirpool gauge')
        ewrs['weirpool_gauge'] =str(weirpool_gauge)
    if 'MG' in components:
        multiGaugeDict = data_inputs.getMultiGauges('all')
        ewrs['second_gauge'] = multiGaugeDict[PU][gauge]        
    if 'SG' in components:
        simultaneousGaugeDict = data_inputs.getSimultaneousGauges('all')
        ewrs['second_gauge'] = simultaneousGaugeDict[PU][gauge]
    if 'TF' in components:
        try:
            ewrs['frequency'] = component_pull(EWR_table, gauge, PU, EWR, 'frequency')
        except IndexError:
            ewrs['frequency'] = None
    if 'MIE' in components:
        try:
            ewrs['max_inter-event'] = component_pull(EWR_table, gauge, PU, EWR, 'max inter-event')
        except IndexError:
            ewrs['max_inter-event'] = None
    return ewrs 

#Handling support functions ----------------------#

def mask_dates(EWR_info, input_df):
    if EWR_info['start_day'] == None or EWR_info['end_day'] == None:
        input_df_timeslice = get_month_mask(EWR_info['start_month'],
                                            EWR_info['end_month'],
                                            input_df)
    else:
        input_df_timeslice = get_day_month_mask(EWR_info['start_day'],
                                                EWR_info['end_day'],
                                                EWR_info['start_month'],
                                                EWR_info['end_month'],
                                                input_df)
    return input_df_timeslice

def get_month_mask(start, end, input_df):
    ''' takes in a start date, end date, and dataframe,
    masks the dataframe to these dates'''
    
    if start > end:
        month_mask = (input_df.index.month >= start) | (input_df.index.month <= end)
    elif start <= end:
        month_mask = (input_df.index.month >= start) & (input_df.index.month <= end)  
        
    input_df_timeslice = input_df.loc[month_mask]
    
    return input_df_timeslice

def get_day_month_mask(startDay, startMonth, endDay, endMonth, input_df):
    ''' for the ewrs with a day and month requirement, takes in a start day, start month, 
    end day, end month, and dataframe,
    masks the dataframe to these dates'''

    if startMonth > endMonth:
        month_mask = (((input_df.index.month >= startMonth) & (input_df.index.day >= startDay)) |\
                      ((input_df.index.month <= endMonth) & (input_df.index.day <= endDay)))
        input_df_timeslice = input_df.loc[month_mask]
        
    elif startMonth <= endMonth:
        #Filter the first and last month, and then get the entirety of the months in between
        month_mask1 = ((input_df.index.month == startMonth) & (input_df.index.day >= startDay))
        month_mask2 = ((input_df.index.month == endMonth) & (input_df.index.day <= endDay))
        month_mask3 = ((input_df.index.month > startMonth) & (input_df.index.month < endMonth))
        
        input_df_timeslice1 = input_df.loc[month_mask1]
        input_df_timeslice2 = input_df.loc[month_mask2]
        input_df_timeslice3 = input_df.loc[month_mask3]
        frames = [input_df_timeslice1, input_df_timeslice2, input_df_timeslice3]
        input_df_timeslice4 = pd.concat(frames)
        input_df_timeslice = input_df_timeslice4.sort_index()
        
    return input_df_timeslice

def wateryear_daily(input_df, ewrs):
    '''Creating a daily water year timeseries'''
    # check if water years needs to change:

    years = input_df.index.year.values
    months = input_df.index.month.values

    def appenderStandard(year, month):
        if month < 7:
            year = year - 1
        return year
    
    def appenderNonStandard(year, month):
        if month < ewrs['start_month']:
            year = year - 1
        return year
    
    if ((ewrs['start_month'] <= 6) and (ewrs['end_month'] >= 7)):
        waterYears = np.vectorize(appenderNonStandard)(years, months)
    else:     
        waterYears = np.vectorize(appenderStandard)(years, months)    
    
    return waterYears

# EWR handling functions --------------------------------------------------------------------------

def ctf_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('cease to flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_F)
    water_years = wateryear_daily(df, EWR_info)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(df, catchment, climate)
    # Check data against EWR requirements:
    E, NE, D = ctf_calc(EWR_info, df[gauge].values, water_years, climates)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df

def lowflow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('low flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_F) 
    water_years = wateryear_daily(df, EWR_info)
    catchment = data_inputs.gauge_to_catchment(gauge) 
    climates = data_inputs.wy_to_climate(df, catchment, climate)
    # Check data against EWR requirements:
    E, NE, D = lowflow_calc(EWR_info, df[gauge].values, water_years, climates)    
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df

def flow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_F) 
    water_years = wateryear_daily(df, EWR_info)
    # Check data against EWR requirements:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D = flow_calc_anytime(EWR_info, df[gauge].values, water_years)
    else:
        E, NE, D = flow_calc(EWR_info, df[gauge].values, water_years)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df

def cumulative_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('cumulative')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_F) 
    water_years = wateryear_daily(df, EWR_info)   
    # Check data against EWR requirements:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D = cumulative_calc_anytime(EWR_info, df[gauge].values, water_years)
    else:
        E, NE, D = cumulative_calc(EWR_info, df[gauge].values, water_years)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)

    return PU_df

def level_handle(PU, gauge, EWR, EWR_table, df_L, PU_df, allowance):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('level')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_L) 
    water_years = wateryear_daily(df, EWR_info)  
    # Check data against EWR requirements:   
    E, NE, D = lake_calc(EWR_info, df[gauge].values, water_years)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df

def weirpool_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance):
    # Get ewrs and relevants data:
    weirpool_type = data_inputs.weirpool_type(EWR)
    if weirpool_type == 'raising':
        pull = data_inputs.get_EWR_components('weirpool-raising')
    elif weirpool_type == 'falling':
        pull = data_inputs.get_EWR_components('weirpool-falling')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df_flow = mask_dates(EWR_info, df_F)
    df_lev = mask_dates(EWR_info, df_L)
    water_years = wateryear_daily(df_flow, EWR_info)
    # Check data against EWR requirements:
    try:
        levels = df_lev[EWR_info['weirpool_gauge']].values
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['weirpool_gauge']))
        return PU_df
    E, NE, D = weirpool_calc(EWR_info, df_flow[gauge].values, levels, water_years, weirpool_type)
    PU_df = event_stats(df_flow, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df

def nest_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance):
    # Get ewrs and relevants data:
    requires_second_gauge = gauge in ['414203', '425010', '4260505', '4260507', '4260509']
    if requires_second_gauge:
        pull = data_inputs.get_EWR_components('nest-level')
    else:
        pull = data_inputs.get_EWR_components('nest-percent')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    EWR_info = data_inputs.complex_EWR_pull(EWR_info, gauge, EWR, allowance)
    df_flow = mask_dates(EWR_info, df_F)
    df_lev = mask_dates(EWR_info, df_L)
    water_years = wateryear_daily(df_flow, EWR_info)
    # Check data against EWR requirements:
    if ((EWR_info['trigger_day'] != None) and (EWR_info['trigger_month'] != None)):
        E, NE, D = nest_calc_percent_trigger(EWR_info, df_flow[gauge].values, water_years, df_flow.index)
    elif ((EWR_info['trigger_day'] == None) and (EWR_info['trigger_month'] == None)):
        if '%' in EWR_info['drawdown_rate']:
            E, NE, D = nest_calc_percent(EWR_info, df_flow[gauge].values, water_years)
        else:
            try:
                levels = df_lev[EWR_info['weirpool_gauge']].values
            except KeyError:
                print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
                also needs data for {}'''.format(gauge, EWR, EWR_info['weirpool_gauge']))
                return PU_df
            E, NE, D = nest_calc_weirpool(EWR_info, df_flow[gauge].values, levels, water_years)
    PU_df = event_stats(df_flow, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df

def flow_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('multi-gauge-flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_F)
    water_years = wateryear_daily(df, EWR_info)
    flows1 = df[gauge].values
    try:
        flows2 = df[EWR_info['second_gauge']].values
        flows = flows1 + flows2
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['second_gauge']))
        return PU_df  
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D = flow_calc_anytime(EWR_info, flows, water_years)
    else:
        E, NE, D = flow_calc(EWR_info, flows, water_years)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df

def lowflow_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):  
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('multi-gauge-low flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_F)
    water_years = wateryear_daily(df, EWR_info)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(df, catchment, climate)
    flows1 = df[gauge].values
    try:
        flows2 = df[EWR_info['second_gauge']].values
        flows = flows1 + flows2
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['second_gauge']))
        return PU_df
    E, NE, D = lowflow_calc(EWR_info, flows, water_years, climates)       
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df   
 
def ctf_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('multi-gauge-cease to flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_F)
    water_years = wateryear_daily(df, EWR_info)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(df, catchment, climate)
    flows1 = df[gauge].values  
    try:
        flows2 = df[EWR_info['second_gauge']].values
        flows = flows1 + flows2
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['second_gauge']))
        return PU_df 
    E, NE, D = ctf_calc(EWR_info, df[gauge].values, water_years, climates)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df    

def cumulative_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('multi-gauge-cumulative')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info, df_F)
    water_years = wateryear_daily(df, EWR_info)
    flows1 = df[gauge].values
    try:
        flows2 = df[EWR_info['second_gauge']].values
        flows = flows1 + flows2
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['second_gauge']))
        return PU_df
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D = cumulative_calc_anytime(EWR_info, flows, water_years)
    else:
        E, NE, D = cumulative_calc(EWR_info, flows, water_years)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)    
    return PU_df    

def flow_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('simul-gauge-flow')
    EWR_info1 = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info1, df_F)
    water_years = wateryear_daily(df, EWR_info1)
    EWR_info2 = get_EWRs(PU, EWR_info1['second_gauge'], EWR, EWR_table, allowance, pull)
    flows1 = df[gauge].values
    try:
        flows2 = df[EWR_info1['second_gauge']].values
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge, EWR, EWR_info1['second_gauge']))
        return PU_df
    if ((EWR_info1['start_month'] == 7) and (EWR_info1['end_month'] == 6)):
        E, NE, D = flow_calc_anytime_sim(EWR_info1, EWR_info2, flows1, flows2, water_years)
    else:
        E, NE, D = flow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info1, E, NE, D, water_years)
    return PU_df

def lowflow_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('simul-gauge-low flow')
    EWR_info1 = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info1, df_F)
    water_years = wateryear_daily(df, EWR_info1)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(df, catchment, climate)
    EWR_info2 = get_EWRs(PU, EWR_info1['second_gauge'], EWR, EWR_table, allowance, pull)
    flows1 = df[gauge].values
    try:
        flows2 = df[EWR_info1['second_gauge']].values
    except KeyError: 
        print('''Cannot evaluate this ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge, EWR, EWR_info1['second_gauge']))
        return PU_df
    E1, E2, NE1, NE2, D = lowflow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates)
    PU_df = event_stats_sim(df, PU_df, gauge, EWR_info1['second_gauge'], EWR, EWR_info1, E1, E2, NE1, NE2, D, water_years)
    return PU_df

def ctf_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    # Get ewrs and relevants data:
    pull = data_inputs.get_EWR_components('simul-gauge-cease to flow')
    EWR_info1 = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    df = mask_dates(EWR_info1, df_F)
    water_years = wateryear_daily(df, EWR_info1)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(df, catchment, climate)
    EWR_info2 = get_EWRs(PU, EWR_info1['second_gauge'], EWR, EWR_table, allowance, pull)
    flows1 = df[gauge].values
    try:
        flows2 = df[EWR_info1['second_gauge']].values
    except KeyError: 
        print('''Cannot evaluate this ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge, EWR, EWR_info1['second_gauge']))
        return PU_df
    E1, E2, NE1, NE2, D = ctf_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates)
    PU_df = event_stats_sim(df, PU_df, gauge, EWR_info1['second_gauge'], EWR, EWR_info1, E1, E2, NE1, NE2, D, water_years)
    return PU_df

def complex_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    '''Handling complex EWRs (complex EWRs are hard coded into the tool).
    returns a dataframe yearly results for each ewr within the planning unit'''
    pull = data_inputs.get_EWR_components('complex')
    EWR_info = EWR_pull(PU, gauge, EWR_name, EWR_table, allowance, components)
    EWR_info = data_inputs.complex_EWR_pull(EWR_info, gauge, EWR, allowance)
    df = mask_dates(EWR_info, df_F)
    water_years = wateryear_daily(df, EWR_info)
    # Check the flow timeseries against the ewr
    if '2' in EWR:
        E, NE, D = flow_calc_post_req(EWR_info, df[gauge].values, water_years)
    elif '3' in EWR:
        E, NE, D = flow_calc_outside_req(EWR_info, df[gauge].values, water_years)
    PU_df = event_stats(df, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df

# Checking EWRs -----------------------------------#

def which_water_year(iteration, event, water_years):
    '''Used to find which water year the majority of the event fell in'''
    
    event_wateryears = water_years[iteration-len(event):iteration]
    midway_iteration = (len(event_wateryears))/2
    mid_event = event_wateryears[int(midway_iteration)]

    return mid_event

def flow_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, gap_track, 
               water_years, total_event):
    '''Checks daily flow against ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list, event dictionary, and time between events
    '''
    no_event += 1
    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        event.append(flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) >= EWR_info['min_event']:
                water_year = which_water_year(iteration, event, water_years)
                all_events[water_year].append(np.array(event))
                total_event_gap = no_event - total_event
                all_no_events[water_year].append(np.array(total_event_gap))
                no_event = 0
                
            event = []

    return event, all_events, no_event, all_no_events, gap_track, total_event

def lowflow_check(EWR_info, flow, event, all_events, no_event, all_no_events, water_year):
    '''Checks daily flow against the low flow ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list and event dictionary
    '''    
    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        event.append(flow)
        if no_event > 0:
            all_no_events[water_year].append(np.array(no_event))
        no_event = 0
    else:
        no_event += 1
        if len(event) > 0:
            all_events[water_year].append(np.array(event))
            
        event = []
        
    return event, all_events, no_event, all_no_events

def ctf_check(EWR_info, flow, event, all_events, no_event, all_no_events, water_year):
    '''Checks daily flow against the cease to flow ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list and event dictionary
    '''
    no_event += 1
    if flow <= EWR_info['min_flow']:
        event.append(flow)
    else:
        if len(event) > 0:
            all_events[water_year].append(np.array(event))
            total_event_gap = no_event - len(event)
            all_no_events[water_year].append(np.array(total_event_gap))
            no_event = 0
        event = []
    
    return event, all_events, no_event, all_no_events

def level_check(EWR_info, level, level_change, event, all_events, no_event, all_no_events, water_year):
    '''Checks daily level against the ewr level threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list and event dictionary
    '''
    no_event += 1
    if ((level >= EWR_info['min_level']) and (level <= EWR_info['max_level'])):
        event.append(level)
    else:
        if (len(event) >= EWR_info['duration']):
            all_events[water_year].append(np.array(event))
            total_event_gap = no_event - len(event)
            all_no_events[water_year].append(np.array(total_event_gap))
            no_event = 0
            event = []

    return event, all_events, no_event, all_no_events

def flow_check_sim(iteration, EWR_info1, EWR_info2, water_years, flow1, flow2, event, all_events,
                   no_event, all_no_events, gap_track, total_event):
    '''Checks daily flow against ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list, event dictionary, and time between events
    '''
    no_event += 1
    if ((flow1 >= EWR_info1['min_flow']) and (flow1 <= EWR_info1['max_flow']) and\
        (flow2 >= EWR_info2['min_flow']) and (flow2 <= EWR_info2['max_flow'])):
        event.append(flow1)
        total_event += 1
        gap_track = EWR_info1['gap_tolerance'] # reset the gapTolerance after threshold is reached
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) >= EWR_info1['min_event']:
                water_year = which_water_year(iteration, event, water_years)
                all_events[water_year].append(np.array(event))
                total_event_gap = no_event - total_event
                all_no_events[water_year].append(np.array(total_event_gap))
                no_event = 0
                
            event = []

    return event, all_events, no_event, all_no_events, gap_track, total_event
# Calculation functions ----------------------------------------------------------

def get_duration(climate, EWR_info):
    '''Determines the relevant duration for the water year'''
    if ((climate == 'Very Dry') and (EWR_info['duration_VD'] !=None)):
        duration = EWR_info['duration_VD']
    else:
        duration = EWR_info['duration']
    
    return duration

def construct_event_dict(water_years):
    ''' Pulling together a dictionary with a key per year in the timeseries,
    and an empty list as each value, where events will be addded to
    '''
    all_events = {}
    water_years_unique = sorted(set(water_years))
    all_events = dict.fromkeys(water_years_unique)
    for k, _ in all_events.items():
        all_events[k] = []
        
    return all_events

def lowflow_calc(EWR_info, flows, water_years, climates):
    '''For calculating low flow ewrs. These have no consecutive requirement on their durations'''        
    # Decalre variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    for i, flow in enumerate(flows[:-1]): 
        event, all_events, no_event, all_no_events = lowflow_check(EWR_info, flow, event, all_events, no_event, all_no_events, water_years[i])
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(np.array(event))
            event = [] # Reset at the end of the water year
            no_event = 0 # Reset at the end of the water year
            durations.append(get_duration(climates[i], EWR_info))
            min_events.append(1)
    #--- Check final iteration: ---#
    event, all_events, no_event, all_no_events = lowflow_check(EWR_info, flows[-1], event, all_events, no_event, all_no_events, water_years[-1])
    if len(event) > 0:
        all_events[water_years[-1]].append(np.array(event))
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    durations.append(get_duration(climates[-1], EWR_info))
    min_events.append(1)
    return all_events, all_no_events, durations

def ctf_calc(EWR_info, flows, water_years, climates):
    '''For calculating cease to flow type ewrs'''
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    for i, flow in enumerate(flows[:-1]): 
        event, all_events, no_event, all_no_events = ctf_check(EWR_info, flow, event, all_events, no_event, all_no_events, water_years[i])                              
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(np.array(event))
            event = []
            durations.append(get_duration(climates[i], EWR_info))
    #--- Check final iteration ---#
    event, all_events, no_event, all_no_events = ctf_check(EWR_info, flows[-1], event, all_events, no_event, all_no_events, water_years[-1])   
    if len(event) > 0:
        all_events[water_years[-1]].append(np.array(event))
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    durations.append(get_duration(climates[-1], EWR_info))
    
    return all_events, all_no_events, durations

def flow_calc(EWR_info, flows, water_years):
    event = []
    total_event = 0
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    for i, flow in enumerate(flows[:-1]):
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, gap_track, water_years, total_event)
        if water_years[i] != water_years[i+1]:
            if len(event) >= EWR_info['min_event']:
                all_events[water_years[i]].append(np.array(event))
            event = []
            min_events.append(EWR_info['min_event'])
            durations.append(EWR_info['duration'])
            
    event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, gap_track, water_years, total_event)   
    if len(event) >= EWR_info['min_event']: 
        all_events[water_years[-1]].append(np.array(event))
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    min_events.append(EWR_info['min_event'])
    durations.append(EWR_info['duration'])
    
    return all_events, all_no_events, durations
    
def flow_calc_anytime(EWR_info, flows, water_years):
    ''''''
    event = []
    no_event = 0
    total_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    for i, flow in enumerate(flows[:-1]):
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, gap_track, water_years, total_event)  
        if water_years[i] != water_years[i+1]:
            min_events.append(EWR_info['min_event'])
            durations.append(EWR_info['duration'])
            
    event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, gap_track, water_years, total_event)
    if len(event) >= EWR_info['min_event']:
        water_year = which_water_year(-1, event, water_years)
        all_events[water_year].append(np.array(event))
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    min_events.append(EWR_info['min_event'])
    durations.append(EWR_info['duration'])   

    return all_events, all_no_events, durations

def lake_calc(EWR_info, levels, water_years):
    ''''''
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    for i, level in enumerate(levels[:-1]):
        if i == 0:
            level_change = 0
        else:
            level_change = levels[i-1]-levels[i]
        event, all_events, no_event, all_no_events = level_check(EWR_info, level, level_change, event, all_events, no_event, all_no_events, water_years[i])
        if water_years[i] != water_years[i+1]:
            if len(event) >= EWR_info['duration']:
                all_events[water_years[i]].append(np.array(event))
            event = []
            min_events.append(EWR_info['min_event'])
            durations.append(EWR_info['duration'])
    level_change = levels[-2]-levels[-1]        
    event, all_events, no_event, all_no_events = level_check(EWR_info, levels[-1], level_change, event, all_events, no_event, all_no_events, water_years[-1])
    if len(event) >= EWR_info['duration']:
        all_events[water_years[-1]].append(np.array(event))
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    min_events.append(EWR_info['min_event'])
    durations.append(EWR_info['duration'])
    
    return all_events, all_no_events, durations

def cumulative_calc(EWR_info, flows, water_years):
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    unique_water_years = sorted(set(water_years))
    skip_lines = 0
    for year in unique_water_years:
        mask = water_years == year
        year_flows = flows[mask]
        durations.append(EWR_info['duration'])
        for i, flow in enumerate(year_flows[:-EWR_info['duration']]):
            if skip_lines > 0:
                skip_lines -= 1
            else:
                subset_flows = year_flows[i:i+EWR_info['duration']]
                large_enough_flows = subset_flows[subset_flows >= EWR_info['min_flow']]
                if sum(large_enough_flows) >= EWR_info['min_volume']:
                    all_events[year].append(np.array(large_enough_flows))
                    skip_lines = EWR_info['duration']
                    if no_event > 0:
                        all_no_events[year].append(np.array(no_event))
                        no_event = 0
                else:
                    no_event += 1
        final_subset_flows = year_flows[-EWR_info['duration']:]
        final_large_enough_flows = final_subset_flows[final_subset_flows >= EWR_info['min_flow']]
        if sum(final_large_enough_flows) >= EWR_info['min_volume']:
            all_events[year].append(np.array(final_large_enough_flows))
            if no_event > 0:
                all_no_events[year].append(np.array(no_event))
                no_event = 0
        else:
            no_event = no_event + EWR_info['duration']
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    
    return all_events, all_no_events, durations

def which_water_year_start(iteration, event, water_years):
    '''Used to find which water year the majority of the event fell in'''
    event_wateryears = water_years[iteration:iteration+len(event)]
    midway_iteration = (len(event_wateryears))/2
    mid_event = event_wateryears[int(midway_iteration)]

    return mid_event

def which_water_year_end(iteration, event, water_years):
    '''Used to find which water year the majority of the event fell in'''
    event_wateryears = water_years[iteration:]
    midway_iteration = (len(event_wateryears))/2
    mid_event = event_wateryears[int(midway_iteration)]

    return mid_event

def cumulative_calc_anytime(EWR_info, flows, water_years):
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations = len(set(water_years))*[EWR_info['duration']]
    skip_lines = 0
    for i, flow in enumerate(flows[:-EWR_info['duration']]):
        if skip_lines > 0:
            skip_lines -= 1
        else:
            subset_flows = flows[i:i+EWR_info['duration']]
            large_enough_flows = subset_flows[subset_flows >= EWR_info['min_flow']]
            if sum(large_enough_flows) >= EWR_info['min_volume']:
                water_year = which_water_year_start(i, subset_flows, water_years)
                all_events[water_year].append(np.array(large_enough_flows))
                skip_lines = EWR_info['duration']
                if no_event > 0:
                    all_no_events[water_year].append(np.array(no_event))
                    no_event = 0
            else:
                no_event += 1
    final_subset_flows = flows[-EWR_info['duration']:]
    final_large_enough_flows = final_subset_flows[final_subset_flows >= EWR_info['min_flow']]
    if sum(final_large_enough_flows) >= EWR_info['min_volume']:
        water_year = which_water_year_end(-EWR_info['duration'], final_subset_flows, water_years)
        all_events[water_year].append(np.array(final_large_enough_flows))
        if no_event > 0:
            all_no_events[water_year].append(np.array(no_event))
            no_event = 0
    else:
        no_event = no_event + EWR_info['duration']
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))  
    return all_events, all_no_events, durations
    

def check_requirements(list_of_lists):
    '''Iterate through the lists, if there is a False in any, return False'''
    result = True
    for list_ in list_of_lists:
        if False in list_:
            result = False

    return result

def nest_calc_weirpool(EWR_info, flows, levels, water_years):
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    unique_water_years = sorted(set(water_years))
    skip_lines = 0
    for year in unique_water_years:
        mask = water_years == year
        year_flows = flows[mask]
        year_levels = levels[mask]
        durations.append(EWR_info['duration'])
        for i, flow in enumerate(year_flows[:-EWR_info['duration']]):
            if skip_lines > 0:
                skip_lines -= 1
            else:
                subset_flows = year_flows[i:i+EWR_info['duration']]
                subset_levels = levels[i:i+EWR_info['duration']]
                
                min_flow_check = subset_flows >= EWR_info['min_flow']
                max_flow_check = subset_flows <= EWR_info['max_flow']
                level_change = np.diff(subset_levels)
                level_change_check = level_change <= float(EWR_info['drawdown_rate'])

                checks_passed = check_requirements([min_flow_check, max_flow_check, level_change_check])

                if checks_passed:
                    all_events[year].append(np.array(subset_flows))
                    if no_event > 0:
                        all_no_events[year].append(no_event)
                    skip_lines = len(subset_flows)
                else:
                    no_event += 1
                    
        final_subset_flows = year_flows[-EWR_info['duration']:]
        final_subset_levels = year_levels[-EWR_info['duration']:]

        min_flow_check = subset_flows >= EWR_info['min_flow']
        max_flow_check = subset_flows <= EWR_info['max_flow']
        level_change = np.diff(subset_levels)
        level_change_check = level_change <= float(EWR_info['drawdown_rate'])

        checks_passed = check_requirements([min_flow_check, max_flow_check, level_change_check])

        if checks_passed:
            all_events[year].append(np.array(subset_flows))
            if no_event > 0:
                all_no_events[water_years[i]].append(np.array(no_event))
                no_event = 0
        else:
            # There won't be any events for the rest of the year, reflect this in the counter:
            no_event = no_event + EWR_info['duration']
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    
    return all_events, all_no_events, durations  
            
def nest_calc_percent(EWR_info, flows, water_years):
    ''''''
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    unique_water_years = sorted(set(water_years))
    skip_lines = 0
    drawdown_rate = int(EWR_info['drawdown_rate'][:-1])
    for year in unique_water_years:
        mask = water_years == year
        year_flows = flows[mask]
        durations.append(EWR_info['duration'])
        for i, flow in enumerate(year_flows[:-EWR_info['duration']]):
            if skip_lines > 0:
                skip_lines -= 1
            else:
                subset_flows = year_flows[i:i+EWR_info['duration']]
                
                min_flow_check = subset_flows >= EWR_info['min_flow']
                max_flow_check = subset_flows <= EWR_info['max_flow']
                flow_change = np.array(np.diff(subset_flows),dtype=float)
                divide_flows = subset_flows[:-1]
                difference = np.divide(flow_change, divide_flows, out=np.zeros_like(flow_change), where=divide_flows!=0)*100
                flow_change_check = difference <= drawdown_rate

                checks_passed = check_requirements([min_flow_check, max_flow_check, flow_change_check])

                if checks_passed:
                    all_events[year].append(np.array(subset_flows))
                    skip_lines = len(subset_flows)
                    
        final_subset_flows = year_flows[-EWR_info['duration']:]

        min_flow_check = final_subset_flows >= EWR_info['min_flow']
        max_flow_check = final_subset_flows <= EWR_info['max_flow']
        flow_change = np.array(np.diff(final_subset_flows),dtype=float)
        divide_flows = final_subset_flows[:-1]
        difference = np.divide(flow_change, divide_flows, out=np.zeros_like(flow_change), where=divide_flows!=0)*100
        flow_change_check = difference <= drawdown_rate

        checks_passed = check_requirements([min_flow_check, max_flow_check, flow_change_check])

        if checks_passed:
            all_events[year].append(np.array(final_subset_flows))
            if no_event > 0:
                all_no_events[water_years[i]].append(np.array(no_event))
                no_event = 0
        else:
            # There won't be any events for the rest of the year, reflect this in the counter:
            no_event = no_event + EWR_info['duration']
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    
    return all_events, all_no_events, durations

def nest_calc_percent_trigger(EWR_info, flows, water_years, dates):
    ''''''
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    durations = len(set(water_years))*[EWR_info['duration']]
    skip_lines = 0
    drawdown_rate = int(EWR_info['drawdown_rate'][:-1])
    days = dates.day.values
    months = dates.month.values    
    for i, flow in enumerate(flows[:-EWR_info['duration']]):
        if skip_lines > 0:
            skip_lines -= 1
        else:
            trigger_day = days[i] == EWR_info['trigger_day']
            trigger_month = months[i] == EWR_info['trigger_month']
            
            if trigger_day and trigger_month:
                subset_flows = flows[i:i+EWR_info['duration']]
                
                min_flow_check = subset_flows >= EWR_info['min_flow']
                max_flow_check = subset_flows <= EWR_info['max_flow']
                flow_change = np.array(np.diff(subset_flows),dtype=float)
                divide_flows = subset_flows[:-1]
                difference = np.divide(flow_change, divide_flows, out=np.zeros_like(flow_change), where=divide_flows!=0)*100
                flow_change_check = difference <= drawdown_rate

                checks_passed = check_requirements([min_flow_check, max_flow_check, flow_change_check])

                if checks_passed:
                    all_events[water_years[i]].append(np.array(subset_flows))
                    skip_lines = len(subset_flows)
                    if no_event > 0:
                        all_no_events[water_years[i]].append(np.array(no_event))
                        no_event = 0
                else:
                    no_event = no_event + 1
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    return all_events, all_no_events, durations
       
def weirpool_calc(EWR_info, flows, levels, water_years, weirpool_type):
    ''''''
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    skip_lines = 0
    unique_water_years = sorted(set(water_years))
    for year in unique_water_years:
        mask = water_years == year
        year_flows = flows[mask]
        year_levels = levels[mask]
        durations.append(EWR_info['duration'])
        for i, flow in enumerate(year_flows[:-EWR_info['duration']]):
            subset_flows = year_flows[i:i+EWR_info['duration']]
            subset_levels = year_levels[i:i+EWR_info['duration']]
            # Check achievements:
            min_flow_check = subset_flows >= EWR_info['min_flow']
            max_flow_check = subset_flows <= EWR_info['max_flow']
            
            levels_change = np.array(np.diff(subset_levels),dtype=float)
            divide_levels = subset_levels[:-1]
            difference = np.divide(levels_change, divide_levels, out=np.zeros_like(levels_change), where=divide_levels!=0)*100
            level_change_check = difference <= float(EWR_info['drawdown_rate'])
            if weirpool_type == 'raising':
                check_levels = subset_levels >= EWR_info['min_level']
            elif weirpool_type == 'falling':
                check_levels = subset_levels <= EWR_info['max_level']
            checks_passed = check_requirements([min_flow_check, max_flow_check, 
                                                level_change_check, check_levels])
            if checks_passed:
                all_events[year].append(np.array(subset_flows))
                skip_lines = len(subset_flows)
                if no_event > 0:
                    all_no_events[year].append(np.array(no_event))
                    no_event = 0
            else:
                no_event += 1
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    return all_events, all_no_events, durations  

def flow_calc_anytime_sim(EWR_info1, EWR_info2, flows1, flows2, water_years):
    ''''''
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    for i, flow in enumerate(flows1[:-1]):
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                               EWR_info2, water_years,
                                                                               flow, flows2[i], event,
                                                                               all_events, no_event,
                                                                               all_no_events,gap_track,
                                                                                           total_event)
        if water_years[i] != water_years[i+1]:
            min_events.append(EWR_info1['min_event'])
            durations.append(EWR_info1['duration'])
    event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                           EWR_info2, water_years,
                                                                           flows1[-1], flows2[-1], event,
                                                                           all_events, no_event,
                                                                           all_no_events,gap_track,
                                                                          total_event)           
    if len(event) >= EWR_info1['min_event']:
        water_year = which_water_year(-1, event, water_years)
        all_events[water_year].append(np.array(event))
    min_events.append(EWR_info1['min_event'])
    durations.append(EWR_info1['duration'])   
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    return all_events, all_no_events, durations
    
def flow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years):
    ''''''
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    lines_to_skip = 0
    for i, flow in enumerate(flows1[:-1]):
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,EWR_info2,
                                                                               water_years, flow,
                                                                               flows2[i],event,
                                                                               all_events, no_event,
                                                                               all_no_events, gap_track,
                                                                                           total_event)  
        if water_years[i] != water_years[i+1]:
            if len(event) >= EWR_info1['min_event']: 
                all_events[water_years[i]].append(np.array(event))
            event = []
            min_events.append(EWR_info1['min_event'])
            durations.append(EWR_info1['duration'])
    event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                           EWR_info2, water_years,
                                                                           flows1[-1], flows2[-1], event,
                                                                           all_events, no_event,
                                                                           all_no_events,gap_track,
                                                                                       total_event)            
    if len(event) >= EWR_info1['min_event']: 
        all_events[water_years[-1]].append(np.array(event))
    min_events.append(EWR_info1['min_event'])
    durations.append(EWR_info1['duration'])
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))  
    return all_events, all_no_events, durations    

def lowflow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates):
    '''For calculating low flow ewrs. These have no consecutive requirement on their durations'''        
    # Decalre variables:
    event1, event2 = [], []
    no_event1, no_event2 = 0, 0
    all_events1 = construct_event_dict(water_years)
    all_events2 = construct_event_dict(water_years)
    all_no_events1 = construct_event_dict(water_years)
    all_no_events2 = construct_event_dict(water_years)
    durations, min_events = [], []
    for i, flow in enumerate(flows1[:-1]):
        event1, all_events1, no_event1, all_no_events1 = lowflow_check(EWR_info1, flow, event1, all_events1, no_event1, all_no_events1, water_years[i])
        event2, all_events2, no_event2, all_no_events2 = lowflow_check(EWR_info2, flows2[i], event2, all_events2, no_event2, all_no_events2, water_years[i])
        if water_years[i] != water_years[i+1]:
            if len(event1) > 0:
                all_events1[water_years[i]].append(np.array(event1))
            if len(event2) > 0:
                all_events2[water_years[i]].append(np.array(event2))
            event1, event2 = [], []
            no_event1, no_event2 = 0, 0
            durations.append(get_duration(climates[i], EWR_info1))
            min_events.append(1)
            
    event1, all_events1, no_event1, all_no_events1 = lowflow_check(EWR_info1, flows1[-1], event1, all_events1, no_event1, all_no_events1, water_years[-1])
    event2, all_events2, no_event2, all_no_events2 = lowflow_check(EWR_info2, flows2[-1], event2, all_events2, no_event2, all_no_events2, water_years[-1])
    if len(event1) > 0:
        all_events1[water_years[-1]].append(np.array(event1))
    if len(event2) > 0:
        all_events2[water_years[-1]].append(np.array(event2))    
    durations.append(get_duration(climates[-1], EWR_info1))
    min_events.append(1)
    if no_event1 > 0:
        all_no_events1[water_years[-1]].append(np.array(no_event1))
    if no_event2 > 0:
        all_no_events2[water_years[-1]].append(np.array(no_event2))
    return all_events1, all_events2, all_no_events1, all_no_events2, durations

def ctf_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates):
    '''For calculating cease to flow type ewrs'''
    # Decalre variables:
    event1, event2 = [], []
    no_event1, no_event2 = 0, 0
    all_events1 = construct_event_dict(water_years)
    all_events2 = construct_event_dict(water_years)
    all_no_events1 = construct_event_dict(water_years)
    all_no_events2 = construct_event_dict(water_years)
    durations, min_events = [], []
    for i, flow in enumerate(flows1[:-1]): 
        event1, all_events1, no_event1, all_no_events1 = ctf_check(EWR_info1, flow, event1, all_events1, no_event1, all_no_events1, water_years[i])
        event2, all_events2, no_event2, all_no_events2 = ctf_check(EWR_info2, flows2[i], event2, all_events2, no_event2, all_no_events2, water_years[i])
        if water_years[i] != water_years[i+1]:
            if len(event1) > 0:
                all_events1[water_years[i]].append(np.array(event1))
            if len(event2) > 0:
                all_events2[water_years[i]].append(np.array(event2))
            event1, event2 = [], [] 
            # Spot for adding the min event if relevant
            if no_event1 > 0:
                all_no_events1[water_years[i]].append(np.array(no_event1))
            if no_event2 > 0:
                all_no_events2[water_years[i]].append(np.array(no_event2))
            durations.append(get_duration(climates[i], EWR_info1))
            min_events.append(1)
    #--- Check final iteration ---#
    event1, all_events1, no_event1, all_no_events1 = ctf_check(EWR_info1, flows1[-1], event1, all_events1, no_event1, all_no_events1, water_years[-1])
    event2, all_events2, no_event2, all_no_events2 = ctf_check(EWR_info2, flows2[-1], event2, all_events2, no_event2, all_no_events2, water_years[-1])  
    if len(event1) > 0:
        all_events1[water_years[-1]].append(np.array(event1))
    if len(event2) > 0:
        all_events2[water_years[-1]].append(np.array(event2))
    durations.append(get_duration(climates[-1], EWR_info1))
    # Spot for adding the min event if relevant
    if no_event1 > 0:
        all_no_events1[water_years[-1]].append(np.array(no_event1))
    if no_event2 > 0:
        all_no_events2[water_years[-1]].append(np.array(no_event2))
    return all_events1, all_events2, all_no_events1, all_no_events2, durations

def check_trigger(iteration, min_flow, max_flow, gap_tolerance, min_event, water_years, flow,
                  event, gap_track, trigger):
    '''Checks daily flow against ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list, event dictionary, and time between events
    '''

    if ((flow >= min_flow) and (flow <= max_flow)):
        event.append(flow)
        gap_track = gap_tolerance # reset the gapTolerance after threshold is reached
        if len(event) >= min_event:
            trigger = True
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            event.append(flow)
        else:
            gap_track = -1
            event = []
            trigger = False

    return event, gap_track, trigger

def flow_calc_post_req(EWR_info, flows, water_years):
    ''''''
    trigger, post_trigger = False, False
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    skip_lines = 0
    for i, flow in enumerate(flows):
        if skip_lines > 0:
            skip_lines -= 1
        else:
            no_event += 1
            if gap_track == -1:
                trigger, post_trigger = False, False
            if ((trigger == False) and (post_trigger == False)):
                event, gap_track, trigger = check_trigger(i, EWR_info['min_flow'], EWR_info['max_flow'], EWR_info['gap_tolerance'], EWR_info['min_event'], water_years, flow, event, gap_track, trigger)
            elif ((trigger == True) and (post_trigger == False)):
                # Send to check the post requirement
                event, gap_track, post_trigger = check_trigger(i, EWR_info['min_flow_post'], EWR_info['max_flow_post'], EWR_info['gap_tolerance'], EWR_info['duration_post'], water_years, flow, event, gap_track, post_trigger)
            elif ((trigger == True) and (post_trigger == True)):
                water_year = which_water_year(iteration, event, water_years)
                all_events[water_year].append(np.array(event))
                all_no_events[water_year].append(np.array(no_event-len(event)))
                trigger1, trigger2 = False, False
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    return all_events, all_no_events, durations

def flow_calc_outside_req(EWR_info, flows, water_years):
    trigger, pre_trigger, post_trigger = False, False, False
    event, pre_event, post_event = [], [], []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    skip_lines = 0
    gap_track = 0
    for i, flow in enumerate(flows):
        if skip_lines > 0:
            skip_lines -= 1
        else:
            no_event += 1
            if gap_track == -1:
                trigger = False
            if trigger == False:
                event, gap_track, trigger = check_trigger(i, EWR_info['min_flow'], EWR_info['max_flow'], EWR_info['gap_tolerance'], EWR_info['min_event'], water_years, flow, event, gap_track, trigger)
            elif trigger == True: # Event registered, now check for event pre/post this
                gap_track = EWR_info['gap_tolerance']
                # First check if there was an event before the main event:
                for pre_i, pre_flow in enumerate(reversed(flows[:(i-len(event)+1)])):  
                    event, gap_track, pre_trigger = check_trigger(i, EWR_info['min_flow_outside'], EWR_info['max_flow_outside'], EWR_info['gap_tolerance'], EWR_info['duration_outside'], water_years, flow, event, gap_track, pre_trigger)
                    if pre_gap_track == -1: # If the pre event gap tolerance is exceeded, break
                        pre_trigger = False
                        break
                    if pre_trigger == True:
                        event = list(reversed(pre_event)) + event
                        water_year = which_water_year(iteration, event, water_years)
                        all_events[water_year].append(np.array(event))
                        all_no_events[water_year].append(np.array(no_event-len(event)))
                # If the above fails, enter sub routine to check for an event after:
                if pre_trigger == False:
                    gap_track = EWR_info['gap_tolerance']
                    for post_i, post_flow in enumerate(flows[i:]):
                        post_event, gap_track, post_trigger = check_trigger(i, EWR_info['min_flow_outside'], EWR_info['max_flow_outside'], EWR_info['gap_tolerance'], EWR_info['duration_outside'], water_years, flow, event, gap_track, post_trigger)
                        if post_gap_track == -1:
                            trigger, pre_trigger, post_trigger = False, False, False
                            break
                        if trigger_after == True:
                            water_year = which_water_year(iteration, event, water_years) #Check loc
                            all_no_events[water_year].append(np.array(no_event-len(event)))
                            event = event + post_event
                            all_events[water_year].append(np.array(event))
                            skip_lines = len(post_event)
                    if pre_trigger == False and post_trigger == False:
                        trigger, pre_trigger, post_trigger =  False, False, False
    if no_event > 0:
        all_no_events[water_years[-1]].append(np.array(no_event))
    return all_events, all_no_events, durations

# Stats on EWR events--------------------------------------------------------------

def get_event_years(EWR_info, events, water_years, durations):
    ''''''
    event_years = []
    for year in water_years:
        combined_len = 0
        for i in events[year]:
            combined_len += i.size
        if ((combined_len >= EWR_info['duration'] and len(events[year])>=EWR_info['events_per_year'])):
            event_years.append(1)
        else:
            event_years.append(0)
    
    return event_years
    
def number_events(EWR_info, events, water_years):
    ''''''
    num_events = []
    for year in water_years:
        combined_len = 0
        yearly_events = 0
        for i in events[year]:
            combined_len += i.size
            if combined_len >= EWR_info['duration']:
                yearly_events += 1
                combined_len = 0
        total = yearly_events/EWR_info['events_per_year']
        num_events.append(int(total))
    
    return num_events

def get_average_event_length(events, water_years):
    ''''''
    av_length = list()
    for year in water_years:
        count = len(events[year])
        if count > 0:
            joined = np.concatenate(events[year])
            length = len(joined)
            av_length.append(length/count)
        else:
            av_length.append(0)
            
    return av_length

def get_total_days(events, water_years):
    # total days
    total_days = list()
    for year in water_years:
        count = len(events[year])
        if count > 0:
            joined = np.concatenate(events[year])
            length = len(joined)
            total_days.append(length)
        else:
            total_days.append(0)
            
    return total_days

def get_days_between(events, EWR, EWR_info, unique_water_years, water_years):
    ''''''
    LOWFLOW_EWR = 'VF' in EWR or 'BF' in EWR
    YEARLY_INTEREVENT = EWR_info['max_inter-event'] >= 1
    if EWR_info['max_inter-event'] == None:
        # If there is no max interevent period defined in the EWR, return all interevent periods:
        return list(events.values())
    else:
        max_interevent = data_inputs.convert_max_interevent(unique_water_years, water_years, EWR_info)
        if LOWFLOW_EWR and YEARLY_INTEREVENT:
            # Need to sum the events together before calcuating:
            temp = {}
            for year in events:
                total_yearly = np.array([sum(events[year])])
                mask = total_yearly >= max_interevent
                temp[year] = np.array(total_yearly)[mask]
            return list(temp.values())
        else: 
            temp = {}
            for year in events:
                mask = np.array(events[year]) >= max_interevent
                temp[year] = np.array(events[year])[mask]
            return list(temp.values())            
        

def get_data_gap(input_df, water_years, gauge):
    '''Input a dataframe, 
    calculate how much missing data there is, 
    send yearly results back
    '''
    temp_df = input_df.copy(deep=True)
    masked = ~temp_df.notna()
    masked['water year'] = water_years
    group_df = masked.groupby('water year').sum()
    
    return group_df[gauge]

def get_total_series_days(water_years):
    '''Input a series of missing days and a possible maximum days,
    returns the percentage of data available for each year
    '''
    unique, counts = np.unique(water_years, return_counts=True)
    intoSeries = pd.Series(index=unique, data=counts)
    
    return intoSeries

def event_years_sim(events1, events2):
    '''add the event lists, event only occurs when both are met'''
    added = np.array(list(events1)) + np.array(list(events2))
    mask = added == 2
    results = mask*1
    return results

def number_events_sim(events1, events2):
    e1 = np.array(list(events1))
    e2 = np.array(list(events2))
    results = []
    for i, event in enumerate(e1):
        results.append(min([event, e2[i]]))
    return results

def average_event_length_sim(events1, events2):
    e1 = np.array([list(events1)])
    e2 = np.array([list(events2)])
    average = (e1 + e2)/2
    return average[0]

def event_stats(df, PU_df, gauge, EWR, EWR_info, events, no_events, durations, water_years):
    ''' Depending on the requests, produces statistics for each'''
    unique_water_years = set(water_years)
    # Years with events
    years_with_events = get_event_years(EWR_info, events, unique_water_years, durations)
    YWE = pd.Series(name = str(EWR + '_eventYears'), data = years_with_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, YWE], axis = 1)
    # Number of events
    num_events = number_events(EWR_info, events, unique_water_years)
    NE = pd.Series(name = str(EWR + '_numEvents'), data= num_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, NE], axis = 1)
    # Average length of events
    av_length = get_average_event_length(events, unique_water_years)
    AL = pd.Series(name = str(EWR + '_eventLength'), data = av_length, index = unique_water_years)
    PU_df = pd.concat([PU_df, AL], axis = 1)
    # Total event days
    total_days = get_total_days(events, unique_water_years)
    TD = pd.Series(name = str(EWR + '_totalEventDays'), data = total_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, TD], axis = 1)
    # Days between events
    days_between = get_days_between(no_events, EWR, EWR_info, unique_water_years, water_years)
    DB = pd.Series(name = str(EWR + '_daysBetweenEvents'), data = days_between, index = unique_water_years)
    PU_df = pd.concat([PU_df, DB], axis = 1)
    # Append information around available and missing data:
    yearly_gap = get_data_gap(df, water_years, gauge)
    total_days = get_total_series_days(water_years)
    YG = pd.Series(name = str(EWR + '_missingDays'), data = yearly_gap, index = unique_water_years)
    TD = pd.Series(name = str(EWR + '_totalPossibleDays'), data = total_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, YG], axis = 1)
    PU_df = pd.concat([PU_df, TD], axis = 1)
    
    return PU_df

def event_stats_sim(df, PU_df, gauge1, gauge2, EWR, EWR_info, events1, events2, no_events1, no_events2, durations, water_years):
    ''' Depending on the requests, produces statistics for each'''
    unique_water_years = set(water_years)
    # Years with events
    years_with_events1 = get_event_years(EWR_info, events1, unique_water_years, durations)
    years_with_events2 = get_event_years(EWR_info, events2, unique_water_years, durations)
    years_with_events = pd.Series(event_years_sim(years_with_events1, years_with_events2))
    YWE = pd.Series(name = str(EWR + '_eventYears'), data = years_with_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, YWE], axis = 1)
    # Number of events
    num_events1 = number_events(EWR_info, events1, unique_water_years)
    num_events2 = number_events(EWR_info, events2, unique_water_years)
    num_events = pd.Series(number_events_sim(num_events1, num_events2))
    NE = pd.Series(name = str(EWR + '_numEvents'), data= num_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, NE], axis = 1)
    # Average length of events
    av_length1 = get_average_event_length(events1, unique_water_years)
    av_length2 = get_average_event_length(events2, unique_water_years)  
    av_length = pd.Series(average_event_length_sim(av_length1, av_length2))
    AL = pd.Series(name = str(EWR + '_eventLength'), data = av_length, index = unique_water_years)
    PU_df = pd.concat([PU_df, AL], axis = 1)
    # Total event days
    total_days1 = get_total_days(events1, unique_water_years)
    total_days2 = get_total_days(events2, unique_water_years)
    av_total_days = pd.Series(average_event_length_sim(total_days1, total_days2))
    TD = pd.Series(name = str(EWR + '_totalEventDays'), data = av_total_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, TD], axis = 1)
    # Days between events
    days_between1 = pd.Series(get_days_between(no_events1, EWR_info, unique_water_years, water_years))
#     days_between2 = pd.Series(get_days_between(no_events2, unique_water_years, water_years)
    DB = pd.Series(name = str(EWR + '_daysBetweenEvents'), data = days_between1, index = unique_water_years)
    PU_df = pd.concat([PU_df, DB], axis = 1) # Only adding the main gauge
    # Append information around available and missing data:
    yearly_gap = get_data_gap(df, water_years, gauge1) # Only adding data gap for main gauge
    total_days = get_total_series_days(water_years)
    YG = pd.Series(name = str(EWR + '_missingDays'), data = yearly_gap, index = unique_water_years)
    TD = pd.Series(name = str(EWR + '_totalPossibleDays'), data = total_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, YG], axis = 1)
    PU_df = pd.concat([PU_df, TD], axis = 1)
    
    return PU_df

# Sorting and distributing to calculation functions -------------------------------------------------------

def calc_sorter(df_F, df_L, gauge, allowance, climate):
    '''Sends to handling functions to get calculated depending on the type of EWR''' 
    # Get ewr tables:
    EWR_table, see_notes_ewrs, undefined_ewrs, noThresh_df, no_duration, DSF_ewrs = data_inputs.get_EWR_table()
    menindee_gauges, wp_gauges = data_inputs.getLevelGauges()
    multi_gauges = data_inputs.getMultiGauges('all')
    simultaneous_gauges = data_inputs.getSimultaneousGauges('all')
    complex_EWRs = data_inputs.getComplexCalcs()
    # Extract relevant sections of the EWR table:
    gauge_table = EWR_table[EWR_table['gauge'] == gauge]
    # save the planning unit dataframes to this dictionary:
    location_results = {}
    for PU in set(gauge_table['PlanningUnitID']):
        PU_table = gauge_table[gauge_table['PlanningUnitID'] == PU]
        EWR_categories = PU_table['flow level volume'].values
        EWR_codes = PU_table['code']
        PU_df = pd.DataFrame()
        for i, EWR in enumerate(tqdm(EWR_codes, position = 0, leave = False,
                                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                                     desc= str('Evaluating ewrs for '+ gauge))):
            # Get the EWRs with the very dry year tag to exclude
            VERYDRY = '_VD' in EWR
            # Get the overarching EWR category (flow/cumulative flow/level)
            CAT_FLOW = EWR_categories[i] == 'F'
            CAT_CUMUL = EWR_categories[i] == 'V'
            CAT_LEVEL = EWR_categories[i] == 'L'
            # Get the specific type of EWR:
            EWR_CTF = 'CF' in EWR
            EWR_LOWFLOW = 'BF' in EWR or 'VF' in EWR
            EWR_FLOW = 'SF' in EWR or 'LF' in EWR or 'BK' in EWR or 'OB' in EWR or 'AC' in EWR
            EWR_WP = 'WP' in EWR
            EWR_NEST = 'Nest' in EWR
            EWR_CUMUL = 'LF' in EWR or 'OB' in EWR or 'WL' in EWR # Some LF and OB are cumulative
            EWR_LEVEL = 'LLLF' in EWR or 'MLLF' in EWR or 'HLLF' in EWR or 'VHLL' in EWR
            # Determine if its classified as a complex EWR:
            COMPLEX = gauge in complex_EWRs and EWR in complex_EWRs[gauge]
            MULTIGAUGE = PU in multi_gauges and gauge in multi_gauges[PU]
            SIMULTANEOUS = PU in simultaneous_gauges and gauge in simultaneous_gauges[PU]
            if CAT_FLOW and EWR_CTF and not VERYDRY:
                if MULTIGAUGE:
                    PU_df = ctf_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
                elif SIMULTANEOUS:
                    PU_df = ctf_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
                else:
                    PU_df = ctf_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
            elif CAT_FLOW and EWR_LOWFLOW and not VERYDRY:
                if MULTIGAUGE:
                    PU_df = lowflow_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
                elif SIMULTANEOUS:
                    PU_df = lowflow_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
                else:
                    PU_df = lowflow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
            elif CAT_FLOW and EWR_FLOW and not VERYDRY:
                if COMPLEX:
                    PU_df = complex_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
                elif MULTIGAUGE:
                    PU_df = flow_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
                elif SIMULTANEOUS:
                    PU_df = flow_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
                else:
                    PU_df = flow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
            elif CAT_FLOW and EWR_WP and not VERYDRY:
                PU_df = weirpool_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance)
            elif CAT_FLOW and EWR_NEST and not VERYDRY:
                PU_df = nest_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance)
            elif CAT_CUMUL and EWR_CUMUL and not VERYDRY:
                if MULTIGAUGE:
                    PU_df = cumulative_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
                else:
                    PU_df = cumulative_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
            elif CAT_LEVEL and EWR_LEVEL and not VERYDRY:
                PU_df = level_handle(PU, gauge, EWR, EWR_table, df_L, PU_df, allowance)
            else:
                continue

        location_results[PU] = PU_df
        
    return location_results