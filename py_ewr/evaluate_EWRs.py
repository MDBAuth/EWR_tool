import pandas as pd
import re
import numpy as np
from datetime import date, timedelta
from datetime import time

from tqdm import tqdm

from . import data_inputs

#----------------------------------- Getting EWRs from the database ------------------------------#

def component_pull(EWR_table, gauge, PU, EWR, component):
    '''Pass EWR details (planning unit, gauge, EWR, and EWR component) and the EWR table, 
    this function will then pull the component from the table
    '''
    component = list(EWR_table[((EWR_table['gauge'] == gauge) & 
                           (EWR_table['code'] == EWR) &
                           (EWR_table['PlanningUnitID'] == PU)
                          )][component])[0]
    return component

def apply_correction(info, correction):
    '''Applies a correction to the EWR component (based on user request)'''
    return info*correction
    
def get_EWRs(PU, gauge, EWR, EWR_table, allowance, components):
    '''Pulls the relevant EWR componenets for each EWR, and applies any relevant corrections'''
    ewrs = {}
    # Save identifying information to dictionary:
    ewrs['gauge'] = gauge
    ewrs['planning_unit'] = PU
    ewrs['EWR_code'] = EWR
    
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
        min_flow = int(component_pull(EWR_table, gauge, PU, EWR, 'flow threshold min'))
        corrected = apply_correction(min_flow, allowance['minThreshold'])
        ewrs['min_flow'] = int(corrected)
    if 'MAXF' in components:
        max_flow = int(component_pull(EWR_table, gauge, PU, EWR, 'flow threshold max'))
        corrected = apply_correction(max_flow, allowance['maxThreshold'])
        ewrs['max_flow'] = int(corrected)
    if 'MINL' in components:
        min_level = float(component_pull(EWR_table, gauge, PU, EWR, 'level threshold min'))
        corrected = apply_correction(min_level, allowance['minThreshold'])
        ewrs['min_level'] = corrected
    if 'MAXL' in components:
        max_level = float(component_pull(EWR_table, gauge, PU, EWR, 'level threshold max'))
        corrected = apply_correction(max_level, allowance['maxThreshold'])
        ewrs['max_level'] = corrected
    if 'MINV' in components:
        min_volume = int(component_pull(EWR_table, gauge, PU, EWR, 'volume threshold'))
        corrected = apply_correction(min_volume, allowance['minThreshold'])
        ewrs['min_volume'] = int(corrected)
    if 'DUR' in components:
        duration = int(component_pull(EWR_table, gauge, PU, EWR, 'duration'))
        corrected = apply_correction(duration, allowance['duration'])
        ewrs['duration'] = int(corrected)
    if 'GP' in components:
        gap_tolerance = int(component_pull(EWR_table, gauge, PU, EWR, 'within event gap tolerance'))
        ewrs['gap_tolerance'] = gap_tolerance
    if 'EPY' in components:
        events_per_year = int(component_pull(EWR_table, gauge, PU, EWR, 'events per year'))
        ewrs['events_per_year'] = events_per_year       
    if 'ME' in components:
        min_event = int(component_pull(EWR_table, gauge, PU, EWR, 'min event'))
        if min_event != 1:
            corrected = apply_correction(min_event, allowance['duration'])
        else:
            corrected = min_event
        ewrs['min_event'] = int(corrected)
    if 'MD' in components:
        try: # There may not be a recommended drawdown rate
            max_drawdown = component_pull(EWR_table, gauge, PU, EWR, 'drawdown rate')
            if '%' in str(max_drawdown):
                value_only = int(max_drawdown.replace('%', ''))
                corrected = apply_correction(value_only, allowance['drawdown'])
                ewrs['drawdown_rate'] = str(int(corrected))+'%'
            else:
                corrected = apply_correction(float(max_drawdown), allowance['drawdown'])
                ewrs['drawdown_rate'] = str(corrected/100)
        except ValueError: # In this case set a large number
            ewrs['drawdown_rate'] = str(1000000)          
    if 'DURVD' in components:
        try: # There may not be a very dry duration available for this EWR
            EWR_VD = str(EWR + '_VD')
            duration_vd = int(component_pull(EWR_table, gauge, PU, EWR_VD, 'duration'))
            corrected = apply_correction(duration_vd, allowance['duration'])
            ewrs['duration_VD'] =int(corrected)
        except IndexError: # In this case return None type for this component
            ewrs['duration_VD'] = None
    if 'WPG' in components:
        weirpool_gauge = component_pull(EWR_table, gauge, PU, EWR, 'weirpool gauge')
        ewrs['weirpool_gauge'] =str(weirpool_gauge)
    if 'MG' in components:
        multiGaugeDict = data_inputs.get_multi_gauges('all')
        ewrs['second_gauge'] = multiGaugeDict[PU][gauge]        
    if 'SG' in components:
        simultaneousGaugeDict = data_inputs.get_simultaneous_gauges('all')
        ewrs['second_gauge'] = simultaneousGaugeDict[PU][gauge]
    if 'TF' in components:
        try:
            ewrs['frequency'] = component_pull(EWR_table, gauge, PU, EWR, 'frequency')
            
        except IndexError:
            ewrs['frequency'] = None
    if 'MIE' in components:
        try:
            ewrs['max_inter-event'] = float(component_pull(EWR_table, gauge, PU, EWR, 'max inter-event'))
        except IndexError:
            ewrs['max_inter-event'] = None

    return ewrs

#------------------------ Masking timeseries data to dates in EWR requirement --------------------#

def mask_dates(EWR_info, input_df):
    '''Distributes flow/level dataframe to functions for masking over dates'''
    if EWR_info['start_day'] == None or EWR_info['end_day'] == None:
        # A month mask is required here as there are no day requirements:
        input_df_timeslice = get_month_mask(EWR_info['start_month'],
                                            EWR_info['end_month'],
                                            input_df)       
    else:
        # this function masks to the day:
        input_df_timeslice = get_day_mask(EWR_info['start_day'],
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
    
    return set(input_df_timeslice.index)


def get_day_mask(startDay, endDay, startMonth, endMonth, input_df):
    ''' for the ewrs with a day and month requirement, takes in a start day, start month, 
    end day, end month, and dataframe,
    masks the dataframe to these dates'''

    if startMonth > endMonth:
        month_mask = (((input_df.index.month >= startMonth) & (input_df.index.day >= startDay)) |\
                      ((input_df.index.month <= endMonth) & (input_df.index.day <= endDay)))
        input_df_timeslice = input_df.loc[month_mask]
        
    elif startMonth < endMonth:
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
    
    elif startMonth == endMonth:
        month_mask = ((input_df.index.month == startMonth) & (input_df.index.day >= startDay) & (input_df.index.day <= endDay))
        input_df_timeslice = input_df.loc[month_mask]
    
    return set(input_df_timeslice.index)

#---------------------------- Creating a daily time series with water years ----------------------#

def wateryear_daily(input_df, ewrs):
    '''Creating a daily time series with water years'''

    years = input_df.index.year.values
    months = input_df.index.month.values

    def appenderStandard(year, month):
        '''Handles standard water years'''
        if month < 7:
            year = year - 1
        return year
    
    def appenderNonStandard(year, month):
        '''Handles non-standard water years'''
        if month < ewrs['start_month']:
            year = year - 1
        return year
    
    if ((ewrs['start_month'] <= 6) and (ewrs['end_month'] >= 7)):
        waterYears = np.vectorize(appenderNonStandard)(years, months)
    else:     
        waterYears = np.vectorize(appenderStandard)(years, months)    
    
    return waterYears

#----------------------------------- EWR handling functions --------------------------------------#

def ctf_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    '''For handling Cease to flow type EWRs'''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('cease to flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years and climate categorisation for this catchment
    water_years = wateryear_daily(df_F, EWR_info)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(water_years, catchment, climate)
    # Check flow data against EWR requirements and then perform analysis on the results:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D, ME = ctf_calc_anytime(EWR_info, df_F[gauge].values, water_years, climates)
    else:
        E, NE, D, ME = ctf_calc(EWR_info, df_F[gauge].values, water_years, climates, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])

def lowflow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    '''For handling low flow type EWRs (Very low flows and baseflows)'''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('low flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years and climate categorisation for this catchment
    water_years = wateryear_daily(df_F, EWR_info)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(water_years, catchment, climate)
    # Check flow data against EWR requirements and then perform analysis on the results:
    E, NE, D, ME = lowflow_calc(EWR_info, df_F[gauge].values, water_years, climates, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])

def flow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    '''For handling non low flow based flow EWRs (freshes, bankfulls, overbanks)'''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow data against EWR requirements and then perform analysis on the results:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D, ME = flow_calc_anytime(EWR_info, df_F[gauge].values, water_years)
    else:
        E, NE, D, ME = flow_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])

def cumulative_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    '''For handling cumulative flow EWRs (some large freshes and overbanks, wetland flows)'''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('cumulative')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow data against EWR requirements and then perform analysis on the results:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D, ME = cumulative_calc_anytime(EWR_info, df_F[gauge].values, water_years)
    else:
        E, NE, D, ME = cumulative_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)

    return PU_df, tuple([E])

def level_handle(PU, gauge, EWR, EWR_table, df_L, PU_df, allowance):
    '''For handling level type EWRs (low, mid, high and very high level lake fills)'''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('level')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_L) 
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_L, EWR_info)  
    # Check flow data against EWR requirements and then perform analysis on the results: 
    E, NE, D, ME = lake_calc(EWR_info, df_L[gauge].values, water_years, df_L.index, masked_dates)
    PU_df = event_stats(df_L, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])

def weirpool_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance):
    '''For handling weirpool type EWRs'''
    # Get information about EWR (changes depending on the weirpool type):
    weirpool_type = data_inputs.weirpool_type(EWR)
    if weirpool_type == 'raising':
        pull = data_inputs.get_EWR_components('weirpool-raising')
    elif weirpool_type == 'falling':
        pull = data_inputs.get_EWR_components('weirpool-falling')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates for both the flow and level dataframes:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years:
    water_years = wateryear_daily(df_F, EWR_info)
    # If there is no level data loaded in, let user know and skip the analysis
    try:
        levels = df_L[EWR_info['weirpool_gauge']].values
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['weirpool_gauge']))
        return PU_df, None
    # Check flow and level data against EWR requirements and then perform analysis on the results: 
    E, NE, D, ME = weirpool_calc(EWR_info, df_F[gauge].values, levels, water_years, weirpool_type, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])

def nest_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance):
    '''For handling nest style EWRs'''
    # Get information about EWR (changes depending on if theres a weirpool level gauge in the EWR)
    requires_weirpool_gauge = gauge in ['414203', '425010', '4260505', '4260507', '4260509']
    if requires_weirpool_gauge:
        pull = data_inputs.get_EWR_components('nest-level')
    else:
        pull = data_inputs.get_EWR_components('nest-percent')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    EWR_info = data_inputs.additional_nest_pull(EWR_info, gauge, EWR, allowance)
    # Mask dates for both the flow and level dataframes:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years:
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow/level data against EWR requirements and then perform analysis on the results: 
    if ((EWR_info['trigger_day'] != None) and (EWR_info['trigger_month'] != None)):
        # If a trigger requirement for EWR (i.e. flows must be between x and y on Z day of year)
        E, NE, D, ME = nest_calc_percent_trigger(EWR_info, df_F[gauge].values, water_years, df_F.index)
    elif ((EWR_info['trigger_day'] == None) and (EWR_info['trigger_month'] == None)):
        if '%' in EWR_info['drawdown_rate']:
            E, NE, D, ME = nest_calc_percent(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
        else:
            try:
                # If its a nest with a weirpool requirement, do not analyse without the level data:
                levels = df_L[EWR_info['weirpool_gauge']].values
            except KeyError:
                print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
                also needs data for {}'''.format(gauge, EWR, EWR_info['weirpool_gauge']))
                return PU_df, None
            E, NE, D, ME = nest_calc_weirpool(EWR_info, df_F[gauge].values, levels, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])

def flow_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    '''For handling flow EWRs where flow needs to be combined at two gauges'''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('multi-gauge-flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask the dates:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years:
    water_years = wateryear_daily(df_F, EWR_info)
    # Save flows for the two gauges to vars. If there is no flow data for the second, dont analyse:
    flows1 = df_F[gauge].values
    try:
        flows2 = df_F[EWR_info['second_gauge']].values
        flows = flows1 + flows2
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['second_gauge']))
        return PU_df, None
    # Check flow data against EWR requirements and then perform analysis on the results: 
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D, ME = flow_calc_anytime(EWR_info, flows, water_years)
    else:
        E, NE, D, ME = flow_calc(EWR_info, flows, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])

def lowflow_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    '''For handling low flow EWRs where flow needs to be combined at two gauges'''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('multi-gauge-low flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years and climates in the catchment:
    water_years = wateryear_daily(df_F, EWR_info)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(water_years, catchment, climate)
    # Save flows for the two gauges to vars. If there is no flow data for the second, dont analyse:
    flows1 = df_F[gauge].values
    try:
        flows2 = df_F[EWR_info['second_gauge']].values
        flows = flows1 + flows2
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['second_gauge']))
        return PU_df, None
    # Check flow data against EWR requirements and then perform analysis on the results: 
    E, NE, D, ME = lowflow_calc(EWR_info, flows, water_years, climates, df_F.index, masked_dates)       
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])
 
def ctf_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    '''For handling cease to flow EWRs where flow needs to be combined at two gauges'''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('multi-gauge-cease to flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years and climates in the catchment:
    water_years = wateryear_daily(df_F, EWR_info)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(water_years, catchment, climate)
    # Save flows for the two gauges to vars. If there is no flow data for the second, dont analyse:
    flows1 = df_F[gauge].values
    try:
        flows2 = df_F[EWR_info['second_gauge']].values
        flows = flows1 + flows2
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['second_gauge']))
        return PU_df, None
    # Check flow data against EWR requirements and then perform analysis on the results:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D, ME = ctf_calc_anytime(EWR_info, df_F[gauge].values, water_years, climates)
    else:
        E, NE, D, ME = ctf_calc(EWR_info, df_F[gauge].values, water_years, climates, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)
    return PU_df, tuple([E])

def cumulative_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    '''For handling cumulative volume EWRs where flow needs to be combined at two gauges'''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('multi-gauge-cumulative')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Save flows for the two gauges to vars. If there is no flow data for the second, dont analyse:
    flows1 = df_F[gauge].values
    try:
        flows2 = df_F[EWR_info['second_gauge']].values
        flows = flows1 + flows2
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data. Specifically this EWR 
        also needs data for {}'''.format(gauge, EWR, EWR_info['second_gauge']))
        return PU_df, None
    # Check flow data against EWR requirements and then perform analysis on the results:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, NE, D, ME = cumulative_calc_anytime(EWR_info, flows, water_years)
    else:
        E, NE, D, ME = cumulative_calc(EWR_info, flows, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, ME, water_years)    
    return PU_df, tuple([E])

def flow_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    '''For handling flow EWRs that need to be met simultaneously with other sites'''
    # Get information about the EWR for the main EWR:
    pull = data_inputs.get_EWR_components('simul-gauge-flow')
    EWR_info1 = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info1, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info1)
    # Get information about the EWR for the secondary EWR:
    EWR_info2 = get_EWRs(PU, EWR_info1['second_gauge'], EWR, EWR_table, allowance, pull)
    # Save flows for the two gauges to vars. If there is no flow data for the second, dont analyse:
    flows1 = df_F[gauge].values
    try:
        flows2 = df_F[EWR_info1['second_gauge']].values
    except KeyError:
        print('''Cannot evaluate this ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge, EWR, EWR_info1['second_gauge']))
        return PU_df, None
    # Check flow data against EWR requirements and then perform analysis on the results:
    if ((EWR_info1['start_month'] == 7) and (EWR_info1['end_month'] == 6)):
        E, NE, D, ME = flow_calc_anytime_sim(EWR_info1, EWR_info2, flows1, flows2, water_years)
    else:
        E, NE, D, ME = flow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info1, E, NE, D, ME, water_years)
    return PU_df, tuple([E])
    
def lowflow_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    '''For handling lowflow EWRs that need to be met simultaneously with other sites'''
    # Get information about the EWR for the main EWR:
    pull = data_inputs.get_EWR_components('simul-gauge-low flow')
    EWR_info1 = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info1, df_F)
    # Extract a daily timeseries for water years and climates for the catchment
    water_years = wateryear_daily(df_F, EWR_info1)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(water_years, catchment, climate)
    # Get information about the EWR for the secondary EWR:
    EWR_info2 = get_EWRs(PU, EWR_info1['second_gauge'], EWR, EWR_table, allowance, pull)
    # Save flows for the two gauges to vars. If there is no flow data for the second, dont analyse:
    flows1 = df_F[gauge].values
    try:
        flows2 = df_F[EWR_info1['second_gauge']].values
    except KeyError: 
        print('''Cannot evaluate this ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge, EWR, EWR_info1['second_gauge']))
        return PU_df, None
    # Check flow data against EWR requirements and then perform analysis on the results:
    E1, E2, NE1, NE2, D, ME = lowflow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates, df_F.index, masked_dates)
    PU_df = event_stats_sim(df_F, PU_df, gauge, EWR_info1['second_gauge'], EWR, EWR_info1, E1, E2, NE1, NE2, D, ME, water_years)
    return PU_df, tuple([E1, E2])

def ctf_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate):
    '''For handling cease to flow EWRs that need to be met simultaneously with other sites'''
    # Get information about the EWR for the main EWR:
    pull = data_inputs.get_EWR_components('simul-gauge-cease to flow')
    EWR_info1 = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info1, df_F)
    # Extract a daily timeseries for water years and climates for the catchment
    water_years = wateryear_daily(df_F, EWR_info1)
    catchment = data_inputs.gauge_to_catchment(gauge)
    climates = data_inputs.wy_to_climate(water_years, catchment, climate)
    # Get information about the EWR for the secondary EWR:
    EWR_info2 = get_EWRs(PU, EWR_info1['second_gauge'], EWR, EWR_table, allowance, pull)
    # Save flows for the two gauges to vars. If there is no flow data for the second, dont analyse:
    flows1 = df_F[gauge].values
    try:
        flows2 = df_F[EWR_info1['second_gauge']].values
    except KeyError: 
        print('''Cannot evaluate this ewr for {} {}, due to missing data.
        Specifically, this EWR also needs data for {}'''.format(gauge, EWR, EWR_info1['second_gauge']))
        return PU_df, None
    # Check flow data against EWR requirements and then perform analysis on the results:
    E1, E2, NE1, NE2, D, ME = ctf_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates, df_F.index, masked_dates)
    PU_df = event_stats_sim(df_F, PU_df, gauge, EWR_info1['second_gauge'], EWR, EWR_info1, E1, E2, NE1, NE2, D, ME, water_years)
    
    return PU_df, tuple([E1, E2])

def complex_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance):
    '''Handling complex EWRs (complex EWRs are hard coded into the tool)'''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('complex')
    EWR_info1 = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Instead of pulling the other complex information, pull the second part and save it to a new EWR_info variable:
    if 'a' in EWR:
        EWR2 = EWR.replace('a', 'b')
    elif 'b' in EWR:
        EWR2 = EWR.replace('b', 'a')
    EWR_info2 = get_EWRs(PU, gauge, EWR2, EWR_table, allowance, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info1, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info1)
    # If preferred, skip, timing window too small to meet requirements:
    if 'P' in EWR:
        return PU_df, E
    else:
        if '2' in EWR:
            E, NE, D, ME = flow_calc_post_req(EWR_info1, EWR_info2, df_F[gauge].values, water_years, df_F.index, masked_dates)
        elif '3' in EWR:
            E, NE, D, ME = flow_calc_outside_req(EWR_info1, EWR_info2, df_F[gauge].values, water_years, df_F.index, masked_dates)
        PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info1, E, NE, D, ME, water_years)
        
        return PU_df, tuple([E])

#---------------------------------------- Checking EWRs ------------------------------------------#

def which_water_year_no_event(iteration, total_event, water_years):
    '''Finding which water year the event gap was finished in - the start of the event that broke the gap'''
    
    start_event = water_years[iteration-total_event]
    
    return start_event
    
    
def which_water_year(iteration, total_event, water_years):
    '''Finding which water year the majority of the event fell in. If equal, defaults to latter'''
    event_wateryears = water_years[iteration-total_event:iteration]
    midway_iteration = int((len(event_wateryears))/2)
    mid_event = event_wateryears[int(midway_iteration)]

    return mid_event

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

# def which_water_year_complex(iteration, event, water_years, reference_loc):
#     '''Finding which water year the majority of the event fell in. If equal, defaults to latter'''
#     if reference_loc == 'before':
        
        
#     elif reference_loc == ''
#     event_wateryears = water_years[iteration-len(event):iteration]
#     midway_iteration = int((len(event_wateryears))/2)
#     mid_event = event_wateryears[int(midway_iteration)]

#     return mid_event

def flow_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, gap_track, 
               water_years, total_event):
    '''Checks daily flow against EWR threshold. Builds on event lists and no event counters.
    At the end of the event, if it was long enough, the event is saved against the relevant
    water year in the event dictionary. All event gaps are saved against the relevant water 
    year in the no event dictionary
    '''

    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        event.append(flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
        no_event += 1
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) >= EWR_info['min_event']:
                water_year = which_water_year(iteration, total_event, water_years)
                all_events[water_year].append(event)
                total_event_gap = no_event - total_event
                if total_event_gap > 0:
                    ne_water_year = which_water_year_no_event(iteration, total_event, water_years)
                    all_no_events[ne_water_year].append([total_event_gap])
                no_event = 0
                total_event = 0
                
            event = []
        no_event += 1
        
    return event, all_events, no_event, all_no_events, gap_track, total_event

def lowflow_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, water_years):
    '''Checks daily flow against the EWR threshold. Saves all events to the relevant water year
    in the event tracking dictionary. Saves all event gaps to the relevant water year in the 
    no event dictionary.
    '''
    
    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        event.append(flow)
        if no_event > 0:
            ne_water_year = which_water_year_no_event(iteration, len(event), water_years)
            all_no_events[ne_water_year].append([no_event])
        no_event = 0
    else:
        no_event += 1
        if len(event) > 0:
            all_events[water_years[iteration]].append(event)
            
        event = []
        
    return event, all_events, no_event, all_no_events

def ctf_check(EWR_info, iteration, flow, event, all_events, no_event, all_no_events, water_years):
    '''Checks daily flow against the cease to flow EWR threshold. Saves all events to the relevant
    water year in the event tracking dictionary. Saves all no events to the relevant water year
    in the no event dictionary.
    '''

    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        event.append(flow)
    else:
        if len(event) > 0:
            water_year = which_water_year(iteration, len(event), water_years)
            all_events[water_year].append(event)
            if no_event > 0:
                ne_water_year = which_water_year_no_event(iteration, len(event), water_years)
                all_no_events[ne_water_year].append([no_event])
                no_event = 0
        event = []
        no_event += 1
    
    return event, all_events, no_event, all_no_events

def level_check(EWR_info, iteration, level, level_change, event, all_events, no_event, all_no_events, water_years):
    '''Checks daily level against the EWR threshold. Saves events meeting the minimum duration req
    to the relevant water year in the event tracking dictionary. Saves all event gaps to the
    relevant water year in the no event dictionary.
    '''

    if ((level >= EWR_info['min_level']) and (level <= EWR_info['max_level']) and\
        (level_change <= float(EWR_info['drawdown_rate']))):
        event.append(level)
        no_event += 1
    else:
        if (len(event) >= EWR_info['duration']):
            all_events[water_years[iteration]].append(event)
            total_event_gap = no_event - len(event)
            if total_event_gap > 0:
                ne_water_year = which_water_year_no_event(iteration, len(event), water_years)
                all_no_events[ne_water_year].append([total_event_gap])
            no_event = 0
        event = []
        no_event += 1

    return event, all_events, no_event, all_no_events

def flow_check_sim(iteration, EWR_info1, EWR_info2, water_years, flow1, flow2, event, all_events,
                   no_event, all_no_events, gap_track, total_event):
    '''Checks daily flow for both sites against EWR thresholds. Saves events to the relevant 
    water year in the event tracking dictionary. Saves all event gaps to the relevant
    water year in the no event dictionary.
    '''

    if ((flow1 >= EWR_info1['min_flow']) and (flow1 <= EWR_info1['max_flow']) and\
        (flow2 >= EWR_info2['min_flow']) and (flow2 <= EWR_info2['max_flow'])):
        event.append(flow1)
        total_event += 1
        gap_track = EWR_info1['gap_tolerance'] # reset the gapTolerance after threshold is reached
        no_event += 1
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) >= EWR_info1['min_event']:
                water_year = which_water_year(iteration, total_event, water_years)
                all_events[water_year].append(event)
                total_event_gap = no_event - total_event
                ne_water_year = which_water_year_no_event(iteration, total_event, water_years)
                if total_event_gap > 0:
                    all_no_events[ne_water_year].append([total_event_gap])
                no_event = 0
                total_event = 0
                
            event = []
            total_event = 0
        no_event += 1

    return event, all_events, no_event, all_no_events, gap_track, total_event

def date_check(date, masked_dates):
    '''Pass in a date, if the date is within the range of accepted dates, return True, else False'''
    return True if data in masked_dates else False

#------------------------------------ Calculation functions --------------------------------------#

def get_duration(climate, EWR_info):
    '''Determines the relevant duration for the water year'''
    
    if ((climate == 'Very Dry') and (EWR_info['duration_VD'] !=None)):
        duration = EWR_info['duration_VD']
    else:
        duration = EWR_info['duration']
    
    return duration

def construct_event_dict(water_years):
    ''' Pulling together a dictionary with a key per year in the timeseries,
    and an empty list as each value, where events will be saved into
    '''
    all_events = {}
    water_years_unique = sorted(set(water_years))
    all_events = dict.fromkeys(water_years_unique)
    for k, _ in all_events.items():
        all_events[k] = []
        
    return all_events

def check_requirements(list_of_lists):
    '''Iterate through the lists, if there is a False in any, return False'''
    result = True
    for list_ in list_of_lists:
        if False in list_:
            result = False

    return result

def next_water_year():
    '''When moving to the next water year, this function calculates the missing days and returns'''
    '''Option 2 is to have another series getting generated, showing missed days per year - perhaps coming out of the masking calculation functions
    this will be a dictionary(?), and each year will have a corresponding days masked out, which can be called and then added onto the end'''

def lowflow_calc(EWR_info, flows, water_years, climates, dates, masked_dates):
    '''For calculating low flow ewrs. These have no consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of 
    each water year.
    '''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    # Iterate over daily flow, sending to the lowflow_check function for each iteration 
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            event, all_events, no_event, all_no_events = lowflow_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, water_years)
        else:
            no_event += 1
        # At the end of each water year, save any ongoing events and reset the list
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
            event = [] # Reset at the end of the water year
            durations.append(get_duration(climates[i], EWR_info))
            min_events.append(EWR_info['min_event'])
        
    # Check the final iteration, saving any ongoing events/event gaps to their spots in the dictionaries
    if dates[-1] in masked_dates:
        event, all_events, no_event, all_no_events = lowflow_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, water_years)
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        if no_event > 0:
            ne_water_year = which_water_year_no_event(-1, len(event), water_years)
            all_no_events[ne_water_year].append([no_event])
        no_event = 0
    if no_event > 0 :
        all_no_events[water_years[-1]].append([no_event]) # no event to finish, save no event period to final year
    durations.append(get_duration(climates[-1], EWR_info))
    min_events.append(EWR_info['min_event'])
    return all_events, all_no_events, durations, min_events

def ctf_calc_anytime(EWR_info, flows, water_years, climates):
    '''For calculating cease to flow ewrs. These have a consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of each
    water year.
    '''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    # Iterate over daily flow, sending to the ctf_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        event, all_events, no_event, all_no_events = ctf_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, water_years)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            durations.append(get_duration(climates[i], EWR_info))
            min_events.append(EWR_info['min_event'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    event, all_events, no_event, all_no_events = ctf_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, water_years) 
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        if no_event > 0:
            ne_water_year = which_water_year_no_event(-1, len(event), water_years)
            all_no_events[ne_water_year].append([no_event])
        no_event = 0
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event]) # No event to finish so save to final year in dictionary
    durations.append(get_duration(climates[-1], EWR_info))
    min_events.append(EWR_info['min_event'])
    return all_events, all_no_events, durations, min_events


def ctf_calc(EWR_info, flows, water_years, climates, dates, masked_dates):
    '''For calculating cease to flow ewrs. These have a consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of each
    water year.
    '''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    # Iterate over daily flow, sending to the ctf_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            event, all_events, no_event, all_no_events = ctf_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, water_years)
        else:
            no_event += 1
            # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
                if no_event > 0:
                    ne_water_year = which_water_year_no_event(i, len(event), water_years)
                    all_no_events[ne_water_year].append([no_event])
                    no_event = 0
                event = []
            durations.append(get_duration(climates[i], EWR_info))
            min_events.append(EWR_info['min_event'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        event, all_events, no_event, all_no_events = ctf_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, water_years) 
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        if no_event > 0:
            ne_water_year = which_water_year_no_event(-1, len(event), water_years)
            all_no_events[ne_water_year].append([no_event])
        no_event = 0
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event]) # No event to finish so save to final year in the dictionary
    durations.append(get_duration(climates[-1], EWR_info))
    min_events.append(EWR_info['min_event'])
    
    return all_events, all_no_events, durations, min_events

def flow_calc(EWR_info, flows, water_years, dates, masked_dates):
    '''For calculating flow EWRs with a time constraint within their requirements. Events are
    therefore reset at the end of each water year.
    '''
    # Declare variables:
    event = []
    total_event = 0
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, gap_track, water_years, total_event)
        else:
            no_event += 1
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) >= EWR_info['min_event']:
                all_events[water_years[i]].append(event)
                if no_event - total_event > 0:
                    ne_water_year = which_water_year_no_event(i, total_event, water_years)
                    all_no_events[ne_water_year].append([no_event-total_event])
                no_event = 0
                total_event = 0
            event = []
            durations.append(EWR_info['duration'])
            min_events.append(EWR_info['min_event'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, gap_track, water_years, total_event)   
    if len(event) >= EWR_info['min_event']:
        all_events[water_years[-1]].append(event)
        if no_event - total_event > 0:
            ne_water_year = which_water_year_no_event(i, total_event, water_years)
            all_no_events[ne_water_year].append([no_event-total_event])
        no_event = 0
        total_event = 0
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
    durations.append(EWR_info['duration'])
    min_events.append(EWR_info['min_event'])

    return all_events, all_no_events, durations, min_events
    
def flow_calc_anytime(EWR_info, flows, water_years):
    '''For calculating flow EWRs with no time constraint within their requirements. Events crossing
    water year boundaries will be saved to the water year where the majority of event days were.
    '''
    # Declare variables:
    event = []
    no_event = 0
    total_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    # Iterate over flows:
    for i, flow in enumerate(flows[:-1]):
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, gap_track, water_years, total_event)  
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
            min_events.append(EWR_info['min_event'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, gap_track, water_years, total_event)
    if len(event) >= EWR_info['min_event']:
        water_year = which_water_year(-1, total_event, water_years)
        all_events[water_year].append(event)
        total_event_gap = no_event-total_event
        if total_event_gap > 0:
            ne_water_year = which_water_year_no_event(-1, total_event, water_years)
            all_no_events[ne_water_year].append([total_event_gap])
        no_event = 0
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event]) # No event so add to the final year
    durations.append(EWR_info['duration'])
    min_events.append(EWR_info['min_event'])

    return all_events, all_no_events, durations, min_events

def lake_calc(EWR_info, levels, water_years, dates, masked_dates):
    '''For calculating lake level EWRs. The EWRs are checked on an annual basis, events and
    event gaps are therefore reset at the end of each water year.
    '''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    # Check first level and then iterate through, until last day:
    for i, level in enumerate(levels[:-1]):
        if i == 0:
            level_change = 0
        if dates[i] in masked_dates:
            level_change = levels[i-1]-levels[i]
            event, all_events, no_event, all_no_events = level_check(EWR_info, i, level, level_change, event, all_events, no_event, all_no_events, water_years)
        else:
            no_event += 1
        # At the end of each water year, save any ongoing events to the dictionaries, and reset the list
        if water_years[i] != water_years[i+1]:
            if len(event) >= EWR_info['duration']:
                all_events[water_years[i]].append(event)
                if no_event > 0:
                    ne_water_year = which_water_year_no_event(i, len(event), water_years)
                    all_no_events[water_years[i]].append([no_event-len(event)])
                    no_event = 0
            event = []
            durations.append(EWR_info['duration'])
            min_events.append(EWR_info['min_event'])
    if dates[-1] in masked_dates:
        level_change = levels[-2]-levels[-1]        
        event, all_events, no_event, all_no_events = level_check(EWR_info, -1, levels[-1], level_change, event, all_events, no_event, all_no_events, water_years)
        
    if len(event) >= EWR_info['duration']:
        all_events[water_years[-1]].append(event)
        if no_event > 0:
            ne_water_year = which_water_year_no_event(-1, len(event), water_years)
            all_no_events[ne_water_year].append([no_event-len(event)])
        no_event = 0
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event]) # if there is an unsaved event gap, save this to the final year of the dictionary
    durations.append(EWR_info['duration'])
    min_events.append(EWR_info['min_event'])

    return all_events, all_no_events, durations, min_events

def cumulative_calc(EWR_info, flows, water_years, dates, masked_dates):
    '''For calculating cumulative flow EWRs with time constraints. A 'window' is moved over the
    flow timeseries for each year with a length of the EWR duration, these flows in the window
    that are over the minimum flow threshold are summed together, and if this sum meets the min
    volume requirement, and event is achieved. The program the skips over the flows included so
    as to not double count'''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    unique_water_years = sorted(set(water_years))
    # Iterate over unique water years:
    for year in unique_water_years:
        mask = water_years == year
        year_flows = flows[mask]
        year_dates = dates[mask]
        durations.append(EWR_info['duration'])
        min_events.append(EWR_info['min_event'])
        skip_lines = 0
        # Within the water year, iterate over the flows, checking the future (duration length) of days for an event:
        for i, flow in enumerate(year_flows[:-EWR_info['duration']]):
            if year_dates[i] in masked_dates and year_dates[i+EWR_info['duration']] in masked_dates:
                if skip_lines > 0:
                    skip_lines -= 1
                else:
                    subset_flows = year_flows[i:i+EWR_info['duration']]
                    large_enough_flows = subset_flows[subset_flows >= EWR_info['min_flow']]
                    # If there are enough flows over the threshold to meet the volume requirement, save event and event gap:
                    if sum(large_enough_flows) >= EWR_info['min_volume']:
                        all_events[year].append(list(large_enough_flows))
                        skip_lines = EWR_info['duration'] - 1
                        if no_event > 0:
                            all_no_events[year].append([no_event])
                            no_event = 0
                    else:
                        no_event += 1
            else:
                no_event += 1
                
        if year_dates[-1] in masked_dates and skip_lines == 0:
            final_subset_flows = year_flows[-EWR_info['duration']:]
            final_large_enough_flows = final_subset_flows[final_subset_flows >= EWR_info['min_flow']]
            # If there are enough flows over the threshold to meet the volume requirement, save event and event gap:
            if sum(final_large_enough_flows) >= EWR_info['min_volume']:
                all_events[year].append(list(final_large_enough_flows))
                if no_event > 0:
                    all_no_events[year].append([no_event])
                    no_event = 0
            else:
                no_event = no_event + EWR_info['duration']
        else:
            no_event = no_event + EWR_info['duration'] - skip_lines
                
    
    return all_events, all_no_events, durations, min_events

def cumulative_calc_anytime(EWR_info, flows, water_years):
    '''For calculating cumulative flow EWRs with no time constraints. A 'window' is moved over the
    flow timeseries for each year with a length of the EWR duration, these flows in the window
    that are over the minimum flow threshold are summed together, and if this sum meets the min
    volume requirement, and event is achieved. The program the skips over the flows included so
    as to not double count'''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations = len(set(water_years))*[EWR_info['duration']]
    min_events = len(set(water_years))*[EWR_info['min_event']]
    skip_lines = 0
    # Iterate over the flows, checking the future (duration length) of days for an event:
    for i, flow in enumerate(flows[:-EWR_info['duration']]):
        if skip_lines > 0:
            skip_lines -= 1
        else:
            subset_flows = flows[i:i+EWR_info['duration']]
            large_enough_flows = subset_flows[subset_flows >= EWR_info['min_flow']]
            # If there are enough flows over the threshold to meet the volume requirement, save event and event gap:
            if sum(large_enough_flows) >= EWR_info['min_volume']:
                water_year = which_water_year_start(i, subset_flows, water_years)
                all_events[water_year].append(list(large_enough_flows))
                skip_lines = EWR_info['duration'] -1
                if no_event > 0:
                    all_no_events[water_year].append([no_event])
                no_event = 0
            else:
                no_event += 1
    # Check for final window:
    final_subset_flows = flows[-EWR_info['duration']+skip_lines:]
    final_large_enough_flows = final_subset_flows[final_subset_flows >= EWR_info['min_flow']]
    if sum(final_large_enough_flows) >= EWR_info['min_volume']:
        water_year = which_water_year_end(-EWR_info['duration'], final_subset_flows, water_years)
        all_events[water_year].append(list(final_large_enough_flows))
        if no_event > 0:
            all_no_events[water_years[-1]].append([no_event])
        no_event = 0
    else:
        no_event = no_event + EWR_info['duration']
        
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
    return all_events, all_no_events, durations, min_events

def nest_calc_weirpool(EWR_info, flows, levels, water_years, dates, masked_dates):
    ''' For calculating Nest type EWRs with a weirpool element in the requirement. For an event
    to be registered, the requirements for flow at the flow gauge, level at the level gauge,
    and drawdown rate at the level gauge are all required to be met.'''
    # Declare variables:
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    unique_water_years = sorted(set(water_years))
    # Iterate over the years:
    for year in unique_water_years:
        mask = water_years == year
        year_flows = flows[mask]
        year_levels = levels[mask]
        year_dates = dates[mask]
        durations.append(EWR_info['duration'])
        min_events.append(EWR_info['min_event'])
        skip_lines = 0
        # Iterate over flows in the water year:
        for i, flow in enumerate(year_flows[:-EWR_info['duration']]):
            if skip_lines > 0:
                skip_lines -= 1
            else:
                if year_dates[i] in masked_dates and year_dates[i+EWR_info['duration']] in masked_dates:
                    # Perform checks on flow, level, and drawdown rate in duration window
                    subset_flows = year_flows[i:i+EWR_info['duration']]
                    subset_levels = year_levels[i:i+EWR_info['duration']]

                    min_flow_check = subset_flows >= EWR_info['min_flow']
                    max_flow_check = subset_flows <= EWR_info['max_flow']
                    level_change = np.diff(subset_levels)
                    level_change_check = level_change >= -float(EWR_info['drawdown_rate'])
                    checks_passed = check_requirements([min_flow_check, max_flow_check, level_change_check])

                    if checks_passed:
                        all_events[year].append(list(subset_flows))
                        if no_event > 0:
                            all_no_events[year].append([no_event])
                        no_event = 0
                        skip_lines = len(subset_flows) -1
                    else:
                        no_event += 1
                else:
                    no_event += 1
        
        if year_dates[-1] in masked_dates and skip_lines == 0:
            final_subset_flows = year_flows[-EWR_info['duration']:]
            final_subset_levels = year_levels[-EWR_info['duration']:]

            min_flow_check = final_subset_flows >= EWR_info['min_flow']
            max_flow_check = final_subset_flows <= EWR_info['max_flow']
            level_change = np.diff(final_subset_levels)
            level_change_check = level_change >= -float(EWR_info['drawdown_rate'])
            checks_passed = check_requirements([min_flow_check, max_flow_check, level_change_check])

            if checks_passed:
                all_events[year].append(list(final_subset_flows))
                if no_event > 0:
                    all_no_events[year].append([no_event])
                no_event = 0
            else:
                no_event = no_event + EWR_info['duration']
        else:
            no_event = no_event + EWR_info['duration'] - skip_lines
        
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])

    return all_events, all_no_events, durations, min_events
            
def nest_calc_percent(EWR_info, flows, water_years, dates, masked_dates):
    ''' For calculating Nest type EWRs with a percentage drawdown requirement. For an event
    to be registered, the requirements for flow and drawdown rate at the flow gauge are all
    required to be met.'''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    unique_water_years = sorted(set(water_years))
    drawdown_rate = int(EWR_info['drawdown_rate'][:-1])
    # Iterate over unique water years:
    for year in unique_water_years:
        mask = water_years == year
        year_flows = flows[mask]
        year_dates = dates[mask]
        durations.append(EWR_info['duration'])
        min_events.append(EWR_info['min_event'])
        skip_lines = 0
        # Iterate over flows in the water year:
        for i, flow in enumerate(year_flows[:-EWR_info['duration']]):
            if skip_lines > 0:
                skip_lines -= 1
            else:
                if year_dates[i] in masked_dates and year_dates[i+EWR_info['duration']] in masked_dates:
                    # Perform checks on flow and drawdown rate (percentage) in duration window
                    subset_flows = year_flows[i:i+EWR_info['duration']]

                    min_flow_check = subset_flows >= EWR_info['min_flow']
                    max_flow_check = subset_flows <= EWR_info['max_flow']
                    flow_change = np.array(np.diff(subset_flows),dtype=float)
                    divide_flows = subset_flows[:-1]
                    difference = np.divide(flow_change, divide_flows, out=np.zeros_like(flow_change), where=divide_flows!=0)*100
                    flow_change_check = difference >= -drawdown_rate

                    checks_passed = check_requirements([min_flow_check, max_flow_check, flow_change_check])

                    if checks_passed:
                        all_events[year].append(list(subset_flows))
                        if no_event > 0:
                            all_no_events[year].append([no_event])
                        no_event = 0
                        skip_lines = len(subset_flows) -1
                    else:
                        no_event = no_event + 1
                else:
                    no_event += 1
        if year_dates[-1] in masked_dates and skip_lines == 0:
            final_subset_flows = year_flows[-EWR_info['duration']:]
            min_flow_check = final_subset_flows >= EWR_info['min_flow']
            max_flow_check = final_subset_flows <= EWR_info['max_flow']
            flow_change = np.array(np.diff(final_subset_flows),dtype=float)
            divide_flows = final_subset_flows[:-1]
            difference = np.divide(flow_change, divide_flows, out=np.zeros_like(flow_change), where=divide_flows!=0)*100
            flow_change_check = difference >= -drawdown_rate

            checks_passed = check_requirements([min_flow_check, max_flow_check, flow_change_check])

            if checks_passed:
                all_events[year].append(list(final_subset_flows))
                if no_event > 0:
                    all_no_events[year].append([no_event])
                no_event = 0
                skip_lines = len(subset_flows) -1
            else:
                no_event = no_event + EWR_info['duration']
        else:
            no_event = no_event + EWR_info['duration'] - skip_lines
            
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
    
    return all_events, all_no_events, durations, min_events

def nest_calc_percent_trigger(EWR_info, flows, water_years, dates):
    ''' For calculating Nest type EWRs with a percentage drawdown requirement and a trigger day.
    A trigger day is when there is required to be a flow between x and y on DD/MM for the EWR to be
    checked. The requirements for flow and drawdown rate at the flow gauge are all required 
    to be met.'''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    durations = len(set(water_years))*[EWR_info['duration']]
    min_events = len(set(water_years))*[EWR_info['min_event']]
    skip_lines = 0
    drawdown_rate = int(EWR_info['drawdown_rate'][:-1])
    days = dates.day.values
    months = dates.month.values
    # Iterate over flows
    for i, flow in enumerate(flows[:-EWR_info['duration']]):
        if skip_lines > 0:
            skip_lines -= 1
        else:
            # Only perform check on trigger day, looking ahead to see if there is an event:
            trigger_day = days[i] == EWR_info['trigger_day']
            trigger_month = months[i] == EWR_info['trigger_month']
            if trigger_day and trigger_month:
                subset_flows = flows[i:i+EWR_info['duration']]
                min_flow_check = subset_flows >= EWR_info['min_flow']
                max_flow_check = subset_flows <= EWR_info['max_flow']
                
                flow_change = np.array(np.diff(subset_flows),dtype=float)
                divide_flows = subset_flows[:-1]
                difference = np.divide(flow_change, divide_flows, out=np.zeros_like(flow_change), where=divide_flows!=0)*100
                flow_change_check = difference >= -drawdown_rate
                checks_passed = check_requirements([min_flow_check, max_flow_check, flow_change_check])
                if checks_passed:
                    all_events[water_years[i]].append(list(subset_flows))
                    if no_event > 0:
                        all_no_events[water_years[i]].append([no_event])
                    no_event = 0
                    skip_lines = EWR_info['duration'] -1
                else:
                    no_event = no_event + 1
            else:
                no_event = no_event + 1
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event+EWR_info['duration']])
    return all_events, all_no_events, durations, min_events
       
def weirpool_calc(EWR_info, flows, levels, water_years, weirpool_type, dates, masked_dates):
    '''For calculating weirpool type EWRs, these have a required flow rate at a flow gauge,
    a required river level at a level gauge, a drawdown rate, and are either a weirpool raising
    or weirpool drawdown. Weirpool raising EWRs have a minimum height, and weirpool falling have
    a maximum level.'''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    unique_water_years = sorted(set(water_years))
    # Iterate over unique water years:
    for year in unique_water_years:
        mask = water_years == year
        year_flows = flows[mask]
        year_levels = levels[mask]
        year_dates = dates[mask]
        durations.append(EWR_info['duration'])
        min_events.append(EWR_info['min_event'])
        skip_lines = 0
        # Iterate over flows:
        for i, flow in enumerate(year_flows[:-EWR_info['duration']]):
            if skip_lines > 0:
                skip_lines = skip_lines - 1
            else:
                if year_dates[i] in masked_dates and year_dates[i+EWR_info['duration']] in masked_dates:
                    # Get a subset of flows and levels, check the requirements are met:
                    subset_flows = year_flows[i:i+EWR_info['duration']]
                    subset_levels = year_levels[i:i+EWR_info['duration']]

                    min_flow_check = subset_flows >= EWR_info['min_flow']
                    max_flow_check = subset_flows <= EWR_info['max_flow']
                    levels_change = np.array(np.diff(subset_levels),dtype=float)
                    level_change_check = levels_change >= -float(EWR_info['drawdown_rate'])
                    if weirpool_type == 'raising':
                        check_levels = subset_levels >= EWR_info['min_level']
                    elif weirpool_type == 'falling':
                        check_levels = subset_levels <= EWR_info['max_level']
                    checks_passed = check_requirements([min_flow_check, max_flow_check, 
                                                        level_change_check, check_levels])
                    if checks_passed:
                        all_events[year].append(list(subset_flows))
                        skip_lines = EWR_info['duration'] -1
                        if no_event > 0:
                            all_no_events[year].append([no_event])
                        no_event = 0
                    else:
                        no_event += 1
                else:
                    no_event += 1
                    
        if year_dates[-1] in masked_dates and skip_lines == 0:
            final_subset_flows = year_flows[-EWR_info['duration']:]
            final_subset_levels = year_levels[-EWR_info['duration']:]

            min_flow_check = final_subset_flows >= EWR_info['min_flow']
            max_flow_check = final_subset_flows <= EWR_info['max_flow']
            levels_change = np.array(np.diff(final_subset_levels),dtype=float)
            level_change_check = levels_change >= -float(EWR_info['drawdown_rate'])
            if weirpool_type == 'raising':
                check_levels = final_subset_levels >= EWR_info['min_level']
            elif weirpool_type == 'falling':
                check_levels = final_subset_levels <= EWR_info['max_level']
            checks_passed = check_requirements([min_flow_check, max_flow_check, 
                                                level_change_check, check_levels])
            if checks_passed:
                all_events[year].append(list(final_subset_flows))
                skip_lines = EWR_info['duration'] -1
                if no_event > 0:
                    all_no_events[year].append([no_event])
                no_event = 0
            else:
                no_event = no_event + EWR_info['duration']
        else:
            no_event = no_event + EWR_info['duration'] - skip_lines
            
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
    return all_events, all_no_events, durations, min_events

def flow_calc_anytime_sim(EWR_info1, EWR_info2, flows1, flows2, water_years):
    '''For calculating flow EWRs with no time constraint within their requirements. Events crossing
    water year boundaries will be saved to the water year where the majority of event days were. 
    These EWRs need to be met simultaneously with EWRs at partner sites.
    '''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    total_event = 0
    # Iterate over flows
    for i, flow in enumerate(flows1[:-1]):
        # Each iteration send to a simultaneous flow check function, to see if both sites requirements are met:
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                               EWR_info2, water_years,
                                                                               flow, flows2[i], event,
                                                                               all_events, no_event,
                                                                               all_no_events,gap_track,
                                                                                           total_event)
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info1['duration'])
            min_events.append(EWR_info1['min_event'])
    # Check final iteration:
    event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                           EWR_info2, water_years,
                                                                           flows1[-1], flows2[-1], event,
                                                                           all_events, no_event,
                                                                           all_no_events,gap_track,
                                                                          total_event)           
    if len(event) >= EWR_info1['min_event']:
        water_year = which_water_year(-1, total_event, water_years)
        all_events[water_year].append(event)
        if no_event > 0:
            ne_water_year = which_water_year_no_event(-1, total_event, water_years)
            all_no_events[ne_water_year].append([no_event-total_event])
        no_event = 0
    durations.append(EWR_info1['duration'])
    min_events.append(EWR_info1['min_event'])
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event-total_event])
    return all_events, all_no_events, durations, min_events
    
def flow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, dates, masked_dates):
    '''For calculating flow EWRs with a time constraint within their requirements. Events are
    therefore reset at the end of each water year. These EWRs need to be met simultaneously 
    with EWRs at partner sites.
    '''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    lines_to_skip = 0
    total_event = 0
    # Iterate over flows
    for i, flow in enumerate(flows1[:-1]):
        if dates[i] in masked_dates:
            # Each iteration send to a simultaneous flow check function, to see if both sites requirements are met:
            event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,EWR_info2,
                                                                                   water_years, flow,
                                                                                   flows2[i],event,
                                                                                   all_events, no_event,
                                                                                   all_no_events, gap_track,
                                                                                               total_event)
        else:
            no_event += 1
        if water_years[i] != water_years[i+1]:
            if len(event) >= EWR_info1['min_event']: 
                all_events[water_years[i]].append(event)
                if no_event - total_event > 0:
                    ne_water_year = which_water_year_no_event(i, total_event, water_years)
                    all_no_events[ne_water_year].append([no_event-total_event])
                no_event = 0
                total_event = 0
            event = []
            durations.append(EWR_info1['duration'])
            min_events.append(EWR_info1['min_event'])
        
    # Check final iteration:
    if dates[-1] in masked_dates:
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                               EWR_info2, water_years,
                                                                               flows1[-1], flows2[-1], event,
                                                                               all_events, no_event,
                                                                               all_no_events,gap_track,
                                                                                           total_event)
    if len(event) >= EWR_info1['min_event']:
        all_events[water_years[-1]].append(event)
        if no_event - total_event > 0:
            ne_water_year = which_water_year_no_event(-1, total_event, water_years)
            all_no_events[ne_water_year].append([no_event-total_event])
        no_event = 0
        total_event = 0
    durations.append(EWR_info1['duration'])
    min_events.append(EWR_info1['min_event'])
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])  
    return all_events, all_no_events, durations, min_events

def lowflow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates, dates, masked_dates):
    '''For calculating low flow ewrs. These have no consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of 
    each water year. These EWRs need to be met simultaneously with EWRs at partner sites.
    '''      
    # Decalre variables:
    event1, event2 = [], []
    no_event1, no_event2 = 0, 0
    all_events1 = construct_event_dict(water_years)
    all_events2 = construct_event_dict(water_years)
    all_no_events1 = construct_event_dict(water_years)
    all_no_events2 = construct_event_dict(water_years)
    durations, min_events = [], []
    for i, flow in enumerate(flows1[:-1]):
        if dates[i] in masked_dates:
            # Check flows at each site against their respective EWR requirements:
            event1, all_events1, no_event1, all_no_events1 = lowflow_check(EWR_info1, i, flow, event1, all_events1, no_event1, all_no_events1, water_years)
            event2, all_events2, no_event2, all_no_events2 = lowflow_check(EWR_info2, i, flows2[i], event2, all_events2, no_event2, all_no_events2, water_years)
        else:
            no_event1 += 1
            no_event2 += 1
        if water_years[i] != water_years[i+1]:
            if len(event1) > 0:
                all_events1[water_years[i]].append(event1)
            if len(event2) > 0:
                all_events2[water_years[i]].append(event2)

            event1, event2 = [], []
            durations.append(get_duration(climates[i], EWR_info1))
            min_events.append(EWR_info1['min_event'])

    # Check final iteration:
    if dates[-1] in masked_dates:
        event1, all_events1, no_event1, all_no_events1 = lowflow_check(EWR_info1, -1, flows1[-1], event1, all_events1, no_event1, all_no_events1, water_years)
        event2, all_events2, no_event2, all_no_events2 = lowflow_check(EWR_info2, -1, flows2[-1], event2, all_events2, no_event2, all_no_events2, water_years)
    if len(event1) > 0:
        all_events1[water_years[-1]].append(event1)
        if no_event1 > 0:
            ne_water_year = which_water_year_no_event(-1, len(event), water_years)
            all_no_events1[ne_water_year].append([no_event1])
        no_event1 = 0
    if len(event2) > 0:
        all_events2[water_years[-1]].append(event2)
        if no_event2 > 0:
            ne_water_year = which_water_year_no_event(-1, len(event), water_years)
            all_no_events2[ne_water_year].append([no_event2])
        no_event2 = 0
    durations.append(get_duration(climates[-1], EWR_info1))
    min_events.append(EWR_info1['min_event'])
    # If there are event gaps left over, save these to the last year in the dictionary
    if no_event1 > 0:
        all_no_events1[water_years[-1]].append([no_event1])
    if no_event2 > 0:
        all_no_events2[water_years[-1]].append([no_event2])
    return all_events1, all_events2, all_no_events1, all_no_events2, durations, min_events

def ctf_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, climates, dates, masked_dates):
    '''For calculating cease to flow ewrs. These have a consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of each
    water year. These EWRs need to be met simultaneously with EWRs at partner sites.
    '''
    # Declare variables:
    event1, event2 = [], []
    no_event1, no_event2 = 0, 0
    all_events1 = construct_event_dict(water_years)
    all_events2 = construct_event_dict(water_years)
    all_no_events1 = construct_event_dict(water_years)
    all_no_events2 = construct_event_dict(water_years)
    durations, min_events = [], []
    for i, flow in enumerate(flows1[:-1]):
        # Check flows at each site against their respective EWR requirements:
        if dates[i] in masked_dates:
            event1, all_events1, no_event1, all_no_events1 = ctf_check(EWR_info1, i, flow, event1, all_events1, no_event1, all_no_events1, water_years)
            event2, all_events2, no_event2, all_no_events2 = ctf_check(EWR_info2, i, flows2[i], event2, all_events2, no_event2, all_no_events2, water_years)
            if water_years[i] != water_years[i+1]:
                if len(event1) > 0:
                    all_events1[water_years[i]].append(event1)
                    if no_event1 > 0:
                        ne_water_year = which_water_year_no_event(i, len(event1), water_years)
                        all_no_events1[water_years[i]].append([no_event1])
                        no_event1 = 0
                if len(event2) > 0:
                    all_events2[water_years[i]].append(event2)
                    if no_event2 > 0:
                        ne_water_year = which_water_year_no_event(i, len(event2), water_years)
                        all_no_events2[water_years[i]].append([no_event2])
                        no_event2 = 0
                event1, event2 = [], []
                durations.append(get_duration(climates[i], EWR_info1))
                min_events.append(EWR_info1['min_event'])
        else:
            no_event1 += 1
            no_event2 += 1
    # Check final iteration:
    if dates[-1] in masked_dates:
        event1, all_events1, no_event1, all_no_events1 = ctf_check(EWR_info1, -1, flows1[-1], event1, all_events1, no_event1, all_no_events1, water_years)
        event2, all_events2, no_event2, all_no_events2 = ctf_check(EWR_info2, -1, flows2[-1], event2, all_events2, no_event2, all_no_events2, water_years)  
    if len(event1) > 0:
        all_events1[water_years[-1]].append(event1)
    if len(event2) > 0:
        all_events2[water_years[-1]].append(event2)
    durations.append(get_duration(climates[-1], EWR_info1))
    min_events.append(EWR_info1['min_event'])
    if no_event1 > 0:
        all_no_events1[water_years[-1]].append([no_event1])
    if no_event2 > 0:
        all_no_events2[water_years[-1]].append([no_event2])
    return all_events1, all_events2, all_no_events1, all_no_events2, durations, min_events

def check_trigger(iteration, min_flow, max_flow, gap_tolerance, min_event, water_years, flow, event, gap_track, trigger, total_event):
    '''Checks daily flow against ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list, event dictionary, and time between events
    '''

    if ((flow >= min_flow) and (flow <= max_flow)):
        event.append(flow)
        total_event += 1
        gap_track = gap_tolerance # reset the gapTolerance after threshold is reached
        if len(event) >= min_event:
            trigger = True
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            gap_track = -1
            event = []
            total_event = 0
            trigger = False

    return event, gap_track, trigger, total_event

def flow_calc_post_req(EWR_info1, EWR_info2, flows, water_years, dates, masked_dates):
    ''' For flow EWRs with a main requirement, and a secondary requirement which needs to be 
    satisfied immediately after the main requirement. Currently only two EWRs have this requirement,
    gauge 409025 OB2_S and OB2_P
    '''
    trigger, post_trigger = False, False
    event = []
    post_event = []
    total_event = 0
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    durations = len(set(water_years))*[EWR_info1['duration']]
    min_events = len(set(water_years))*[EWR_info1['min_event']]
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
                event, gap_track, trigger, total_event = check_trigger(i, EWR_info1['min_flow'], EWR_info1['max_flow'], EWR_info1['gap_tolerance'], EWR_info1['min_event'], water_years, flow, event, gap_track, trigger, total_event)
            elif ((trigger == True) and (post_trigger == False)):
                # Send to check the post requirement
                post_event, gap_track, post_trigger, total_event = check_trigger(i, EWR_info2['min_flow'], EWR_info2['max_flow'], EWR_info2['gap_tolerance'], EWR_info2['duration'], water_years, flow, post_event, gap_track, post_trigger, total_event)
            elif ((trigger == True) and (post_trigger == True)):
                water_year = which_water_year(i, total_event, water_years)
                full_event = event + post_event
                all_events[water_year].append(full_event)
                ne_water_year = which_water_year_no_event(i, total_event, water_years)
                all_no_events[ne_water_year].append([no_event-total_event])
                no_event = 0
                trigger, post_trigger = False, False
                event, post_event = [], []
                total_event = 0
    if ((trigger == True) and (post_trigger == True)):
        water_year = which_water_year(i, total_event, water_years)
        full_event = event + post_event
        all_events[water_year].append(full_event)
        ne_water_year = which_water_year_no_event(i, total_event, water_years)
        all_no_events[ne_water_year].append([no_event-total_event])
        no_event = 0
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
    return all_events, all_no_events, durations, min_events

def flow_calc_outside_req(EWR_info1, EWR_info2, flows, water_years, dates, masked_dates):
    ''' For flow EWRs with a main requirement, and a secondary requirement which can either be 
    satisfied immediately after the main requirement, or immediately before. 
    Currently only two EWRs have this requirement, gauge 409025 OB3_S and OB3_P
    '''
    trigger, pre_trigger, post_trigger = False, False, False
    event, pre_event, post_event = [], [], []
    no_event = 0
    total_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    durations = len(set(water_years))*[EWR_info1['duration']]
    min_events = len(set(water_years))*[EWR_info1['min_event']]
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
                event, gap_track, trigger, total_event = check_trigger(i, EWR_info1['min_flow'], EWR_info1['max_flow'], EWR_info1['gap_tolerance'], EWR_info1['min_event'], water_years, flow, event, gap_track, trigger, total_event)
            elif trigger == True: # Event registered, now check for event pre/post this
                gap_track = EWR_info1['gap_tolerance']
                # First check if there was an event before the main event:
                total_event_pre = total_event
                for pre_i, pre_flow in enumerate(reversed(flows[:(i-len(event))])):  
                    pre_event, gap_track, pre_trigger, total_event_pre = check_trigger(pre_i, EWR_info2['min_flow'], EWR_info2['max_flow'], EWR_info2['gap_tolerance'], EWR_info2['duration'], water_years, pre_flow, pre_event, gap_track, pre_trigger, total_event_pre)
                    if gap_track == -1: # If the pre event gap tolerance is exceeded, break
                        pre_trigger = False
                        pre_event = []
                        total_event_pre = 0
                        break
                    if pre_trigger == True:
                        event = list(reversed(pre_event)) + event
                        water_year = which_water_year(i, total_event_pre, water_years)
                        all_events[water_year].append(event)
                        ne_water_year = which_water_year_no_event(i, total_event_pre, water_years)
                        all_no_events[ne_water_year].append([no_event-len(event)])
                        pre_event = []
                        total_event_pre = 0
                        trigger, pre_trigger, post_trigger = False, False, False
                        break
                # If the above fails, enter sub routine to check for an event after:
                if pre_trigger == False:
                    gap_track = EWR_info1['gap_tolerance']
                    total_event_post = total_event
                    for post_i, post_flow in enumerate(flows[i:]):
                        post_event, gap_track, post_trigger, total_event_post = check_trigger(post_i, EWR_info2['min_flow'], EWR_info2['max_flow'], EWR_info2['gap_tolerance'], EWR_info2['duration'], water_years, post_flow, post_event, gap_track, post_trigger, total_event_post)
                        if gap_track == -1:
                            post_event = []
                            total_event_post = 0
                            trigger, pre_trigger, post_trigger = False, False, False
                            break
                        if post_trigger == True:
                            water_year = which_water_year((i+post_i+2), total_event_post, water_years) #Check loc
                            ne_water_year = which_water_year_no_event((i+post_i+1), total_event_post, water_years)
                            all_no_events[ne_water_year].append([no_event-len(event)])
                            no_event = 0
                            event = event + post_event
                            all_events[water_year].append(event)
                            skip_lines = len(post_event) -1
                            total_event_post = 0
                            post_event = []
                            trigger, pre_trigger, post_trigger = False, False, False
                            break
                if pre_trigger == False and post_trigger == False:
                    trigger, pre_trigger, post_trigger =  False, False, False
                    event, pre_event, post_event = [], [], []
                    total_event = 0

    if trigger == True and post_trigger == True:
        water_year = which_water_year(i, total_event, water_years)
        all_events[water_year].append(event)
        ne_water_year = which_water_year_no_event(i, total_event, water_years)
        all_no_events[ne_water_year].append([no_event-len(event)])
        no_event = 0
    if trigger == True and pre_trigger==True:
        water_year = which_water_year(i, total_event, water_years)
        all_events[water_year].append(event)
        ne_water_year = which_water_year_no_event(i, total_event, water_years)
        all_no_events[ne_water_year].append([no_event-len(event)])
        no_event = 0     

    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
        
    return all_events, all_no_events, durations, min_events

#------------------------------------ Stats on EWR events ----------------------------------------#

def get_event_years(EWR_info, events, unique_water_years, durations, min_events):
    '''Returns a list of years with events (represented by a 1), and years without events (0)'''
    event_years = []
    for index, year in enumerate(unique_water_years):
        combined_len = 0
        for e in events[year]:
            if len(e) >= min_events[index]:
                combined_len += len(e)
        if ((combined_len >= durations[index] and len(events[year])>=EWR_info['events_per_year'])):
            event_years.append(1)
        else:
            event_years.append(0)
    
    return event_years



def get_achievements(EWR_info, events, unique_water_years, durations, min_events):
    '''Returns a list of number of events per year'''
    num_events = []
    for index, year in enumerate(unique_water_years):
        combined_len = 0
        yearly_events = 0
        for e in events[year]:
            if len(e) >= min_events[index]:
                combined_len += len(e)
            if combined_len >= durations[index]:
                yearly_events += 1
                combined_len = 0
        total = yearly_events/EWR_info['events_per_year']
        num_events.append(int(total))
    
    return num_events

def get_number_events(EWR_info, events, unique_water_years, durations, min_events):
    '''Returns a list of number of events per year'''
    num_events = []
    for index, year in enumerate(unique_water_years):
        combined_len = 0
        yearly_events = 0
        for e in events[year]:
            if len(e) >= min_events[index]:
                combined_len += len(e)
            if combined_len >= durations[index]:
                yearly_events += 1
                combined_len = 0
        total = yearly_events
        num_events.append(int(total))
    
    return num_events

def get_average_event_length(events, unique_water_years):
    '''Returns a list of average event length per year'''
    av_length = list()
    for year in unique_water_years:
        count = len(events[year])
        if count > 0:
            joined = sum(events[year], [])
            length = len(joined)
            av_length.append(length/count)
        else:
            av_length.append(0.0)
            
    return av_length

def get_total_days(events, unique_water_years):
    '''Returns a list with total event days per year'''
    total_days = list()
    for year in unique_water_years:
        count = len(events[year])
        if count > 0:
            joined = sum(events[year], [])
            length = len(joined)
            total_days.append(length)
        else:
            total_days.append(0)
            
    return total_days

def get_days_between(years_with_events, no_events, EWR, EWR_info, unique_water_years, water_years):
    '''Calculates the days/years between events. For certain EWRs (cease to flow, lowflow, 
    and level EWRs), event gaps are calculated on an annual basis, others will calculate on a daily basis'''
    
    CTF_EWR = 'CF' in EWR
    LOWFLOW_EWR = 'VF' in EWR or 'BF' in EWR
    YEARLY_INTEREVENT = EWR_info['max_inter-event'] >= 1
    if EWR_info['max_inter-event'] == None:
        # If there is no max interevent period defined in the EWR, return all interevent periods:
        return list(no_events.values())
    else:
        max_interevent = data_inputs.convert_max_interevent(unique_water_years, water_years, EWR_info)
        # If its a cease to flow/low flow/level EWR and has an interevent duration of equal to or more than a year,
        # This one will need to be worked out on an annual basis, looking at the total days between the years with events,
        # Rather than the individual sub event gaps, as these are not applicable here:
        if ((CTF_EWR and YEARLY_INTEREVENT) or (LOWFLOW_EWR and YEARLY_INTEREVENT)): #  or (LEVEL_EWR and YEARLY_INTEREVENT))
            temp = np.array(years_with_events)
            temp[temp==0] = 365
            temp[temp==1] = 0
            total_list = []
            temp_count = 0
            for i in temp:
                if i > 0:
                    temp_count = temp_count + i
        
                    total_list.append([])
                else:
                    if temp_count >= max_interevent:
                        total_list.append([temp_count])
                    else:
                        total_list.append([])
                    temp_count = 0
            return total_list
        # If its another EWR type of EWR, use the daily interevent counts
        else:
            temp = {}
            for year in no_events:
                temp[year] = []
                for n_e in no_events[year]:
                    if n_e[0] >= max_interevent:
                        temp[year].append(n_e[0])
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
    
    return list(group_df[gauge].values)

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

def get_achievements_sim(events1, events2):
    '''get the minimum number of events for simultaneous EWRs'''
    e1 = np.array(list(events1))
    e2 = np.array(list(events2))
    results = []
    for i, event in enumerate(e1):
        results.append(min([event, e2[i]]))
    return results

def get_number_events_sim(events1, events2):
    '''get the minimum number of events for simultaneous EWRs'''
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

def event_stats(df, PU_df, gauge, EWR, EWR_info, events, no_events, durations, min_events, water_years):
    ''' Produces statistics based on the event dictionaries and event gap dictionaries'''
    unique_water_years = set(water_years)
    # Years with events
    years_with_events = get_event_years(EWR_info, events, unique_water_years, durations, min_events)
    YWE = pd.Series(name = str(EWR + '_eventYears'), data = years_with_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, YWE], axis = 1)
    # Number of event achievements:
    num_event_achievements = get_achievements(EWR_info, events, unique_water_years, durations, min_events)
    NEA = pd.Series(name = str(EWR + '_numAchieved'), data= num_event_achievements, index = unique_water_years)
    PU_df = pd.concat([PU_df, NEA], axis = 1)
    # Total number of events
    num_events = get_number_events(EWR_info, events, unique_water_years, durations, min_events)
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
    days_between = get_days_between(years_with_events, no_events, EWR, EWR_info, unique_water_years, water_years)
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

def event_stats_sim(df, PU_df, gauge1, gauge2, EWR, EWR_info, events1, events2, no_events1, no_events2, durations, min_events, water_years):
    ''' Produces statistics based on the event dictionaries and event gap dictionaries for simultaneous EWRs'''
    unique_water_years = set(water_years)
    # Years with events
    years_with_events1 = get_event_years(EWR_info, events1, unique_water_years, durations, min_events)
    years_with_events2 = get_event_years(EWR_info, events2, unique_water_years, durations, min_events)
    years_with_events = event_years_sim(years_with_events1, years_with_events2)
    YWE = pd.Series(name = str(EWR + '_eventYears'), data = years_with_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, YWE], axis = 1)
    # Number of event achievements per year
    num_events_ach_1 = get_achievements(EWR_info, events1, unique_water_years, durations, min_events)
    num_events_ach_2 = get_achievements(EWR_info, events2, unique_water_years, durations, min_events)
    num_events_ach = get_achievements_sim(num_events_ach_1, num_events_ach_2)
    NEA = pd.Series(name = str(EWR + '_numAchieved'), data= num_events_ach, index = unique_water_years)
    PU_df = pd.concat([PU_df, NEA], axis = 1)
    # Total number of event per year
    num_events1 = get_number_events(EWR_info, events1, unique_water_years, durations, min_events)
    num_events2 = get_number_events(EWR_info, events2, unique_water_years, durations, min_events)
    num_events = get_number_events_sim(num_events1, num_events2)
    NE = pd.Series(name = str(EWR + '_numEvents'), data= num_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, NE], axis = 1)
    # Average length of events
    av_length1 = get_average_event_length(events1, unique_water_years)
    av_length2 = get_average_event_length(events2, unique_water_years)  
    av_length = average_event_length_sim(av_length1, av_length2)
    AL = pd.Series(name = str(EWR + '_eventLength'), data = av_length, index = unique_water_years)
    PU_df = pd.concat([PU_df, AL], axis = 1)
    # Total event days
    total_days1 = get_total_days(events1, unique_water_years)
    total_days2 = get_total_days(events2, unique_water_years)
    av_total_days = average_event_length_sim(total_days1, total_days2)
    TD = pd.Series(name = str(EWR + '_totalEventDays'), data = av_total_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, TD], axis = 1)
    # Days between events
    days_between1 = get_days_between(years_with_events1, no_events1, EWR, EWR_info, unique_water_years, water_years)
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

#---------------------------- Sorting and distributing to handling functions ---------------------#

def calc_sorter(df_F, df_L, gauge, allowance, climate):
    '''Sends to handling functions to get calculated depending on the type of EWR''' 
    # Get ewr tables:
    PU_items = data_inputs.get_planning_unit_info()
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    menindee_gauges, wp_gauges = data_inputs.get_level_gauges()
    multi_gauges = data_inputs.get_multi_gauges('all')
    simultaneous_gauges = data_inputs.get_simultaneous_gauges('all')
    complex_EWRs = data_inputs.get_complex_calcs()
    # Extract relevant sections of the EWR table:
    gauge_table = EWR_table[EWR_table['gauge'] == gauge]
    # save the planning unit dataframes to this dictionary:
    location_results = {}
    location_events = {}
    for PU in set(gauge_table['PlanningUnitID']):
        PU_table = gauge_table[gauge_table['PlanningUnitID'] == PU]
        EWR_categories = PU_table['flow level volume'].values
        EWR_codes = PU_table['code']
        PU_df = pd.DataFrame()
        PU_events = {}
        for i, EWR in enumerate(tqdm(EWR_codes, position = 0, leave = False,
                                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                                     desc= str('Evaluating ewrs for '+ gauge))):
            events = {}
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
                    PU_df, events = ctf_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
                elif SIMULTANEOUS:
                    PU_df, events = ctf_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
                else:
                    PU_df, events = ctf_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
            elif CAT_FLOW and EWR_LOWFLOW and not VERYDRY:
                if MULTIGAUGE:
                    PU_df, events = lowflow_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
                elif SIMULTANEOUS:
                    PU_df, events = lowflow_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
                else:
                    PU_df, events = lowflow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance, climate)
            elif CAT_FLOW and EWR_FLOW and not VERYDRY:
                if COMPLEX:
                    PU_df, events = complex_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
                elif MULTIGAUGE:
                    PU_df, events = flow_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
                elif SIMULTANEOUS:
                    PU_df, events = flow_handle_sim(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
                else:
                    PU_df, events = flow_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
            elif CAT_FLOW and EWR_WP and not VERYDRY:
                PU_df, events = weirpool_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance)
            elif CAT_FLOW and EWR_NEST and not VERYDRY:
                PU_df, events = nest_handle(PU, gauge, EWR, EWR_table, df_F, df_L, PU_df, allowance)
            elif CAT_CUMUL and EWR_CUMUL and not VERYDRY:
                if MULTIGAUGE:
                    PU_df, events = cumulative_handle_multi(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
                else:
                    PU_df, events = cumulative_handle(PU, gauge, EWR, EWR_table, df_F, PU_df, allowance)
            elif CAT_LEVEL and EWR_LEVEL and not VERYDRY:
                PU_df, events = level_handle(PU, gauge, EWR, EWR_table, df_L, PU_df, allowance)
            else:
                continue
            # Add the events to the dictionary:
            if events != {}:
                PU_events[str(EWR)]=events
            
        PU_name = PU_items['PlanningUnitName'].loc[PU_items[PU_items['PlanningUnitID'] == PU].index[0]]
        
        location_results[PU_name] = PU_df
        location_events[PU_name] = PU_events
    return location_results, location_events