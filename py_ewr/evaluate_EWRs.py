from collections import defaultdict
from copy import deepcopy
from enum import unique
import sunau
from typing import Any, List, Dict
import pandas as pd
import numpy as np
from datetime import date, timedelta
import datetime
import calendar
from itertools import chain

from tqdm import tqdm

from . import data_inputs, summarise_results

#----------------------------------- Getting EWRs from the database ------------------------------#

# def cast_str_to_float(component:str)->int:
#     return int(float(component))

def component_pull(EWR_table: pd.DataFrame, gauge: str, PU: str, EWR: str, component: str) -> str:
    '''Pass EWR details (planning unit, gauge, EWR, and EWR component) and the EWR table, 
    this function will then pull the component from the table.

    Args:
        EWR_table (pd.DataFrame): Dataframe of EWRs
        gauge (str): Gauge number
        PU (str): Planning Unit ID
        EWR (str): EWR code
        component (str): EWR parameter (data from which column in the EWR table)

    Results:
        str: value of requested parameter from the EWR table

    '''
    component = list(EWR_table[((EWR_table['Gauge'] == gauge) & 
                           (EWR_table['Code'] == EWR) &
                           (EWR_table['PlanningUnitID'] == PU)
                          )][component])[0]
    return component if component else 0

def apply_correction(val: float, correction: float) -> float:
    '''Applies a correction to the EWR component (based on user request)
    
    Args:
        val (float): value to be corrected
        correction (float): the correction to be applied to the value (0-1)
    
    Results:
        float: new value with correction applied

    '''
    return val*correction

def get_second_multigauge(parameter_sheet: pd.DataFrame, gauge:float, ewr:str, pu:str) -> str:
    """get the second gauge number for a multiguage

    Args:
        parameter_sheet (pd.DataFrame): parameter sheet used in the calculation
        gauge (float): gauge number
        ewr (str): ewr code
        pu (str): planning unit code

    Returns:
        bool: second gauge code
    """
    item = parameter_sheet[(parameter_sheet['Gauge']==gauge) & (parameter_sheet['Code']==ewr) & (parameter_sheet['PlanningUnitID']==pu)]
    gauge_array = item['Multigauge'].to_list()
    gauge_number = gauge_array[0] if gauge_array else ''
    return gauge_number
    
def get_EWRs(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, allowance: dict, components: list) -> dict:
    '''Pulls the relevant EWR componenets for each EWR, and applies any relevant corrections
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge ID
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset
        allowance (dict): How much to scale components by (0-1)
        components (list): List of parameters needing to be pulled from the EWR dataset
    
    Results:
        dict: The EWR components and their values
    
    '''
    ewrs = {}
    # Save identifying information to dictionary:
    ewrs['gauge'] = gauge
    ewrs['planning_unit'] = PU
    ewrs['EWR_code'] = EWR
    
    if 'SM' in components:
        start_date = str(component_pull(EWR_table, gauge, PU, EWR, 'StartMonth'))
        if '.' in start_date:
            ewrs['start_day'] = int(start_date.split('.')[1])
            ewrs['start_month'] = int(start_date.split('.')[0])
        else:
            ewrs['start_day'] = None
            ewrs['start_month'] = int(start_date)
    if 'EM' in components:
        end_date = str(component_pull(EWR_table, gauge, PU, EWR, 'EndMonth'))
        if '.' in end_date:  
            ewrs['end_day'] = int(end_date.split('.')[1])
            ewrs['end_month'] = int(end_date.split('.')[0])
        else:
            ewrs['end_day'] = None
            ewrs['end_month'] =int(end_date)
    if 'MINF' in components:
        min_flow = int(component_pull(EWR_table, gauge, PU, EWR, 'FlowThresholdMin'))
        corrected = apply_correction(min_flow, allowance['minThreshold'])
        ewrs['min_flow'] = int(corrected)
    if 'MAXF' in components:
        max_flow = int(component_pull(EWR_table, gauge, PU, EWR, 'FlowThresholdMax'))
        corrected = apply_correction(max_flow, allowance['maxThreshold'])
        ewrs['max_flow'] = int(corrected)
    if 'MINL' in components:
        min_level = float(component_pull(EWR_table, gauge, PU, EWR, 'LevelThresholdMin'))
        corrected = apply_correction(min_level, allowance['minThreshold'])
        ewrs['min_level'] = corrected
    if 'MAXL' in components:
        max_level = float(component_pull(EWR_table, gauge, PU, EWR, 'LevelThresholdMax'))
        corrected = apply_correction(max_level, allowance['maxThreshold'])
        ewrs['max_level'] = corrected
    if 'MINV' in components:
        min_volume = int(component_pull(EWR_table, gauge, PU, EWR, 'VolumeThreshold'))
        corrected = apply_correction(min_volume, allowance['minThreshold'])
        ewrs['min_volume'] = int(corrected)
    if 'DUR' in components:
        duration = int(component_pull(EWR_table, gauge, PU, EWR, 'Duration'))
        corrected = apply_correction(duration, allowance['duration'])
        ewrs['duration'] = int(corrected)
    if 'GP' in components:
        gap_tolerance = int(component_pull(EWR_table, gauge, PU, EWR, 'WithinEventGapTolerance'))
        ewrs['gap_tolerance'] = gap_tolerance
    if 'EPY' in components:
        events_per_year = int(component_pull(EWR_table, gauge, PU, EWR, 'EventsPerYear'))
        ewrs['events_per_year'] = events_per_year       
    if 'ME' in components:
        min_event = int(component_pull(EWR_table, gauge, PU, EWR, 'MinSpell'))
        if min_event != 1:
            corrected = apply_correction(min_event, allowance['duration'])
        else:
            corrected = min_event
        ewrs['min_event'] = int(corrected)
    if 'MD' in components:
        try: # There may not be a recommended drawdown rate
            max_drawdown = component_pull(EWR_table, gauge, PU, EWR, 'DrawdownRate')
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
            duration_vd = int(component_pull(EWR_table, gauge, PU, EWR_VD, 'Duration'))
            corrected = apply_correction(duration_vd, allowance['duration'])
            ewrs['duration_VD'] =int(corrected)
        except IndexError: # In this case return None type for this component
            ewrs['duration_VD'] = None
    if 'WPG' in components:
        weirpool_gauge = component_pull(EWR_table, gauge, PU, EWR, 'WeirpoolGauge')
        ewrs['weirpool_gauge'] =str(weirpool_gauge)
    if 'MG' in components:       
        ewrs['second_gauge'] = get_second_multigauge(EWR_table, gauge, EWR, PU)    
    if 'SG' in components:
        simultaneousGaugeDict = data_inputs.get_simultaneous_gauges('all')
        ewrs['second_gauge'] = simultaneousGaugeDict[PU][gauge]
    if 'TF' in components:
        try:
            ewrs['frequency'] = component_pull(EWR_table, gauge, PU, EWR, 'TargetFrequency')
            
        except IndexError:
            ewrs['frequency'] = None
    if 'MIE' in components:
        try:
            ewrs['max_inter-event'] = float(component_pull(EWR_table, gauge, PU, EWR, 'MaxInter-event'))
        except IndexError:
            ewrs['max_inter-event'] = None
    if 'AP' in components:
        accumulation_period = component_pull(EWR_table, gauge, PU, EWR, 'AccumulationPeriod')
        ewrs['accumulation_period'] = int(accumulation_period)
    if 'FLV' in components:
        flow_level_volume = component_pull(EWR_table, gauge, PU, EWR, 'FlowLevelVolume')
        ewrs['flow_level_volume'] = flow_level_volume
    if 'MAXD' in components:
        max_duration = component_pull(EWR_table, gauge, PU, EWR, 'MaxSpell')
        ewrs['max_duration'] = int(max_duration) if max_duration else 1_000_000
    if 'TD' in components:
        trigger_day = component_pull(EWR_table, gauge, PU, EWR, 'TriggerDay')
        ewrs['trigger_day'] = int(trigger_day)
    if 'TM' in components:
        trigger_month = component_pull(EWR_table, gauge, PU, EWR, 'TriggerMonth')
        ewrs['trigger_month'] = int(trigger_month)
    if 'WDD' in components:
        try: # The rate is represented in cm
            drawdown_rate_week = component_pull(EWR_table, gauge, PU, EWR, 'DrawDownRateWeek')
            corrected = apply_correction(float(drawdown_rate_week), allowance['drawdown'])
            ewrs['drawdown_rate_week'] = str(corrected/100)
        except ValueError: # In this case set a large number
            ewrs['drawdown_rate_week'] = str(1000000)   

    return ewrs

def is_multigauge(parameter_sheet: pd.DataFrame, gauge:float, ewr:str, pu:str) -> bool:
    """check in the parameter sheet if currently iterated EWR is a multigauge

    Args:
        parameter_sheet (pd.DataFrame): parameter sheet used in the calculation
        gauge (float): gauge number
        ewr (str): ewr code
        pu (str): planning unit code

    Returns:
        bool: returns True if it is a multigauge and False if not
    """
    item = parameter_sheet[(parameter_sheet['Gauge']==gauge) & (parameter_sheet['Code']==ewr) & (parameter_sheet['PlanningUnitID']==pu)]
    mg = item['Multigauge'].to_list()
    if not mg:
        return False
    if mg[0] == '':
        return False
    return int(mg[0]) > 0

def is_weirpool_gauge(parameter_sheet: pd.DataFrame, gauge:float, ewr:str, pu:str) -> bool:
    """check in the parameter sheet if currently iterated EWR is a weirpool gauge

    Args:
        parameter_sheet (pd.DataFrame): parameter sheet used in the calculation
        gauge (float): gauge number
        ewr (str): ewr code
        pu (str): planning unit code

    Returns:
        bool: returns True if it is a weirpool gauge and False if not
    """
    item = parameter_sheet[(parameter_sheet['Gauge']==gauge) & (parameter_sheet['Code']==ewr) & (parameter_sheet['PlanningUnitID']==pu)]
    wp = item['WeirpoolGauge'].to_list()
    if not wp:
        return False
    if wp[0] == '':
        return False
    return int(wp[0]) > 0

#------------------------ Masking timeseries data to dates in EWR requirement --------------------#

def mask_dates(EWR_info: dict, input_df: pd.DataFrame) -> set:
    '''Distributes flow/level dataframe to functions for masking over dates
    
    Args:
        EWR_info (dict): The EWR components and their values
        input_df (pd.DataFrame): Flow/water level dataframe
    
    Results:
        set: A set of dates from the dataframe that fall within the required date range
    
    '''
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

def get_month_mask(start: int, end: int, input_df: pd.DataFrame) -> set:
    ''' takes in a start date, end date, and dataframe,
    masks the dataframe to these dates
    
    Args:
        start (int): start month
        end (int): end month
        input_df (pd.DataFrame): Flow/water level dataframe

    Results:
        set: A set of dates from the dataframe that fall within the required date range
    
    '''
    
    if start > end:
        month_mask = (input_df.index.month >= start) | (input_df.index.month <= end)
    elif start <= end:
        month_mask = (input_df.index.month >= start) & (input_df.index.month <= end)  
        
    input_df_timeslice = input_df.loc[month_mask]
    
    return set(input_df_timeslice.index)


def get_day_mask(startDay: int, endDay: int, startMonth: int, endMonth: int, input_df: pd.DataFrame) -> set:
    ''' for the ewrs with a day and month requirement, takes in a start day, start month, 
    end day, end month, and dataframe, masks the dataframe to these dates
    
    Args:
        startDay (int): start day of required date range
        endDay (int): end day of required date range
        startMonth (int): start month of required date range 
        endMonth (int): end month of required date range
        input_df (pd.DataFrame): Flow/water level dataframe

    Results:
        set: A set of dates from the dataframe that fall within the required date range
    
    '''

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

def wateryear_daily(input_df: pd.DataFrame, ewrs: dict) -> np.array:
    '''Creating a daily time series with water years.
    
    Args:
        input_df (pd.DataFrame): Flow/water level dataframe
        ewrs (dict): The EWR components and their values

    Results:
        np.array: array containing the daily assignment of water year
    
    '''

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

def get_index_date(date_index:Any)-> datetime.date:
    if type(date_index) == pd._libs.tslibs.timestamps.Timestamp:
        return date_index.date()
    if type(date_index) == pd._libs.tslibs.period.Period:
        return date_index.to_timestamp().date()
    else:
        return date_index

#----------------------------------- EWR handling functions --------------------------------------#

def ctf_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict, climate: str) -> tuple:
    '''For handling Cease to flow type EWRs
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)
        climate (str): name of the climate categorisation file to be used

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 

    '''
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
        E, NE, D, ME = ctf_calc_anytime(EWR_info, df_F[gauge].values, water_years, climates, df_F.index)
    else:
        E, NE, D, ME = ctf_calc(EWR_info, df_F[gauge].values, water_years, climates, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])

def lowflow_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict, climate: str) -> tuple:
    '''For handling low flow type EWRs (Very low flows and baseflows)
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)
        climate (str): name of the climate categorisation file to be used

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 
    
    '''
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
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])

def flow_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict) -> tuple:
    '''For handling non low flow based flow EWRs (freshes, bankfulls, overbanks)
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow data against EWR requirements and then perform analysis on the results:
    # if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
    #     E, NE, D, ME = flow_calc_anytime(EWR_info, df_F[gauge].values, water_years, df_F.index)
    # else:
    E, NE, D, ME = flow_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])

def cumulative_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict):
    '''For handling cumulative flow EWRs (some large freshes and overbanks, wetland flows).
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information    
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('cumulative')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    E, NE, D, ME = cumulative_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)

    return PU_df, tuple([E])

def level_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict) -> tuple:
    '''For handling level type EWRs (low, mid, high and very high level lake fills).
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information        
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('level')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_L) 
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_L, EWR_info)  
    # Check flow data against EWR requirements and then perform analysis on the results:
    try:    
        E, NE, D, ME = lake_calc(EWR_info, df_L[gauge].values, water_years, df_L.index, masked_dates)
    except ValueError:
        print(f'''Cannot evaluate this ewr for {gauge} {EWR}, due wrong value in the parameter sheet 
        give level drawdown in cm not in % {EWR_info.get('drawdown_rate', 'no drawdown rate')}''')
        return PU_df, None

    PU_df = event_stats(df_L, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])

def weirpool_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict) -> tuple:
    '''For handling weirpool type EWRs.
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information
    
    '''
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
        print(f'''Cannot evaluate this ewr for {gauge} {EWR}, due to missing data. Specifically this EWR 
        also needs data for level gauge {EWR_info.get('weirpool_gauge', 'no wp gauge')}''')
        return PU_df, None
    # Check flow and level data against EWR requirements and then perform analysis on the results: 
    E, NE, D, ME = weirpool_calc(EWR_info, df_F[gauge].values, levels, water_years, weirpool_type, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])

def nest_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict) -> tuple:
    '''For handling nest style EWRs.

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information    
    
    '''
    # Get information about EWR (changes depending on if theres a weirpool level gauge in the EWR)
    requires_weirpool_gauge =  is_weirpool_gauge(EWR_table, gauge, EWR, PU)
    if requires_weirpool_gauge:
        pull = data_inputs.get_EWR_components('nest-level')
    else:
        pull = data_inputs.get_EWR_components('nest-percent')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, allowance, pull)
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years:
    water_years = wateryear_daily(df_F, EWR_info)
    # there are 2 types of Nesting. 1. with trigger date with daily % drawdown rate and 2. Nesting Weirpool. 
    # no longer required a non-trigger version
    if not requires_weirpool_gauge:
        try:
            # calculate based on a trigger date and % drawdown drop
            E, NE, D, ME = nest_calc_percent_trigger(EWR_info, df_F[gauge].values, water_years, df_F.index)
        except ValueError:
            print(f"""Please pass a value to TriggerMonth between 1..12 and TriggerDay you passed 
            TriggerMonth:{EWR_info['trigger_month']} TriggerDay:{EWR_info['trigger_day']} """)
            return PU_df, None
        
    else:
        try:
            # If its a nest with a weirpool requirement, do not analyses without the level data:
            levels = df_L[EWR_info['weirpool_gauge']].values
        except KeyError:
            print(f'''Cannot evaluate this ewr for {gauge} {EWR}, due to missing data. Specifically this EWR 
            also needs data for level gauge {EWR_info.get('weirpool_gauge', 'no wp gauge')}''')
            return PU_df, None
        # handle any error in missing values in parameter sheet
        try:
            E, NE, D, ME = nest_calc_weirpool(EWR_info, df_F[gauge].values, levels, water_years, df_F.index, masked_dates)
        except KeyError:
            print(f'''Cannot evaluate this ewr for {gauge} {EWR}, due to missing parameter data. Specifically this EWR 
            also needs data for level threshold min or level threshold max''')
            return PU_df, None
        
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])

def flow_handle_multi(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict) -> tuple:
    '''For handling flow EWRs where flow needs to be combined at two gauges
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information     
    
    '''
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
        also needs data for gauge'''.format(gauge, EWR))
        return PU_df, None

    E, NE, D, ME = flow_calc(EWR_info, flows, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])

def lowflow_handle_multi(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict, climate: str) -> tuple:
    '''For handling low flow EWRs where flow needs to be combined at two gauges.
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)
        climate (str): name of the climate categorisation file to be used

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information  
    
    '''
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
        also needs data for gauge'''.format(gauge, EWR))
        return PU_df, None
    # Check flow data against EWR requirements and then perform analysis on the results: 
    E, NE, D, ME = lowflow_calc(EWR_info, flows, water_years, climates, df_F.index, masked_dates)       
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])
 
def ctf_handle_multi(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict, climate: str) -> tuple:
    '''For handling cease to flow EWRs where flow needs to be combined at two gauges

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)
        climate (str): name of the climate categorisation file to be used

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information  
    
    '''
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
        E, NE, D, ME = ctf_calc_anytime(EWR_info, df_F[gauge].values, water_years, climates, df_F.index)
    else:
        E, NE, D, ME = ctf_calc(EWR_info, df_F[gauge].values, water_years, climates, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)
    return PU_df, tuple([E])

def cumulative_handle_multi(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict) -> tuple:
    '''For handling cumulative volume EWRs where flow needs to be combined at two gauges.

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information      
    
    '''
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
        also needs data for gauge'''.format(gauge, EWR))
        return PU_df, None
    E, NE, D, ME = cumulative_calc(EWR_info, flows, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, NE, D, water_years)    
    return PU_df, tuple([E])

def flow_handle_sim(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict):
    '''For handling flow EWRs that need to be met simultaneously with other sites
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 
    
    '''
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
        Specifically, this EWR also needs data for gauge'''.format(gauge, EWR))
        return PU_df, None
        
    E, NE, D, ME = flow_calc_sim(EWR_info1, EWR_info2, flows1, flows2, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info1, E, NE, D, water_years)
    return PU_df, tuple([E])
    
def lowflow_handle_sim(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict, climate: str) -> tuple:
    '''For handling lowflow EWRs that need to be met simultaneously with other sites
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)
        climate (str): name of the climate categorisation file to be used

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 
    
    '''
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
    PU_df = event_stats_sim(df_F, PU_df, gauge, EWR, EWR_info1, E1, E2, NE1, D, water_years)
    return PU_df, tuple([E1, E2])

def ctf_handle_sim(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict, climate: str) -> tuple:
    '''For handling cease to flow EWRs that need to be met simultaneously with other sites
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)
        climate (str): name of the climate categorisation file to be used

    Results:
        tuple[pd.DataFrame, tuple[dict, dict]]: EWR results for the current planning unit iteration (updated); [dictionary of EWR event information site 1; dictionary of EWR event information site 2] 
    
    '''
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
    PU_df = event_stats_sim(df_F, PU_df, gauge, EWR, EWR_info1, E1, E2, NE1, D, water_years)
    
    return PU_df, tuple([E1, E2])

def complex_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame, allowance: dict):
    '''Handling complex EWRs (complex EWRs are hard coded into the tool)

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration
        allowance (dict): How much to scale EWR components by (0-1)

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); [dictionary of EWR event information]     
    
    
    '''
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
        PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info1, E, NE, D, water_years)
        
        return PU_df, tuple([E])

#---------------------------------------- Checking EWRs ------------------------------------------#

def which_water_year_no_event(iteration: int, total_event: int, water_years: np.array) -> int:
    '''Finding which water year the event gap was finished in - the start of the event that broke the gap
    
    Args:
        iteration (int): current iteration in the timeseries
        total_event (int): total length of the current event
        water_years (np.array): daily array of water year values

    Results:
        int: water year assigned for the event gap
    
    '''
    
    start_event = water_years[iteration-total_event]
    
    return start_event
    
    
def which_water_year(iteration: int, total_event: int, water_years: np.array) -> int:
    '''Finding which water year the majority of the event fell in. If equal, defaults to latter

    Args:
        iteration (int): current iteration in the timeseries
        total_event (int): total length of the current event
        water_years (np.array): daily array of water year values

    Results:
        int: water year assigned for the event gap

    '''
    event_wateryears = water_years[iteration-total_event:iteration]
    midway_iteration = int((len(event_wateryears))/2)
    mid_event = event_wateryears[int(midway_iteration)]

    return mid_event

def water_year(flow_date:date)-> int:
    """given a date it returns the wateryear the date is in

    Args:
        flow_date (date): date

    Returns:
        int: water year. e.g. 2022 is the 2022-2023 year start 2022-07-01 end 2023-06-01
    """
    month = flow_date.month
    return flow_date.year if month > 6 else flow_date.year -1

def water_year_touches(start_date:date, end_date:date)->List[int]:
    """given a start and end date of an event return a list of water years
    that the events touches.

    Args:
        start_date (date): Event start date
        end_date (date): Event end date

    Returns:
        list: List of years
    """
    start_wy = water_year(start_date)
    end_wy = water_year(end_date)
    span = end_wy - start_wy
    return [start_wy + i for i in range(span + 1)]

def return_event_info(event:list)-> tuple:
    """given an event return information about an event
    containing start_date, end_date, length and the water_years the event touches

    Args:
        event (list): gauge event list

    Returns:
        tuple: start_date, end_date, length and the water_years the event touches
    """
    start_date, _ = event[0]
    end_date, _ = event[-1]
    length = (end_date - start_date).days
    water_years = water_year_touches(start_date, end_date)
    event_info = (start_date, end_date, length+1, water_years)
    return event_info

def return_events_list_info(gauge_events:dict)-> List[tuple]:
    """It iterates through a gauge events dictionary and returns a list
    with a summary information of the event in a tuple.
    tuple contains 
    start_date, end_date, length and the water_years the event touches

    Args:
        gauge_events (dict): gauge events

    Returns:
        list: list of tuples with the information described above.
    """
    events_list = []
    for _, events in gauge_events.items():
        for i, event in enumerate(events):
            event_info = return_event_info(event)
            events_list.append(event_info)
    return events_list

def years_lengths(event_info: tuple)-> list:
    """return for each year event it touches the number of day up to
    the year boundary from the start of the event unless the last year from the
    first day of the water year to the end of the event.

    Args:
        event_info (tuple): start_date, end_date, length and the water_years the event touches

    Returns:
        list: list of integers with the number of days for each year
    """
    years_lengths_list = []
    start, end, length, wys = event_info
    if len(wys) == 1:
        years_lengths_list.append(length)
    else:
        for wy in wys[:-1]:
            up_to_boundary = (date(wy+1,6,30) - start).days + 1
            years_lengths_list.append(up_to_boundary)
        # last water year of collection is the days from start of the water year to last day of the event
        tail_length = (end - date(wys[-1],7, 1)).days + 1
        years_lengths_list.append(tail_length)        
    return years_lengths_list

def which_year_lake_event(event_info: tuple, min_duration: int)-> int:
    """given a event info and a event min duration it returns the
    year the event has to be recorded according to the lake level EWR rule

    If not at year boundary the event will be recorded.

    - Year it ends if o All years event touches the duration is less then min, 
    but the total is within duration range o If all years the duration within duration range
    - Year prior to its end if o Last year event touches has duration less then min duration

    Args:
        event_info (tuple): start_date, end_date, length and the water_years the event touches
        min_duration (int): min duration of the event

    Returns:
        int: year event to be recorded
    """
    
    _, _, _, wys = event_info
    years_lengths_list = years_lengths(event_info) 
   
    if len(years_lengths_list) == 1:
        return wys[0]
        
    if years_lengths_list[-1] < min_duration:
        year = wys[-1] if all([i< min_duration 
                               for i in years_lengths_list]) else wys[-2]
    else:
        year = wys[-1]
        
    return year

def flow_check(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, no_event: list, all_no_events: dict, gap_track: int, 
               water_years: np.array, total_event: int, flow_date: date) -> tuple:
    '''Checks daily flow against EWR threshold. Builds on event lists and no event counters.
    At the end of the event, if it was long enough, the event is saved against the relevant
    water year in the event dictionary. All event gaps are saved against the relevant water 
    year in the no event dictionary

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        no_event (list): current no_event state
        all_no_events (dict): current all no events state
        gap_track (int): current gap_track state
        water_years (np.array): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date

    Returns:
        tuple: the current state of the event, all_events, no_event, all_no_events, gap_track, total_event

    '''

    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
        no_event += 1
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                # breakpoint()
                water_year = which_water_year(iteration, total_event, water_years)
                all_events[water_year].append(event)
                total_event_gap = no_event - total_event
                if total_event_gap > 0 and (len(event) >= EWR_info['min_event']):
                    ne_water_year = which_water_year_no_event(iteration, total_event, water_years)
                    all_no_events[ne_water_year].append([total_event_gap])
                    no_event = 0
                if (len(event) >= EWR_info['min_event']):
                    no_event = 0
            total_event = 0
                
            event = []
        no_event += 1
        
    return event, all_events, no_event, all_no_events, gap_track, total_event

def level_check(EWR_info: dict, iteration: int, level:float, level_change:float, 
               event: list, all_events: dict, no_event: list, all_no_events: dict, gap_track: int, 
               water_years: np.array, total_event: int, level_date: date)-> tuple:
    """Checks daily level against EWR threshold. Builds on event lists and no event counters.
    At the end of the event, if it was long enough, the event is saved against the relevant
    water year in the event dictionary. All event gaps are saved against the relevant water 
    year in the no event dictionary
    NOTE: this EWR is a slight variation of the level_check_ltwp as it records the event in a different year depending on
     the rules in the function which_year_lake_event

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        level (float): current level
        level_change (float): level change in meters from previous day to current day
        event (list): current event state
        all_events (dict): current all events state
        no_event (list): current no_event state
        all_no_events (dict): current all no events state
        gap_track (int): current gap_track state
        water_years (np.array): list of water year for every flow iteration
        total_event (int): current total event state
        level_date (date): current level date

    Returns:
        tuple: the current state of the event, all_events, no_event, all_no_events, gap_track, total_event
    """
    iteration_date = get_index_date(level_date)
    if ((level >= EWR_info['min_level']) and (level <= EWR_info['max_level']) and \
        (level_change <= float(EWR_info['drawdown_rate']))):
        threshold_level = (iteration_date, level)
        event.append(threshold_level)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
        no_event += 1
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if (len(event) >= EWR_info['duration'] and len(event) <= EWR_info['max_duration']):
                if (iteration_date.month == 7 and iteration_date.day ==1):
                    pass
                else:
                    event_info = return_event_info(event)
                    lake_event_year =  which_year_lake_event(event_info, EWR_info['duration'])
                    all_events[lake_event_year].append(event)
                total_event_gap = no_event - total_event
                if total_event_gap > 0:
                    ne_water_year = which_water_year_no_event(iteration, total_event, water_years)
                    all_no_events[ne_water_year].append([total_event_gap])
                no_event = 0
                total_event = 0
                
            event = []
        no_event += 1
        
    return event, all_events, no_event, all_no_events, gap_track, total_event

def nest_flow_check(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, 
                    no_event: list, all_no_events: dict, gap_track: int, water_years: list, total_event: int, 
                    flow_date: date, flow_percent_change: float, iteration_no_event: int)-> tuple:

    """Checks daily flows against EWR threshold. Builds on event lists and no_event counters.
    At the end of the event, if it was long enough, the event is saved against the relevant
    water year in the event dictionary. All event gaps are saved against the relevant water 
    year in the no event dictionary.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        no_event (list): current no_event state
        all_no_events (dict): current all no events state
        gap_track (int): current gap_track state
        water_years (np.array): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        flow_percent_change (float): change from previous day to current day
        iteration_no_event (int): iteration_no_event count

    Returns:
        tuple: the current state of the event, all_events, no_event, all_no_events, gap_track, total_event, iteration_no_event
    """

    iteration_date = get_index_date(flow_date)
    if flow >= EWR_info['min_flow'] and check_nest_percent_drawdown(flow_percent_change, EWR_info, flow):
        threshold_flow = (iteration_date, flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
        no_event += 1
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            iteration_no_event = 1 
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
                total_event_gap = no_event - total_event
                if total_event_gap > 0 and (len(event) >= EWR_info['min_event']):
                    ne_water_year = which_water_year_no_event(iteration, total_event, water_years)
                    all_no_events[ne_water_year].append([total_event_gap])
                    no_event = 0
            total_event = 0    
            event = []
        no_event += 1
        
    return event, all_events, no_event, all_no_events, gap_track, total_event, iteration_no_event


def lowflow_check(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, no_event: list, all_no_events: dict, water_years: np.array, flow_date: date) -> tuple:
    '''Checks daily flow against the EWR threshold. Saves all events to the relevant water year
    in the event tracking dictionary. Saves all event gaps to the relevant water year in the 
    no event dictionary.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        no_event (list): current no_event state
        all_no_events (dict): current all no events state
        water_years (np.array): list of water year for every flow iteration
        flow_date (date): current flow date

    Returns:
        tuple: the current state of the event, all_events, no_event, all_no_events

    '''
    
    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
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

def ctf_check(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, no_event: list, all_no_events: dict, water_years: np.array, flow_date: date) -> tuple:
    '''Checks daily flow against the cease to flow EWR threshold. Saves all events to the relevant
    water year in the event tracking dictionary. Saves all no events to the relevant water year
    in the no event dictionary.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        no_event (list): current no_event state
        all_no_events (dict): current all no events state
        water_years (np.array): list of water year for every flow iteration
        flow_date (date): current flow date

    Returns:
        tuple: the current state of the event, all_events, no_event, all_no_events

    '''

    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
    else:
        if len(event) > 0:
            all_events[water_years[iteration-1]].append(event)
            if no_event > 0:
                ne_water_year = which_water_year_no_event(iteration, len(event), water_years)
                all_no_events[ne_water_year].append([no_event])
                no_event = 0
        event = []
        no_event += 1
    
    return event, all_events, no_event, all_no_events



def flow_check_sim(iteration: int, EWR_info1: dict, EWR_info2: dict, water_years: np.array, flow1: float, flow2: float, event: list, all_events: dict,
                   no_event: list, all_no_events: dict, gap_track: int, total_event: int, flow_date: date) -> tuple:
    '''Checks daily flow for both sites against EWR thresholds. Saves events to the relevant 
    water year in the event tracking dictionary. Saves all event gaps to the relevant
    water year in the no event dictionary.

    Args:
        iteration (int): current iteration
        EWR_info1 (dict): dictionary with the parameter info of the EWR being calculated (site 1)
        EWR_info2 (dict): dictionary with the parameter info of the EWR being calculated (site 2)
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        no_event (list): current no_event state
        all_no_events (dict): current all no events state
        water_years (np.array): list of water year for every flow iteration
        flow_date (date): current flow date

    Returns:
        tuple: the current state of the event, all_events, no_event, all_no_events

    '''

    if ((flow1 >= EWR_info1['min_flow']) and (flow1 <= EWR_info1['max_flow']) and\
        (flow2 >= EWR_info2['min_flow']) and (flow2 <= EWR_info2['max_flow'])):
        threshold_flow = (get_index_date(flow_date), flow1)
        event.append(threshold_flow)
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
                
            event = []
            total_event = 0
        no_event += 1

    return event, all_events, no_event, all_no_events, gap_track, total_event

def date_check(date, masked_dates):
    '''Pass in a date, if the date is within the range of accepted dates, return True, else False'''
    return True if date in masked_dates else False

def check_roller_reset_points(roller:int, flow_date:date, EWR_info:Dict):
    """given a date check if roller needs reset to 0
    It happens either at the start of a water year or the start of a window check
    period. Which ever comes first.

    Args:
        roller (int): how many days to look back on the volume checker window
        flow_date (date): date of the current flow
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated

    Returns:
        (int): roller value either the same or a reset value
    """
    if flow_date.month == EWR_info['start_month'] and flow_date.day == 1:
        roller = 0       
    return roller

def volume_check(EWR_info:Dict, iteration:int, flow:int, event:List, all_events:Dict, no_event:int, all_no_events:Dict, gap_track:int, 
               water_years:List, total_event:int, flow_date:date, roller:int, max_roller:int, flows:List)-> tuple:
    """Check in the current iteration of flows if the volume meet the ewr requirements.
    It looks back in a window of the size of the Accumulation period in(Days)

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (int): current flow
        event (List[float]): current event state
        all_events (Dict): current all events state
        no_event (List): current no_event state
        all_no_events (Dict): current all no events state
        gap_track (int): current gap_track state 
        water_years (List): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        roller (int): current roller state
        max_roller (int): current EWR max roller window
        flows (List): current list of all flows being iterated

    Returns:
        tuple: the current state of the event, all_events, no_event, all_no_events, gap_track, total_event and roller
    """
    
    flows_look_back = flows[iteration - roller:iteration+1]
    if roller < max_roller-1:
        roller += 1
    valid_flows = filter(lambda x: (x >= EWR_info['min_flow']) and (x <= EWR_info['max_flow']) , flows_look_back)
    volume = sum(valid_flows)
    if volume >= EWR_info['min_volume']:
        threshold_flow = (get_index_date(flow_date), volume)
        event.append(threshold_flow)
        total_event += 1
        no_event += 1
        gap_track = EWR_info['gap_tolerance']
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) >=  1:
                all_events[water_years[iteration]].append(event)
                total_event_gap = no_event - total_event
                if total_event_gap > 0:
                    ne_water_year = which_water_year_no_event(iteration, total_event, water_years)
                    all_no_events[ne_water_year].append([total_event_gap])
                no_event = 0
            total_event = 0

            event = []
        no_event += 1

    return event, all_events, no_event, all_no_events, gap_track, total_event, roller

def weirpool_check(EWR_info: dict, iteration: int, flow: float, level: float, event: list, all_events: dict, no_event: int, all_no_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, weirpool_type: str, level_change: float) -> tuple:
    """Check weirpool flow and level if meet condition and update state of the events

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        level (float): current level
        event (list): current event state
        all_events (dict): current all events state
        no_event (int): current no_event state
        all_no_events (dict): current all no events state
        gap_track (int): current gap_track state
        water_years (list): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        weirpool_type (str): type of weirpool ewr raising of falling
        level_change (float): level change in meters

    Returns:
        tuple: after the check return the current state of the event, all_events, no_event, all_no_events, gap_track, total_event
    """

    if flow >= EWR_info['min_flow'] and check_wp_level(weirpool_type, level, EWR_info) and check_draw_down(level_change, EWR_info) :
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
        no_event += 1
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
                total_event_gap = no_event - total_event
                if total_event_gap > 0:
                    ne_water_year = which_water_year_no_event(iteration, total_event, water_years)
                    all_no_events[ne_water_year].append([total_event_gap])
                no_event = 0
            total_event = 0
                
            event = []
        no_event += 1
        
    return event, all_events, no_event, all_no_events, gap_track, total_event

def nest_weirpool_check(EWR_info: dict, iteration: int, flow: float, level: float, event: list, all_events: dict, no_event: int, all_no_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, weirpool_type: str, levels: list)-> tuple:
    """Check weirpool flow and level if meet condition and update state of the events

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        level (float): current level
        event (List): current event state
        all_events (Dict): current all events state
        no_event (int): current no_event state
        all_no_events (Dict): current all no events state
        gap_track (int): current gap_track state
        water_years (List): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        weirpool_type (str): type of weirpool ewr raising of falling
        level_change (float): level change in meters

    Returns:
        tuple: after the check return the current state of the event, all_events, no_event, all_no_events, gap_track, total_event
    """

    if flow >= EWR_info['min_flow'] and check_wp_level(weirpool_type, level, EWR_info) and check_weekly_drawdown(levels, EWR_info, iteration, len(event)) :
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
        no_event += 1
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
                total_event_gap = no_event - total_event
                if total_event_gap > 0:
                    ne_water_year = which_water_year_no_event(iteration, total_event, water_years)
                    all_no_events[ne_water_year].append([total_event_gap])
                no_event = 0
            total_event = 0
                
            event = []
        no_event += 1
        
    return event, all_events, no_event, all_no_events, gap_track, total_event



#------------------------------------ Calculation functions --------------------------------------#

def get_duration(climate: str, EWR_info: dict) -> int:
    '''Determines the relevant duration for the water year
    
    Args:
        climate (str): Climate value for this year 
        EWR_info (dict): Dictionary with the parameter info of the EWR being calculated
    
    Results:
        int: Duration value
    
    '''
    
    if ((climate == 'Very Dry') and (EWR_info['duration_VD'] !=None)):
        duration = EWR_info['duration_VD']
    else:
        duration = EWR_info['duration']
    
    return duration

def construct_event_dict(water_years: np.array) -> dict:
    ''' Pulling together a dictionary with a key per year in the timeseries,
    and an empty list as each value, where events will be saved into

    Args:
        water_years (np.array): Daily array of water year values 
    
    Results:
        dict]: A dictionary with years for keys, and empty lists for values 

    '''
    all_events = {}
    water_years_unique = sorted(set(water_years))
    all_events = dict.fromkeys(water_years_unique)
    for k, _ in all_events.items():
        all_events[k] = []
        
    return all_events

def check_wp_level(weirpool_type: str, level: float, EWR_info: dict)-> bool:
    """check if current level meets weirpool requirement. If meets returns True otherwise False

    Args:
        weirpool_type (str): type of weirpool either 'raising' or 'falling'
        level (float): current level
        EWR_info (dict): EWR parameters

    Returns:
        bool: if meet requirements True else False
    """
    return level >= EWR_info['min_level'] if weirpool_type == 'raising' else level <= EWR_info['max_level']

def check_draw_down(level_change: float, EWR_info: dict) -> bool:
    """Check if the level change from yesterday to today changed more than the maximum allowed in the day.
    It will return True if the drawdown is within the allowed rate in cm/day and False if it is above.

    Args:
        level_change (float): change in meters
        EWR_info (dict): EWR parameters

    Returns:
        bool: if pass test returns True and fail return False
    """
    return level_change <= float(EWR_info['drawdown_rate']) if float(EWR_info['drawdown_rate']) else True


def check_weekly_drawdown(levels: list, EWR_info: dict, iteration: int, event_length: int) -> bool:
    """Check if the level change from 7 days ago to today changed more than the maximum allowed in a week.
    It will return True if the drawdown is within the allowed drawdown_rate_week in cm/week and False if it is above.
    drawdown will be assessed only looking at levers within the event window
    looking from the current level to the fist level since event started up to day 7 then
    will check 7 days back.

    Args:
        levels (float): Level time series values
        EWR_info (dict): EWR parameters

    Returns:
        bool: if pass test returns True and fail return False
    """
    drawdown_rate_week = float(EWR_info["drawdown_rate_week"])
    
    if event_length < 6 :
        current_weekly_dd = levels[iteration - event_length] - levels[iteration]
    else:
        current_weekly_dd = levels[iteration - 6 ] - levels[iteration]
        
    return current_weekly_dd <= drawdown_rate_week

def calc_flow_percent_change(iteration: int, flows: list) -> float:
    """Calculate the percentage change in flow from yesterday to today

    Args:
        iteration (int): current iteration
        flows (List): flows timeseries values

    Returns:
        float: returns value
    """
    if iteration == 0:
        return .0
    if iteration != 0:
        return ( ( float(flows[iteration]) / float(flows[iteration -1]) ) -1 )*100 if flows[iteration -1] != .0 else .0


def check_nest_percent_drawdown(flow_percent_change: float, EWR_info: dict, flow:float) -> bool:
    """check if current flow sustain a nest event based on the flow_percent_change
    if it is within the flow band and the drop is greater than the max_drawdown
    then it does not meet

    Args:
        flow_percent_change (float): flow percent change
        EWR_info (Dict): EWR parameters
        flow (float): current flow

    Returns:
        bool: True if meets condition otherwise False
    """
    percent_drawdown = float(EWR_info['drawdown_rate'][:-1])
    if flow >= EWR_info['max_flow']:
        return True
    if flow_percent_change < - percent_drawdown:
        return False 
    else:
        return True


def calc_nest_cut_date(EWR_info: dict, iteration: int, dates: list) -> date:
    """Calculates the last date (date of the month) the nest EWR event is valid

    Args:
        EWR_info (Dict): EWR parameters
        iteration (int): current iteration
        dates (List): time series dates

    Returns:
        date: cut date for the current iteration
    """
    return date(dates[iteration].year, EWR_info['end_month'], calendar.monthrange(dates[0].year,EWR_info['end_month'])[1])

def lowflow_calc(EWR_info: dict, flows: np.array, water_years: np.array, climates: np.array, dates: np.array, masked_dates: set) -> tuple:
    '''For calculating low flow ewrs. These have no consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of 
    each water year.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        climates (np.array): array of daily climate values
        dates (np.array): array of dates
        masked_dates (set): Dates within required date range
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations/min_events

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
            flow_date = dates[i]
            event, all_events, no_event, all_no_events = lowflow_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, water_years, flow_date)
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
        flow_date = dates[-1]
        event, all_events, no_event, all_no_events = lowflow_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, water_years, flow_date)
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

def ctf_calc_anytime(EWR_info: dict, flows: np.array, water_years: np.array, climates: np.array, dates: np.array) -> tuple:
    '''For calculating cease to flow ewrs. These have a consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of each
    water year.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        climates (np.array): array of daily climate values
        dates (np.array): array of dates
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations/min_events

    '''
    # Declare variables:
    event = []
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    # Iterate over daily flow, sending to the ctf_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        flow_date = dates[i]
        event, all_events, no_event, all_no_events = ctf_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, water_years, flow_date)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            durations.append(get_duration(climates[i], EWR_info))
            min_events.append(EWR_info['min_event'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    flow_date = dates[-1]
    event, all_events, no_event, all_no_events = ctf_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, water_years, flow_date) 
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


def ctf_calc(EWR_info: dict, flows: np.array, water_years: np.array, climates: np.array, dates: np.array, masked_dates: set) -> tuple:
    '''For calculating cease to flow ewrs. These have a consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of each
    water year.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        climates (np.array): array of daily climate values
        dates (np.array): array of dates
        masked_dates (set): Dates within required date range
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations/min_events

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
            flow_date = dates[i]
            event, all_events, no_event, all_no_events = ctf_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, water_years, flow_date)
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
        flow_date = dates[-1]
        event, all_events, no_event, all_no_events = ctf_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, water_years, flow_date) 
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

def flow_calc(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array, masked_dates: set) -> tuple:
    '''For calculating flow EWRs with a time constraint within their requirements. Events are
    therefore reset at the end of each water year.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        dates (np.array): array of dates
        masked_dates (set): Dates within required date range
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations/min_events

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
            flow_date = dates[i]
            event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, gap_track, water_years, total_event, flow_date)
        else:
            no_event += 1
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
                if (no_event - total_event) > 0 and (len(event) >= EWR_info['min_event']):
                    ne_water_year = which_water_year_no_event(i, total_event, water_years)
                    all_no_events[ne_water_year].append([no_event-total_event])
                    no_event = 0
                total_event = 0
                if (len(event) >= EWR_info['min_event']):
                    no_event = 0
            event = []
            durations.append(EWR_info['duration'])
            min_events.append(EWR_info['min_event'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, gap_track, water_years, total_event,flow_date)   
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        # if no_event - total_event > 0:
        if (no_event - total_event) > 0 and (len(event) >= EWR_info['min_event']):
            ne_water_year = which_water_year_no_event(i, total_event, water_years)
            all_no_events[ne_water_year].append([no_event-total_event])
            no_event = 0
        if (len(event) >= EWR_info['min_event']):
            no_event = 0
        total_event = 0
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
    durations.append(EWR_info['duration'])
    min_events.append(EWR_info['min_event'])

    return all_events, all_no_events, durations, min_events
    
def flow_calc_anytime(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array) -> tuple:
    '''For calculating flow EWRs with no time constraint within their requirements. Events crossing
    water year boundaries will be saved to the water year where the majority of event days were.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        dates (np.array): array of dates
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations/min_events

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
        flow_date = dates[i]
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, i, flow, event, all_events, no_event, all_no_events, gap_track, water_years, total_event, flow_date)  
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
            min_events.append(EWR_info['min_event'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    flow_date = dates[-1]
    event, all_events, no_event, all_no_events, gap_track, total_event = flow_check(EWR_info, -1, flows[-1], event, all_events, no_event, all_no_events, gap_track, water_years, total_event, flow_date)
    if len(event) > 0:
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


def lake_calc(EWR_info: dict, levels: np.array, water_years: np.array, dates: np.array, masked_dates: set)-> tuple:
    """For calculating lake level EWR with or without time constraint (anytime).
     At the end of each water year save ongoing event, however not resetting the event list. 
     Let the level_check_ltwp_alt record the event when it finishes and reset the event list.
     NOTE: this EWR is a slight variation of the lake_calc_ltwp as it records the event in a different year depending on
     the rules in the function which_year_lake_event

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        levels (np.array): List with all the levels for the current calculated EWR
        water_years (np.array): List of the water year of each day of the current calculated EWR
        dates (np.array): List of the dates of the current calculated EWR
        masked_dates (set): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events
    """

    # Declare variables:
    event = []
    total_event = 0
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    for i, level in enumerate(levels[:-1]):
        if dates[i] in masked_dates:
            level_date = dates[i]
            level_change = levels[i-1]-levels[i] if i > 0 else 0
             # use the same logic as WP
            event, all_events, no_event, all_no_events, gap_track, total_event = level_check(EWR_info, i, level, level_change, event, all_events, no_event,
                                                                                    all_no_events, gap_track, water_years, total_event, level_date)
        else:
            no_event += 1
        # At the end of each water year save ongoing event, however not resetting the list. Let the flow_check record the event when it finishes
        if water_years[i] != water_years[i+1]:
            if len(event) >= EWR_info['duration'] and len(event) <= EWR_info['max_duration']:
                event_at_year_end = deepcopy(event)
                all_events[water_years[i]].append(event_at_year_end)
                if no_event - total_event > 0:
                    ne_water_year = which_water_year_no_event(i, total_event, water_years)
                    all_no_events[ne_water_year].append([no_event-total_event])
                no_event = 0
                total_event = 0
            durations.append(EWR_info['duration'])
            min_events.append(EWR_info['min_event'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        level_change = levels[-2]-levels[-1]   
        level_date = dates[-1]     
        event, all_events, no_event, all_no_events, gap_track, total_event = level_check(EWR_info, -1, levels[-1], level_change, event, all_events, no_event,
                                                                                    all_no_events, gap_track, water_years, total_event, level_date)
        
    if len(event) >= EWR_info['duration'] and len(event) <= EWR_info['max_duration']:
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

def cumulative_calc(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array, masked_dates: set)-> tuple:
    """ Calculate and manage state of the Volume EWR calculations. It delegates to volume_check function
    the record of events when they not end at the end of a water year, otherwise it resets the event at year boundary
    adopting the hybrid method

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): List with all the flows for the current calculated EWR
        water_years (np.array): List of the water year of each day of the current calculated EWR
        dates (np.array): List of the dates of the current calculated EWR
        masked_dates (set): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events
    """
    event = []
    total_event = 0
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    roller = 0
    max_roller = EWR_info['accumulation_period']

    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            roller = check_roller_reset_points(roller, dates[i], EWR_info)
            flow_date = dates[i]
            event, all_events, no_event, all_no_events, gap_track, total_event, roller = volume_check(EWR_info, i, flow, event, all_events, 
                                                                                            no_event, all_no_events, gap_track, water_years, 
                                                                                            total_event, flow_date, roller, max_roller, flows)
        else:
            no_event += 1
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) >= 1:
                all_events[water_years[i]].append(event)
                if no_event - total_event > 0:
                    ne_water_year = which_water_year_no_event(i, total_event, water_years)
                    all_no_events[ne_water_year].append([no_event-total_event])
                no_event = 0
                total_event = 0
            event = []
            durations.append(EWR_info['duration'])
            min_events.append(EWR_info['min_event'])
    
    if dates[-1] in masked_dates:
        roller = check_roller_reset_points(roller, dates[-1], EWR_info)
        flow_date = dates[-1]
        event, all_events, no_event, all_no_events, gap_track, total_event, roller = volume_check(EWR_info, i, flow, event, all_events, 
                                                                                            no_event, all_no_events, gap_track, water_years, 
                                                                                            total_event, flow_date, roller, max_roller, flows)   
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
    durations.append(EWR_info['duration'])
    min_events.append(EWR_info['min_event'])


    return all_events, all_no_events, durations, min_events


def nest_calc_weirpool(EWR_info: dict, flows: list, levels: list, water_years: list, 
    dates: list, masked_dates:List, weirpool_type: str = "raising")-> tuple:
    """For calculating Nest type EWRs with a weirpool element in the requirement. For an event
    to be registered, the requirements for flow at the flow gauge, level at the level gauge,
    and drawdown rate at the level gauge are all required to be met.
    different from the pure weirpool:
    - The drawdown rate is assessed weekly not daily. 
	- To start and end an event the event needs to be in a time window (masked dates).

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        flows (List):  List with all the flows measurements for the current calculated EWR
        levels (List): List with all the levels measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.
        weirpool_type (str, optional): type of weirpool either 'raising' or 'falling'. Defaults to "raising".

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events
    """
    # Declare variables:
    event = []
    total_event = 0
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    # Iterate over flow timeseries, sending to the weirpool_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            # level_change = levels[i-1]-levels[i] if i > 0 else 0
            event, all_events, no_event, all_no_events, gap_track, total_event = nest_weirpool_check(EWR_info, i, flow, levels[i], event,
                                                                                all_events, no_event, all_no_events, gap_track, 
                                                                                water_years, total_event, flow_date, weirpool_type, levels)
        else:
            no_event += 1
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
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
        flow_date = dates[-1]
        # level_change = levels[-2]-levels[-1] if i > 0 else 0
        event, all_events, no_event, all_no_events, gap_track, total_event = nest_weirpool_check(EWR_info, -1, flows[-1], levels[-1], event,
                                                                                all_events, no_event, all_no_events, gap_track, 
                                                                              water_years, total_event, flow_date, weirpool_type, levels)   
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        if no_event - total_event > 0:
            ne_water_year = which_water_year_no_event(i, total_event, water_years)
            all_no_events[ne_water_year].append([no_event-total_event])
        no_event = 0
        total_event = 0
        
    if no_event > 0:
        no_event += 1
        all_no_events[water_years[-1]].append([no_event])
    durations.append(EWR_info['duration'])
    min_events.append(EWR_info['min_event'])

    return all_events, all_no_events, durations, min_events


def nest_calc_percent_trigger(EWR_info:Dict, flows:List, water_years:List, dates:List)->tuple:
    """Do the calculation of the nesting EWR with trigger and % drawdown
    To start and event it needs to be in a trigger window
    To sustain the EWR needs to 
    - Be above the flow (min threshold)
    - Does not fall more than the % in a day if between the min and max flow i.e If it is above max flow threshold, percent drawn 
    down rate does not matter 
    - Days above max threshold count towards the duration count for the event
    Event ends if:
    - fall below min flow threshold
    - drop more than the % drawdown rate when in the flow band 
    - When timing window ends 


    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        flows (List):  List with all the flows measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events
    """
    event = []
    total_event = 0
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    for i, flow in enumerate(flows[:-1]):   
            flow_date = dates[i]
            flow_percent_change = calc_flow_percent_change(i, flows)
            trigger_day = date(dates[i].year,EWR_info["trigger_month"], EWR_info["trigger_day"])
            cut_date = calc_nest_cut_date(EWR_info, i, dates)
            is_in_trigger_window = dates[i].to_timestamp().date() >= trigger_day - timedelta(days=7) \
            and dates[i].to_timestamp().date() <= trigger_day + timedelta(days=7)
            iteration_no_event = 0
            
            ## if there IS an ongoing event check if we are on the trigger season window 
            # if yes then check the current flow
            if total_event > 0:
                if (dates[i].to_timestamp().date() >= trigger_day - timedelta(days=7)) and (dates[i].to_timestamp().date() <= cut_date):
                    event, all_events, no_event, all_no_events, gap_track, total_event, iteration_no_event = nest_flow_check(EWR_info, i, flow, event, all_events, no_event, 
                                                        all_no_events, gap_track, water_years, total_event, flow_date, flow_percent_change, iteration_no_event)

                # this path will only be executed if an event extends beyond the cut date    
                else:
                    if len(event) > 0:
                        all_events[water_years[i]].append(event)
                        total_event_gap = no_event - total_event
                        if total_event_gap > 0 :
                            ne_water_year = which_water_year_no_event(i, total_event, water_years)
                            all_no_events[ne_water_year].append([total_event_gap])
                        no_event = 0
                        total_event = 0
                    event = []
                    no_event += 1
                    iteration_no_event = 1    
            ## if there is NOT an ongoing event check if we are on the trigger window before sending to checker
            if total_event == 0:
                if is_in_trigger_window and iteration_no_event == 0:
                    event, all_events, no_event, all_no_events, gap_track, total_event, iteration_no_event = nest_flow_check(EWR_info, i, flow, event, all_events, no_event, 
                                                        all_no_events, gap_track, water_years, total_event, flow_date, flow_percent_change, iteration_no_event)

                else:
                    # only add an extra no_event count if this iteration_no_event = 0
                    if iteration_no_event == 0: 
                        no_event += 1
                    
            # at end of water year record duration and min event values
            if water_years[i] != water_years[i+1]:
                durations.append(EWR_info['duration'])
                min_events.append(EWR_info['min_event'])
    
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    # reset all variable to last flow
    flow_date = dates[-1].to_timestamp().date()
    flow_percent_change = calc_flow_percent_change(-1, flows)
    trigger_day = date(dates[-1].year,EWR_info["trigger_month"], EWR_info["trigger_day"])
    cut_date = calc_nest_cut_date(EWR_info, -1, dates)
    is_in_trigger_window = dates[-1].to_timestamp().date() >= trigger_day - timedelta(days=7) \
    and dates[-1].to_timestamp().date() <= trigger_day + timedelta(days=7)
    iteration_no_event = 0

    if total_event > 0:

        if (flow_date >= trigger_day - timedelta(days=7)) \
            and (flow_date <= cut_date):
            event, all_events, no_event, all_no_events, gap_track, total_event, iteration_no_event = nest_flow_check(EWR_info, -1, flows[-1], event, all_events, no_event, 
                                                            all_no_events, gap_track, water_years, total_event, flow_date, flow_percent_change, iteration_no_event)
        else:
            no_event += 1

    if total_event == 0:
        if is_in_trigger_window and iteration_no_event == 0:
            event, all_events, no_event, all_no_events, gap_track, total_event, iteration_no_event = nest_flow_check(EWR_info, i, flow, event, all_events, no_event, 
                                                all_no_events, gap_track, water_years, total_event, flow_date, flow_percent_change, iteration_no_event)
        else:
        # only add an extra no_event count if this iteration_no_event = 0
            if iteration_no_event == 0:
                no_event += 1

    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        if no_event - total_event > 0 and (len(event) >= EWR_info['min_event']):
            ne_water_year = which_water_year_no_event(i, total_event, water_years)
            all_no_events[ne_water_year].append([no_event-total_event])
            no_event = 0
        total_event = 0
        
    if no_event > 0:
        all_no_events[water_years[-1]].append([no_event])
    durations.append(EWR_info['duration'])
    min_events.append(EWR_info['min_event'])
    
    return all_events, all_no_events, durations, min_events
       

def weirpool_calc(EWR_info: Dict, flows: List, levels: List, water_years: List, weirpool_type: str, 
                        dates:List, masked_dates:List)-> tuple:
    """ Iterates each yearly flows to manage calculation of weirpool EWR. It delegates to weirpool_check function
    the record of events which will check the flow and level threshold as all as drawdown of event 

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        flows (List):  List with all the flows measurements for the current calculated EWR
        levels (List): List with all the levels measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        weirpool_type (str): type of weirpool either 'raising' or 'falling'
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events
    """
    # Declare variables:
    event = []
    total_event = 0
    no_event = 0
    all_events = construct_event_dict(water_years)
    all_no_events = construct_event_dict(water_years)
    durations, min_events = [], []
    gap_track = 0
    # Iterate over flow timeseries, sending to the weirpool_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            level_change = levels[i-1]-levels[i] if i > 0 else 0
            event, all_events, no_event, all_no_events, gap_track, total_event = weirpool_check(EWR_info, i, flow, levels[i], event,
                                                                                all_events, no_event, all_no_events, gap_track, 
                                                                                water_years, total_event, flow_date, weirpool_type, level_change)
        else:
            no_event += 1
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
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
        flow_date = dates[-1]
        level_change = levels[-2]-levels[-1] if i > 0 else 0
        event, all_events, no_event, all_no_events, gap_track, total_event = weirpool_check(EWR_info, -1, flows[-1], levels[-1], event,
                                                                                all_events, no_event, all_no_events, gap_track, 
                                                                              water_years, total_event, flow_date, weirpool_type, level_change)   
    if len(event) > 0:
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

def flow_calc_anytime_sim(EWR_info1: dict, EWR_info2: dict, flows1: np.array, flows2: np.array, water_years: np.array, dates: np.array) -> tuple:
    '''For calculating flow EWRs with no time constraint within their requirements. Events crossing
    water year boundaries will be saved to the water year where the majority of event days were. 
    These EWRs need to be met simultaneously with EWRs at partner sites.

    Args:
        EWR_info1 (Dict): dictionary with the parameter info of the EWR being calculated (site 1)
        EWR_info2 (Dict): dictionary with the parameter info of the EWR being calculated (site 2)
        flows1 (List):  List with all the flows measurements for the current calculated EWR (site 1)
        flows2 (List):  List with all the flows measurements for the current calculated EWR (site 2)
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events

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
        flow_date = dates[i]
        # Each iteration send to a simultaneous flow check function, to see if both sites requirements are met:
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                               EWR_info2, water_years,
                                                                               flow, flows2[i], event,
                                                                               all_events, no_event,
                                                                               all_no_events,gap_track,
                                                                                           total_event, flow_date)
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info1['duration'])
            min_events.append(EWR_info1['min_event'])
    # Check final iteration:
    flow_date = dates[-1]
    event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                           EWR_info2, water_years,
                                                                           flows1[-1], flows2[-1], event,
                                                                           all_events, no_event,
                                                                           all_no_events,gap_track,
                                                                          total_event, flow_date)           
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
    
def flow_calc_sim(EWR_info1: dict, EWR_info2: dict, flows1: np.array, flows2: np.array, water_years: np.array, dates: np.array, masked_dates: set):
    '''For calculating flow EWRs with a time constraint within their requirements. Events are
    therefore reset at the end of each water year. These EWRs need to be met simultaneously 
    with EWRs at partner sites.

    Args:
        EWR_info1 (Dict): dictionary with the parameter info of the EWR being calculated (site 1)
        EWR_info2 (Dict): dictionary with the parameter info of the EWR being calculated (site 2)
        flows1 (List):  List with all the flows measurements for the current calculated EWR (site 1)
        flows2 (List):  List with all the flows measurements for the current calculated EWR (site 2)
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events

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
            flow_date = dates[i]
            # Each iteration send to a simultaneous flow check function, to see if both sites requirements are met:
            event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,EWR_info2,
                                                                                   water_years, flow,
                                                                                   flows2[i],event,
                                                                                   all_events, no_event,
                                                                                   all_no_events, gap_track,
                                                                                               total_event, flow_date)
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
        flow_date = dates[-1]
        event, all_events, no_event, all_no_events, gap_track, total_event = flow_check_sim(i,EWR_info1,
                                                                               EWR_info2, water_years,
                                                                               flows1[-1], flows2[-1], event,
                                                                               all_events, no_event,
                                                                               all_no_events,gap_track,
                                                                                           total_event,flow_date)
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

def lowflow_calc_sim(EWR_info1: dict, EWR_info2: dict, flows1: np.array, flows2: np.array, water_years: np.array, climates: np.array, dates: np.array, masked_dates: set) -> set:
    '''For calculating low flow ewrs. These have no consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of 
    each water year. These EWRs need to be met simultaneously with EWRs at partner sites.

    Args:
        EWR_info1 (Dict): dictionary with the parameter info of the EWR being calculated (site 1)
        EWR_info2 (Dict): dictionary with the parameter info of the EWR being calculated (site 2)
        flows1 (List):  List with all the flows measurements for the current calculated EWR (site 1)
        flows2 (List):  List with all the flows measurements for the current calculated EWR (site 2)
        water_years (List): List of the water year of each day of the current calculated EWR
        climates (np.array): array of daily climate values
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events

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
            flow_date = dates[i]
            # Check flows at each site against their respective EWR requirements:
            event1, all_events1, no_event1, all_no_events1 = lowflow_check(EWR_info1, i, flow, event1, all_events1, no_event1, all_no_events1, water_years, flow_date)
            event2, all_events2, no_event2, all_no_events2 = lowflow_check(EWR_info2, i, flows2[i], event2, all_events2, no_event2, all_no_events2, water_years, flow_date)
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
        flow_date = dates[-1]
        event1, all_events1, no_event1, all_no_events1 = lowflow_check(EWR_info1, -1, flows1[-1], event1, all_events1, no_event1, all_no_events1, water_years, flow_date)
        event2, all_events2, no_event2, all_no_events2 = lowflow_check(EWR_info2, -1, flows2[-1], event2, all_events2, no_event2, all_no_events2, water_years, flow_date)
    if len(event1) > 0:
        all_events1[water_years[-1]].append(event1)
        if no_event1 > 0:
            ne_water_year = which_water_year_no_event(-1, len(event1), water_years)
            all_no_events1[ne_water_year].append([no_event1])
        no_event1 = 0
    if len(event2) > 0:
        all_events2[water_years[-1]].append(event2)
        if no_event2 > 0:
            ne_water_year = which_water_year_no_event(-1, len(event2), water_years)
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

def ctf_calc_sim(EWR_info1: dict, EWR_info2: dict, flows1: np.array, flows2: np.array, water_years: np.array, climates: np.array, dates: np.array, masked_dates: set) -> tuple:
    '''For calculating cease to flow ewrs. These have a consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of each
    water year. These EWRs need to be met simultaneously with EWRs at partner sites.

    Args:
        EWR_info1 (Dict): dictionary with the parameter info of the EWR being calculated (site 1)
        EWR_info2 (Dict): dictionary with the parameter info of the EWR being calculated (site 2)
        flows1 (List):  List with all the flows measurements for the current calculated EWR (site 1)
        flows2 (List):  List with all the flows measurements for the current calculated EWR (site 2)
        water_years (List): List of the water year of each day of the current calculated EWR
        climates (np.array): array of daily climate values
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, all_no_events, durations and min_events

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
            flow_date = dates[i]
            event1, all_events1, no_event1, all_no_events1 = ctf_check(EWR_info1, i, flow, event1, all_events1, no_event1, all_no_events1, water_years, flow_date)
            event2, all_events2, no_event2, all_no_events2 = ctf_check(EWR_info2, i, flows2[i], event2, all_events2, no_event2, all_no_events2, water_years, flow_date)
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
        flow_date = dates[-1]
        event1, all_events1, no_event1, all_no_events1 = ctf_check(EWR_info1, -1, flows1[-1], event1, all_events1, no_event1, all_no_events1, water_years, flow_date)
        event2, all_events2, no_event2, all_no_events2 = ctf_check(EWR_info2, -1, flows2[-1], event2, all_events2, no_event2, all_no_events2, water_years, flow_date)  
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

def check_trigger(EWR_info: dict, flow: float, event: list, gap_track: int, trigger: bool, total_event: int) -> tuple:
    '''Checks daily flow against ewr threshold requirement.
    Saves events to the relevant water year in the event tracking dictionary.
    returns the event list, event dictionary, and time between events

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flow (float): Flow of current day
        event (list): Current state of the event list
        gap_track (int): Current state of tracker for gap between event days
        trigger (bool): Current state of the trigger
        total_event (int): Current state of the total event

    Results:
        tuple: Updated instances of any ongoing events, event gaps, and event triggers

    '''

    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        event.append(flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
        if len(event) >= EWR_info['min_event']:
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

def flow_calc_post_req(EWR_info1: dict, EWR_info2: dict, flows: np.array, water_years: np.array, dates: np.array, masked_dates: set) -> tuple:
    ''' For flow EWRs with a main requirement, and a secondary requirement which needs to be 
    satisfied immediately after the main requirement. Currently only two EWRs have this requirement,
    gauge 409025 OB2_S and OB2_P

    Args:
        EWR_info1 (dict): dictionary with the parameter info of the EWR being calculated (Part 1 of EWR)
        EWR_info2 (dict): dictionary with the parameter info of the EWR being calculated (Part 2 of EWR)
        flows (np.array): List with all the flows measurements for the current calculated EWR
        water_years (np.array): Daily sequence of water year values
        dates (np.array): Daily sequence of date values
        masked_dates (set): Set of dates within the required timing window

    Results:
        tuple: final output with the calculation of all_events, all_no_events, durations and min_events
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
                event, gap_track, trigger, total_event = check_trigger(EWR_info1, flow, event, gap_track, trigger, total_event)
            elif ((trigger == True) and (post_trigger == False)):
                # Send to check the post requirement
                post_event, gap_track, post_trigger, total_event = check_trigger(EWR_info2, flow, post_event, gap_track, post_trigger, total_event)
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

def flow_calc_outside_req(EWR_info1: dict, EWR_info2: dict, flows: np.array, water_years: np.array, dates: np.array, masked_dates: set) -> tuple:
    ''' For flow EWRs with a main requirement, and a secondary requirement which can either be 
    satisfied immediately after the main requirement, or immediately before. 
    Currently only two EWRs have this requirement, gauge 409025 OB3_S and OB3_P

    Args:
        EWR_info1 (dict): dictionary with the parameter info of the EWR being calculated (Part 1 of EWR)
        EWR_info2 (dict): dictionary with the parameter info of the EWR being calculated (Part 2 of EWR)
        flows (np.array): List with all the flows measurements for the current calculated EWR
        water_years (np.array): Daily sequence of water year values
        dates (np.array): Daily sequence of date values
        masked_dates (set): Set of dates within the required timing window

    Results:
        tuple: final output with the calculation of all_events, all_no_events, durations and min_events

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
                event, gap_track, trigger, total_event = check_trigger(EWR_info1, flow, event, gap_track, trigger, total_event)
            elif trigger == True: # Event registered, now check for event pre/post this
                gap_track = EWR_info1['gap_tolerance']
                # First check if there was an event before the main event:
                total_event_pre = total_event
                for pre_i, pre_flow in enumerate(reversed(flows[:(i-len(event))])):  
                    pre_event, gap_track, pre_trigger, total_event_pre = check_trigger(EWR_info2, pre_flow, pre_event, gap_track, pre_trigger, total_event_pre)
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
                        post_event, gap_track, post_trigger, total_event_post = check_trigger(EWR_info2, post_flow, post_event, gap_track, post_trigger, total_event_post)
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

def filter_min_events(EWR_info:Dict, events:Dict)-> Dict:
    """Given an events dictionary, filter out all events that are 
    below min_event as they should not contribute to total duration.

    Args:
        EWR_info (Dict): EWR parameters
        events (Dict): EWR calculation events dictionary

    Returns:
        Dict: events dictionary with only events that is above minimum
    """
    filtered_events = {}
    for year, evts in events.items():
        filtered_events[year] = [e for e in evts if len(e) >= EWR_info["min_event"] ] 

    return filtered_events

def get_event_years(EWR_info:Dict, events:Dict, unique_water_years:set, durations:List) -> List:
    '''Returns a list of years with events (represented by a 1), and years without events (0)
    
    Args:
        EWR_info (Dict): EWR parameters
        events (Dict): Dictionary with water years as keys, and a list of event lists for values.
        unique_water_years (set): Set of unique water years in timeseries
        durations (List): List of durations - 1 value per year

    Results:
        list: A list of years with events (represented by a 1), and years without events (0)
    
    '''
    events_filtered = filter_min_events(EWR_info, events)
    event_years = []
    for index, year in enumerate(unique_water_years):
        combined_len = 0
        for e in events_filtered[year]:
            combined_len += len(e)
        if ((combined_len >= durations[index] and len(events_filtered[year])>=EWR_info['events_per_year'])):
            event_years.append(1)
        else:
            event_years.append(0)
    
    return event_years


def get_achievements(EWR_info:Dict, events:Dict, unique_water_years:set, durations:List) -> List:
    '''Returns a list of number of events per year.
    
    Args:
        EWR_info (Dict): EWR parameters
        events (Dict): Dictionary with water years as keys, and a list of event lists for values.
        unique_water_years (set): Set of unique water years in timeseries
        durations (List): List of durations - 1 value per year

    Results:
        list: A list of years with the number of times the EWR requirements were achieved
    '''
    events_filtered = filter_min_events(EWR_info, events)
    num_events = []
    for index, year in enumerate(unique_water_years):
        combined_len = 0
        yearly_events = 0
        for e in events_filtered[year]:
            combined_len += len(e)
            if combined_len >= durations[index]:
                yearly_events += 1
                combined_len = 0
        total = yearly_events/EWR_info['events_per_year']
        num_events.append(int(total))
    
    return num_events

def get_number_events(EWR_info:Dict, events:Dict, unique_water_years:set, durations:List) -> List:
    '''Returns a list of number of events per year
    
    Args:
        EWR_info (Dict): EWR parameters
        events (Dict): Dictionary with water years as keys, and a list of event lists for values.
        unique_water_years (set): Set of unique water years in timeseries
        durations (List): List of durations - 1 value per year
    
    Results:
        list: A list of years with the number of events achieved throughout the year

    '''
    events_filtered = filter_min_events(EWR_info, events)
    num_events = []
    for index, year in enumerate(unique_water_years):
        combined_len = 0
        yearly_events = 0
        for e in events_filtered[year]:
            combined_len += len(e)
            if combined_len >= durations[index]:
                yearly_events += 1
                combined_len = 0
        total = yearly_events
        num_events.append(int(total))
    
    return num_events

def get_average_event_length(events:Dict, unique_water_years:set) -> List:
    '''Returns a list of average event length per year
    
    Args:
        events (Dict): Dictionary with water years as keys, and a list of event lists for values.
        unique_water_years (set): Set of unique water years in timeseries
    
    Results:
        list: A list with the average length of the events for each year
    '''
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

def get_total_days(events:Dict, unique_water_years:set) -> List:
    '''Returns a list with total event days per year
    
    Args:
        events (Dict): Dictionary with water years as keys, and a list of event lists for values.
        unique_water_years (set): Set of unique water years in timeseries

    Results:
        list: A list of total event days per year

    '''
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

def get_max_event_days(events:Dict, unique_water_years:set)-> List:
    """Given the events in the yearly time series calculates what was the event
    in each year with the maximum number of days and appends to a list.

    Args:
        events (dict): Dict of lists with events flow/level
        unique_water_years (set): Set of unique water years for the events 

    Returns:
        list: List with the max event days per year
    """
    max_events = []
    for year in unique_water_years:
        events_length = [len(e) for e in events[year]]
        max_event =  max(events_length) if events_length else 0
        max_events.append(max_event)
    return max_events

def get_max_volume(events:Dict, unique_water_years:set)-> List:
    """Given the events in the yearly time series calculates what was the maximum 
    volume achieved in the year and appends to a list.

    Args:
        events (dict): Dict of lists with events flow/level
        unique_water_years (set): Set of unique water years for the events 

    Returns:
        list: List with the max volume days per year
    """
    max_volumes = []
    for year in unique_water_years:
        max_volume = 0
        year_events = events[year]
        for event in year_events:
            if event:
                volumes = [vol for _ , vol in event]
                event_max_vol = max(volumes)
                if event_max_vol > max_volume:
                    max_volume = event_max_vol
        max_volumes.append(max_volume)
    return max_volumes

def get_max_inter_event_days(no_events:Dict, unique_water_years:set)-> List:
    """Given event gaps in a all no_event dict. return the maximum
    gap each year

    Args:
        no_events (dict): Dict of lists with no_events gap
        unique_water_years (set): Set of unique water years for the events 

    Returns:
        list: List with the max inter event gap
    """
    max_inter_event_gaps = []
    for year in unique_water_years:
        max_inter_event = 0
        year_events = no_events[year]
        for event in year_events:
            if event:
                if event[0] > max_inter_event:
                    max_inter_event = event[0]
        max_inter_event_gaps.append(max_inter_event)
    return max_inter_event_gaps


def lengths_to_years(events:List)-> defaultdict:
    """iterates through the events_list_info and returns a dictionary
    with all the events length to each year. It handles events that crosses
    year boundary and assign to the year a rolling sum of event days from the
    event start up to the end of the year boundary

    Args:
        events (list): events list info

    Returns:
        defaultdict: all events length for each water year
    """
    years_event_lengths = defaultdict(list)
    for event in events:
        start, _, length, wys = event
        if len(wys) == 1:
            years_event_lengths[wys[0]].append(length)
        else:
            for wy in wys[:-1]:
                up_to_boundary = (date(wy+1,6,30) - start).days + 1
                years_event_lengths[wy].append(up_to_boundary)
            # last water year of the collection is always the total event length
            years_event_lengths[wys[-1]].append(length)        
    return years_event_lengths

def get_max_consecutive_event_days(events:Dict, unique_water_years:set)-> List:
    """Given gauge events it calculates the max rolling event duration
    at the end of each water year. If an event goes beyond an year it will count the 
    days from the start of the event up to the last day that of the boundary cross i.e June 30th.

    Args:
        events (dict): Dict of lists with events flow/level
        unique_water_years (set): Set of unique water years for the events 

    Returns:
        list: List with the max event days per year
    """
    events_list = return_events_list_info(events)
    water_year_maxs = lengths_to_years(events_list)
    max_consecutive_events = []
    for year in unique_water_years:
        maximum_event_length = max(water_year_maxs.get(year)) if water_year_maxs.get(year) else 0
        max_consecutive_events.append(maximum_event_length)

    return max_consecutive_events

def get_event_years_max_rolling_days(events:Dict , unique_water_years:List[int]):
    '''Returns a list of years with events (represented by a 1), where the max rolling duration passes the
    test of ANY duration'''
    try:
        max_consecutive_days = get_max_consecutive_event_days(events, unique_water_years)
    except Exception as e:
        max_consecutive_days = [0]*len(unique_water_years)
        print(e)
    return [1 if (max_rolling > 0) else 0 for max_rolling in max_consecutive_days]

def get_event_years_volume_achieved(events:Dict , unique_water_years:List[int])->List:
    """Returns a list of years with events (represented by a 1), where the volume threshold was 
    achieved

    Args:
        events (Dict): events dictionary
        unique_water_years (List[int]): unique water years for the time series

    Returns:
        List: List of 1 and 0. 1 achieved and 0 not achieved
    """
    try:
        max_volumes = get_max_volume(events, unique_water_years)
    except Exception as e:
        max_volumes = [0]*len(unique_water_years)
        print(e)
    return [1 if (max_vol > 0) else 0 for max_vol in max_volumes]

def get_event_max_inter_event_achieved(EWR_info:Dict, no_events:Dict , unique_water_years:List[int])->List:
    """Returns a list of years where the event gap was achieved (represented by a 1), 
    Args:
        no_events (Dict): no_events dictionary
        unique_water_years (List[int]): unique water years for the time series

    Returns:
        List: List of 1 and 0. 1 achieved and 0 not achieved
    """
    try:
        max_inter_event_achieved = get_max_inter_event_days(no_events, unique_water_years)
    except Exception as e:
        max_inter_event_achieved = [0]*len(unique_water_years)
        print(e)
    return [0 if (max_inter_event > EWR_info['max_inter-event']*365) else 1 for max_inter_event in max_inter_event_achieved]

def get_max_rolling_duration_achievement(durations:List[int], max_consecutive_days:List[int])-> List[int]:
    """test if in a given year the max rolling duration was equals or above the min duration.

    Args:
        durations (List[int]):  minimum days in a year to meet the requirement
        max_consecutive_days (List[int]): max rolling duration

    Returns:
        List[int]: a list of 1 and 0 where 1 is achievement and 0 is no achievement.
    """
    return [1 if (max_rolling >= durations[index]) else 0 for index, max_rolling in enumerate(max_consecutive_days)]

def get_all_events(yearly_events:dict)-> List:
    """count the events in a collection of years

    Args:
        yearly_events (dict): ewr yearly events dictionary of lists of lists

    Returns:
        List: total event count per year in order
    """
    return [len(yearly_events[year]) for year in sorted(yearly_events.keys())]

def get_all_event_days(yearly_events:dict)-> List:
    """count the events in a collection of years

    Args:
        yearly_events (dict): ewr yearly events dictionary of lists of lists

    Returns:
        List: total event days count per year in order
    """
    return [len(list(chain(*yearly_events[year]))) for year in sorted(yearly_events.keys())]

def get_achieved_event_days(EWR_info:Dict, yearly_events:dict)-> List:
    """count the events days in a collection of years. Filter events below min_event

    Args:
        yearly_events (dict): ewr yearly events dictionary of lists of lists

    Returns:
        List: total event days count per year in order
    """
    filtered_events = filter_min_events(EWR_info, yearly_events)
    return [len(list(chain(*filtered_events[year]))) for year in sorted(filtered_events.keys())]

def get_average_event_length_achieved(EWR_info:Dict, events:Dict)-> List:
    '''Returns a list of average event length per year of achieved events'''
    filtered_events = filter_min_events(EWR_info, events)
    events_length = [[float(len(event)) for event in filtered_events[year]] for year in sorted(filtered_events.keys())]
    year_average_lengths = [sum(year) / len(year) if len(year) != 0 else float(0) for year in events_length]
    return year_average_lengths

def get_data_gap(input_df:pd.DataFrame, water_years:List, gauge:str) -> List:
    '''Input a dataframe, 
    calculate how much missing data there is, 
    send yearly results back.

    Args:
        input_df (pd.DataFrame): 
        water_years (List):
        gauge (str): 

    Results:
        list: List of missing days. 1 value per water year.

    '''
    temp_df = input_df.copy(deep=True)
    masked = ~temp_df.notna()
    masked['water year'] = water_years
    group_df = masked.groupby('water year').sum()
    
    return list(group_df[gauge].values)

def get_total_series_days(water_years:List) -> pd.Series:
    '''Input a series of missing days and a possible maximum days,
    returns the percentage of data available for each year.

    Args:
        water_years (List): List of daily water year values
        
    Results:
        pd.Series: index - unique water years, col - number of daily occurences

    '''
    unique, counts = np.unique(water_years, return_counts=True)
    intoSeries = pd.Series(index=unique, data=counts)
    
    return intoSeries

def event_years_sim(events1:List, events2:List) -> np.array:
    '''add the event lists, event only occurs when both are met.
    
    Args:
        events1 (List): Annual years with events - (site 1)
        events2 (List): Annual years with events - (site 2)

    Results:
        np.array: returns combined years with events array
    
    '''
    added = np.array(list(events1)) + np.array(list(events2))
    mask = added == 2
    results = mask*1
    return results

def get_achievements_sim(events1:List, events2:List) -> List:
    '''get the minimum number of events for simultaneous EWRs.
    
    Args:
        events1 (List): Annual EWR achievements - (site 1)
        events2 (List): Annual EWR achievements - (site 2)

    Results:
        List: Annual achievements (minimum from each year chosen)
    
    '''
    e1 = np.array(list(events1))
    e2 = np.array(list(events2))
    results = []
    for i, event in enumerate(e1):
        results.append(min([event, e2[i]]))
    return results

def get_number_events_sim(events1:List, events2:List) -> List:
    '''get the minimum number of events for simultaneous EWRs.
    
    Args:
        events1 (List): Annual number of events - (site 1)
        events2 (List): Annual number of events - (site 2)

    Results:
        List: Annual events (minimum from each year chosen)    
    
    '''
    e1 = np.array(list(events1))
    e2 = np.array(list(events2))
    results = []
    for i, event in enumerate(e1):
        results.append(min([event, e2[i]]))
    return results

def average_event_length_sim(events1:List, events2:List) -> List:
    '''get the average event length between two series.
    
    Args:
        events1 (List): Annual average events lengths - (site 1)
        events2 (List): Annual average events lengths - (site 2)

    Results:
        List: Annual average event length    
    
    '''

    e1 = np.array([list(events1)])
    e2 = np.array([list(events2)])
    average = (e1 + e2)/2
    return average[0]

def event_stats(df:pd.DataFrame, PU_df:pd.DataFrame, gauge:str, EWR:str, EWR_info:Dict, events:Dict, no_events:Dict, durations:List, water_years:List) -> pd.DataFrame:
    ''' Produces statistics based on the event dictionaries and event gap dictionaries.
    
    Args:
        df (pd.DataFrame): Raw flow/level dataframe
        PU_df (pd.DataFrame): Dataframe with the results from the EWRs in the current planning unit
        gauge (str): current iteration gauge string
        EWR (str): current iteration EWR string
        EWR_info (Dict): Parameter information for current EWR
        events (Dict): Detailed event information 
        no_events (Dict): Detailed event gap information
        durations (List): List of annual required durations
        water_years (List): Daily water year values

    Results:
        pd.DataFrame: Updated results dataframe for this current planning unit

    
    '''
    unique_water_years = set(water_years)
    # Years with events
    years_with_events = get_event_years(EWR_info, events, unique_water_years, durations)
    
    if EWR_info['EWR_code'] in ['CF1_c','CF1_C']:
        years_with_events = get_event_years_max_rolling_days(events, unique_water_years)

    if EWR_info['flow_level_volume'] == 'V':
        years_with_events = get_event_years_volume_achieved(events, unique_water_years)

    YWE = pd.Series(name = str(EWR + '_eventYears'), data = years_with_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, YWE], axis = 1)
    # Number of event achievements:
    num_event_achievements = get_achievements(EWR_info, events, unique_water_years, durations)
    NEA = pd.Series(name = str(EWR + '_numAchieved'), data= num_event_achievements, index = unique_water_years)
    PU_df = pd.concat([PU_df, NEA], axis = 1)
    # Total number of events THIS ONE IS ONLY ACHIEVED due to Filter Applied
    num_events = get_number_events(EWR_info, events, unique_water_years, durations)
    NE = pd.Series(name = str(EWR + '_numEvents'), data= num_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, NE], axis = 1)
    # Total number of events THIS ONE IS ALL EVENTS
    num_events_all = get_all_events(events)
    NEALL = pd.Series(name = str(EWR + '_numEventsAll'), data= num_events_all, index = unique_water_years)
    PU_df = pd.concat([PU_df, NEALL], axis = 1)
    # Max inter event period
    max_inter_period = get_max_inter_event_days(no_events, unique_water_years)
    MIP = pd.Series(name = str(EWR + '_maxInterEventDays'), data= max_inter_period, index = unique_water_years)
    PU_df = pd.concat([PU_df, MIP], axis = 1)
    # Max inter event period achieved
    max_inter_period_achieved = get_event_max_inter_event_achieved(EWR_info, no_events, unique_water_years)
    MIPA = pd.Series(name = str(EWR + '_maxInterEventDaysAchieved'), data= max_inter_period_achieved, index = unique_water_years)
    PU_df = pd.concat([PU_df, MIPA], axis = 1)
    # Average length of events
    av_length = get_average_event_length(events, unique_water_years)
    AL = pd.Series(name = str(EWR + '_eventLength'), data = av_length, index = unique_water_years)
    PU_df = pd.concat([PU_df, AL], axis = 1)
    # Average length of events ONLY the ACHIEVED
    av_length_achieved = get_average_event_length_achieved(EWR_info, events)
    ALA = pd.Series(name = str(EWR + '_eventLengthAchieved' ), data = av_length_achieved, index = unique_water_years)
    PU_df = pd.concat([PU_df, ALA], axis = 1)
    # Total event days
    total_days = get_total_days(events, unique_water_years)
    TD = pd.Series(name = str(EWR + '_totalEventDays'), data = total_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, TD], axis = 1)
    # Total event days ACHIEVED
    total_days_achieved = get_achieved_event_days(EWR_info, events)
    TDA = pd.Series(name = str(EWR + '_totalEventDaysAchieved'), data = total_days_achieved, index = unique_water_years)
    PU_df = pd.concat([PU_df, TDA], axis = 1)
    # Max event days
    max_days = get_max_event_days(events, unique_water_years)
    MD = pd.Series(name = str(EWR + '_maxEventDays'), data = max_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, MD], axis = 1)
    # Max rolling consecutive event days
    try:
        max_consecutive_days = get_max_consecutive_event_days(events, unique_water_years)
        MR = pd.Series(name = str(EWR + '_maxRollingEvents'), data = max_consecutive_days, index = unique_water_years)
        PU_df = pd.concat([PU_df, MR], axis = 1)
    except Exception as e:
        max_consecutive_days = [0]*len(unique_water_years)
        MR = pd.Series(name = str(EWR + '_maxRollingEvents'), data = max_consecutive_days, index = unique_water_years)
        PU_df = pd.concat([PU_df, MR], axis = 1)
        print(e)
    # Max rolling duration achieved
    achieved_max_rolling_duration = get_max_rolling_duration_achievement(durations, max_consecutive_days)
    MRA = pd.Series(name = str(EWR + '_maxRollingAchievement'), data = achieved_max_rolling_duration, index = unique_water_years)
    PU_df = pd.concat([PU_df, MRA], axis = 1)
    # Append information around available and missing data:
    yearly_gap = get_data_gap(df, water_years, gauge)
    total_days = get_total_series_days(water_years)
    YG = pd.Series(name = str(EWR + '_missingDays'), data = yearly_gap, index = unique_water_years)
    TD = pd.Series(name = str(EWR + '_totalPossibleDays'), data = total_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, YG], axis = 1)
    PU_df = pd.concat([PU_df, TD], axis = 1)
    
    return PU_df

def event_stats_sim(df:pd.DataFrame, PU_df:pd.DataFrame, gauge:str, EWR:str, EWR_info:Dict, events1:Dict, events2:Dict, no_events1:Dict, durations:List, water_years:List) -> pd.DataFrame:
    ''' Produces statistics based on the event dictionaries and event gap dictionaries for simultaneous EWRs
    
    Args:
        df (pd.DataFrame): Raw flow/level dataframe
        PU_df (pd.DataFrame): Dataframe with the results from the EWRs in the current planning unit
        gauge (str): current iteration gauge string
        EWR (str): current iteration EWR string
        EWR_info (Dict): Parameter information for current EWR
        events1 (Dict): Detailed event information (site 1)
        events2 (Dict): Detailed event information (site 2)
        no_events1 (Dict): Detailed event gap information (site 1)
        durations (List): List of annual required durations
        water_years (List): Daily water year values

    Results:
        pd.DataFrame: Updated results dataframe for this current planning unit
    
    '''
    unique_water_years = set(water_years)
    # Years with events
    years_with_events1 = get_event_years(EWR_info, events1, unique_water_years, durations)
    years_with_events2 = get_event_years(EWR_info, events2, unique_water_years, durations)
    years_with_events = event_years_sim(years_with_events1, years_with_events2)
    YWE = pd.Series(name = str(EWR + '_eventYears'), data = years_with_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, YWE], axis = 1)
    # Number of event achievements per year
    num_events_ach_1 = get_achievements(EWR_info, events1, unique_water_years, durations)
    num_events_ach_2 = get_achievements(EWR_info, events2, unique_water_years, durations)
    num_events_ach = get_achievements_sim(num_events_ach_1, num_events_ach_2)
    NEA = pd.Series(name = str(EWR + '_numAchieved'), data= num_events_ach, index = unique_water_years)
    PU_df = pd.concat([PU_df, NEA], axis = 1)
    # Total number of event per year
    num_events1 = get_number_events(EWR_info, events1, unique_water_years, durations)
    num_events2 = get_number_events(EWR_info, events2, unique_water_years, durations)
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
    # Append information around available and missing data:
    yearly_gap = get_data_gap(df, water_years, gauge) # Only adding data gap for main gauge
    total_days = get_total_series_days(water_years)
    YG = pd.Series(name = str(EWR + '_missingDays'), data = yearly_gap, index = unique_water_years)
    TD = pd.Series(name = str(EWR + '_totalPossibleDays'), data = total_days, index = unique_water_years)
    PU_df = pd.concat([PU_df, YG], axis = 1)
    PU_df = pd.concat([PU_df, TD], axis = 1)
    
    return PU_df

#-----------------------------Post processing-----------------------------------------------------#

def merge_weirpool_with_freshes(wp_freshes:List, PU_df:pd.DataFrame)-> pd.DataFrame:
    """Perform the post processing of WP2 and WP3 with equivalent freshes

    Args:
        wp_freshes (List): freshed that need to be proceed to produced merged EWR
        PU_df (pd.DataFrame): Dataframe with all the statistics so for this calculation run

    Returns:
        pd.DataFrame: Return Dataframe with the statistics of the merged EWR
    """
    
    weirpool_pair = {'SF_WP':'WP3',
                      'LF2_WP': 'WP4' }
    # iterate wp_freshes
    for fresh in wp_freshes:
        fresh_event_year = PU_df[f"{fresh}_eventYears"].values
        try:    
            weirpool_event_year = PU_df[f"{weirpool_pair[fresh]}_eventYears"].values
            merged_ewr_event_year = np.maximum(fresh_event_year, weirpool_event_year)
            # add merged results
            PU_df[f"{fresh}/{weirpool_pair[fresh]}_eventYears"] = merged_ewr_event_year
            # add remaninig columns with N/As
            column_attributes = list(set([col.split("_")[-1] for col in PU_df.columns if "eventYears" not in col ]))
            for col in column_attributes:
                PU_df[f"{fresh}/{weirpool_pair[fresh]}_{col}"] = np.nan
        except KeyError as err:
            print(f"While trying to calculate {fresh}/{weirpool_pair[fresh]} there is missing data for {weirpool_pair[fresh]}")
            continue

    return PU_df


#---------------------------- Sorting and distributing to handling functions ---------------------#

def calc_sorter(df_F:pd.DataFrame, df_L:pd.DataFrame, gauge:str, allowance:Dict, climate:str, EWR_table:pd.DataFrame) -> tuple:
    '''Sends to handling functions to get calculated depending on the type of EWR
    
    Args:
        df_F (pd.DataFrame): Dataframe with the daily flows
        df_L (pd.DataFrame): Dataframe with the daily levels
        gauge (str): gauge string of current iteration
        allowance (Dict): allowance applied to EWR elements
        climate (str): location string for the climate file
        EWR_table (pd.DataFrame): Dataframe of EWR parameters

    Results:
        tuple[dict, dict]: annual results summary and detailed event information
    
    ''' 
    # Get ewr tables:
    PU_items = data_inputs.get_planning_unit_info()
    # menindee_gauges, wp_gauges = data_inputs.get_level_gauges()
    # simultaneous_gauges = data_inputs.get_simultaneous_gauges('all')
    complex_EWRs = data_inputs.get_complex_calcs()
    # Extract relevant sections of the EWR table:
    gauge_table = EWR_table[EWR_table['Gauge'] == gauge]
    # save the planning unit dataframes to this dictionary:
    location_results = {}
    location_events = {}
    for PU in set(gauge_table['PlanningUnitID']):
        PU_table = gauge_table[gauge_table['PlanningUnitID'] == PU]
        EWR_categories = PU_table['FlowLevelVolume'].values
        EWR_codes = PU_table['Code']
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
            EWR_WP = 'WP' in EWR and 'SF' not in EWR and 'LF' not in EWR # added for the WP3 and WP4 dependencies
            EWR_NEST = 'Nest' in EWR
            EWR_CUMUL = 'LF' in EWR or 'OB' in EWR or 'WL' in EWR # Some LF and OB are cumulative
            EWR_LEVEL = 'LLLF' in EWR or 'MLLF' in EWR or 'HLLF' in EWR or 'VHLL' in EWR
            # Determine if its classified as a complex EWR:
            COMPLEX = gauge in complex_EWRs and EWR in complex_EWRs[gauge]
            MULTIGAUGE = is_multigauge(EWR_table, gauge, EWR, PU)
            # SIMULTANEOUS calculations is switched off
            # SIMULTANEOUS = PU in simultaneous_gauges and gauge in simultaneous_gauges[PU]
            SIMULTANEOUS = False
            if COMPLEX:
                print(f"skipping due to not validated calculations for {PU}-{gauge}-{EWR}")
                continue
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
        
        wp_freshes = [ewr for ewr in EWR_codes if ewr in ["SF_WP","LF2_WP"]]
        if wp_freshes:
            PU_df = merge_weirpool_with_freshes(wp_freshes, PU_df)
            

        PU_name = PU_items['PlanningUnitName'].loc[PU_items[PU_items['PlanningUnitID'] == PU].index[0]]
        
        location_results[PU_name] = PU_df
        location_events[PU_name] = PU_events
    return location_results, location_events