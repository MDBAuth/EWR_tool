from collections import defaultdict
from copy import deepcopy
import inspect
from typing import Any, List, Dict
from datetime import date, timedelta
import datetime
import calendar
from itertools import chain
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

from . import data_inputs

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#----------------------------------- Getting EWRs from the database ------------------------------#


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
    
def get_EWRs(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, components: list) -> dict:
    '''Pulls the relevant EWR componenets for each EWR
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge ID
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset
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
        ewrs['min_flow'] = int(min_flow)
    if 'MAXF' in components:
        max_flow = int(component_pull(EWR_table, gauge, PU, EWR, 'FlowThresholdMax'))
        ewrs['max_flow'] = int(max_flow)
    if 'MINL' in components:
        min_level = float(component_pull(EWR_table, gauge, PU, EWR, 'LevelThresholdMin'))
        ewrs['min_level'] = min_level
    if 'MAXL' in components:
        max_level = float(component_pull(EWR_table, gauge, PU, EWR, 'LevelThresholdMax'))
        ewrs['max_level'] = max_level
    if 'MINV' in components:
        min_volume = int(component_pull(EWR_table, gauge, PU, EWR, 'VolumeThreshold'))
        ewrs['min_volume'] = int(min_volume)
    if 'DUR' in components:
        duration = int(component_pull(EWR_table, gauge, PU, EWR, 'Duration'))
        ewrs['duration'] = int(duration)
    if 'GP' in components:
        gap_tolerance = int(component_pull(EWR_table, gauge, PU, EWR, 'WithinEventGapTolerance'))
        ewrs['gap_tolerance'] = gap_tolerance
    if 'EPY' in components:
        events_per_year = int(component_pull(EWR_table, gauge, PU, EWR, 'EventsPerYear'))
        ewrs['events_per_year'] = events_per_year       
    if 'ME' in components:
        min_event = int(component_pull(EWR_table, gauge, PU, EWR, 'MinSpell'))
        ewrs['min_event'] = int(min_event)
    if 'MD' in components:
        max_drawdown = component_pull(EWR_table, gauge, PU, EWR, 'DrawdownRate')
        if '%' in str(max_drawdown):
            value_only = int(max_drawdown.replace('%', ''))
            ewrs['drawdown_rate'] = str(int(value_only))+'%'
        else:
            ewrs['drawdown_rate'] = str(float(max_drawdown)) #TODO check this works
        if max_drawdown == 0:
            # Large value set to ensure that drawdown check is always passed in this case
            ewrs['drawdown_rate'] = int(1000000)          
    if 'WPG' in components:
        weirpool_gauge = component_pull(EWR_table, gauge, PU, EWR, 'WeirpoolGauge')
        ewrs['weirpool_gauge'] =str(weirpool_gauge)
    if 'MG' in components:       
        ewrs['second_gauge'] = get_second_multigauge(EWR_table, gauge, EWR, PU)    
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
            ewrs['drawdown_rate_week'] = str(float(drawdown_rate_week)/100)#TODO check this works
        except ValueError: # In this case set a large number
            ewrs['drawdown_rate_week'] = int(1000000)
    if 'ML' in components:
        max_level = component_pull(EWR_table, gauge, PU, EWR, 'MaxLevelRise')
        ewrs['max_level_raise'] = float(max_level)
    if 'ABF' in components:
        annual_barrage_flow = component_pull(EWR_table, gauge, PU, EWR, 'AnnualBarrageFlow')
        ewrs['annual_barrage_flow'] = int(annual_barrage_flow)
    if 'TYBF' in components:
        three_years_barrage_flow = component_pull(EWR_table, gauge, PU, EWR, 'ThreeYearsBarrageFlow')
        ewrs['three_years_barrage_flow'] = int(three_years_barrage_flow)
    if 'HRWS' in components:
        high_release_window_start = component_pull(EWR_table, gauge, PU, EWR, 'HighReleaseWindowStart')
        ewrs['high_release_window_start'] = int(high_release_window_start)
    if 'HRWE' in components:
        high_release_window_end = component_pull(EWR_table, gauge, PU, EWR, 'HighReleaseWindowEnd')
        ewrs['high_release_window_end'] = int(high_release_window_end)
    if 'LRWS' in components:
        low_release_window_start = component_pull(EWR_table, gauge, PU, EWR, 'LowReleaseWindowStart')
        ewrs['low_release_window_start'] = int(low_release_window_start)
    if 'LRWE' in components:
        low_release_window_end = component_pull(EWR_table, gauge, PU, EWR, 'LowReleaseWindowEnd')
        ewrs['low_release_window_end'] = int(low_release_window_end)
    if 'PLWS' in components:
        peak_level_window_start = component_pull(EWR_table, gauge, PU, EWR, 'PeakLevelWindowStart')
        ewrs['peak_level_window_start'] = int(peak_level_window_start)
    if 'PLWE' in components:
        peak_level_window_end = component_pull(EWR_table, gauge, PU, EWR, 'PeakLevelWindowEnd')
        ewrs['peak_level_window_end'] = int(peak_level_window_end)
    if 'LLWS' in components:
        low_level_window_start = component_pull(EWR_table, gauge, PU, EWR, 'LowLevelWindowStart')
        ewrs['low_level_window_start'] = int(low_level_window_start)
    if 'LLWE' in components:
        low_level_window_end = component_pull(EWR_table, gauge, PU, EWR, 'LowLevelWindowEnd')
        ewrs['low_level_window_end'] = int(low_level_window_end)
    if 'NFS' in components:
        non_flow_spell = component_pull(EWR_table, gauge, PU, EWR, 'NonFlowSpell')
        ewrs['non_flow_spell'] = int(non_flow_spell)
    if 'EDS' in components: 
        non_flow_spell = component_pull(EWR_table, gauge, PU, EWR, 'EggsDaysSpell')
        ewrs['eggs_days_spell'] = int(non_flow_spell)
    if 'LDS' in components: 
        non_flow_spell = component_pull(EWR_table, gauge, PU, EWR, 'LarvaeDaysSpell')
        ewrs['larvae_days_spell'] = int(non_flow_spell)
    if 'MLR' in components: 
        min_level_rise = component_pull(EWR_table, gauge, PU, EWR, 'MinLevelRise')
        ewrs['min_level_rise'] = float(min_level_rise)
    if 'RRM1' in components:
        rate_of_rise_max1 = component_pull(EWR_table, gauge, PU, EWR, 'RateOfRiseMax1')
        ewrs['rate_of_rise_max1'] = float(rate_of_rise_max1)
    if 'RRM2' in components:
        rate_of_rise_max1 = component_pull(EWR_table, gauge, PU, EWR, 'RateOfRiseMax2')
        ewrs['rate_of_rise_max2'] = float(rate_of_rise_max1)
    if 'RFM' in components:
        rate_of_fall_min = component_pull(EWR_table, gauge, PU, EWR, 'RateOfFallMin')
        ewrs['rate_of_fall_min'] = float(rate_of_fall_min)
    if 'RRT1' in components:
        rate_of_rise_threshold1 = component_pull(EWR_table, gauge, PU, EWR, 'RateOfRiseThreshold1')
        ewrs['rate_of_rise_threshold1'] = float(rate_of_rise_threshold1)
    if 'RRT2' in components:
        rate_of_rise_threshold2 = component_pull(EWR_table, gauge, PU, EWR, 'RateOfRiseThreshold2')
        ewrs['rate_of_rise_threshold2'] = float(rate_of_rise_threshold2)
    if 'RRL' in components:
        rate_of_rise_river_level = component_pull(EWR_table, gauge, PU, EWR, 'RateOfRiseRiverLevel')
        ewrs['rate_of_rise_river_level'] = float(rate_of_rise_river_level)
    if 'RFL' in components:
        rate_of_fall_river_level = component_pull(EWR_table, gauge, PU, EWR, 'RateOfFallRiverLevel')
        ewrs['rate_of_fall_river_level'] = float(rate_of_fall_river_level)
    if 'CTFT' in components:
        ctf_threshold = component_pull(EWR_table, gauge, PU, EWR, 'CtfThreshold')
        ewrs['ctf_threshold'] = float(ctf_threshold)

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
    else:
        return True

def calculate_n_day_moving_average(df: pd.DataFrame, days: int) -> pd.DataFrame:
    '''Calculates the n day moving average for a given gauges
    
    Args:
        df (pd.DataFrame): Daily flow data
        gauge (str): Gauge ID
        n (int): Number of days to calculate moving average over
    
    Results:
        pd.DataFrame: Daily flow data with an additional column for the moving average
    
    '''
    gauges = [col for col in df.columns]
    original_df = df[gauges]
    original_df = original_df.expanding(min_periods=1).mean() 
    original_df = original_df[:days-1]


    for gauge in gauges:
        df[gauge] = df[gauge].rolling(window=days).mean()
    df = df[days-1:]

    result_df = pd.concat([original_df, df], sort = False, axis = 0)

    return result_df

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
    """convert the data index of the dataframe to required format

    Args:
        date_index (Any): date index

    Returns:
        datetime.date: return on the correct format
    """
    if type(date_index) == pd._libs.tslibs.timestamps.Timestamp:
        return date_index.date()
    if type(date_index) == pd._libs.tslibs.period.Period:
        return date_index.to_timestamp().date()
    else:
        return date_index

#----------------------------------- EWR handling functions --------------------------------------#

def ctf_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling Cease to flow type EWRs
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 

    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('cease to flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow data against EWR requirements and then perform analysis on the results:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, D = ctf_calc_anytime(EWR_info, df_F[gauge].values, water_years, df_F.index)
    else:
        E, D = ctf_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def lowflow_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling low flow type EWRs (Very low flows and baseflows)
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('low flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow data against EWR requirements and then perform analysis on the results:
    E, D = lowflow_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def flow_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling non low flow based flow EWRs (freshes, bankfulls, overbanks)
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow data against EWR requirements and then perform analysis on the results:
    E, D = flow_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def flow_handle_anytime(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling flow based flow EWRs (freshes, bankfulls, overbanks) to allow flows to continue to record
    if it crosses water year boundaries.
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration


    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow data against EWR requirements and then perform analysis on the results
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E,  D = flow_calc_anytime(EWR_info, df_F[gauge].values, water_years, df_F.index)
    else:
        E,  D = flow_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def flow_handle_check_ctf(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling non low flow based flow EWRs 
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information 
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('flow-ctf')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    # Check flow data against EWR requirements and then perform analysis on the results:
    E, D = flow_calc_check_ctf(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def cumulative_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame):
    '''For handling cumulative flow EWRs (some large freshes and overbanks, wetland flows).
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information    
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('cumulative')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    E, D = cumulative_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)

    return PU_df, tuple([E])

def cumulative_handle_qld(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame):
    '''For handling cumulative flow EWRs this to meet QLD requirements for bird breeding type 2.
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information    
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('cumulative')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    E, D = cumulative_calc_qld(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)

    return PU_df, tuple([E])

def cumulative_handle_bbr(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame):
    '''For handling cumulative flow EWRs (for bird breeding ewr QLD).
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information    
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('cumulative_bbr') 
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
     # If there is no level data loaded in, let user know and skip the analysis
    try:
        levels = df_L[EWR_info['weirpool_gauge']].values
    except KeyError:
        print(f'''Cannot evaluate this ewr for {gauge} {EWR}, due to missing data. Specifically this EWR 
        also needs data for level gauge {EWR_info.get('weirpool_gauge', 'gauge data')}''')
        return PU_df, None
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    E, D = cumulative_calc_bbr(EWR_info, df_F[gauge].values, levels, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)

    return PU_df, tuple([E])

def water_stability_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, df_L: pd.DataFrame, 
                           PU_df: pd.DataFrame):
    '''For handling Fish Recruitment with water stability requirement (QLD).
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information    
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('water_stability') 
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
     # If there is no level data loaded in, let user know and skip the analysis
    try:
        levels = df_L[EWR_info['weirpool_gauge']].values
    except KeyError:
        print(f'''Cannot evaluate this ewr for {gauge} {EWR}, due to missing data. Specifically this EWR 
        also needs data for level gauge {EWR_info.get('weirpool_gauge', 'gauge data')}''')
        return PU_df, None
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)
    E, D = water_stability_calc(EWR_info, df_F[gauge].values, levels, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)

    return PU_df, tuple([E])

def water_stability_level_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame):
    '''For handling Fish Recruitment with water stability requirement (QLD).
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information    
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('water_stability_level') 
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_L)
     # If there is no level data loaded in, let user know and skip the analysis
    try:
        levels = df_L[EWR_info['weirpool_gauge']].values
    except KeyError:
        print(f'''Cannot evaluate this ewr for {gauge} {EWR}, due to missing data. Specifically this EWR 
        also needs data for level gauge {EWR_info.get('weirpool_gauge', 'gauge data')}''')
        return PU_df, None
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_L, EWR_info)
    E, D = water_stability_level_calc(EWR_info, levels, water_years, df_L.index, masked_dates)
    PU_df = event_stats(df_L, PU_df, gauge, EWR, EWR_info, E, D, water_years)

    return PU_df, tuple([E])

def level_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling level type EWRs (low, mid, high and very high level lake fills).
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information        
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('level')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_L) 
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_L, EWR_info)  
    E, D = lake_calc(EWR_info, df_L[gauge].values, water_years, df_L.index, masked_dates)
  

    PU_df = event_stats(df_L, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def level_change_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling level type EWRs (low, mid, high and very high level lake fills).
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information        
    
    '''
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('level')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_L) 
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_L, EWR_info)
    E, D = level_change_calc(EWR_info, df_L[gauge].values, water_years, df_L.index, masked_dates)
  

    PU_df = event_stats(df_L, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def weirpool_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling weirpool type EWRs.
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information
    
    '''
    # Get information about EWR (changes depending on the weirpool type):
    weirpool_type = data_inputs.weirpool_type(EWR)
    if weirpool_type == 'raising':
        pull = data_inputs.get_EWR_components('weirpool-raising')
    elif weirpool_type == 'falling':
        pull = data_inputs.get_EWR_components('weirpool-falling')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
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
    E, D = weirpool_calc(EWR_info, df_F[gauge].values, levels, water_years, weirpool_type, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def nest_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling nest style EWRs.

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information    
    
    '''
    # Get information about EWR (changes depending on if theres a weirpool level gauge in the EWR)
    requires_weirpool_gauge =  is_weirpool_gauge(EWR_table, gauge, EWR, PU)
    if requires_weirpool_gauge:
        pull = data_inputs.get_EWR_components('nest-level')
    else:
        pull = data_inputs.get_EWR_components('nest-percent')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years:
    water_years = wateryear_daily(df_F, EWR_info)
    # there are 2 types of Nesting. 1. with trigger date with daily % drawdown rate and 2. Nesting Weirpool. 
    # no longer required a non-trigger version
    if not requires_weirpool_gauge:
        try:
            # calculate based on a trigger date and % drawdown drop
            E, D = nest_calc_percent_trigger(EWR_info, df_F[gauge].values, water_years, df_F.index)
        except ValueError:
            log.info(f"""Please pass a value to TriggerMonth between 1..12 and TriggerDay you passed 
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
            E, D = nest_calc_weirpool(EWR_info, df_F[gauge].values, levels, water_years, df_F.index, masked_dates)
        except KeyError:
            log.info(f'''Cannot evaluate this ewr for {gauge} {EWR}, due to missing parameter data. Specifically this EWR 
            also needs data for level threshold min or level threshold max''')
            return PU_df, None
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def flow_handle_multi(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling flow EWRs where flow needs to be combined at two gauges
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information     
    
    '''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('multi-gauge-flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
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
        print(f'''This {EWR} at the gauge {gauge} sums the flows at two gauges ({gauge} and {EWR_info['second_gauge']}.
        The EWR tool has not been able to find the flow data for {EWR_info["second_gauge"]} so it will only evaluate EWRs against the
        flow at the gauge {gauge}. If you are running a model scenario through please disregard this message - most hydrology models have already
        summed flows at these two gauges.''')
        flows = flows1

    E, D = flow_calc(EWR_info, flows, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def lowflow_handle_multi(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling low flow EWRs where flow needs to be combined at two gauges.
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information  
    
    '''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('multi-gauge-low flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
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
        print(f'''This {EWR} at the gauge {gauge} sums the flows at two gauges ({gauge} and {EWR_info['second_gauge']}.
        The EWR tool has not been able to find the flow data for {EWR_info["second_gauge"]} so it will only evaluate EWRs against the
        flow at the gauge {gauge}. If you are running a model scenario through please disregard this message - most hydrology models have already
        summed flows at these two gauges.''')
        flows = flows1
    # Check flow data against EWR requirements and then perform analysis on the results: 
    E, D = lowflow_calc(EWR_info, flows, water_years, df_F.index, masked_dates)  
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])
 
def ctf_handle_multi(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling cease to flow EWRs where flow needs to be combined at two gauges

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information  
    
    '''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('multi-gauge-cease to flow')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
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
        print(f'''This {EWR} at the gauge {gauge} sums the flows at two gauges ({gauge} and {EWR_info['second_gauge']}.
        The EWR tool has not been able to find the flow data for {EWR_info["second_gauge"]} so it will only evaluate EWRs against the
        flow at the gauge {gauge}. If you are running a model scenario through please disregard this message - most hydrology models have already
        summed flows at these two gauges.''')
        flows = flows1
    # Check flow data against EWR requirements and then perform analysis on the results:
    if ((EWR_info['start_month'] == 7) and (EWR_info['end_month'] == 6)):
        E, D = ctf_calc_anytime(EWR_info, df_F[gauge].values, water_years, df_F.index)
    else:
        E, D = ctf_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def cumulative_handle_multi(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling cumulative volume EWRs where flow needs to be combined at two gauges.

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information      
    
    '''
    # Get information about the EWR:
    pull = data_inputs.get_EWR_components('multi-gauge-cumulative')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
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
        print(f'''This {EWR} at the gauge {gauge} sums the flows at two gauges ({gauge} and {EWR_info['second_gauge']}.
        The EWR tool has not been able to find the flow data for {EWR_info["second_gauge"]} so it will only evaluate EWRs against the
        flow at the gauge {gauge}. If you are running a model scenario through please disregard this message - most hydrology models have already
        summed flows at these two gauges.''')
        flows = flows1
    E, D = cumulative_calc(EWR_info, flows, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)    
    return PU_df, tuple([E])


def flow_handle_sa(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    '''For handling SA IC(in channel) and FP (flood plain) type EWRs.
    It checks Flow thresholds, and check for Flow raise and fall.
    
    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily water level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Results:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information
    
    '''

    pull = data_inputs.get_EWR_components('flood-plains')
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates for both the flow and level dataframes:
    masked_dates = mask_dates(EWR_info, df_F)
    # Extract a daily timeseries for water years:
    water_years = wateryear_daily(df_F, EWR_info)

    E, D = flow_calc_sa(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
    return PU_df, tuple([E])

def barrage_flow_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    """handle function to calculate barrage flow type EWRs

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_F (pd.DataFrame): Daily flow data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Returns:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information
    """
    barrage_flow_gauges = data_inputs.get_barrage_flow_gauges()
    all_required_gauges = barrage_flow_gauges.get(gauge)
    if all_required_gauges:
        all_required_gauges_in_df_F = all(gauge in df_F.columns for gauge in all_required_gauges)
    # check if current gauge is the main barrage gauge
        if all_required_gauges_in_df_F:
            pull = data_inputs.get_EWR_components('barrage-flow')
            EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
            # Mask dates for both the flow and level dataframes:
            # Extract a daily timeseries for water years:
            water_years = wateryear_daily(df_F, EWR_info)
            # If there is no level data loaded in, let user know and skip the analysis
            df = df_F.copy(deep=True)
            df['combined_flow'] = df[all_required_gauges].sum(axis=1)
            E, D = barrage_flow_calc(EWR_info, df['combined_flow'], water_years, df_F.index)
            PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)
            return PU_df, tuple([E])
    else:
        print(f'Missing data for barrage gauges {" ".join(all_required_gauges)}')
        return PU_df, None

def barrage_level_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    """handle function to calculate barrage level type EWRs

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset 
        df_L (pd.DataFrame): Daily level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Returns:
        tuple[pd.DataFrame, tuple[dict]]: EWR results for the current planning unit iteration (updated); dictionary of EWR event information
    """
    barrage_level_gauges = data_inputs.get_barrage_level_gauges()
    all_required_gauges = barrage_level_gauges.get(gauge)
    if all_required_gauges:
        all_required_gauges_in_df_L = all(gauge in df_L.columns for gauge in all_required_gauges)
    # check if current gauge is the main barrage gauge
        if all_required_gauges_in_df_L:
            pull = data_inputs.get_EWR_components('barrage-level')
            EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
            masked_dates = mask_dates(EWR_info, df_L)
            # Extract a daily timeseries for water years:
            water_years = wateryear_daily(df_L, EWR_info)
            # If there is no level data loaded in, let user know and skip the analysis
            df = df_L.copy(deep=True)
            # calculate 5 day moving average and average of all required gauges
            df_5_day_averages = calculate_n_day_moving_average(df,5)
            df_5_day_averages['mean'] = df[all_required_gauges].mean(axis=1)
            cllmm_type = what_cllmm_type(EWR_info)
            if cllmm_type == 'c':
                E, D = lower_lakes_level_calc(EWR_info, df_5_day_averages['mean'], water_years, df_L.index, masked_dates)
            if cllmm_type == 'd':
                E, D = coorong_level_calc(EWR_info, df_5_day_averages['mean'], water_years, df_L.index, masked_dates)
        
        PU_df = event_stats(df_L, PU_df, gauge, EWR, EWR_info, E, D, water_years)    
        return PU_df, tuple([E])

    else:
        print(f'skipping calculation because gauge {" ".join(all_required_gauges)} is not the main barrage level gauge ')
        return PU_df, None

def rise_and_fall_handle(PU: str, gauge: str, EWR: str, EWR_table: pd.DataFrame, df_F: pd.DataFrame, df_L: pd.DataFrame, PU_df: pd.DataFrame) -> tuple:
    """For handling rise and fall EWRs of type FLOW and LEVEL.

    Args:
        PU (str): Planning unit ID
        gauge (str): Gauge number
        EWR (str): EWR code
        EWR_table (pd.DataFrame): EWR dataset
        df_F (pd.DataFrame): Daily flow data
        df_L (pd.DataFrame): Daily level data
        PU_df (pd.DataFrame): EWR results for the current planning unit iteration

    Returns:
        tuple[pd.DataFrame, tuple[dict]]: EWRS results for the current planning unit iteration (updated); dictionary of EWR event information
    """
   
    # Get information about EWR:
    pull = data_inputs.get_EWR_components('rise_fall') 
    EWR_info = get_EWRs(PU, gauge, EWR, EWR_table, pull)
    # Mask dates:
    masked_dates = mask_dates(EWR_info, df_F)
     # If there is no level data loaded in, let user know and skip the analysis
    # Extract a daily timeseries for water years
    water_years = wateryear_daily(df_F, EWR_info)

    if 'RRF' in EWR:
        E, D = rate_rise_flow_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    if 'RFF' in EWR:
        E, D = rate_fall_flow_calc(EWR_info, df_F[gauge].values, water_years, df_F.index, masked_dates)
    if 'RRL' in EWR:
        E, D = rate_rise_level_calc(EWR_info, df_L[gauge].values, water_years, df_F.index, masked_dates)
    if 'RFL' in EWR:
        E, D = rate_fall_level_calc(EWR_info, df_L[gauge].values, water_years, df_F.index, masked_dates)

    PU_df = event_stats(df_F, PU_df, gauge, EWR, EWR_info, E, D, water_years)

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

def achieved_min_volume(event: List[tuple], EWR_info: Dict)-> bool:
    flows = [flow for _, flow in event]
    flows_accumulation_period = flows[:EWR_info["accumulation_period"]]
    return sum(flows_accumulation_period) >= EWR_info['min_volume'] 

def flow_check(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, gap_track: int, 
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
        gap_track (int): current gap_track state
        water_years (np.array): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date

    Returns:
        tuple: the current state of the event, all_events, gap_track, total_event

    '''

    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                water_year = which_water_year(iteration, total_event, water_years)
                all_events[water_year].append(event)
            total_event = 0
                
            event = []
        
    return event, all_events, gap_track, total_event

def level_change_check(EWR_info: dict, iteration: int, levels: list, event: list, all_events: dict, gap_track: int, 
               water_years: np.array, total_event: int, level_date: date) -> tuple:
    '''Checks daily levels and evaluate level change in the last n days. 
    if the change os equals or greater the event accrue and is saved against the relevant
    water year in the event dictionary when the change is below. 

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        levels (list): levels for the current gauge
        event (list): current event state
        all_events (dict): current all events state
        gap_track (int): current gap_track state
        water_years (np.array): list of water year for every flow iteration
        total_event (int): current total event state
        level_date (date): current level date

    Returns:
        tuple: the current state of the event, all_events, gap_track, total_event

    '''
    period = EWR_info['min_event']
    season_start = date(level_date.year, EWR_info['start_month'], 1)
    start_day = get_index_date(level_date) - timedelta(days=period-1)
    meet_level_change_condition = evaluate_level_change(EWR_info, levels, iteration, period=period)
    if meet_level_change_condition and start_day >= season_start:
        if len(event) > 0:
            threshold_level = (get_index_date(level_date), levels[iteration])
            event.append(threshold_level)
        if len(event) == 0:
            levels_to_append = levels[iteration-(period-1):iteration+1]
            start_of_event = [ (start_day + timedelta(days=i), l)  for i, l in zip(range(period), levels_to_append)]
            event.extend(start_of_event)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0
                
            event = []
        
    return event, all_events, gap_track, total_event

def get_flows_in_between_dry_spells(flows:list, iteration:int, ctf_state:dict)-> list:
    """get flows in between dry spells from the ctf_state dictionary

    Args:
        flows (list): current flows
        iteration (int): current iteration
        ctf_state (dict): state_of_ctf_events

    Returns:
        flows_in_between (list): the flows in between the dry spells to be evaluated
    """

    last_day_first_event = ctf_state['events'][0][-1][0]
    first_day_second_event = ctf_state['events'][1][0][0]
    distance_iteration = (first_day_second_event - last_day_first_event).days
    days_second_event = len(ctf_state['events'][1])
    flows_in_between = flows[iteration - (distance_iteration + days_second_event) + 1: iteration - days_second_event]

    return flows_in_between

def get_full_failed_event(flows:list, iteration:int, ctf_state:dict)->list:
    """get full failed event inclusive of dry spells from the ctf_state dictionary

    Args:
        flows (list): current flows
        iteration (int): current iteration
        ctf_state (dict): state_of_ctf_events

    Returns:
        event: failed event inclusive of dry spells
    """

    first_day_first_event = ctf_state['events'][0][0][0]
    last_day_second_event = ctf_state['events'][1][-1][0]
    distance_iteration = (last_day_second_event - first_day_first_event).days + 1
    full_failed_event_flows = flows[iteration - (distance_iteration) : iteration]
    event = [ (first_day_first_event + timedelta(days=i), f)  for i, f in zip(range(distance_iteration), full_failed_event_flows)]

    return event

def get_threshold_events(EWR_info:dict, flows:list)->list:
    """get events that meed a threshold flow and return a list of events

    Args:
        EWR_info (dict): EWRs parameters
        flows (list): current flows

    Returns:
        list: events
    """
    events = []
    current_sublist = []

    for value in flows:
        if value >= EWR_info['min_flow']:
            current_sublist.append(value)
        else:
            if len(current_sublist) > 0:
                events.append(current_sublist)
                current_sublist = []
    if len(current_sublist) > 0:
        events.append(current_sublist)
    return events

def flow_check_ctf(EWR_info: dict, iteration: int, flows: List,  all_events: dict, water_years: np.array, flow_date: date, ctf_state: dict) -> tuple:
    '''Checks daily flow against EWR threshold and records dry spells
    in the ctf_state dictionary in the events key.
    When there are 2 events in the events key it evaluates if there is at least 1 phase 2 event 
    (i.e. event that allow Fish Dispersal) in between the dry spells. If there is no phase 2 event i.e. 
    the event fail then it records in the all_events dictionary, otherwise does nothing.
    It records in the all_events dictionary from the beginning of the first dry spell to the end of the second dry spell

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        water_years (np.array): list of water year for every flow iteration
        flow_date (date): current flow date

    Returns:
        tuple: all_events, ctf_state

    '''
    period = EWR_info["non_flow_spell"]
    flow = flows[iteration]
    if flow <= EWR_info["ctf_threshold"]:
        threshold_flow = (get_index_date(flow_date), flow)
        if ctf_state['in_event']:
            ctf_state['events'][-1].append(threshold_flow)
        if not ctf_state['in_event']:
            new_event = []
            new_event.append(threshold_flow)
            ctf_state['events'].append(new_event)
            ctf_state['in_event'] = True

    if flow > 1:
        if ctf_state['in_event']:
            ctf_state['in_event'] = False
            if len(ctf_state['events'][-1]) < period:
                ctf_state['events'].pop()
            if len(ctf_state['events']) == 2:
                flows_in_between_dry_spells = get_flows_in_between_dry_spells(flows, iteration, ctf_state)
                events_in_between_dry_spells = get_threshold_events(EWR_info, flows_in_between_dry_spells)
                at_least_one_dispersal_opportunity = any([len(event) >= EWR_info['min_event'] for event in events_in_between_dry_spells])
                if at_least_one_dispersal_opportunity:
                    ctf_state['events'].pop(0)
                if not at_least_one_dispersal_opportunity:
                    full_failed_event = get_full_failed_event(flows, iteration, ctf_state)
                    # records the failed event inclusive of dry spells in the all event year dictionary
                    all_events[water_years[iteration]].append(full_failed_event)
                    ctf_state['events'].pop(0)
        
    return all_events, ctf_state

def level_check(EWR_info: dict, iteration: int, level:float, level_change:float, 
               event: list, all_events: dict, gap_track: int, 
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
        gap_track (int): current gap_track state
        water_years (np.array): list of water year for every flow iteration
        total_event (int): current total event state
        level_date (date): current level date

    Returns:
        tuple: the current state of the event, all_events, gap_track, total_event
    """
    iteration_date = get_index_date(level_date)
    if ((level >= EWR_info['min_level']) and (level <= EWR_info['max_level']) and \
        (level_change <= float(EWR_info['drawdown_rate']))):
        threshold_level = (iteration_date, level)
        event.append(threshold_level)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
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
            total_event = 0
                
            event = []
        
    return event, all_events, gap_track, total_event

def nest_flow_check(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, 
                     gap_track: int, water_years: list, total_event: int, 
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
        gap_track (int): current gap_track state
        water_years (np.array): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        flow_percent_change (float): change from previous day to current day
        iteration_no_event (int): iteration_no_event count

    Returns:
        tuple: the current state of the event, all_events,  gap_track, total_event, iteration_no_event
    """

    iteration_date = get_index_date(flow_date)
    if flow >= EWR_info['min_flow'] and check_nest_percent_drawdown(flow_percent_change, EWR_info, flow):
        threshold_flow = (iteration_date, flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] # reset the gapTolerance after threshold is reached
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            iteration_no_event = 1 
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0    
            event = []
        
    return event, all_events, gap_track, total_event, iteration_no_event


def lowflow_check(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict,  water_years: np.array, flow_date: date) -> tuple:
    '''Checks daily flow against the EWR threshold. Saves all events to the relevant water year
    in the event tracking dictionary. Saves all event gaps to the relevant water year in the 
    no event dictionary.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        water_years (np.array): list of water year for every flow iteration
        flow_date (date): current flow date

    Returns:
        tuple: the current state of the event, all_events

    '''
    
    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
    else:
        if len(event) > 0:
            all_events[water_years[iteration]].append(event)
            
        event = []
        
    return event, all_events

def ctf_check(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, water_years: np.array, flow_date: date) -> tuple:
    '''Checks daily flow against the cease to flow EWR threshold. Saves all events to the relevant
    water year in the event tracking dictionary. Saves all no events to the relevant water year
    in the no event dictionary.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        water_years (np.array): list of water year for every flow iteration
        flow_date (date): current flow date

    Returns:
        tuple: the current state of the event, all_events

    '''

    if ((flow >= EWR_info['min_flow']) and (flow <= EWR_info['max_flow'])):
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
    else:
        if len(event) > 0:
            all_events[water_years[iteration-1]].append(event)
        event = []
    
    return event, all_events

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

def volume_check(EWR_info:Dict, iteration:int, flow:int, event:List, all_events:Dict, gap_track:int, 
               water_years:List, total_event:int, flow_date:date, roller:int, max_roller:int, flows:List)-> tuple:
    """Check in the current iteration of flows if the volume meet the ewr requirements.
    It looks back in a window of the size of the Accumulation period in(Days)

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (int): current flow
        event (List[float]): current event state
        all_events (Dict): current all events state
        gap_track (int): current gap_track state 
        water_years (List): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        roller (int): current roller state
        max_roller (int): current EWR max roller window
        flows (List): current list of all flows being iterated

    Returns:
        tuple: the current state of the event, all_events, gap_track, total_event and roller
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
        gap_track = EWR_info['gap_tolerance']
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) >=  1:
                all_events[water_years[iteration]].append(event)
            total_event = 0

            event = []

    return event, all_events, gap_track, total_event, roller

def volume_check_qld(EWR_info:Dict, iteration:int, event:List, all_events:Dict,
               water_years:List, total_event:int, flow_date:date, roller:int, max_roller:int, flows:List)-> tuple:
    """Check in the current iteration of flows if the volume meet the ewr requirements.
    It looks back in a window of the size up to the Accumulation period in(Days). 
    This is the QLD version of the volume check

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        event (List): current event state
        all_events (Dict): current all events state
        water_years (List):  list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date):  current flow date
        roller (int):  current roller state
        max_roller (int): current EWR max roller window
        flows (List): current list of all flows being iterated

    Returns:
        tuple: returns event, all_events, total_event, roller
    """
    
    flows_look_back = flows[iteration - roller:iteration+1]
    if roller < max_roller-1:
        roller += 1
    volume = sum(flows_look_back)
    if volume >= EWR_info['min_volume']:
        threshold_flow = (get_index_date(flow_date), volume)
        event.append(threshold_flow)
        total_event += 1
    else:
        if len(event) >=  1:
            all_events[water_years[iteration]].append(event)
        total_event = 0

        event = []

    return event, all_events, total_event, roller

def volume_level_check_bbr(EWR_info:Dict, iteration:int, flow:float, event:List, all_events:Dict, 
               water_years:List, total_event:int, flow_date:date, event_state:dict, levels:List)-> tuple:
    """Check in the current iteration of flows if the volume meet the ewr requirements and if the level 
    threshold drop grant an event exit
    It looks back in a window of the size of the Accumulation period in(Days)

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (int): current flow
        event (List[float]): current event state
        all_events (Dict): current all events state
        gap_track (int): current gap_track state 
        water_years (List): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        roller (int): current roller state
        max_roller (int): current EWR max roller window
        flows (List): current list of all flows being iterated
        levels (List): current list of all levels being iterated

    Returns:
        tuple: the current state of the event, all_events, gap_track, total_event and roller
    """

    # if there is not an event happening then check condition
    if total_event == 0:
        meet_min_flow_condition = flow >= EWR_info['min_flow']

    # if there is an event happening then the condition is always true
    if total_event > 0:
        meet_min_flow_condition = True
    
    if meet_min_flow_condition and not event_state["level_crossed_down"]:
        if not event_state["level_crossed_up"]:
            event_state["level_crossed_up"] = False if levels[iteration] < EWR_info['max_level'] else True
        
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1

        if event_state["level_crossed_up"]:
            event_state["level_crossed_down"] = False if levels[iteration] > EWR_info['max_level'] else True
        
    # if go back to cease to flow and level is below and never crosses up 
    if flow <= 1:
        total_event = 0
        event_state["level_crossed_up"] = False
        event_state["level_crossed_down"] = False
        event = []

    if event_state["level_crossed_down"]:
        if achieved_min_volume(event, EWR_info) :
            all_events[water_years[iteration]].append(event)
        total_event = 0
        event_state["level_crossed_up"] = False
        event_state["level_crossed_down"] = False
        event = []

    return event, all_events, total_event, event_state

def is_date_in_window(current_date:date, window_end:date, event_length:int)->bool:
    """Given a date plus days forward days check if days forward is less than last day of the window

    Args:
        current_date (date): current iteration data
        window_end (date): ewr end window
        days_forward (int): length of the event

    Returns:
        bool: Returns True if it end of event is in the window otherwise returns False 
    """
    event_last_day = current_date + timedelta(days=(event_length - 1))
    return event_last_day <= window_end

def get_last_day_of_window(iteration_date:date, month_window_end:int)->date:
    """Get the last day seasonal window of ewr

    Args:
        iteration_date (date): date where the code is iterating
        month_window_end (int): month that window ends

    Returns:
        date: return the last day of the window
    """

    iteration_year = iteration_date.year
    last_day_window_year = (iteration_year 
                            if month_window_end >= iteration_date.month 
                            else iteration_year + 1)
    _, last_day = calendar.monthrange(last_day_window_year, month_window_end)
    return date(last_day_window_year, month_window_end,last_day)

def water_stability_check(EWR_info:Dict, iteration:int, flows:List, all_events:Dict, water_years:List, flow_date:date, levels:List)-> tuple:
    """Check in the current iteration of flows and levels
    and look forwards for eggs (EggDaysSpell)+ larvae (LarvaeDaysSpell) days (parameter for ewr)
	If potential opportunity is still within the seasonal window
	then check stability for flow and level for egg and larvae
    if there is an opportunity record the event otherwise go to next day

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (int): current flow
        event (List[float]): current event state
        all_events (Dict): current all events state
        water_years (List): list of water year for every flow iteration
        flow_date (date): current flow date
        levels (List): current list of all levels being iterated

    Returns:
        tuple: the current state of the event, all_events, gap_track, total_event and roller
    """
    flows_are_stable = check_water_stability_flow(flows, iteration, EWR_info)
    
    levels_are_stable = False

    if flows_are_stable:
        levels_are_stable = check_water_stability_level(levels, iteration, EWR_info)
    if levels_are_stable:
        # record event opportunity for the next n days for the total period of (EggDaysSpell)+ larvae (LarvaeDaysSpell)
        # if the last day of the event is not over the last day of the event window
        iteration_date = flow_date.to_timestamp().date()
        last_day_window = get_last_day_of_window(iteration_date, EWR_info['end_month'])
        event_size = EWR_info['eggs_days_spell'] + EWR_info['larvae_days_spell']
        if is_date_in_window(iteration_date, last_day_window, event_size):
            event = create_water_stability_event(flow_date, flows, iteration, EWR_info)
            all_events[water_years[iteration]].append(event)
    return all_events


def water_stability_level_check(EWR_info:Dict, iteration:int, all_events:Dict, water_years:List, flow_date:date, levels:List)-> tuple:
    """Check in the current iteration water level height and levels
    and look forwards for eggs (EggDaysSpell)+ larvae (LarvaeDaysSpell) days (parameter for ewr)
	If potential opportunity is still within the seasonal window
	then check stability for water height and level for egg and larvae
    if there is an opportunity record the event otherwise go to next day

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        event (List[float]): current event state
        all_events (Dict): current all events state
        water_years (List): list of water year for every flow iteration
        flow_date (date): current flow date
        levels (List): current list of all levels being iterated

    Returns:
        tuple: the current state of the event, all_events, gap_track, total_event and roller
    """
    heights_are_stable = check_water_stability_height(levels, iteration, EWR_info)
    
    levels_are_stable = False

    if heights_are_stable:
        levels_are_stable = check_water_stability_level(levels, iteration, EWR_info)
    if levels_are_stable:
        # record event opportunity for the next n days for the total period of (EggDaysSpell)+ larvae (LarvaeDaysSpell)
        # if the last day of the event is not over the last day of the event window
        iteration_date = flow_date.to_timestamp().date()
        last_day_window = get_last_day_of_window(iteration_date, EWR_info['end_month'])
        event_size = EWR_info['eggs_days_spell'] + EWR_info['larvae_days_spell']
        if is_date_in_window(iteration_date, last_day_window, event_size):
            event = create_water_stability_event(flow_date, levels, iteration, EWR_info)
            all_events[water_years[iteration]].append(event)
    return all_events

def weirpool_check(EWR_info: dict, iteration: int, flow: float, level: float, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, weirpool_type: str, level_change: float) -> tuple:
    """Check weirpool flow and level if meet condition and update state of the events

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        level (float): current level
        event (list): current event state
        all_events (dict): current all events state
        gap_track (int): current gap_track state
        water_years (list): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        weirpool_type (str): type of weirpool ewr raising of falling
        level_change (float): level change in meters

    Returns:
        tuple: after the check return the current state of the event, all_events, gap_track, total_event
    """

    if flow >= EWR_info['min_flow'] and check_wp_level(weirpool_type, level, EWR_info) and check_draw_down(level_change, EWR_info) :
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0
                
            event = []
        
    return event, all_events, gap_track, total_event

def nest_weirpool_check(EWR_info: dict, iteration: int, flow: float, level: float, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, weirpool_type: str, levels: list)-> tuple:
    """Check weirpool flow and level if meet condition and update state of the events

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        level (float): current level
        event (List): current event state
        all_events (Dict): current all events state
        gap_track (int): current gap_track state
        water_years (List): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        weirpool_type (str): type of weirpool ewr raising of falling
        level_change (float): level change in meters

    Returns:
        tuple: after the check return the current state of the event, all_events, gap_track, total_event
    """

    if flow >= EWR_info['min_flow'] and check_weekly_drawdown(levels, EWR_info, iteration, len(event)) :
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0
                
            event = []
        
    return event, all_events, gap_track, total_event

def flow_level_check(EWR_info: dict, iteration: int, flow: float, level: float, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, level_change: float, levels:list) -> tuple:
    """Check weirpool flow and level if meet condition and update state of the events

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        level (float): current level
        event (list): current event state
        all_events (dict): current all events state
        gap_track (int): current gap_track state
        water_years (list): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        weirpool_type (str): type of weirpool ewr raising of falling
        level_change (float): level change in meters

    Returns:
        tuple: after the check return the current state of the event, all_events, gap_track, total_event
    """
    if flow >= EWR_info['min_flow'] and check_weekly_level_change(levels, EWR_info, iteration, len(event)) and level > 0:
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0
            event = []
        
    return event, all_events, gap_track, total_event

def flow_check_rise_fall(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, flows: list) -> tuple:
    """ Check if current flow meets the treshold and manages the recording of events based on 
        previous 30 days flow rise to validade event.
        Also check at the end of the event ig leading 30 days after the evend fall of flow validates the event
        
        The raise and fall of flow is calculated accorsing to the following rules
        Calculated as a rolling 3-day average Rate of rise to be assessed on the rising limb 
        for one month immediately prior to the target minimum discharge metric being met 
        Rate of fall to be assessed on the falling limb for one month immediately after discharge 
        has fallen below the target minimum discharge metric

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        gap_track (int): current gap_track state
        water_years (list): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        flows (list): flows of the iteration

    Returns:
        tuple: after the check return the current state of the event, all_events, gap_track, total_event
    """

    # if there is not an event hapening then check condition
    if total_event == 0:
        meet_condition_previous_30_days = check_period_flow_change(flows, EWR_info, iteration, "backwards", 3)

    # if there is an event hapening then the condition is always true and only need to check flow threshold
    if total_event > 0:
        meet_condition_previous_30_days = True

    if flow >= EWR_info['min_flow'] and meet_condition_previous_30_days:
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                meet_condition_next_30_days = check_period_flow_change(flows, EWR_info, iteration, "forwards", 3)
                if meet_condition_next_30_days:
                    all_events[water_years[iteration]].append(event)
            total_event = 0
            event = []
        
    return event, all_events, gap_track, total_event


def flow_check_rise_fall_stepped(EWR_info: dict, iteration: int, flow: float, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, flows: list) -> tuple:
    """ Check if current flow meets the treshold and manages the recording of events based on 
        previous 30 days flow rise to validade event.
        Also check at the end of the event if leading 30 days after the evend fall of flow validates the event
        
    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        iteration (int): current iteration
        flow (float): current flow
        event (list): current event state
        all_events (dict): current all events state
        gap_track (int): current gap_track state
        water_years (list): list of water year for every flow iteration
        total_event (int): current total event state
        flow_date (date): current flow date
        flows (list): flows of the iteration

    Returns:
        tuple: after the check return the current state of the event, all_events, gap_track, total_event
    """

    # if there is not an event hapening then check condition
    if total_event == 0:
        meet_condition_previous_30_days = check_period_flow_change_stepped(flows, EWR_info, iteration, "backwards", 3)

    # if there is an event hapening then the condition is always true and only need to check flow threshold
    if total_event > 0:
        meet_condition_previous_30_days = True

    if flow >= EWR_info['min_flow'] and meet_condition_previous_30_days:
        threshold_flow = (get_index_date(flow_date), flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                meet_condition_next_30_days = check_period_flow_change(flows, EWR_info, iteration, "forwards", 3)
                if meet_condition_next_30_days:
                    all_events[water_years[iteration]].append(event)
            total_event = 0
            event = []
        
    return event, all_events, gap_track, total_event


def rate_rise_flow_check(EWR_info: dict, iteration: int, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, flows: list)-> tuple:
    
    current_flow = flows[iteration]
    previous_day_flow_change = flows[iteration] / flows[iteration - 1] if flows[iteration - 1] else 0.
    allowed_rate_rise =  EWR_info['rate_of_rise_max2'] if current_flow >= EWR_info['rate_of_rise_threshold2'] else EWR_info['rate_of_rise_max1']

    if previous_day_flow_change > allowed_rate_rise and current_flow >= EWR_info['rate_of_rise_threshold1']:
        threshold_flow = (get_index_date(flow_date), current_flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0
            event = []
        
    return event, all_events, gap_track, total_event

def rate_fall_flow_check(EWR_info: dict, iteration: int, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, flows: list)-> tuple:
    
    current_flow = flows[iteration]
    previous_day_flow_change = flows[iteration] / flows[iteration - 1] if flows[iteration - 1] else 1.
    allowed_rate_fall =  EWR_info['rate_of_fall_min']

    if previous_day_flow_change < allowed_rate_fall:
        threshold_flow = (get_index_date(flow_date), current_flow)
        event.append(threshold_flow)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0
            event = []
        
    return event, all_events, gap_track, total_event

def rate_rise_level_check(EWR_info: dict, iteration: int, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, levels: list)-> tuple:
    
    current_level = levels[iteration]
    previous_day_level_change = levels[iteration] - levels[iteration - 1]
    if previous_day_level_change > EWR_info['rate_of_rise_river_level']:
        threshold_level = (get_index_date(flow_date), current_level)
        event.append(threshold_level)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0
            event = []
        
    return event, all_events, gap_track, total_event


def rate_fall_level_check(EWR_info: dict, iteration: int, event: list, all_events: dict, gap_track: int, 
               water_years: list, total_event: int, flow_date: date, levels: list)-> tuple:
    
    current_level = levels[iteration]
    previous_day_level_change = levels[iteration] - levels[iteration - 1] 

    if previous_day_level_change < EWR_info['rate_of_fall_river_level']*-1:
        threshold_level = (get_index_date(flow_date), current_level)
        event.append(threshold_level)
        total_event += 1
        gap_track = EWR_info['gap_tolerance'] 
     
    else:
        if gap_track > 0:
            gap_track = gap_track - 1
            total_event += 1
        else:
            if len(event) > 0:
                all_events[water_years[iteration]].append(event)
            total_event = 0
            event = []
        
    return event, all_events, gap_track, total_event

def is_leap_year(year:int)-> bool:
    """ check if the year is a leap year
    A leap year occurs once every four years, except for years that are divisible 
    by 100 but not divisible by 400. 
    For example, 1900 was not a leap year because it is divisible by 100 
    but not divisible by 400. However, 1904 was a leap year because it is divisible by 4 
    and not divisible by 100, or it is divisible by 400.

    Args:
        year (int): year to check

    Returns:
        bool: True if the year is a leap year
    """
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return True
    else:
        return False

def get_water_year(month: int, year:int) -> int:
    """Get the water year for the given date
    if month is 1st half then get current year
    otherwise get previous year

    Args:
        date (date): date to get the water year

    Returns:
        int: water year
    """
    return year if month <= 6 else year - 1

def filter_timing_window_std(flows: pd.Series, flow_date: date, start_month:int, end_month:int) -> pd.Series:
    """Filter the flows/levels to only include start and end months from a given date
    in a standard way where start and end do not cross water year boundary

    Args:
        flows (pd.Series): series of flows
        flow_date (date): current flow date
        start_month (int): start month of the window
        end_month (int): end month of the window

    Returns:
        pd.Series: series of flows filtered to only include the high release window
    """ 

    window_day_end = str(get_days_in_month(end_month, get_water_year(end_month, flow_date.year)))
    high_release_window_mask = (flows.index >= f'{get_water_year(start_month, flow_date.year)}-{start_month:02d}-01') & \
                               (flows.index <= f'{get_water_year(end_month, flow_date.year)}-{end_month:02d}-{window_day_end}')
    
    return flows[high_release_window_mask]

def filter_timing_window_non_std(flows: pd.Series, flow_date: date, start_month:int, end_month:int) -> pd.Series:
    """Filter the flows/levels to only include start and end months from a given date
    in a non-standard way where start and end cross water year boundary

    Args:
        flows (pd.Series): series of flows
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flow_date (date): current flow date

    Returns:
        pd.Series: series of flows filtered to only include the low release window
    """
    window_day_end = str(get_days_in_month(end_month, get_water_year(end_month, flow_date.year)))
    low_release_window_mask_front = (flows.index >= f'{get_water_year(start_month, flow_date.year)}-{start_month:02d}-01') & \
                               (flows.index <= f'{flow_date.year}-06-30')
    low_release_window_mask_back = (flows.index >= f'{flow_date.year - 1}-07-01') & \
                              (flows.index <= f'{flow_date.year - 1}-{end_month:02d}-{window_day_end}')
                               
    
    return pd.concat([flows[low_release_window_mask_front],flows[low_release_window_mask_back]])

def filter_last_year_flows(flows: pd.Series, flow_date: date) -> pd.Series:
    """Filter the flows to only include the last year

    Args:
        flows (pd.Series): series of flows
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flow_date (date): current flow date

    Returns:
        pd.Series: series of flows filtered to only include the last year
    """
    last_year = flow_date.year - 1
    last_year_start = f'{last_year}-07-01'
    last_year_end = f'{flow_date.year}-06-30'
    last_year_mask = (flows.index >= last_year_start) & (flows.index <= last_year_end)
    
    return flows[last_year_mask]

def filter_n_year_shift_flows(flows: pd.Series, flow_date: date, years_shift:int) -> pd.Series:
    """Filter the flows to only include year shift n from flow date

    Args:
        flows (pd.Series): series of flows
        flow_date (date): current flow date
        years_back (int): number of years to go back

    Returns:
        pd.Series: series of flows filtered to only include the last year
    """
    period_start_year = flow_date.year - years_shift
    period_end_year = (flow_date.year - years_shift ) + 1
    start_date = f'{period_start_year}-07-01'
    end_date = f'{period_end_year}-06-30'
    last_year_mask = (flows.index >= start_date) & (flows.index <= end_date)
    
    return flows[last_year_mask]

def get_min_each_last_three_years_volume(flows_series: pd.Series, flow_date: date) -> float:
    """It gets the minimum volume of the last three years and then returns the minimum of the three

    Args:
        flows (pd.Series): series of flows
        flow_date (date): current flow date

    Returns:
        float: minimum volume of the last three years
    """
    volume_flows = []
    for shift in range(1,4,1):
        flows = filter_n_year_shift_flows(flows_series, flow_date, shift)
        volume_flows.append(flows.sum())
    return min(volume_flows)


def filter_last_three_years_flows(flows: pd.Series, flow_date: date) -> pd.Series:
    """Filter the flows to only include the last three years

    Args:
        flows (pd.Series): series of flows
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flow_date (date): current flow date

    Returns:
        pd.Series: series of flows filtered to only include the last three years
    """
    last_three_years = flow_date.year - 3
    last_three_years_start = f'{last_three_years}-07-01'
    last_three_years_end = f'{flow_date.year}-06-30'
    last_three_years_mask = (flows.index >= last_three_years_start) & (flows.index <= last_three_years_end)
    
    return flows[last_three_years_mask]

def what_cllmm_type(EWR_info: dict) -> str:
    """Determine the CLLMM type based on the EWR code.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated

    Returns:
        str: 'a' if the EWR code contains '_a', 'b' otherwise
    """
    ewr_code = EWR_info['EWR_code']

    return ewr_code.split('_')[0][-1]


def barrage_flow_check(EWR_info: dict, flows: pd.Series, event: list, all_events: dict, flow_date: date) -> tuple:
    """Check if barrage total volume has met the barrage flow SA ERW parameters
    then save the results in the events list and all events dictionary

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (pd.Series): series of flows
        event (list): current event state
        all_events (dict): current all events state
        flow_date (date): current flow date

    Returns:
        tuple: after the check return the current state of the event, all_events
    """
    cllmm_type = what_cllmm_type(EWR_info)

    if cllmm_type == 'a':
        last_year_flows = filter_last_year_flows(flows, flow_date)
        
        if 'S' in EWR_info['EWR_code']:
            if last_year_flows.sum() >= EWR_info['annual_barrage_flow']:
                threshold_flow = (get_index_date(flow_date), last_year_flows.sum())
                event.append(threshold_flow)
                all_events[flow_date.year -1 ].append(event)
        else:
            start_date_peak = EWR_info['high_release_window_start']
            end_date_peak = EWR_info['high_release_window_end']
            high_release_window_flows = filter_timing_window_std(flows, flow_date, start_date_peak, end_date_peak)
            if start_date_peak < 7 and end_date_peak > 7:
                high_release_window_flows = filter_timing_window_non_std(flows, flow_date, start_date_peak, end_date_peak)

            start_date_min = EWR_info['low_release_window_start']
            end_date_min = EWR_info['low_release_window_end']
            low_release_window_flows = filter_timing_window_std(flows, flow_date, start_date_min, end_date_min)
            if start_date_min < 7 and end_date_min > 7:
                low_release_window_flows = filter_timing_window_non_std(flows, flow_date, start_date_min, end_date_min)

            if last_year_flows.sum() >= EWR_info['annual_barrage_flow'] and high_release_window_flows.sum() > low_release_window_flows.sum():
                threshold_flow = (get_index_date(flow_date), last_year_flows.sum())
                event.append(threshold_flow)
                all_events[flow_date.year -1 ].append(event)

    if cllmm_type == 'b':
        min_last_three_years_flows = get_min_each_last_three_years_volume(flows, flow_date)
        last_three_years_flows = filter_last_three_years_flows(flows, flow_date)
        if (min_last_three_years_flows >= EWR_info['annual_barrage_flow'] and last_three_years_flows.sum() > EWR_info['three_years_barrage_flow']
            and len(last_three_years_flows) >= 3*365 ):
            threshold_flow = (get_index_date(flow_date), last_three_years_flows.sum())
            event.append(threshold_flow)
            all_events[flow_date.year -1 ].append(event)

    return event, all_events

def coorong_check(EWR_info: dict, levels: pd.Series, event: list, all_events: dict, level_date: date,
                        water_years: List, iteration: int, total_event: int) -> tuple:
    """check if current level meet the minimal level requirement for coorong levels.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        levels (pd.Series): series of levels
        event (list): current event state
        all_events (dict): current all events state
        level_date (date): current level date
        water_years (List): list of water year for every flow iteration
        iteration (int): current iteration
        total_event (int): current total event state

    Returns:
        tuple: after the check return the current state of the event, all_events
    """
    
    level = levels[level_date]

    if level >= EWR_info['min_level'] and level <= EWR_info['max_level']:
        threshold_level = (get_index_date(level_date), level)
        event.append(threshold_level)
    else:
        if len(event) > 0:
            all_events[water_years[iteration]].append(event)
            event = []

    return event, all_events


def lower_lakes_level_check(EWR_info: dict, levels: pd.Series, event: list, all_events: dict, 
                                 level_date: date) -> tuple:
    """	Check  if
    last year peak is above max_level for full year and the max data point is also contained in the peak period
	last year low is above min_level for full year and the min data point is also contained in the low period	
	If both conditions are met event is counted. 

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        levels (pd.Series): series of levels
        event (list): current event state
        all_events (dict): current all events state
        level_date (date): current level date

    Returns:
        tuple: after the check return the current state of the event, all_events
    """
    
    last_year_peak = get_last_year_peak(levels, level_date)
    last_year_low = get_last_year_low(levels, level_date)
    last_year_levels = filter_last_year_flows(levels, level_date)
    if ( last_year_peak >= EWR_info['max_level'] and last_year_peak_within_window(last_year_peak, last_year_levels, EWR_info) and
        last_year_low >= EWR_info['min_level'] and last_year_low_within_window(last_year_low, last_year_levels, EWR_info)):
        threshold_flow = (get_index_date(level_date), last_year_peak)
        event.append(threshold_flow)
        all_events[level_date.year -1 ].append(event)
    
    return event, all_events

#------------------------------------ Calculation functions --------------------------------------#


def create_water_stability_event(flow_date: pd.Period, flows:List, iteration: int, EWR_info:dict)->List:
    """create overlapping event that meets an achievement for fish recruitment water stability

    Args:
        flow_date (datetime.date): current iteration date
        flows (List): flows series
        event_state (Dict): state of the event in the current iteration

    Returns:
        List: event list with flows and dates
    """
    event_size = EWR_info['eggs_days_spell'] + EWR_info['larvae_days_spell']
    event_flows = flows[iteration: iteration + event_size]
    start_event_date = flow_date.to_timestamp().date()
    event_dates = [ start_event_date + timedelta(i) for i in range(event_size)]
    
    return [(d, flow)  for d, flow in zip(event_dates, event_flows)]


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

def check_daily_level_change(level_change: float, EWR_info: dict) -> bool:
    """ check if the daily level changes has breached the min and max boundaries
    if not returns True if yes then False

    Args:
        level_change (float): change in meters
        EWR_info (dict): EWR parameters

    Returns:
        bool: if pass test returns True and fail return False
    """
    if level_change < 0:
        return level_change*-1 <= float(EWR_info['drawdown_rate'])
    else:
        return level_change <= float(EWR_info['max_level_raise'])


def check_water_stability_flow(flows: List, iteration:int, EWR_info:Dict)-> bool:
    """Check if water flows for in the next n days if is within water flow
    range for eggs and larvae stability

    Args:
        flows (List): flows time series
        iteration (int): current iteration
        EWR_info (Dict): ewr parameters

    Returns:
        bool: Returns True if levels are stable as per parameters and False otherwise
    """
    evaluation_period = EWR_info['eggs_days_spell'] + EWR_info['larvae_days_spell']
    period_to_check = flows[iteration: iteration + evaluation_period]
    max_period = max(period_to_check)
    min_period = min(period_to_check)
    return max_period < EWR_info['max_flow'] and min_period > EWR_info['min_flow']


def check_water_stability_height(levels: List, iteration:int, EWR_info:Dict)-> bool:
    """Check if water flows for in the next n days if is within water flow
    range for eggs and larvae stability

    Args:
        levels (List): flows time series
        iteration (int): current iteration
        EWR_info (Dict): ewr parameters

    Returns:
        bool: Returns True if levels are stable as per parameters and False otherwise
    """
    evaluation_period = EWR_info['eggs_days_spell'] + EWR_info['larvae_days_spell']
    period_to_check = levels[iteration: iteration + evaluation_period]
    max_period = max(period_to_check)
    min_period = min(period_to_check)
    return max_period <= EWR_info['max_level'] and min_period >= EWR_info['min_level']

def is_egg_phase_stable(levels:list, EWR_info: dict )-> bool:
    """Evaluate if water stability for egg is stable
    It calculated the difference between the max level in the period
    and the minimum and then evaluate if the difference is
    less than the 'max_level_raise' parameter
    True otherwise returns false

    Args:
        levels (list): levels to be evaluates
        EWR_info (dict): ewr parameters

    Returns:
        bool: Returns True if levels are stable as per parameters and False otherwise
    """
    max_level_in_period = max(levels)
    min_level_in_period = min(levels)
    max_level_change = max_level_in_period - min_level_in_period
    return max_level_change <= EWR_info["max_level_raise"]

def is_larva_phase_stable(levels:list, EWR_info: dict )-> bool:
    """Evaluate if water stability for larva is stable
    If calculated the max daily change is less than
    the 'max_level_raise' parameter then return
    True otherwise returns false

    Args:
        levels (list): levels to be evaluates
        EWR_info (dict): ewr parameters

    Returns:
        bool: Returns True if levels are stable as per parameters and False otherwise
    """
    daily_changes = [ abs(levels[i] - levels[i-1]) for i in range(1, len(levels))]
    max_daily_change = max(daily_changes) if daily_changes else 0
    return max_daily_change <= EWR_info["max_level_raise"]


def check_water_stability_level(levels: List, iteration:int, EWR_info:Dict)-> bool:
    """Check if water level for in the next n days if is within water 
    stability parameters for egg and larva for the period
    If it is within the paremetes then returns True otherwise returns False

    Args:
        levels (List): levels time series
        iteration (int): current iteration
        EWR_info (Dict): ewr parameters

    Returns:
        bool: Returns True if levels are stable as per parameters and False otherwise
    """
    # evaluate egg
    egg_stability_length = EWR_info['eggs_days_spell']
    egg_levels_to_check = levels[iteration: iteration + egg_stability_length]
    is_egg_level_stable = True
    if egg_stability_length > 0 :
        is_egg_level_stable = is_egg_phase_stable(egg_levels_to_check, EWR_info )
    
    # evaluate larva
    larva_stability_length = EWR_info['larvae_days_spell']
    larva_levels_to_check = levels[iteration + egg_stability_length - 1: iteration + (egg_stability_length + larva_stability_length)]
    is_larva_level_stable = True
    if larva_stability_length > 1:
        is_larva_level_stable = is_larva_phase_stable(larva_levels_to_check, EWR_info )

    return all([is_egg_level_stable, is_larva_level_stable])
    

def check_weekly_level_change(levels: list, EWR_info: dict, iteration: int, event_length: int) -> bool:
    """Check if the level change from 7 days ago to today is within the maximum allowed in a week
    for raise and fall.

    Args:
        levels (float): Level time series values
        EWR_info (dict): EWR parameters

    Returns:
        bool: if pass test returns True and fail return False
    """
    level_drop_week_max = float(EWR_info["drawdown_rate"])*7
    level_raise_week_max = float(EWR_info["max_level_raise"])*7

    if event_length < 7 :
        current_weekly_change = levels[iteration] - levels[iteration + 1 - event_length]
    else:
        current_weekly_change = levels[iteration] - levels[iteration - 6 ]
    return (current_weekly_change >= level_drop_week_max*-1) if current_weekly_change < 0 else (current_weekly_change <= level_raise_week_max)

def evaluate_level_change(EWR_info:dict, levels:list, iteration:int, period:int = 7) -> bool:
    """Evaluate if in the last period days the level has changes per requirement

    Args:
        EWR_info (dict): EWR parameters
        levels (list): Level time series values
        iteration (int): current iteration
        period (int, optional): lookback period. Defaults to 7.

    Returns:
        bool: True is level change is within the parameters and False otherwise
    """
    level_change = levels[iteration] - levels[iteration - (period - 1) ]

    return False if len(levels[:iteration+1]) < period else ( level_change > EWR_info['min_level_rise'])


def calculate_change(values:List)-> List:
    """Calcualte the change in values for items from a list

    Args:
        values (List): list of values

    Returns:
        List: list of change values
    
    """
    change = []
    for i in range(1, len(values)):
        diff = values[i] - values[i-1]
        change.append(diff)
    return change

def calculate_change_previous_day(values:List)-> List:
    """Calcualte the change in x values from
    previous items in a list

    Args:
        values (List): list of values

    Returns:
        List: list of change values
    
    """
    changes = []
    for i in range(1, len(values)):
        times_change = values[i] / values[i-1]
        changes.append(times_change)
    return changes

def flow_intervals_stepped(flows:List, steps:tuple)-> dict:

    first, second = steps

    intervals = defaultdict(list)

    first_flows = []
    second_flows = []
    for flow in flows:
        if flow >= second:
            second_flows.append(flow)
            if first_flows:
                intervals['first_threshold'].append(first_flows)
                first_flows = [] 
        if flow >= first and flow < second:
            first_flows.append(flow)
            if second_flows:
                intervals['second_threshold'].append(second_flows)
                second_flows = []
        if flow < first:
            if first_flows:
                intervals['first_threshold'].append(first_flows)
                first_flows = []
            if second_flows:
                intervals['second_threshold'].append(second_flows)
                second_flows = []
    # regiter any remaining flows
    if first_flows:
        intervals['first_threshold'].append(first_flows)
        first_flows = []
    if second_flows:
        intervals['second_threshold'].append(second_flows)
        second_flows = []

    return intervals
            
def rolling_average(values: List, period:int)-> List:
    """take a list of values and returns a list with the n period average

    Args:
        values (List): last 30 days flow

    Returns:
        List: rolling period moving averages
    """
    rolling_averages = []
    
    for i in range(period, len(values)+1):
        average = sum(values[i-period:i]) / period
        rolling_averages.append(average)
    
    return rolling_averages

def check_period_flow_change(flows: list, EWR_info: dict, iteration: int, mode: str, period:int) -> bool:
    """Check if the flow change up (raise) or down (fall) from period days ago to current date 
        is within the maximum allowed in a the period

    Args:
        flows (list): Flow time series values
        EWR_info (dict):EWR parameters
        iteration (int): current iteration
        mode (str): mode to look for flow change. Can be backwards to check before start an event or forwards
        to check after an event ends.
        period (int): period to calculate the moving average for the flow change

    Returns:
        bool: Return True is meet condition and False if don't
    """

    max_raise = float(EWR_info["max_level_raise"])
    max_fall = float(EWR_info['drawdown_rate'])


    if mode == "backwards":
        last_30_days_flows = flows[iteration - 29:iteration + 1]
        last_30_days_flows_change = calculate_change(last_30_days_flows) 
        last_30_days_rolling_avg = rolling_average(last_30_days_flows_change, period)
        max_change = max_raise + 1 if len(last_30_days_rolling_avg) == 0 else max(last_30_days_rolling_avg)
        return max_change <= max_raise
    if mode == "forwards":
        next_30_days_flows = flows[iteration -1 :iteration + 29]
        next_30_days_flows_change = calculate_change(next_30_days_flows) 
        next_30_days_rolling_avg = rolling_average(next_30_days_flows_change, period)
        draw_downs = [change for change in next_30_days_rolling_avg if change < 0]
        if len(next_30_days_rolling_avg) == 0 or len(draw_downs ) == 0:
            max_change = 0 
        else:
            max_change = min(draw_downs)  
        return abs(max_change) <= max_fall
     
def check_period_flow_change_stepped(flows: list, EWR_info: dict, iteration: int, mode: str) -> bool:
    """Check if the flow change up (raise) or down (fall) from period days ago to current date 
        is within the maximum allowed in a the period for VIC EWRs

    Args:
        flows (list): Flow time series values
        EWR_info (dict):EWR parameters
        iteration (int): current iteration
        mode (str): mode to look for flow change. Can be backwards to check before start an event or forwards
        to check after an event ends.

    Returns:
        bool: Return True is meet condition and False if don't
    """

    if mode == "backwards_stepped":
        last_30_days_flows = flows[iteration - 29:iteration + 1]
        intervals = flow_intervals_stepped(last_30_days_flows,[1000,5000])
        first_intervals = intervals['first_threshold']
        first_interval_max = EWR_info['rate_of_rise_max1']
        meet_condition_first_interval = all(max(calculate_change_previous_day(interval)) <= first_interval_max 
                                            for interval in first_intervals)
        second_interval_max = EWR_info['rate_of_rise_max2']
        second_intervals = intervals['second_threshold']
        meet_condition_second_interval = all(max(calculate_change_previous_day(interval)) <= second_interval_max
                                            for interval in second_intervals)
        return  meet_condition_first_interval and meet_condition_second_interval
    if mode == "backwards":
        last_30_days_flows = flows[iteration - 29:iteration + 1]
        last_30_days_flows_change = calculate_change_previous_day(last_30_days_flows) 
        return max(last_30_days_flows_change) <= EWR_info['rate_of_rise_stage']
    if mode == "forwards":
        next_30_days_flows = flows[iteration -1 :iteration + 29]
        next_30_days_flows_change = calculate_change_previous_day(next_30_days_flows)
        return min(next_30_days_flows_change) >= EWR_info['rate_of_fall_stage']

def check_cease_flow_period(flows: list, iteration: int, period:int) -> bool:
    """Check if the there is a period of "period" days ending 90 days (hydrological constraint) 
    before the start of the flow event.
    This to allow a ramp up period for the flow event.
    e.g.  Need to have a CTF ie flows <= 1 lasting the period (e.g 365 days) 90 days imediate beore the start
    of the flow thresfold event.
    Rationale from QLD EWR: followign a non-flow spell of 365 days (1 year)
    Args:
        flows (list):  Flow time series values
        iteration (int):  current iteration
        period (int):  period of the minimum duration of the cease to flow in days

    Returns:
        bool: Return True if meet condition and False if not
    """

    hydrological_constraint_days = 90
    end_of_ctf_period = iteration - hydrological_constraint_days
    beginning_of_ctf_period = end_of_ctf_period - period

    return max(flows[beginning_of_ctf_period:end_of_ctf_period]) <= 1 if end_of_ctf_period >= period else False

def check_weekly_drawdown(levels: list, EWR_info: dict, iteration: int, event_length: int) -> bool:
    """Check if the level change from 7 days ago to today changed more than the maximum allowed in a week.
    It will return True if the drawdown is within the allowed drawdown_rate_week in cm/week and False if it is above.
    drawdown will be assessed only looking at levers within the event window
    looking from the current level to the fist level since event started up to day 7 then
    will check 7 days back.

    Args:
        levels (float): Level time series values
        EWR_info (dict): EWR parameters
        iteration (int): current iteration
        event_legth (int): current event length

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


def get_days_in_month(month: int, year: int) -> int:
    """Return how many days there is in a month in a particular month
    and year

    Args:
        month (int): Month number to evaluate
        year (int): Year number to evaluate

    Raises:
        ValueError: If month number is less than 1 ot greater than 12

    Returns:
        int: Number of days in the month
    """
    # Check if the month is a valid integer between 1 and 12
    if not 1 <= month <= 12:
        raise ValueError("Invalid month number")

    # Use the calendar module to get the number of days in the month
    num_days = calendar.monthrange(year, month)[1]

    return num_days

def get_last_year_peak(levels: pd.Series, level_date: date) -> np.int64:
    """Get the last year peak 

    Args:
        levels (pd.Series): Level time series values
        level_date (date): date of the current level

    Returns:
        tuple: last year peak and the date of the peak
    """
    last_year_flows = filter_last_year_flows(levels, level_date)
    last_year_peak = last_year_flows.max()
    return last_year_peak

def get_last_year_low(levels: pd.Series, level_date:date) ->  np.int64:
    """Get the last year low

    Args:
        levels (pd.Series): Level time series values
        level_date (date): date of the current level

    Returns:
        tuple: last year low and the date of the low
    """
    last_year_flows = filter_last_year_flows(levels, level_date)
    last_year_low = last_year_flows.min()
    return last_year_low

def last_year_peak_within_window(last_year_peak: float, levels: pd.Series,  EWR_info: dict) -> bool:
    """Check if the last year peak is within the window

    Args:
        last_year_peak_date (date): date of the last year peak
        EWR_info (Dict): EWR parameters

    Returns:
        bool: True if it is within the window otherwise False
    """
    start_month = EWR_info['peak_level_window_start']
    end_month = EWR_info['peak_level_window_end']

    if start_month > end_month:
        month_mask = (levels.index.month >= start_month) | (levels.index.month <= end_month)
    if start_month <= end_month:
        month_mask = (levels.index.month >= start_month) & (levels.index.month <= end_month) 

    peak_period_levels = levels[month_mask]

    max_in_peak_pediod = peak_period_levels.max()
    return True if max_in_peak_pediod == last_year_peak else False

def last_year_low_within_window(last_year_low: float, levels: pd.Series,  EWR_info: dict) -> bool:
    """Check if the last year peak is within the window

    Args:
        last_year_peak_date (date): date of the last year peak
        EWR_info (Dict): EWR parameters

    Returns:
        bool: True if it is within the window otherwise False
    """
    
    start_month = EWR_info['low_level_window_start']
    end_month = EWR_info['low_level_window_end']
    
    if start_month > end_month:
        month_mask = (levels.index.month >= start_month) | (levels.index.month <= end_month)
    if start_month <= end_month:
        month_mask = (levels.index.month >= start_month) & (levels.index.month <= end_month) 

    low_period_levels = levels[month_mask]

    low_in_peak_pediod = low_period_levels.min()
    return True if low_in_peak_pediod == last_year_low else False


def calc_nest_cut_date(EWR_info: dict, iteration: int, dates: list) -> date:
    """Calculates the last date (date of the month) the nest EWR event is valid

    Args:
        EWR_info (Dict): EWR parameters
        iteration (int): current iteration
        dates (List): time series dates

    Returns:
        date: cut date for the current iteration
    """
    d = date(dates[iteration].year, EWR_info['end_month'], calendar.monthrange(dates[0].year,EWR_info['end_month'])[1])
    if EWR_info['end_day'] != None:
        d = d.replace(day = EWR_info['end_day'])
    return d

def lowflow_calc(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array, masked_dates: set) -> tuple:
    '''For calculating low flow ewrs. These have no consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of 
    each water year.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        dates (np.array): array of dates
        masked_dates (set): Dates within required date range
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations

    '''
    # Declare variables:
    event = []
    all_events = construct_event_dict(water_years)
    durations = []
    # Iterate over daily flow, sending to the lowflow_check function for each iteration 
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events = lowflow_check(EWR_info, i, flow, event, all_events, water_years, flow_date)
        # At the end of each water year, save any ongoing events and reset the list
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
            event = [] # Reset at the end of the water year
            durations.append(EWR_info['duration'])
        
    # Check the final iteration, saving any ongoing events/event gaps to their spots in the dictionaries
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events = lowflow_check(EWR_info, -1, flows[-1], event, all_events, water_years, flow_date)
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
    durations.append(EWR_info['duration'])
    return all_events,  durations

def ctf_calc_anytime(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array) -> tuple:
    '''For calculating cease to flow ewrs. These have a consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of each
    water year.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        dates (np.array): array of dates
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations

    '''
    # Declare variables:
    event = []
    all_events = construct_event_dict(water_years)
    durations = []
    # Iterate over daily flow, sending to the ctf_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        flow_date = dates[i]
        event, all_events = ctf_check(EWR_info, i, flow, event, all_events, water_years, flow_date)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    flow_date = dates[-1]
    event, all_events = ctf_check(EWR_info, -1, flows[-1], event, all_events, water_years, flow_date) 
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
    
    durations.append(EWR_info['duration'])
    
    return all_events, durations


def ctf_calc(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array, masked_dates: set) -> tuple:
    '''For calculating cease to flow ewrs. These have a consecutive requirement on their durations
    Events and event gaps are calculated on an annual basis, and are reset at the end of each
    water year.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        dates (np.array): array of dates
        masked_dates (set): Dates within required date range
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations

    '''
    # Declare variables:
    event = []
    all_events = construct_event_dict(water_years)
    durations = []
    # Iterate over daily flow, sending to the ctf_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events = ctf_check(EWR_info, i, flow, event, all_events, water_years, flow_date)
            # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
                event = []
            durations.append(EWR_info['duration'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events = ctf_check(EWR_info, -1, flows[-1], event, all_events, water_years, flow_date) 
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
    durations.append(EWR_info['duration'])
    
    return all_events, durations

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
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations

    '''
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events, gap_track, total_event = flow_check(EWR_info, i, flow, event, all_events, gap_track, water_years, total_event, flow_date)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
                total_event = 0
            event = []
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events, gap_track, total_event = flow_check(EWR_info, -1, flows[-1], event, all_events, gap_track, water_years, total_event,flow_date)   
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        total_event = 0
    durations.append(EWR_info['duration'])

    return all_events, durations


def level_change_calc(EWR_info: dict, levels: np.array, water_years: np.array, dates: np.array, masked_dates: set) -> tuple:
    '''For calculating level change EWRs with a time constraint within their requirements. Events are
    therefore reset at the end of each water year.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        dates (np.array): array of dates
        masked_dates (set): Dates within required date range
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations

    '''
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    for i, level in enumerate(levels[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events, gap_track, total_event = level_change_check(EWR_info, i, levels, event, all_events, gap_track, water_years, total_event, flow_date)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
                total_event = 0
            event = []
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        level_date = dates[-1]
        event, all_events, gap_track, total_event = level_change_check(EWR_info, -1, levels, event, all_events, gap_track, water_years, total_event, flow_date)   
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        total_event = 0
    durations.append(EWR_info['duration'])

    return all_events, durations

def flow_calc_check_ctf(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array, masked_dates: set) -> tuple:
    '''For calculating flow EWRs with a time constraint within their requirements. Events are
    therefore reset at the end of each water year. Also check at the begining of any event that in the last
    n days there was a period of ctf

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        dates (np.array): array of dates
        masked_dates (set): Dates within required date range
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations

    '''
    # Declare variables:
    event = []
    all_events = construct_event_dict(water_years)
    durations = []
    ctf_state = {'events':[], 'in_event': False}
    # Iterate over flow timeseries, sending to the flow_check_ctf function each iteration:
    for i, _ in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            all_events, ctf_state = flow_check_ctf(EWR_info, i, flows, all_events, water_years, flow_date, ctf_state)
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        all_events, ctf_state = flow_check_ctf(EWR_info, -1, flows, all_events, water_years, flow_date, ctf_state)   
    durations.append(EWR_info['duration'])

    return all_events, durations
    
def flow_calc_anytime(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array) -> tuple:
    '''For calculating flow EWRs with no time constraint within their requirements. Events crossing
    water year boundaries will be saved to the water year where the majority of event days were.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): array of daily flows
        water_years (np.array): array of daily water year values
        dates (np.array): array of dates
    
    Results:
        tuple[dict, dict, list, list]: dictionaries of all events and event gaps in timeseries. Lists of annual required durations

    '''
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    # Iterate over flows:
    for i, flow in enumerate(flows[:-1]):
        flow_date = dates[i]
        event, all_events,  gap_track, total_event = flow_check(EWR_info, i, flow, event, all_events, gap_track, water_years, total_event, flow_date)  
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    flow_date = dates[-1]
    event, all_events, gap_track, total_event = flow_check(EWR_info, -1, flows[-1], event, all_events, gap_track, water_years, total_event, flow_date)
    if len(event) > 0:
        water_year = which_water_year(-1, total_event, water_years)
        all_events[water_year].append(event)
    durations.append(EWR_info['duration'])

    return all_events, durations


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
        tuple: final output with the calculation of volume all_events, durations
    """

    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    for i, level in enumerate(levels[:-1]):
        if dates[i] in masked_dates:
            level_date = dates[i]
            level_change = levels[i-1]-levels[i] if i > 0 else 0
             # use the same logic as WP
            event, all_events, gap_track, total_event = level_check(EWR_info, i, level, level_change, event, all_events,
                                                                                    gap_track, water_years, total_event, level_date)
        # At the end of each water year save ongoing event, however not resetting the list. Let the flow_check record the event when it finishes
        if water_years[i] != water_years[i+1]:
            if len(event) >= EWR_info['duration'] and len(event) <= EWR_info['max_duration']:
                event_at_year_end = deepcopy(event)
                all_events[water_years[i]].append(event_at_year_end)
                total_event = 0
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        level_change = levels[-2]-levels[-1]   
        level_date = dates[-1]     
        event, all_events, gap_track, total_event = level_check(EWR_info, -1, levels[-1], level_change, event, all_events, gap_track,
                                                                 water_years, total_event, level_date)
        
    if len(event) >= EWR_info['duration'] and len(event) <= EWR_info['max_duration']:
        all_events[water_years[-1]].append(event)
    durations.append(EWR_info['duration'])

    return all_events, durations

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
        tuple: final output with the calculation of volume all_events, durations
    """
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    roller = 0
    max_roller = EWR_info['accumulation_period']

    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            roller = check_roller_reset_points(roller, dates[i], EWR_info)
            flow_date = dates[i]
            event, all_events, gap_track, total_event, roller = volume_check(EWR_info, i, flow, event, all_events, 
                                                                                            gap_track, water_years, 
                                                                                            total_event, flow_date, roller, max_roller, flows)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) >= 1:
                all_events[water_years[i]].append(event)
                total_event = 0
            event = []
            durations.append(EWR_info['duration'])
    
    if dates[-1] in masked_dates:
        roller = check_roller_reset_points(roller, dates[-1], EWR_info)
        flow_date = dates[-1]
        event, all_events, gap_track, total_event, roller = volume_check(EWR_info, -1, flows[-1], event, all_events,
                                                                                             gap_track, water_years, 
                                                                                            total_event, flow_date, roller, max_roller, flows)   
    durations.append(EWR_info['duration'])


    return all_events, durations

def cumulative_calc_qld(EWR_info: dict, flows: np.array, water_years: np.array, dates: np.array, masked_dates: set)-> tuple:
    """ Calculate and manage state of the Volume EWR calculations. It delegates to volume_check function
    the record of events when they not end at the end of a water year.
    This version does not reset event at year boundary, it accumulates the event until volume drops below threshold.

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): List with all the flows for the current calculated EWR
        water_years (np.array): List of the water year of each day of the current calculated EWR
        dates (np.array): List of the dates of the current calculated EWR
        masked_dates (set): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    roller = 0
    max_roller = EWR_info['accumulation_period']

    for i, _ in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events, total_event, roller = volume_check_qld(EWR_info, i, event, all_events, 
                                                                                              water_years, total_event, flow_date, 
                                                                                             roller, max_roller, flows)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
    
    if dates[-1] in masked_dates:
        roller = check_roller_reset_points(roller, dates[-1], EWR_info)
        flow_date = dates[-1]
        event, all_events, total_event, roller = volume_check_qld(EWR_info, -1, event, all_events,
                                                                                           water_years, total_event, flow_date, 
                                                                                            roller, max_roller, flows)   
    durations.append(EWR_info['duration'])

    return all_events, durations

def cumulative_calc_bbr(EWR_info: dict, flows: np.array, levels: np.array, water_years: np.array, dates: np.array, masked_dates: set)-> tuple:
    """ Calculate and manage state of the Volume EWR calculations. It delegates to volume_check function
    the record of events when they not end at the end of a water year, otherwise it resets the event at year boundary
    adopting the hybrid method

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): List with all the flows for the current calculated EWR
        levels (np.array): List with all the levels for the current calculated EWR
        water_years (np.array): List of the water year of each day of the current calculated EWR
        dates (np.array): List of the dates of the current calculated EWR
        masked_dates (set): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    # Iterate over flow timeseries, sending to the flow_check function each iteration:
    event_state = {}
    event_state["level_crossed_up"] = False
    event_state["level_crossed_down"] = False

    for i, flow in enumerate(flows[:-1]):
        # if dates[i] in masked_dates:
        flow_date = dates[i]
        event, all_events, total_event, event_state = volume_level_check_bbr(EWR_info, i, flow, event, all_events, 
                                                                                        water_years, total_event, flow_date, event_state, levels)
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
    
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events, total_event, event_state = volume_level_check_bbr(EWR_info, -1, flows[-1], event, all_events,
                                                                                             water_years, total_event, flow_date, event_state, levels)   
    durations.append(EWR_info['duration'])


    return all_events, durations


def water_stability_calc(EWR_info: dict, flows: np.array, levels: np.array, water_years: np.array, dates: np.array, masked_dates: set)-> tuple:
    """ Calculate the water stability EWRs  
    if within season it will look forward if there is an opportunity given the egg and larvae phases are met

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): List with all the flows for the current calculated EWR
        levels (np.array): List with all the levels for the current calculated EWR
        water_years (np.array): List of the water year of each day of the current calculated EWR
        dates (np.array): List of the dates of the current calculated EWR
        masked_dates (set): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    all_events = construct_event_dict(water_years)
    durations = []


    for i, _ in enumerate(flows):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            all_events = water_stability_check(EWR_info, i, flows, all_events, water_years, flow_date, levels)
            durations.append(EWR_info['duration'])
    
    durations.append(EWR_info['duration'])

    return all_events, durations

def water_stability_level_calc(EWR_info: dict, levels: np.array, water_years: np.array, dates: np.array, masked_dates: set)-> tuple:
    """ Calculate the water stability EWRs (LEVEL VERSION)  
    if within season it will look forward if there is an opportunity given the egg and larvae phases are met

    Args:
        EWR_info (dict): dictionary with the parameter info of the EWR being calculated
        flows (np.array): List with all the flows for the current calculated EWR
        levels (np.array): List with all the levels for the current calculated EWR
        water_years (np.array): List of the water year of each day of the current calculated EWR
        dates (np.array): List of the dates of the current calculated EWR
        masked_dates (set): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    all_events = construct_event_dict(water_years)
    durations = []


    for i, _ in enumerate(levels):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            all_events = water_stability_level_check(EWR_info, i, all_events, water_years, flow_date, levels)
            durations.append(EWR_info['duration'])
    
    durations.append(EWR_info['duration'])

    return all_events, durations


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
        tuple: final output with the calculation of volume all_events, durations
    """
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    # Iterate over flow timeseries, sending to the weirpool_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            # level_change = levels[i-1]-levels[i] if i > 0 else 0
            event, all_events, gap_track, total_event = nest_weirpool_check(EWR_info, i, flow, levels[i], event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, weirpool_type, levels)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        # if water_years[i] != water_years[i+1]: # Swapped out because interevent periods were getting saved at the end of the year
        if dates[i] in masked_dates and dates[i+1] not in masked_dates:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
            total_event = 0
            event = []
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        # level_change = levels[-2]-levels[-1] if i > 0 else 0
        event, all_events,  gap_track, total_event = nest_weirpool_check(EWR_info, -1, flows[-1], levels[-1], event,
                                                                                all_events, gap_track, 
                                                                              water_years, total_event, flow_date, weirpool_type, levels)   
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        total_event = 0
    
    durations.append(EWR_info['duration'])

    return all_events, durations


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
        tuple: final output with the calculation of volume all_events, durations
    """
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    for i, flow in enumerate(flows[:-1]):   
            flow_date = dates[i]
            flow_percent_change = calc_flow_percent_change(i, flows)
            trigger_day = date(dates[i].year,EWR_info["trigger_month"], EWR_info["trigger_day"])
            cut_date = calc_nest_cut_date(EWR_info, i, dates)
            is_in_trigger_window = dates[i].to_timestamp().date() >= trigger_day \
                                   and dates[i].to_timestamp().date() <= trigger_day + timedelta(days=14)
            iteration_no_event = 0
            
            ## if there IS an ongoing event check if we are on the trigger season window 
            # if yes then check the current flow
            if total_event > 0:
                if (dates[i].to_timestamp().date() >= trigger_day) and (dates[i].to_timestamp().date() <= cut_date):
                    event, all_events, gap_track, total_event, iteration_no_event = nest_flow_check(EWR_info, i, flow, event, all_events, 
                                                         gap_track, water_years, total_event, flow_date, flow_percent_change, iteration_no_event)

                # this path will only be executed if an event extends beyond the cut date    
                else:
                    if len(event) > 0:
                        all_events[water_years[i]].append(event)
                        total_event = 0
                    event = []
                    iteration_no_event = 1    
            ## if there is NOT an ongoing event check if we are on the trigger window before sending to checker
            if total_event == 0:
                if is_in_trigger_window and iteration_no_event == 0:
                    event, all_events,  gap_track, total_event, iteration_no_event = nest_flow_check(EWR_info, i, flow, event, all_events, 
                                                         gap_track, water_years, total_event, flow_date, flow_percent_change, iteration_no_event)

            # at end of water year record duration and min event values
            if water_years[i] != water_years[i+1]:
                durations.append(EWR_info['duration'])
    
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

        if (flow_date >= trigger_day ) \
            and (flow_date <= cut_date):
            event, all_events,  gap_track, total_event, iteration_no_event = nest_flow_check(EWR_info, -1, flows[-1], event, all_events, 
                                                             gap_track, water_years, total_event, flow_date, flow_percent_change, iteration_no_event)

    if total_event == 0:
        if is_in_trigger_window and iteration_no_event == 0:
            event, all_events, gap_track, total_event, iteration_no_event = nest_flow_check(EWR_info, i, flow, event, all_events, 
                                                 gap_track, water_years, total_event, flow_date, flow_percent_change, iteration_no_event)

    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        total_event = 0
        
    durations.append(EWR_info['duration'])
    return all_events, durations
       

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
        tuple: final output with the calculation of volume all_events, durations
    """
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    # Iterate over flow timeseries, sending to the weirpool_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            level_change = levels[i-1]-levels[i] if i > 0 else 0
            event, all_events, gap_track, total_event = weirpool_check(EWR_info, i, flow, levels[i], event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, weirpool_type, level_change)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
            total_event = 0
            event = []
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        level_change = levels[-2]-levels[-1] if i > 0 else 0
        event, all_events, gap_track, total_event = weirpool_check(EWR_info, -1, flows[-1], levels[-1], event,
                                                                                all_events, gap_track, 
                                                                              water_years, total_event, flow_date, weirpool_type, level_change)   
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        total_event = 0
        
    durations.append(EWR_info['duration'])

    return all_events, durations

def flow_level_calc(EWR_info: Dict, flows: List, levels: List, water_years: List, 
                        dates:List, masked_dates:List)-> tuple:
    """ Iterates each yearly flows to manage calculation of flow and level raise and fall. It delegates to flow_level_check function
    the record of events which will check the flow and level changes against the EWR requirements. 
    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        flows (List):  List with all the flows measurements for the current calculated EWR
        levels (List): List with all the levels measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.
    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    # Iterate over flow timeseries, sending to the weirpool_check function each iteration:
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            level_change = levels[i-1]-levels[i] if i > 0 else 0
            event, all_events, gap_track, total_event = flow_level_check(EWR_info, i, flow, levels[i], event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, level_change, levels)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
            total_event = 0
            event = []
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        level_change = levels[-2]-levels[-1] if i > 0 else 0
        event, all_events, gap_track, total_event = flow_level_check(EWR_info, -1, flows[-1], levels[-1], event,
                                                                                all_events, gap_track, 
                                                                              water_years, total_event, flow_date, level_change, levels)   
    if len(event) > 0:
        all_events[water_years[-1]].append(event)
        total_event = 0
        
    durations.append(EWR_info['duration'])

    return all_events, durations

def flow_calc_sa(EWR_info: Dict, flows: List, water_years: List, 
                        dates:List, masked_dates:List)-> tuple:
    """ Iterates each flows to manage calculation of flow and flows raise and fall. It delegates to flow_check_rise_fall function
    the record of events which will check the flow changes against the EWR requirements. 

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        flows (List):  List with all the flows measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    for i, flow in enumerate(flows[:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events, gap_track, total_event = flow_check_rise_fall(EWR_info, i, flow, event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, flows)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                meet_condition_next_30_days = check_period_flow_change(flows, EWR_info, i, "forwards", 3)
                if meet_condition_next_30_days:
                    all_events[water_years[i]].append(event)
            total_event = 0
            event = []
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events, gap_track, total_event = flow_check_rise_fall(EWR_info, -1, flows[-1], event,
                                                                                all_events, gap_track, 
                                                                              water_years, total_event, flow_date, flows)   
    if len(event) > 0:
        meet_condition_next_30_days = check_period_flow_change(flows, EWR_info, -1, "forwards", 3)
        if meet_condition_next_30_days:
            all_events[water_years[-1]].append(event)
        total_event = 0
        
    durations.append(EWR_info['duration'])

    return all_events, durations

def rate_rise_flow_calc(EWR_info: Dict, flows: List, water_years: List, 
                        dates:List, masked_dates:List)-> tuple:
    """ Iterates each flows to manage identify events that meet the rate of flow rise requirements

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        flows (List):  List with all the flows measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    for i, flow in enumerate(flows[1:-1]):
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events, gap_track, total_event = rate_rise_flow_check(EWR_info, i, event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, flows)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events, gap_track, total_event = rate_rise_flow_check(EWR_info, i, event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, flows)
    durations.append(EWR_info['duration'])

    return all_events, durations

def rate_fall_flow_calc(EWR_info: Dict, flows: List, water_years: List, 
                        dates:List, masked_dates:List)-> tuple:
    """ Iterates each flows to manage identify events that meet the rate of fall requirements

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        flows (List):  List with all the flows measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    for i, _ in enumerate(flows[:-1]):
        if i == 0:
            continue
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events, gap_track, total_event = rate_fall_flow_check(EWR_info, i, event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, flows)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events, gap_track, total_event = rate_fall_flow_check(EWR_info, -1, event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, flows)
    durations.append(EWR_info['duration'])

    return all_events, durations


def rate_rise_level_calc(EWR_info: Dict, levels: List, water_years: List, 
                        dates:List, masked_dates:List)-> tuple:
    """ Iterates each flows to manage identify events that meet the rate of fall requirements

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        levels (List):  List with all the levels for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    for i, _ in enumerate(levels[:-1]):
        if i == 0:
            continue
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events, gap_track, total_event = rate_rise_level_check(EWR_info, i,  event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, levels)
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events, gap_track, total_event =  rate_rise_level_check(EWR_info, i, event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, levels)
    durations.append(EWR_info['duration'])

    return all_events, durations

def rate_fall_level_calc(EWR_info: Dict, levels: List, water_years: List, 
                        dates:List, masked_dates:List)-> tuple:
    """ Iterates each flows to manage identify events that meet the rate of fall requirements

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        levels (List):  List with all the levels for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    # Declare variables:
    event = []
    total_event = 0
    all_events = construct_event_dict(water_years)
    durations = []
    gap_track = 0
    for i, flow in enumerate(levels[1:-1]):
        if i == 0:
            continue
        if dates[i] in masked_dates:
            flow_date = dates[i]
            event, all_events, gap_track, total_event = rate_fall_level_check(EWR_info, i, event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, levels)
        # At the end of each water year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            durations.append(EWR_info['duration'])
        
    # Check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    if dates[-1] in masked_dates:
        flow_date = dates[-1]
        event, all_events, gap_track, total_event =  rate_fall_level_check(EWR_info, i, event,
                                                                                all_events, gap_track, 
                                                                                water_years, total_event, flow_date, levels)
    durations.append(EWR_info['duration'])

    return all_events, durations


def barrage_flow_calc(EWR_info: Dict, flows: pd.Series, water_years: List, dates:List)-> tuple:
    """iterate flow data for barrage combined flow and check at the end of each year
    if barrage cumulative flow min threshold min is met
    as well as the seasonal flow threshold

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        flows (List):  List with all the flows measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    # declare variables:
    event = []
    all_events = construct_event_dict(water_years)
    durations = []

    for i, _ in enumerate(flows.values[:-1]):
        # At the end of each water year check last year barrage flow total if it above minimum threshold
        if water_years[i] != water_years[i+1]:
            flow_date = dates[i]
            event, all_events= barrage_flow_check(EWR_info, flows, event, all_events,  flow_date)
            event = []
            durations.append(EWR_info['duration'])
    
    # check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    event, all_events = barrage_flow_check(EWR_info, flows, event, all_events, dates[-1])
    event = []
    durations.append(EWR_info['duration'])
    return  all_events, durations
    
def coorong_level_calc(EWR_info: Dict, levels: pd.Series, water_years: List, dates:List, masked_dates:List)-> tuple:
    """iterate level data for barrage combined levels are with in minimum levels

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        levels (List):  List with all the levels measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events, durations
    """
    # declare variables:
    event = []
    all_events = construct_event_dict(water_years)
    durations = []
    total_event = 0

    for i, _ in enumerate(levels.values[:-1]):
        if dates[i] in masked_dates:
            level_date = dates[i]
            event, all_events = coorong_check(EWR_info, levels, event, all_events, level_date, water_years, i, total_event)
        # At the end of ecooronger year, save any ongoing events and event gaps to the dictionaries, and reset the list and counter
        if water_years[i] != water_years[i+1]:
            if len(event) > 0:
                all_events[water_years[i]].append(event)
            total_event = 0
            event = []
            durations.append(EWR_info['duration'])
    
    # check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    event, all_events = coorong_check(EWR_info, levels, event, all_events, level_date, water_years, i, total_event)
    event = []
    durations.append(EWR_info['duration'])
    return all_events,  durations

def lower_lakes_level_calc(EWR_info: Dict, levels: pd.Series, water_years: List, dates:List, masked_dates:List)-> tuple:
    """iterate level data for barrage combined levels and check at the end of each year
    if barrage level is at the required minimum as well as the seasonal peak levels threshold

    Args:
        EWR_info (Dict): dictionary with the parameter info of the EWR being calculated
        levels (List):  List with all the levels measurements for the current calculated EWR
        water_years (List): List of the water year of each day of the current calculated EWR
        dates (List): List of the dates of the current calculated EWR
        masked_dates (List): List of the dates that the EWR needs to be calculated i.e. the time window.

    Returns:
        tuple: final output with the calculation of volume all_events,  durations
    """
    # declare variables:
    event = []
    all_events = construct_event_dict(water_years)
    durations = []

    for i, _ in enumerate(levels.values[:-1]):
        # At the end of each water year check last year barrage flow total if it above minimum threshold
        if water_years[i] != water_years[i+1]:
            flow_date = dates[i]
            event, all_events = lower_lakes_level_check(EWR_info, levels, event, all_events, flow_date)
            event = []
            durations.append(EWR_info['duration'])
    
    # check final iteration in the flow timeseries, saving any ongoing events/event gaps to their spots in the dictionaries:
    event, all_events = lower_lakes_level_check(EWR_info, levels, event, all_events, flow_date)
    event = []
    durations.append(EWR_info['duration'])
    return  all_events, durations

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

def get_achievements_connecting_events(events: Dict, unique_water_years:set)->List:
    
    achievements_per_years = []

    for year in unique_water_years:
        year_events = events[year]
        if not year_events:
            achievements_per_years.append(0)
        else:
            events_info = [return_event_info(event) for event in year_events]
            achievement_count = 0
            if len(events_info) == 1:
                _, _, length, _ = events_info[0]
                achievements_per_years.append( 1 if length >= 90 else 0)
            else:
                for i in range(len(events_info)-1):
                    start_first, _, length, _ = events_info[i]
                    start_second, _, _, _ = events_info[i+1]
                    gap = (start_second - start_first).days
                    achievement_count += 1 if (gap >= 27 and gap <= 90 or length >= 90) else 0
                achievements_per_years.append(achievement_count)
    return achievements_per_years

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
        log.error(e)
    return [1 if (max_rolling > 0) else 0 for max_rolling in max_consecutive_days]


def get_min_gap(events:List[List])->int:
    """Retunt the min gap in between events list if there are more than 1 event in the list

    Args:
        events (List[List]): List of events

    Returns:
        int: min gap if there is more than 1 event otherwise return 0
    """


    events_info = [return_event_info(event) for event in events]
    gaps = []

    if len(events_info) > 1:
        for i in range(len(events_info)-1):
            current_event = events_info[i]
            next_event = events_info[i+1]
            _, current_event_end, _, _ = current_event
            next_event_start, _, _, _ = next_event
            gap = (next_event_start - current_event_end).days
            gaps.append(gap)
        return min(gaps)
    else:
        return 0

def get_max_gap(events:List[List])->int:
    """Retunt the max gap in between events list if there are more than 1 event in the list

    Args:
        events (List[List]): List of events

    Returns:
        int: max gap if there is more than 1 event otherwise return 0
    """


    events_info = [return_event_info(event) for event in events]
    gaps = []

    if len(events_info) > 1:
        for i in range(len(events_info)-1):
            current_event = events_info[i]
            next_event = events_info[i+1]
            _, current_event_end, _, _ = current_event
            next_event_start, _, _, _ = next_event
            gap = (next_event_start - current_event_end).days
            gaps.append(gap)
        return max(gaps)
    else:
        return 0

def get_max_event_length(events:List[List])->int:
    """Retunt the max legth of events list 

    Args:
        events (List[List]): List of events

    Returns:
        int: max gap if there is more than 1 event otherwise return 0
    """
    events_info = [return_event_info(event) for event in events]
    lengths = []

    if events_info:
        for info in events_info:
            _, _, length, _ = info
            lengths.append(length)
        return max(lengths)
    else:
        return 0


def get_event_years_connecting_events(events:Dict , unique_water_years:List[int])->List:
    """Return a list of years with events (represented by a 1), where the 2 connecting events
    meet the condition to be between 27 and 90 (3 months) days

    Args:
        events (Dict): Dict with results of Ewr events calculation
        unique_water_years (List[int]): Unique water years for the current result run

    Returns:
        List: a List with 1 or 0 representing which year the event occured 
    """

    achievements = get_achievements_connecting_events(events, unique_water_years)

    return [1 if achievement > 0 else 0 for achievement in achievements]

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
        log.error(e)
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
        log.error(e)
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

def event_stats(df:pd.DataFrame, PU_df:pd.DataFrame, gauge:str, EWR:str, EWR_info:Dict, events:Dict, durations:List, water_years:List) -> pd.DataFrame:
    ''' Produces statistics based on the event dictionaries and event gap dictionaries.
    
    Args:
        df (pd.DataFrame): Raw flow/level dataframe
        PU_df (pd.DataFrame): Dataframe with the results from the EWRs in the current planning unit
        gauge (str): current iteration gauge string
        EWR (str): current iteration EWR string
        EWR_info (Dict): Parameter information for current EWR
        events (Dict): Detailed event information events
        durations (List): List of annual required durations
        water_years (List): Daily water year values

    Results:
        pd.DataFrame: Updated results dataframe for this current planning unit

    
    '''
    unique_water_years = set(water_years)
    # Years with events
    years_with_events = get_event_years(EWR_info, events, unique_water_years, durations)

    ## reset the no_events to keep functionality but switched off
    no_events = construct_event_dict(water_years)

    if EWR_info['EWR_code'] in ['rANA']:
        years_with_events = get_event_years_connecting_events(events, unique_water_years)
    
    if EWR_info['EWR_code'] in ['CF1_c','CF1_C']:
        years_with_events = get_event_years_max_rolling_days(events, unique_water_years)

    if EWR_info['flow_level_volume'] == 'V':
        years_with_events = get_event_years_volume_achieved(events, unique_water_years)

    YWE = pd.Series(name = str(EWR + '_eventYears'), data = years_with_events, index = unique_water_years)
    PU_df = pd.concat([PU_df, YWE], axis = 1)
    # Number of event achievements:
    num_event_achievements = get_achievements(EWR_info, events, unique_water_years, durations)

    if EWR_info['EWR_code'] in ['rANA']:
        num_event_achievements = get_achievements_connecting_events(events, unique_water_years)

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
        log.error(e)
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

    for fresh in wp_freshes:
        try:
            weirpool_event_year = PU_df[f"{weirpool_pair[fresh]}_eventYears"].values
            has_wp = True
        except KeyError as e:
            has_wp = False
        try:
            fresh_event_year = PU_df[f"{fresh}_eventYears"].values
            has_fresh = True
        except KeyError as e:
            has_fresh = False

        if has_wp and has_fresh:
            # has both wp and fresh -> merge years
            merged_ewr_event_year = np.maximum(fresh_event_year, weirpool_event_year)
            # add merged results
            PU_df[f"{fresh}/{weirpool_pair[fresh]}_eventYears"] = merged_ewr_event_year
            # add remaninig columns with N/As
            column_attributes = list(set([col.split("_")[-1] for col in PU_df.columns if "eventYears" not in col]))
            for col in column_attributes:
                PU_df[f"{fresh}/{weirpool_pair[fresh]}_{col}"] = np.nan
        elif (not has_fresh) and has_wp:
            # has wp -> report wp successes
            PU_df[f"{fresh}/{weirpool_pair[fresh]}_eventYears"] = weirpool_event_year
            column_attributes = list(set([col.split("_")[-1] for col in PU_df.columns if "eventYears" not in col]))
            for col in column_attributes:
                PU_df[f"{fresh}/{weirpool_pair[fresh]}_{col}"] = np.nan
        elif (not has_wp) and has_fresh:
            # has fresh -> report fresh successes
            PU_df[f"{fresh}/{weirpool_pair[fresh]}_eventYears"] = fresh_event_year
            column_attributes = list(set([col.split("_")[-1] for col in PU_df.columns if "eventYears" not in col]))
            for col in column_attributes:
                PU_df[f"{fresh}/{weirpool_pair[fresh]}_{col}"] = np.nan
        else:
            # has neither fresh nor wp -> set all columns to NA
            column_attributes = list(set([col.split("_")[-1] for col in PU_df.columns if "eventYears" not in col]))
            for col in column_attributes:
                PU_df[f"{fresh}/{weirpool_pair[fresh]}_{col}"] = np.nan

    return PU_df

# make handling function available to process
HANDLING_FUNCTIONS = {
    'ctf_handle':ctf_handle,
    'ctf_handle_multi': ctf_handle_multi,
    'cumulative_handle': cumulative_handle,
    'cumulative_handle_multi': cumulative_handle_multi,
    'flow_handle': flow_handle,
    'flow_handle_multi': flow_handle_multi,
    'level_handle': level_handle,
    'lowflow_handle': lowflow_handle,
    'lowflow_handle_multi': lowflow_handle_multi,
    'nest_handle': nest_handle,
    'weirpool_handle' : weirpool_handle,
    'flow_handle_sa': flow_handle_sa,
    'barrage_flow_handle': barrage_flow_handle,
    'barrage_level_handle': barrage_level_handle,
    'flow_handle_check_ctf': flow_handle_check_ctf,
    'cumulative_handle_bbr': cumulative_handle_bbr,
    'water_stability_handle': water_stability_handle,
    'water_stability_level_handle' : water_stability_level_handle,
    'flow_handle_anytime': flow_handle_anytime,
    'cumulative_handle_qld': cumulative_handle_qld,
    'level_change_handle': level_change_handle,
    'rise_and_fall_handle' : rise_and_fall_handle
    }

def get_gauge_calc_type(multigauge:bool)-> str:
    """Get the gauge calculation type

    Args:
        complex (bool): is a complex gauge
        multigauge (bool): is a multigauge
        simultaneous (bool): is a simultaneous gauge

    Returns:
        str: gauge calculation type
    """
    if multigauge:
        return 'multigauge'
    else:
        return 'single'

def get_ewr_prefix(ewr_code:str, prefixes:list)-> str:
    """Get the EWR prefix by identifying the prefix in the EWR code

    Args:
        ewr_code (str): EWR code
        prefixes (list): list of prefixes

    Returns:
        str: EWR prefix
    """    
    for prefix in prefixes:
        if prefix in ewr_code:
            return prefix
    return 'unknown'

def get_handle_function(function_name: str) -> object:
    """return handling function

    Args:
        function_name (str): name of function

    Returns:
        object: handling function
    """
    return HANDLING_FUNCTIONS.get(function_name)

def build_args(args:dict, function:object)-> dict:
    """Builds a dictionary of arguments for a function

    Args:
        args (dict): dictionary of arguments
        function (object): function to build arguments for

    Returns:
        dict: dictionary of arguments for the function
    """
    func_object_info = inspect.getfullargspec(function)
    return {k: v for k, v in args.items() if k in func_object_info.args}

def find_function(ewr_key:str, new_config_file:dict) -> str:
    """find handle function name given an ewr key

    Args:
        ewr_key (str): ewr key {code}-{gauge_calc_type}-{flow_level_volume}
        new_config_file (dict): configuration file with the mappings of ewr keys to handle functions

    Returns:
        str: handling function name
    """
    for k, v in new_config_file.items():
        if any([ewr_key == i for i in v]):
            return k
    return 'unknown'

#---------------------------- Sorting and distributing to handling functions ---------------------#

def calc_sorter(df_F:pd.DataFrame, df_L:pd.DataFrame, gauge:str, EWR_table:pd.DataFrame, calc_config: dict) -> tuple:
    '''Sends to handling functions to get calculated depending on the type of EWR
    
    Args:
        df_F (pd.DataFrame): Dataframe with the daily flows
        df_L (pd.DataFrame): Dataframe with the daily levels
        gauge (str): gauge string of current iteration
        EWR_table (pd.DataFrame): Dataframe of EWR parameters

    Results:
        tuple[dict, dict]: annual results summary and detailed event information
    
    ''' 
    # Get EWR tables:
    PU_items = EWR_table.groupby(['PlanningUnitID', 'PlanningUnitName']).size().reset_index().drop([0], axis=1)
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

            MULTIGAUGE = is_multigauge(EWR_table, gauge, EWR, PU)

            # Save dict with function arguments value
            all_args = {"PU": PU, 
            "gauge": gauge, 
            "EWR": EWR, 
            "EWR_table": EWR_table, 
            "df_F": df_F, 
            "df_L": df_L,
            "PU_df": PU_df,
            }

            cat = EWR_categories[i]
            gauge_calc_type = get_gauge_calc_type(MULTIGAUGE)
            ewr_key = f'{EWR}-{gauge_calc_type}-{cat}'
            function_name = find_function(ewr_key, calc_config)
            if function_name == 'unknown':
                log.warning(f"skipping calculation due to ewr key {ewr_key} not in the configuration configuration files")
                continue
            handle_function = get_handle_function(function_name)
            if not handle_function:
                log.warning(f"skipping calculation due to ewr key {ewr_key} not in the configuration configuration files")
                log.warning(f"add {ewr_key} to the configuration file in the appropriate handle function")
                continue
            kwargs = build_args(all_args, handle_function)
        
            PU_df, events = handle_function(**kwargs)
            if events != {}:
                PU_events[str(EWR)]=events
        
        wp_freshes = [ewr for ewr in EWR_codes if ewr in ["SF_WP","LF2_WP"]]
        if wp_freshes:
            PU_df = merge_weirpool_with_freshes(wp_freshes, PU_df)
            
        PU_name = PU_items['PlanningUnitName'].loc[PU_items[PU_items['PlanningUnitID'] == PU].index[0]]
        
        location_results[PU_name] = PU_df
        location_events[PU_name] = PU_events

    return location_results, location_events