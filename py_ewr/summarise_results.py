import pandas as pd
import numpy as np

from . import data_inputs, evaluate_EWRs
#--------------------------------------------------------------------------------------------------

def sum_events(events):
    '''returns a sum of events'''
    return int(round(events.sum(), 0))

def get_frequency(events):
    '''Returns the frequency of years they occur in'''
    if events.count() == 0:
        result = 0
    else:
        result = (int(events.sum())/int(events.count()))*100
    return int(round(result, 0))

def get_average(input_events):
    '''Returns overall average length of events'''
    events = input_events.dropna()
    if len(events) == 0:
        result = 0
    else:    
        result = round(sum(events)/len(events),1)
    return result

def get_event_length(detailed_events):
    '''Takes in the tuple of events calculates average event length over entire timeseries,'''
    length, count = 0, 0
    for index, tuple_ in enumerate(detailed_events):
        for year in detailed_events[index]:
            for event in detailed_events[index][year]:
                length = length + len(event)
                count += 1
    if count == 0:
        average_length = 0
    else:
        average_length = length/count
    return average_length

def get_threshold_days(detailed_events):
    '''Takes in the tuple of events calculates average event length over entire timeseries,'''
    length = 0
    for index, tuple_ in enumerate(detailed_events):
        for year in detailed_events[index]:
            for event in detailed_events[index][year]:
                length = length + len(event)
    return length

def count_exceedence(input_events, EWR_info):
    events = input_events.copy(deep=True)
    if EWR_info['max_inter-event'] == None:
        return 'N/A'
    else:
        masking = events.isna()
        events[masking] = ''
        total = 0
        for year in events.index:
            if list(events[year]) != '':
                count = len(events[year])
                total = total + count
        return int(total)

def initialise_summary_df_columns(input_dict):
    '''Ingest a dictionary of ewr yearly results and a list of statistical tests to perform
    initialises a dataframe with these as a multilevel heading and returns this'''
    analysis = data_inputs.analysis()
    column_list = []
    list_of_arrays = []
    for scenario, scenario_results in input_dict.items():
        for sub_col in analysis:
            column_list = tuple((scenario, sub_col))
            list_of_arrays.append(column_list)
    
    array_of_arrays =tuple(list_of_arrays)    
    multi_col_df = pd.MultiIndex.from_tuples(array_of_arrays, names = ['scenario', 'type'])
    return multi_col_df
    
def initialise_summary_df_rows(input_dict):
    '''Ingests a dictionary of ewr yearly results
    pulls the location information and the assocaited ewrs at each location,
    saves these as respective indexes and return the multi-level index'''
    
    index_1 = list()
    index_2 = list()
    index_3 = list()
    combined_index = list()
    # Get unique col list:
    for scenario, scenario_results in input_dict.items():
        for site, site_results in scenario_results.items():
            for PU in site_results:
                site_list = []
                for col in site_results[PU]:
                    if '_' in col:
                        all_parts = col.split('_')
                        remove_end = all_parts[:-1]
                        if len(remove_end) > 1:
                            EWR_code = '_'.join(remove_end)
                        else:
                            EWR_code = remove_end[0]
                    else:
                        EWR_code = col
                    if EWR_code in site_list:
                        continue
                    else:
                        site_list.append(EWR_code)
                        add_index = tuple((site, PU, EWR_code))
                        if add_index not in combined_index:
                            combined_index.append(add_index)
    unique_index = tuple(combined_index)
    multi_index = pd.MultiIndex.from_tuples(unique_index, names = ['gauge', 'planning unit', 'EWR'])

    return multi_index

def allocate(df, add_this, idx, site, PU, EWR, scenario, category):
    '''Save element to a location in the dataframe'''
    df.loc[idx[[site], [PU], [EWR]], idx[scenario, category]] = add_this
    
    return df
    
def summarise(input_dict, events):
    '''Ingests a dictionary with ewr pass/fails
    summarises these results and returns a single summary dataframe'''
    PU_items = data_inputs.get_planning_unit_info()
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    # Initialise dataframe with multi level column heading and multi-index:
    multi_col_df = initialise_summary_df_columns(input_dict)
    index = initialise_summary_df_rows(input_dict)
    df = pd.DataFrame(index = index, columns=multi_col_df)
    # Run the analysis and add the results to the dataframe created above:
    for scenario, scenario_results in input_dict.items():
        for site, site_results in scenario_results.items():
            for PU in site_results:
                for col in site_results[PU]:
                    all_parts = col.split('_')
                    remove_end = all_parts[:-1]
                    if len(remove_end) > 1:
                        EWR = '_'.join(remove_end)
                    else:
                        EWR = remove_end[0]
                    idx = pd.IndexSlice
                    if ('_eventYears' in col):
                        S = sum_events(site_results[PU][col])
                        df = allocate(df, S, idx, site, PU, EWR, scenario, 'Event years')
                        
                        F = get_frequency(site_results[PU][col])
                        df = allocate(df, F, idx, site, PU, EWR, scenario, 'Frequency')
                        PU_num = PU_items['PlanningUnitID'].loc[PU_items[PU_items['PlanningUnitName'] == PU].index[0]]
                        EWR_info = evaluate_EWRs.get_EWRs(PU_num, site, EWR, EWR_table, None, ['TF'])
                        TF = EWR_info['frequency']
                        df = allocate(df, TF, idx, site, PU, EWR, scenario, 'Target frequency')
                    elif ('_numAchieved' in col):
                        S = sum_events(site_results[PU][col])
                        df = allocate(df, S, idx, site, PU, EWR, scenario, 'Achievement count')
                        
                        ME = get_average(site_results[PU][col])
                        df = allocate(df, ME, idx, site, PU, EWR, scenario, 'Achievements per year')
                    elif ('_numEvents' in col):
                        S = sum_events(site_results[PU][col])
                        df = allocate(df, S, idx, site, PU, EWR, scenario, 'Event count')
                        
                        ME = get_average(site_results[PU][col])
                        df = allocate(df, ME, idx, site, PU, EWR, scenario, 'Events per year')
                    elif ('_eventLength' in col):
                        EL = get_event_length(events[scenario][site][PU][EWR])
                        df = allocate(df, EL, idx, site, PU, EWR, scenario, 'Event length')
                    elif ('_totalEventDays' in col):
                        AD = get_threshold_days(events[scenario][site][PU][EWR])
                        df = allocate(df, AD, idx, site, PU, EWR, scenario, 'Threshold days')
                    elif ('daysBetweenEvents' in col):
                        PU_num = PU_items['PlanningUnitID'].loc[PU_items[PU_items['PlanningUnitName'] == PU].index[0]]
                        EWR_info = evaluate_EWRs.get_EWRs(PU_num, site, EWR, EWR_table, None, ['MIE'])
                        DB = count_exceedence(site_results[PU][col], EWR_info)
                        df = allocate(df, DB, idx, site, PU, EWR, scenario, 'Inter-event exceedence count')
                        # Also save the max inter-event period to the data summary for reference
                        EWR_info = evaluate_EWRs.get_EWRs(PU_num, site, EWR, EWR_table, None, ['MIE'])
                        MIE = EWR_info['max_inter-event']
                        df = allocate(df, MIE, idx, site, PU, EWR, scenario, 'Max inter event period (years)')
                    elif ('_missingDays' in col):
                        MD = sum_events(site_results[PU][col])
                        df = allocate(df, MD, idx, site, PU, EWR, scenario, 'No data days')
                    elif ('_totalPossibleDays' in col):
                        TD = sum_events(site_results[PU][col])
                        df = allocate(df, TD, idx, site, PU, EWR, scenario, 'Total days')
    return df