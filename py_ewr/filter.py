import json
import pandas as pd
# import numpy as np
from pathlib import Path
import os
import logging

BASE_PATH = Path(__file__).resolve().parents[1]   

def get_relevant_EWRs(EWR_df, 
                      relevance_df, 
                      EWR_join = ['LTWPShortName','Gauge', 'Code'], 
                      relevance_join = ['LTWPShortName','gauge','ewr_code']):
    '''Takes in the EWR dataframe, and a data frame with a column representing
    if an EWR is relevant to a project.

    The default is to assume that a project is relevant

    Args:
        EWR_df (pd.DataFrame): Dataframe of EWRs
        relevance_df (pd.DataFrame): Dataframe with identifying which EWRs are relevant. 
            Must contain a column headed "relevant", where 1 == TRUE
        EWR_join (list): List of column names to join on for the EWR_df (foreign key)
        relevance_join (list): List of column names to join on, corresponding to the EWR_df (foreign key)

    Results:
        EWRs (pd.DataFrame): Dataframe containing only those EWRs that are relevant
    '''
    # find duplicates
    find_duplicates(EWR_df, EWR_join)
    find_duplicates(relevance_df, relevance_join)

    # Merge dataframes on their unique identifiers, and if there are duplicates (i.e. the unique ID is not unique) remove them
    merged_df = pd.merge(EWR_df,relevance_df, left_on=EWR_join, right_on=relevance_join, how="left").drop_duplicates(EWR_join)
    print("step1:", merged_df)

    # Filter dataframe for relevant EWRs. Assume that if there is no entry, the EWR is relevant.
    relevant_EWRs = merged_df[(merged_df['relevant'] != 0)]
    print("step 2:",relevant_EWRs)

    # Tidy up and remove the relevance column
    relevant_EWRs = relevant_EWRs.drop('relevant', axis=1)
    print("step3:",relevant_EWRs)
    return relevant_EWRs


def find_duplicates(df,primary_key):

    df_copy = df.loc[:]
    df_copy['duplicate_rows'] = df_copy.duplicated(subset=primary_key)
    duplicates = df_copy[df_copy['duplicate_rows']==True]
    print(duplicates)


if __name__ == "__main__":
    # Set up paths
    base_path = os.path.join(BASE_PATH,"py_ewr/parameter_metadata")
    parameter_sheet = os.path.join(base_path,"parameter_sheet.csv")
    ewr_relevance = os.path.join(base_path,"ewr_relevance.csv")
    small_relevance = os.path.join(base_path,"small_relevance.csv")

    # What column headings we should join on. One-to-one relationship
    # EWR_join = ['PlanningUnitName','LTWPShortName','Gauge', 'Code'] # todo NOTE there are duplicates in this version
    # relevance_join = ['PlanningUnitName','LTWPShortName','gauge','Code']
    EWR_join = ['Gauge']
    relevance_join = ['gauge']

    # Load data
    EWR_df = pd.read_csv(parameter_sheet,
                         usecols=['PlanningUnitID', 'PlanningUnitName',  'LTWPShortName', 'CompliancePoint/Node', 'Gauge', 'Code', 'StartMonth',
                            #   'EndMonth', 'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax', 'EventsPerYear', 'Duration', 'MinSpell', 
                            #   'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                            #   'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'AccumulationPeriod',
                            #   'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek'=
                         ],
                         dtype='str', 
                         encoding='cp1252'
                         )
    # relevance_df = pd.read_csv(ewr_relevance,
    #                            usecols=['PlanningUnitName','LTWPShortName', 'gauge', 'Code', 'relevant'],
    #                            dtype={
    #                                'PlanningUnitName':'str',
    #                                'LTWPShortName':'str', 
    #                                'Gauge':'str', 
    #                                'Code':'str',
    #                                'relevant':'float'
    #                            },
    #                         #    dtype='str', 
    #                            encoding='utf-8-sig'
                            #    )
    relevance_df = pd.read_csv(small_relevance,
                               usecols=['gauge','relevant'],
                               dtype={
                                   'gauge':'str',
                                   'relevant':'float'
                               },
                               encoding='utf-8-sig')
    
    get_relevant_EWRs(EWR_df, relevance_df,EWR_join, relevance_join)