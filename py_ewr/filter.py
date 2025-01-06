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

    Args:
        EWR_df (pd.DataFrame): Dataframe of EWRs
        relevance_df (pd.DataFrame): Dataframe with identifying which EWRs are relevant. 
            Must contain a column headed "relevant", where 1 == TRUE
        EWR_join (list): List of column names to join on for the EWR_df (foreign key)
        relevance_join (list): List of column names to join on, corresponding to the EWR_df (foreign key)

    Results:
        EWRs (pd.DataFrame): Dataframe containing only those EWRs that are relevant
    '''
    # print("step 0:", EWR_df, relevance_df)

    # find duplicates
    find_duplicates(EWR_df, EWR_join)
    find_duplicates(relevance_df, relevance_join)

    merged_df = pd.merge(EWR_df,relevance_df, left_on=EWR_join, right_on=relevance_join, how="left").drop_duplicates(EWR_join)
    print("step1:", merged_df)
    # print("step 1.1: ", merged_df[merged_df['relevant'] == '1'])
    # relevant_EWRs = merged_df[(merged_df['relevant'] == 1) | (merged_df['relevant'] == '') ] # Collect only rows in which relevance == 1
    relevant_EWRs = merged_df[(merged_df['relevant'] != '0')] # Collect only rows in which relevance == 1
    # TODO change the column to be a number not a string
    print("step 2:",relevant_EWRs)
    # Remove the relevance column
    relevant_EWRs = relevant_EWRs.drop('relevant', axis=1)
    print("step3:",relevant_EWRs)


def find_duplicates(df,primary_key):
    # df['duplicate_rows'] = df.duplicated(subset=primary_key)
    df_copy = df[:]
    df_copy['duplicate_rows'] = df_copy.duplicated()
    # print(df)
    duplicates = df_copy[df_copy['duplicate_rows']==True]
    print(duplicates)
    # print(df.duplicated())


if __name__ == "__main__":
    '''
    Still working out how python works, but this should allow us to run the function?
    '''
    print("still working on the get_relevant_EWRs function")
    base_path = os.path.join(BASE_PATH,"py_ewr/parameter_metadata")


    parameter_sheet = os.path.join(base_path,"parameter_sheet.csv")
    # parameter_sheet = "py_ewr/parameter_metadata/parameter_sheet.csv"
    ewr_relevance = os.path.join(base_path,"ewr_relevance.csv")
    # ewr_relevance = "py_ewr/parameter_metadata/ewr_relevance.csv"

    # EWR_join = ['Gauge', 'Code']
    # relevance_join = ['Gauge','Code']
    # EWR_join = ['PlanningUnitName','Gauge', 'Code']
    # relevance_join = ['PlanningUnitName','Gauge','Code']
    EWR_join = ['PlanningUnitName','LTWPShortName','Gauge', 'Code'] # todo NOTE there are duplicates in this version
    relevance_join = ['PlanningUnitName','LTWPShortName','Gauge','Code']


    EWR_df = pd.read_csv(parameter_sheet,
                        usecols=['PlanningUnitID', 'PlanningUnitName',  'LTWPShortName', 'CompliancePoint/Node', 'Gauge', 'Code', 'StartMonth',
                            #   'EndMonth', 'TargetFrequency', 'TargetFrequencyMin', 'TargetFrequencyMax', 'EventsPerYear', 'Duration', 'MinSpell', 
                            #   'FlowThresholdMin', 'FlowThresholdMax', 'MaxInter-event', 'WithinEventGapTolerance', 'WeirpoolGauge', 'FlowLevelVolume', 
                            #   'LevelThresholdMin', 'LevelThresholdMax', 'VolumeThreshold', 'DrawdownRate', 'AccumulationPeriod',
                            #   'Multigauge', 'MaxSpell', 'TriggerDay', 'TriggerMonth', 'DrawDownRateWeek'
                              ],
                     dtype='str', encoding='cp1252'
                    )
    relevance_df = pd.read_csv(ewr_relevance,
                        # usecols=['LTWPShortName', 'Gauge', 'Code', 'relevant'],
                        # usecols=['Gauge','Code','relevant'],
                        # ï»¿LTWPShortName
                     dtype='str', 
                    #  encoding='cp1252'
                     encoding='utf-8-sig'
                    )
    
    # find_duplicates(EWR_df,EWR_join)

    get_relevant_EWRs(EWR_df, relevance_df,EWR_join, relevance_join)