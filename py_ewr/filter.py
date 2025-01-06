import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging



def get_relevant_EWRs(EWR_df, 
                      relevance_df, 
                      EWR_join = ['LTWPShortName','Gauge', 'Code'], 
                      relevance_join = ['LTWPShortName','gauge','ewr_code']):
    '''Takes in the EWR dataframe, and a data frame with a column representing
    if an EWR is relevant to a project.

    Args:
        EWR_df (pd.DataFrame): Dataframe of EWRs
        relevance_df (pd.DataFrame): Dataframe with identifying which EWRs are relevant
        EWR_join (list): List of column names to join on for the EWR_df (foreign key)
        relevance_join (list): List of column names to join on, corresponding to the EWR_df (foreign key)

    Results:
        EWRs (pd.DataFrame): Dataframe containing only those EWRs that are relevant
    '''

    merged_df = pd.merge(EWR_df,relevance_df)

help()

# if __name__ == "__main__":
#     '''
#     Still working out how python works, but this should allow us to run the function?
#     '''
#     print("still working on the get_relevant_EWRs function")

    # get_relevant_EWRs() # todo, add all the parameters