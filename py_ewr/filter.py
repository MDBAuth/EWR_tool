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
        EWR_df
        todo zofia finish this 
    '''

    merged_df = pd.merge(EWR_df,)

print(test)