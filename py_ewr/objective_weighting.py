'''Objective Weighting Handler

This script allows the user to assign weightings to EWRs for each relevant 
objective to improve the nuance of aggregation of the EWR tool outputs. 

It requires TODO list requirements

It includes the following functions:
    * generate_objective_csv - using the ewr2obj csv (or equivalent) creates a simpler csv with the unique EWR identification columns, and one row per EWR per overall objective. i.e. summarises the information in ewr2obj by collapsing sub-objectives into one objective (e.g. NF1, NF2, ect. --> NF)
    * subobj2obj - function that uses string conversion to simplify a subobjective (e.g. WB1) to an objective (e.g. WB)
    * generate_priority_csv - merges the ewr_output csv, with the ewr2obj csv, and finally with the ewr_obj_priority csv (NOTE alternative is to do weighting first????)
    * calculate_weighting - given a csv with ewr identifying columns, objective column, and a priority column, along with a desired grouping (default is by objective), assign a weighting to each row. The sum of all the weightings in each category should equal 1.
'''
import pandas as pd
import os
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1]   

class PriorityHandling:

    def __init__(self):
        pass


def subobj2obj(subobjective: str):
    '''Takes a string an returns the objective from either NF, NV, WB, OS and EF.
    Requires string to have the objective code in the first 2 characters'''
    objective_list = ['NF','NV','WB','OS','EF']
    objective = subobjective[:2]
    if (objective in objective_list):
        # print("It worked",objective)
        return objective
    else:
        print("BOOO", objective)
        # TODO add the different state interpretations of the objective codes to the list of objectives. You may need to make a case statement or something.
        return 'Unknown'

# def generate_objective_csv()


if __name__ == "__main__":
    base_path = os.path.join(BASE_PATH,"py_ewr/parameter_metadata")

    ewr2obj_path = os.path.join(base_path,"ewr2obj.csv")
    ewr2obj_df = pd.read_csv(ewr2obj_path)

    ewr2obj_df['objective'] = ewr2obj_df['env_obj'].map(subobj2obj)
    print(ewr2obj_df)





# Dump of little style notes:
# Module docstrings are similar to class docstrings. Instead of classes and class methods being documented, it’s now the module and any functions found within. Module docstrings are placed at the top of the file even before any imports. Module docstrings should include the following:

# A brief description of the module and its purpose
# A list of any classes, exception, functions, and any other objects exported by the module

# The docstring for a module function should include the same items as a class method:

# A brief description of what the function is and what it’s used for
# Any arguments (both required and optional) that are passed including keyword arguments
# Label any arguments that are considered optional
# Any side effects that occur when executing the function
# Any exceptions that are raised
# Any restrictions on when the function can be called


# his script allows the user to print to the console all columns in the
# spreadsheet. It is assumed that the first row of the spreadsheet is the
# location of the columns.

# This tool accepts comma separated value files (.csv) as well as excel
# (.xls, .xlsx) files.

# This script requires that `pandas` be installed within the Python
# environment you are running this script in.

# This file can also be imported as a module and contains the following
# functions:

#     * get_spreadsheet_cols - returns the column headers of the file
#     * main - the main function of the script

