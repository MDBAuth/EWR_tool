'''Objective Weighting Handler

This script allows the user to assign weightings to EWRs for each relevant 
objective to improve the nuance of aggregation of the EWR tool outputs. 

It requires TODO list requirements

It includes the following functions:
    * generate_objective_csv - using the ewr2obj csv (or equivalent) creates a simpler csv with the unique EWR identification columns, and one row per EWR per overall objective. i.e. summarises the information in ewr2obj by collapsing sub-objectives into one objective (e.g. NF1, NF2, ect. --> NF)
    * subobj2obj - function that uses string conversion to simplify a subobjective (e.g. WB1) to an objective (e.g. WB)
    * generate_priority_csv - merges the ewr_output csv, with the ewr2obj csv, and finally with the ewr_obj_priority csv (NOTE alternative is to do weighting first????)
    * calculate_weighting - given a csv with ewr identifying columns, objective column, and a priority column, along with a desired grouping (default is by objective), assign a weighting to each row. The sum of all the weightings in each category should equal 1.

TODO revibe these functions
    1. generate a csv with the condensed grouping
'''
import pandas as pd
import os
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1]   

class PriorityHandling:

    def __init__(self):
        self.possible_subobjectives=[] # used for me to figure out what's happening
        base_path = os.path.join(BASE_PATH,"py_ewr/parameter_metadata")
        ewr2obj_path = os.path.join(base_path,"ewr2obj.csv")
        self.ewr2obj_df = pd.read_csv(ewr2obj_path)
        self.unique_ewr_id = ['PlanningUnitName','LTWPShortName','gauge','Code']

        # pass


def subobj2obj(subobjective: str, objective_list=['NF','NV','WB','OS','EF']):
    '''Takes a string an returns the objective from either NF, NV, WB, OS and EF.
    Requires string to have the objective code in the first 2 characters'''
    # objective_list = ['NF','NV','WB','OS','EF']
    objective = subobjective[:2]
    if (objective in objective_list):
        # New South Wales and other base stuff
        # print("It worked",objective, subobjective)
        return objective
    else:
        match objective:
            case 'Q-':
                # Queensland objectives
                print('Queensland: ', objective)
                subobj2obj(subobjective[2:])
            case 'SA':
                # South Australia - this is a bit confusing TODO
                print('South Australia, not sure what to do with this rn',subobjective)
                # I think the deal is gonna be that we have SA, _, location, _, objective
                breakdown = subobjective.split('_')
                print(breakdown)
                if (breakdown[2] not in possible_subobjectives):
                    possible_subobjectives.append(breakdown[2])
                subobj2obj(breakdown[2])
                '''
                SA_CLLMM plus:
                    _F11_a/b/d/e1/e2/f1/f2/g1/j (Native fish)
                    _MI1_a1/a2/b1/c/d1/d2/
                    _MI2_a/b
                    CLLMM = Coorong, Lower Lakes and Murray Mouth
                
                FP = flood plains
                [
                Ecosystem Functions? = Biofilms             = 'BF1',
                Ecosystem Functions? = Ecosystem Processes  = 'EP9', 'EP6', 'EP7', 'EP8', 'EP3','EP2', 'EP4', 'EP1', 'EP5', 
                Native Fish?         = Fish                 = 'F11', 'F7', 'F8', 'F10', 'F9','F5', 'F6', 'F2', 'F3', 'F4', 'F1',
                Other Species?       = ???                  = 'FR1',
                Ecosystem Functions? = Ground Water and Soil = 'GWS2', 'GWS3', 'GWS4',  'GWS1', 
                Ecosystem Functions? = ???                  = 'MI1', 'MI2', 
                Other Species?       = ???                  = 'OF1', 
                Native Vegetation?   = Vegetation           = 'V10', 'V9', 'V8', 'V7', 'V3', 'V4', 'V5','V6','V1','V2
                Waterbirds?          = ???                  = 'WB3', 'WB2', 'WB1',  
                Ecosystem Functions? = Water Quality        = 'WQ3', 'WQ2','WQ1',
                ]
                '''
                # NOTE based on this, i think it might be too challenging for SA, and I should just be doing this by the Target column
            # case

        # print("BOOO", objective)
        # TODO add the different state interpretations of the objective codes to the list of objectives. You may need to make a case statement or something.
        return 'Unknown'

def clean_targets(ewr):
    targets = ['Native fish',
               'Ecosystem function',
               'Native vegetation',
               'Waterbirds',
               'Other Species']
    if ewr['Target'] in targets:
        return ewr
    else:
        intended_target = ewr['Target']
        intended_target = intended_target.lower()
        for target in targets:
            # check if it's a case issue
            if intended_target == target.lower():
                ewr['Target'] = target
                return ewr
        # Check for Native Fish
        if 'fish' in intended_target:
            if not 'native' in intended_target:
                print('This ewr might not actually be native fish, but does contain the string fish:', ewr)
            ewr['Target'] = 'Native Fish'
            return ewr
        # Check for Native Vegetation
        if 'veg' in intended_target:
            if not 'native' in intended_target:
                print('This ewr might not actually be native vegetation, but does contain the string veg:', ewr)
            ewr['Target'] = 'Native Vegetation'
            return ewr
        # Check for Ecosystem Function
        if 'ecosystem' in intended_target or 'function' in intended_target:
            if not 'ecosystem' in intended_target or not 'function' in intended_target:
                print('This ewr contains either ecosystem or function, but not both:', ewr)
            ewr['Target'] = 'Ecosystem Function'
            return ewr
        # Check for Other Species
        if 'other' in intended_target or 'species' in intended_target:
            if not 'other' in intended_target or not 'species' in intended_target:
                print('This ewr contains either "other" or "species", but not both:', ewr)
            ewr['Target'] = 'Ecosystem Function'
            return ewr
        # Check if it might be Waterbirds
        if 'water' in intended_target and 'bird' in intended_target:
            ewr['Target'] = 'Waterbirds'
            return ewr

def generate_objective_csv(ewr2obj_df, primary_key) -> pd.DataFrame:
    # remove duplicate rows
    df = ewr2obj_df[primary_key + ['Target']]
    print(df)
    # Doing an initial drop duplicates to reduce the number of rows before doing my slower thing.
    df=df.drop_duplicates(primary_key+['Target'])
    print(df)
    # Clean up the remaining rows, before dropping again
    df.apply(clean_targets,axis=1)
    df=df.drop_duplicates(primary_key+['Target'])
    df = df.assign(priority=2) # Add priority column, default value of 2
    print(df)
    df.to_csv(os.path.join(base_path,'ewr_target_priority.csv'),index=False)
    return df

if __name__ == "__main__":
    possible_subobjectives=[] # used for me to figure out what's happening
    base_path = os.path.join(BASE_PATH,"py_ewr/parameter_metadata")
    ewr2obj_path = os.path.join(base_path,"ewr2obj.csv")
    ewr2obj_df = pd.read_csv(ewr2obj_path, index_col=False)
    unique_ewr_id = ['planning_unit_name','LTWPShortName','gauge','ewr_code_timing']

    # ewr2obj_df['objective'] = ewr2obj_df['env_obj'].map(subobj2obj)
    # print(possible_subobjectives)
    print(ewr2obj_df)
    generate_objective_csv(ewr2obj_df,unique_ewr_id)



'''

'''



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

