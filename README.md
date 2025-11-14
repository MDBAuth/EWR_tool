[![CI](https://github.com/MDBAuth/EWR_tool/actions/workflows/test-release.yml/badge.svg)]()
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py-ewr)](https://pypi.org/project/py-ewr/)
[![PyPI](https://img.shields.io/pypi/v/py-ewr)](https://pypi.org/project/py-ewr/)
[![DOI](https://zenodo.org/badge/342122359.svg)](https://zenodo.org/badge/latestdoi/342122359)

### **ewr tool version 2.3.9 README**

### **Notes on recent version updates**

#### EWR handling and outputs
- Adding handling for cases where there are single MDBA bigmod site IDs mapping to multiple different gauges
- New EWRs: New Qld EWRs - SF_FD and BF_FD used to look into the FD EWRs in closer detail.
- Adding state and Surface Water SDL (SWSDL) to py-ewr output tables
-  Including metadata report (this is still being ironed out and tested)
-  New handling capabilities for FIRM model formated files

#### Model metadata
- Added new FIRM ID file mapping FIRM ID to gauge number.

#### parameter metadata
- Updated parameter sheet and objective mapping correcting some mismatched links between environmental objectives and EWRs
- Fix SDL resource unit mapping in the parameter sheet
- Adding lat and lon to the parameter sheet
- Added in model handling for FIRM model outputs
- Various minor variable renamings for consistency
- Renamed MaxLevelRise to MaxLevelChange
- Removed AnnualFlowSum and MinLevelRise column from parameter sheet
- New format of objective mapping includes the adding of objective mapping back into the parameter sheet and one secondary dataframe (objective_reference.csv) in parameter metadata.

### **Installation**

Note - requires Python 3.9 to 3.13 (inclusive)

Step 1. 
Upgrade pip
```bash
python -m pip install –-upgrade pip
```

Step 2.
```bash
pip install py-ewr
``` 

### Option 1: Running the observed mode of the tool
The ewr tool will use a second program called gauge getter to first download the river data at the locations and dates selected and then run this through the ewr tool.
For more information please visit the [MDBA Gauge Getter](https://github.com/MDBAuth/MDBA_Gauge_Getter) github page. 

```python

from datetime import datetime

# USER INPUT REQUIRED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

dates = {'start_date': datetime(YYYY, 7, 1), 
        'end_date': datetime(YYYY, 6, 30)}

gauges = ['Gauge1', 'Gauge2']

# END USER INPUT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

```

```python

from py_ewr.observed_handling import ObservedHandler

# Running the ewr tool:
ewr_oh = ObservedHandler(gauges=gauges, dates=dates)

# Generating tables:
# Table 1: Summarised ewr results for the entire timeseries
ewr_results = ewr_oh.get_ewr_results()

# Table 2: Summarised ewr results, aggregated to water years:
yearly_results = ewr_oh.get_yearly_results()

# Table 3: All events details regardless of duration 
all_events = ewr_oh.get_all_events()

# Table 4: Inverse of Table 3 showing the interevent periods
all_interEvents = ewr_oh.get_all_interEvents()

# Table 5: All events details that also meet the duration requirement:
all_successfulEvents = ewr_oh.get_all_successful_events()

# Table 6: Inverse of Table 5 showing the interevent periods:
all_successful_interEvents = ewr_oh.get_all_successful_interEvents()

```

### Option 2: Running model scenarios through the ewr tool

1. Tell the tool where the model files are (can either be local or in a remote location). For each scenario list the file path of each file. Python will default to the path relative to the directory it is being run from, but if you put a full file path you can choose any file.
```python
scenarios = {'Scenario1': ['path/to/file1', 'path/to/file2', 'path/to/file3'],
             'Scenario2': ['path/to/file1', 'path/to/file2', 'path/to/file3']}
```
2. Tell the tool what format the model files are in. The current model format options are: 
    - 'Bigmod - MDBA'
        Bigmod formatted outputs
      
    - 'Source - NSW (res.csv)'
        Source res.csv formatted outputs
      
    - 'FIRM - MDBA'
        FIRM ID formatted outputs
      
    - 'Standard time-series'
        The first column header should be *Date* with the date values in the YYYY-MM-DD format.
        The next columns should have the *gauge* followed by *_* followed by either *flow* or *level*
        E.g.
        | Date | 409025_flow | 409025_level | 414203_flow |
        | --- | --- | --- | --- |
        | 1895-07-01 | 8505 | 5.25 | 8500 |
        | 1895-07-02 | 8510 | 5.26 | 8505 |

    - 'ten thousand year'
        This has the same formatting requirements as the 'Standard time-series'. This can handle ten thousand years worth of hydrology data.
        The first column header should be *Date* with the date values in the YYYY-MM-DD format.
        The next columns should have the *gauge* followed by *_* followed by either *flow* or *level*
        E.g.
        | Date | 409025_flow | 409025_level | 414203_flow |
        | --- | --- | --- | --- |
        | 105-07-01 | 8505 | 5.25 | 8500 |
        | 105-07-02 | 8510 | 5.26 | 8505 |


```python
# USER INPUT REQUIRED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Minimum 1 scenario and 1 related file required
scenarios = {'Scenario1': ['path/to/file1', 'path/to/file2', 'path/to/file3'],
             'Scenario2': ['path/to/file1', 'path/to/file2', 'path/to/file3']}

model_format = 'Bigmod - MDBA'

#-----  other model formats include -----#
# 'Bigmod - MDBA'
# 'Source - NSW (res.csv)'
# 'FIRM - MDBA'
# 'Standard time-series'
# 'ten thousand year'

# END USER INPUT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

```
File names will be incorporated into the Scenario column of the ewr tool output tables for tracebility of ewr tool output and corresponding model file. In the code written below, scenario name will be appendeded to the file name. eg. Scenario_name1_all_results.csv, therefore it is suggested to have informative scenario names that can traceback to your model files as well.

``` python
from py_ewr.scenario_handling import ScenarioHandler
import pandas as pd

summary_results_dict = {}
yearly_results_dict = {}
all_events_dict = {}
all_interEvents_dict = {}
all_successful_Events_dict = {}
all_successful_interEvents_dict = {}

for scenario_name, scenario_list in scenarios.items():
    summary_results = pd.DataFrame()
    yearly_results = pd.DataFrame()
    all_events = pd.DataFrame()
    all_interEvents = pd.DataFrame()
    all_successful_Events = pd.DataFrame()
    all_successful_interEvents = pd.DataFrame()
    for file in scenarios[scenario_name]:

        # Running the ewr tool:
        ewr_sh = ScenarioHandler(scenario_file = file, 
                                model_format = model_format)

        # Return each table and stitch the different files of the same scenario together:
        # Table 1: Summarised ewr results for the entire timeseries
        temp_summary_results = ewr_sh.get_ewr_results()
        summary_results = pd.concat([summary_results, temp_summary_results], axis = 0)
        # Table 2: Summarised ewr results, aggregated to water years:
        temp_yearly_results = ewr_sh.get_yearly_ewr_results()
        yearly_results = pd.concat([yearly_results, temp_yearly_results], axis = 0)
        # Table 3: All events details regardless of duration 
        temp_all_events = ewr_sh.get_all_events()
        all_events = pd.concat([all_events, temp_all_events], axis = 0)
        # Table 4: Inverse of Table 3 showing the interevent periods
        temp_all_interEvents = ewr_sh.get_all_interEvents()
        all_interEvents = pd.concat([all_interEvents, temp_all_interEvents], axis = 0)
        # Table 5: All events details that also meet the duration requirement:
        temp_all_successfulEvents = ewr_sh.get_all_successful_events()
        all_successful_Events = pd.concat([all_successful_Events, temp_all_successfulEvents], axis = 0)
        # Table 6: Inverse of Table 5 showing the interevent periods:
        temp_all_successful_interEvents = ewr_sh.get_all_successful_interEvents()
        all_successful_interEvents = pd.concat([all_successful_interEvents, temp_all_successful_interEvents], axis = 0)
        

    # Optional code to output results to csv files:
    summary_results.to_csv(scenario_name + 'summary_results.csv')
    yearly_results.to_csv(scenario_name + 'yearly_results.csv')
    all_events.to_csv(scenario_name + 'all_events.csv')
    all_interEvents.to_csv(scenario_name + 'all_interevents.csv')
    all_successful_Events.to_csv(scenario_name + 'all_successful_Events.csv')
    all_successful_interEvents.to_csv(scenario_name + 'all_successful_interEvents.csv')

    # Save the final tables to the dictionaries:   
    summary_results_dict[scenario_name] = summary_results
    yearly_results_dict[scenario_name] = yearly_results
    all_events_dict[scenario_name] = all_events
    all_interEvents_dict[scenario_name] = all_interEvents
    all_successful_Events_dict[scenario_name] = all_successful_Events
    all_successful_interEvents_dict[scenario_name] = all_successful_interEvents


```
#### Optional arguments for ScenarioHandler
```python
        ewr_sh = ScenarioHandler(scenario_file = file, 
                                model_format = model_format,
                                parameter_sheet = parameter_sheet,
                                calc_config_path = calc_config_path)
```
You may add a custom parameter sheet and or calc_config_file to your EWR tool run using the ```parameter_sheet``` and ```calc_config_path``` arguments.  These arguments take a string file path pointing to files. Please check this ewr_calc_config.json file found in parameter metadata to see if any EWRs in your custom parameter sheet are not represented in the calc_config_file. For an EWR to be calculated, it must be found in both calc_config.json and the parameter sheet.

### **Purpose**
This tool has two purposes:
1. Operational: Tracking ewr success at gauges of interest in real time - option 1 above.
2. Planning: Comparing ewr success between scenarios (i.e. model runs) - option 2 above.

**Support**
For issues relating to the script, a tutorial, or feedback please contact Martin Job at martin.job@mdba.gov.au, Sirous Safari Pour at sirous.safaripour@mdba.gov.au, Elisha Freedman at elisha.freedman@mdba.gov.au, Joel Bailey at joel.bailey@mdba.gov.au, or Lara Palmer at lara.palmer@mdba.gov.au.


**Disclaimer**
Every effort has been taken to ensure the ewr database represents the original EWRs from state Long Term Water Plans (LTWPs) and Environmental Water Management Plans (EWMPs) as best as possible, and that the code within this tool has been developed to interpret and analyse these EWRs in an accurate way. However, there may still be unresolved bugs in the ewr parameter sheet and/or ewr tool. Please report any bugs to the issues tab under the GitHub project so we can investigate further. 


**Notes on development of the dataset of EWRs**
The MDBA has worked with Basin state representatives to ensure scientific integrity of EWRs has been maintained when translating from raw EWRs in the Basin state LTWPs and EWMPs to the machine readable format found in the parameter sheet within this tool. 

Environmental Water Requirements (EWRs) in the tool are subject to change when the relevant documents including Long Term Water Plans (LTWPs) and Environmental Water Management Plans (EWMPs) are updated or move from draft to final versions. LTWPs that are currently in draft form include the ACT and the upper Murrumbidgee section of the NSW Murrumbidgee LTWP.

**Compatibility**

- All Queensland catchments
- All New South Wales catchments
- All South Australian catchments
- All EWRs from river based Environmental Water Management Plans (EWMPs) in Victoria*
- All EWRs for ACT (draft version)

*The wetland EWMPS and mixed wetland-river EWMPs in Victoria are in progress. The MDBA will work with our Victorian colleagues to ensure any updated EWRs in these plans are integrated into the tool where possible.

**Input data**

- Gauge data from the relevant Basin state websites and the Bureau of Meteorology website
- Scenario data input by the user
- Model metadata for location association between gauge ID's and model nodes
#### optional
- ewr parameter sheet
- calc_config.json 

**Objective mapping**
The objective mapping is located in the EWR tool package. This is intended to be used to link EWRs to the detailed objectives, theme level targets and specific goals. The objective reference file is located in the py_ewr/parameter_metadata folder:

parameter_sheet.csv (EnvObj column)
obj_reference.csv

Contains the individual environmnetal objectives listed in the 'EnvObj' column of the parameter sheet and their ecological targets (Target) and plain english description of objectives (Objectives) for each planning unit, long term water plan (LTWPShortName), and surface water sustainable diversion limit (SWSDLName).
the function ```get_obj_mapping()``` is available to automatically merge the information from obj_reference.csv with the parameter sheet to link these objectives with their specific ewr_codes.
