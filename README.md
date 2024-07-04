[![CI](https://github.com/MDBAuth/EWR_tool/actions/workflows/test-release.yml/badge.svg)]()
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py-ewr)](https://pypi.org/project/py-ewr/)
[![PyPI](https://img.shields.io/pypi/v/py-ewr)](https://pypi.org/project/py-ewr/)
[![DOI](https://zenodo.org/badge/342122359.svg)](https://zenodo.org/badge/latestdoi/342122359)

### **EWR tool version 2.1.9 README**

### **Notes on recent version update**
- Standard time-series handling added - each column needs a gauge, followed by and underscore, followed by either flow or level (e.g. 409025_flow). This handling also has missing date filling - so any missing dates will be filled with NaN values in all columns.
- ten thousand year handling - This has been briefly taken offline for this version.
- bug fixes: spells of length equal to the minimum required spell length were getting filtered out of the successful events table and successful interevents table, fixed misclassification of some gauges to flow, level, and lake level categories
- New EWRs: New Qld EWRs - SF_FD and BF_FD used to look into the FD EWRs in closer detail.

### **Installation**

Note - requires Python 3.8 or newer

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
The EWR tool will use a second program called gauge getter to first download the river data at the locations and dates selected and then run this through the EWR tool

```python

from datetime import datetime

#USER INPUT REQUIRED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

dates = {'start_date': datetime(YYYY, 7, 1), 
        'end_date': datetime(YYYY, 6, 30)}

gauges = ['Gauge1', 'Gauge2']

# END USER INPUT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

```

```python

from py_ewr.observed_handling import ObservedHandler

# Running the EWR tool:
ewr_oh = ObservedHandler(gauges=gauges, dates=dates)

# Generating tables:
# Table 1: Summarised EWR results for the entire timeseries
ewr_results = ewr_oh.get_ewr_results()

# Table 2: Summarised EWR results, aggregated to water years:
yearly_ewr_results = ewr_oh.get_yearly_ewr_results()

# Table 3: All events details regardless of duration 
all_events = ewr_oh.get_all_events()

# Table 4: Inverse of Table 3 showing the interevent periods
all_interEvents = ewr_oh.get_all_interEvents()

# Table 5: All events details that also meet the duration requirement:
all_successfulEvents = ewr_oh.get_all_successful_events()

# Table 6: Inverse of Table 5 showing the interevent periods:
all_successful_interEvents = ewr_oh.get_all_successful_interEvents()

```

### Option 2: Running model scenarios through the EWR tool

1. Tell the tool where the model files are (can either be local or in a remote location)
2. Tell the tool what format the model files are in (Current model format options: 'Bigmod - MDBA', 'Source - NSW (res.csv)', 'Standard time-series' - see manual for formatting requirements)

```python
#USER INPUT REQUIRED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Minimum 1 scenario and 1 related file required
scenarios = {'Scenario1': ['file/location/1', 'file/location/2', 'file/location/3'],
             'Scenario2': ['file/location/1', 'file/location/2', 'file/location/3']}

model_format = 'Bigmod - MDBA'

# END USER INPUT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

```

``` python
from py_ewr.scenario_handling import ScenarioHandler
import pandas as pd

ewr_results_dict = {}
yearly_results_dict = {}
all_events_dict = {}
all_interEvents_dict = {}
all_successful_Events_dict = {}
all_successful_interEvents_dict = {}

for scenario_name, scenario_list in scenarios.items():
    ewr_results = pd.DataFrame()
    yearly_ewr_results = pd.DataFrame()
    all_events = pd.DataFrame()
    all_interEvents = pd.DataFrame()
    all_successful_Events = pd.DataFrame()
    all_successful_interEvents = pd.DataFrame()
    for file in scenarios[scenario_name]:

        # Running the EWR tool:
        ewr_sh = ScenarioHandler(scenario_file = file, 
                                model_format = model_format)

        # Return each table and stitch the different files of the same scenario together:
        # Table 1: Summarised EWR results for the entire timeseries
        temp_ewr_results = ewr_sh.get_ewr_results()
        ewr_results = pd.concat([ewr_results, temp_ewr_results], axis = 0)
        # Table 2: Summarised EWR results, aggregated to water years:
        temp_yearly_ewr_results = ewr_sh.get_yearly_ewr_results()
        yearly_ewr_results = pd.concat([yearly_ewr_results, temp_yearly_ewr_results], axis = 0)
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
    ewr_results.to_csv(scenario_name + 'all_results.csv')
    yearly_ewr_results.to_csv(scenario_name + 'yearly_ewr_results.csv')
    all_events.to_csv(scenario_name + 'all_events.csv')
    all_interEvents.to_csv(scenario_name + 'all_interevents.csv')
    all_successful_Events.to_csv(scenario_name + 'all_successful_Events.csv')
    all_successful_interEvents.to_csv(scenario_name + 'all_successful_interEvents.csv')

    # Save the final tables to the dictionaries:   
    ewr_results_dict[scenario_name] = ewr_results
    yearly_results_dict[scenario_name] = yearly_ewr_results
    all_events_dict[scenario_name] = all_events_dict
    all_interEvents_dict[scenario_name] = all_interEvents
    all_successful_Events_dict[scenario_name] = all_successful_Events
    all_successful_interEvents_dict[scenario_name] = all_successful_interEvents


```


### **Purpose**
This tool has two purposes:
1. Operational: Tracking EWR success at gauges of interest in real time - option 1 above.
2. Planning: Comparing EWR success between scenarios (i.e. model runs) - option 2 above.

**Support**
For issues relating to the script, a tutorial, or feedback please contact Lara Palmer at lara.palmer@mdba.gov.au, Martin Job at martin.job@mdba.gov.au, or Joel Bailey at joel.bailey@mdba.gov.au


**Disclaimer**
Every effort has been taken to ensure the EWR database represents the original EWRs from state long term water plans as best as possible, and that the code within this tool has been developed to interpret and analyse these EWRs in an accurate way. However, there may still be unresolved bugs in the EWR parameter sheet and/or EWR tool. Please report any bugs to the issues tab under the GitHub project so we can investigate further. 


**Notes on development of the dataset of EWRs**
The MDBA has worked with Basin state representatives to ensure scientific integrity of EWRs has been maintained when translating from raw EWRs in the Basin state Long Term Water Plans (LTWPs) to the machine readable format found in the parameter sheet within this tool. 

**Compatibility**

NSW:
- All Queensland catchments
- All New South Wales catchments
- All South Australian catchments
- All EWRs from river based Environmental Water Management Plans (EWMPs) in Victoria*

*Currently the wetland EWMPS and mixed wetland-river EWMPs in Victoria contain EWRs that cannot be evaluated by an automated EWR tool so the EWRs from these plans have been left out for now. The MDBA will work with our Victorian colleagues to ensure any updated EWRs in these plans are integrated into the tool where possible.

**Input data**

- Gauge data from the relevant Basin state websites and the Bureau of Meteorology website
- Scenario data input by the user
- Model metadata for location association between gauge ID's and model nodes
- EWR parameter sheet

**Running the tool**

Consult the user manual for instructions on how to run the tool. Please email the above email addresses for a copy of the user manual.

