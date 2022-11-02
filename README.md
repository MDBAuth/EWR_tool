[![CI](https://github.com/MDBAuth/EWR_tool/actions/workflows/test-release.yml/badge.svg)]()
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py-ewr)](https://pypi.org/project/py-ewr/)
[![PyPI](https://img.shields.io/pypi/v/py-ewr)](https://pypi.org/project/py-ewr/)


### **EWR tool beta version 0.8.9 README**

*************
The EWR tool was developed by Ben Wolfenden (NSW DPE), Martin Job (MDBA) and Pedro Junqueira (Agile Analytics).

The method within the code is based on the EWR assessment method created by Ian Burns (NSW DPE).

We would like to thank Carmen Amos (NSW DPE), Markus Buerle (NSW DPE), Lara Palmer (MDBA), Ben Bradshaw (MDBA), Joel Bailey (MDBA) Dennis Stahl (Agile Analytics) and Blake Lawrence (Agile Analytics)  for their work on putting together the EWR dataset, refinement of the codebase, and input to the logic.
*************


### **Installation**

Step 1. 
Upgrade pip
```bash
python -m pip install –-upgrade pip
```

Step 2.
```bash
pip install py-ewr
``` 

### Run Timeseries with ObservedHandler

```python
from datetime import datetime

from py_ewr.observed_handling import ObservedHandler

dates = {'start_date': datetime(2020, 7, 1), 
        'end_date': datetime(2021, 6, 30)}

# Get allowances:

MINT = (100 - 0)/100
MAXT = (100 + 0 )/100
DUR = (100 - 0 )/100
DRAW = (100 -0 )/100

allowance ={'minThreshold': MINT, 'maxThreshold': MAXT, 'duration': DUR, 'drawdown': DRAW}

gauges = ['419039']
climate = "Standard - 1911 to 2018 climate categorisation"

# instantiate ObservedHandler

ewr_oh = ObservedHandler(gauges=gauges, dates=dates , allowance=allowance, climate=climate)


# ObservedHandler methods

# returns a pandas DataFrame with ewr results for the timeseries
ewr_results = ewr_oh.get_ewr_results()

# returns a pandas DataFrame with the yearly ewr results for the timeseries
yearly_ewr_results = ewr_oh.get_yearly_ewr_results()

# returns a pandas DataFrame with all events of the timeseries
all_events = ewr_oh.get_all_events()

# print DataFrame  head of the results
# with the returned object you can use any pandas method like pd.DateFrame.to_csv() etc.

# print("ewr_results","\n")
# print(ewr_results.head())
# print("all_events""\n")
# print(all_events.head())
# print("yearly_ewr_results""\n")
# print(yearly_ewr_results.head())

```

### Run Timeseries with ScenarioHandler

```python
from py_ewr.scenario_handling import ScenarioHandler

# pass a list of location of the scenario files
# this example will pass a IQQDM format scenario read the pdf manual for details
loaded_files = ["419039_gauge_data_IQQDM.csv"]

# Get allowances:

MINT = (100 - 0)/100
MAXT = (100 + 0 )/100
DUR = (100 - 0 )/100
DRAW = (100 -0 )/100

allowance ={'minThreshold': MINT, 'maxThreshold': MAXT, 'duration': DUR, 'drawdown': DRAW}


ewr_sh = ScenarioHandler(scenario_files = loaded_files, 
                         model_format = 'IQQM - NSW 10,000 years', 
                         allowance = allowance, 
                         climate = 'Standard - 1911 to 2018 climate categorisation' )


# ScenarioHandler methods

# returns a pandas DataFrame with ewr results for the timeseries
ewr_results = ewr_sh.get_ewr_results()

# returns a pandas DataFrame with the yearly ewr results for the timeseries
yearly_ewr_results = ewr_sh.get_yearly_ewr_results()

# returns a pandas DataFrame with all events of the timeseries
all_events = ewr_sh.get_all_events()

```

### **Purpose**
This tool has two purposes:
1. Operational: Tracking EWR success at gauges of interest in real time.
2. Planning: Comparing EWR success between scenarios (i.e. model runs)

**Support**
For issues relating to the script, a tutorial, or feedback please contact Martin Job at martin.job@mdba.gov.au or Joel Bailey at joel.bailey@mdba.gov.au

**Notes on development of the tool**

This is the version 0.8.9 of the EWR tool. Testing is still being undertaken.


**Disclaimer**
Every effort has been taken to ensure the EWR database represents the original EWRs from state long term water plans as best as possible, and that the code within this tool has been developed to interpret and analyse these EWRs in an accurate way. However, there may still be unresolved bugs in the database and/or EWR tool. Please report any bugs to the issues tab under this GitHub project so we can investigate further. 


**Notes on development of the database**
The Environmental Assets & Functions Database (EAFD) migration to a machine readable format is underway. This migration may impact on the intricacies of the original EWRs. The MDBA has started working with NSW to ensure the translation from EWRs as they are written in the long term water plans to how they are interpreted by this tool is done in a scientifically robust manner.

**Compatibility**
The tool can currently evaluate most to all of EWRs in the following catchments. Evaluation of EWRs is largely dependent on the migration of the Environmental Assets & Functions Database (EAFD) database into a machine readable format.

NSW:
- All NSW catchments

Work is currently underway to migrate the EWRs in the remaining Basin catchments.

**Input data**
- EWR information: This tool accesses the EWRs in the Environmental Assets & Functions Database (EAFD)
- Climate data from the AWRA-L model
- Gauge data from the relevant state websites
- Scenario data input by the user
- Model metadata for location association between gauge ID's and model nodes

**Running the tool**
Consult the user manual for instructions on how to run the tool. Please email the above email addresses for a copy of the user manual.

**Climate sequence**
NSW Long Term Watering Plans (LTWP) define climate using the Resource Availability Scenarios (RAS). However, until this process can be completed the climate categories defined using outputs from the AWRA-L model will be used.  


