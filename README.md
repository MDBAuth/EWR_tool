[![CI](https://github.com/MDBAuth/EWR_tool/actions/workflows/test-release.yml/badge.svg)]()
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py-ewr)](https://pypi.org/project/py-ewr/)
[![PyPI](https://img.shields.io/pypi/v/py-ewr)](https://pypi.org/project/py-ewr/)
[![DOI](https://zenodo.org/badge/342122359.svg)](https://zenodo.org/badge/latestdoi/342122359)

### **EWR tool version 1.0.8 README**

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

# returns a pandas DataFrame with all interEvents of the timeseries
all_interEvents = ewr_oh.get_all_interEvents()

# returns a pandas DataFrame with all successful events of the timeseries
all_successfulEvents = ewr_oh.get_all_successful_events()

# returns a pandas DataFrame with all interevent periods between the successful events of the timeseries
all_successful_interEvents = ewr_oh.get_all_successful_interEvents()


# with the returned object you can use any pandas method like pd.DateFrame.to_csv() etc.

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

# Current model format options: 'Bigmod - MDBA', 'Source - NSW (res.csv)', 'IQQM - NSW 10,000 years' - see manual for formatting requirements
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

# returns a pandas DataFrame with all interEvents of the timeseries
all_interEvents = ewr_sh.get_all_interEvents()

# returns a pandas DataFrame with all successful events of the timeseries
all_successfulEvents = ewr_sh.get_all_successful_events()

# returns a pandas DataFrame with all interevent periods between the successful events of the timeseries
all_successful_interEvents = ewr_sh.get_all_successful_interEvents()

# with the returned object you can use any pandas method like pd.DateFrame.to_csv() etc.

```

### **Purpose**
This tool has two purposes:
1. Operational: Tracking EWR success at gauges of interest in real time.
2. Planning: Comparing EWR success between scenarios (i.e. model runs)

**Support**
For issues relating to the script, a tutorial, or feedback please contact Lara Palmer at lara.palmer@mdba.gov.au, Martin Job at martin.job@mdba.gov.au, or Joel Bailey at joel.bailey@mdba.gov.au


**Disclaimer**
Every effort has been taken to ensure the EWR database represents the original EWRs from state long term water plans as best as possible, and that the code within this tool has been developed to interpret and analyse these EWRs in an accurate way. However, there may still be unresolved bugs in the database and/or EWR tool. Please report any bugs to the issues tab under this GitHub project so we can investigate further. 


**Notes on development of the dataset of EWRs**
The MDBA has worked with NSW to ensure scientific robustness of EWRs has been maintained when translating from raw EWRs in the LTWPs to the machine readable format found in the dataset used by this tool. 

**Compatibility**

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
In the current version of the tool the climate sequence is not used.


