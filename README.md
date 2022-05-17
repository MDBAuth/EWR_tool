[![CI](https://github.com/agile-analytics-au/EWR_tool/actions/workflows/tox-test.yml/badge.svg)]()


### **EWR tool beta 0.0.6 README**

**Purpose**
This tool has two purposes:
1. Operational: Tracking EWR success at gauges of interest in real time.
2. Planning: Comparing EWR success between scenarios (i.e. model runs)

**Support**
For issues relating to the script, a tutorial, or feedback please contact Martin Job at martin.job@mdba.gov.au or Joel Bailey at joel.bailey@mdba.gov.au

**Notes on development of the tool**
This is the version 0.0.6 of the EWR tool. Testing is still being undertaken.

**Disclaimer**
Every effort has been taken to ensure the EWR database represents the original EWRs from state long term water plans as best as possible, and that the code within this tool has been developed to interpret and analyse these EWRs in an accurate way. However, there may still be unresolved bugs in the database and/or EWR tool. Please report any bugs to the issues tab under this GitHub project so we can investigate further. 


**Notes on development of the database**
The Environmental Assets & Functions Database (EAFD) migration to a machine readable format is underway. This migration may impact on the intricacies of the original EWRs. The MDBA has started working with NSW to ensure the translation from EWRs as they are written in the long term water plans to how they are interpreted by this tool is done in a scientifically robust manner.

**Compatability**
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
