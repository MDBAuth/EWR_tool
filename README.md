### **EWR tool 0.0.3 README**

**Purpose**
This tool has two purposes:
1. Operational: Tracking EWR success at gauges of interest in real time. This option has not been updated since the beta version 0.0.1 of the tool and is currently unstable. Work is being undertaken to update this option.
2. Planning: Comparing EWR success between scenarios (i.e. model runs)

**Support**
For issues relating to the script, a tutorial, or feedback please contact Martin Job at martin.job@mdba.gov.au or Joel Bailey at joel.bailey@mdba.gov.au

**Notes on development of the tool**
This is the version 0.03 of the EWR tool. Testing is still being undertaken.

**Notes on development of the database**
The Environmental Assets & Functions Database (EAFD) migration to a machine readable format is underway. This migration may impact on the intricacies of the original EWRs. The MDBA will begin hosting a series of workshops with the relevant state bodies to ensure the integrity of the EWRs is not lost in this migration.

**Compatability**
The tool can currently evaluate most to all of EWRs in the following catchments. Evaluation of EWRs is largely dependent on the migration of the Environmental Assets & Functions Database (EAFD) database into a machine readable format.

NSW:
- NSW Murray Lower Darling
- Murrumbidgee
- Gwydir
- Lachlan
- Macquarie-Castlereagh

Work is currently underway to migrate the EWRs in the remaining Basin catchments.

**Input data**
- EWR information: This tool accesses the EWRs in the Environmental Assets & Functions Database (EAFD)
- Climate data from the AWRA-L model
- Gauge data from the relevant state websites
- Scenario data input by the user
- Bigmod metadata for location association between gauge ID's and model nodes

**Running the tool**
Consult the user manual for instructions on how to run the tool

**Climate sequence**
NSW Long Term Watering Plans (LTWP) define climate using the Resource Availability Scenarios (RAS). However, until this process can be completed the climate categories defined using outputs from the AWRA-L model will be used.  
