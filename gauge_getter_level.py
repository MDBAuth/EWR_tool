# gauge data loading information for the operational use of the tool

import pandas as pd
from tqdm import tqdm

gauges = pd.read_csv('gauge_data/bom_gauge_data.csv', names=['gauge_name', 'gauge_number', 'gauge_owner', 'lat', 'long'])
gauges["State"] = gauges["gauge_owner"].str[:3]
gauges = gauges.drop(['lat', 'long', 'gauge_owner'], axis=1)

# Sort the gauges into their state lists:

def state_sorter(gauge_list):
    nws_list =[]
    qld_list = [] 
    vic_list =[] 
    other_list=[]
    for gauge in gauge_list:
        
        state = gauges[gauges["gauge_number"]==gauge]["State"]
        if state.isin(["NSW"]).any():
            nws_list.append(gauge)
        if state.isin(["QLD"]).any():
            qld_list.append(gauge)
        if state.isin(["VIC"]).any():
            vic_list.append(gauge)
        if ~state.isin(["NSW","QLD","VIC"]).any():
            other_list.append(gauge)

    return nws_list, qld_list, vic_list, other_list

# Water flow data 
import datetime
from datetime import date, timedelta
# from pyspark.sql import Row
# from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, DateType, DecimalType
import requests
#-------------------------------------------------------------------------------------------------------------------------------
def CallStateAPI (state, indicativeSites, parmStartTime, parmEndTime, parmDataSource, parmVarFrom, parmVarTo, parmInterval, parmDataType):
    sitesString = ''
    if state == 'NSW':
        url = 'realtimedata.waternsw.com.au'
        parmVarFrom = 130
        parmVarTo = 130
    if state == 'QLD':
        url = 'water-monitoring.information.qld.gov.au'
    if state == 'VIC':
        url = 'data.water.vic.gov.au'
        parmVarFrom = 100
        parmVarTo = 100
        
    for site in indicativeSites:
        sitesString = sitesString + site + ','
    sitesString=sitesString[:-1]
    
    requestString = 'https://' + str(url) + '/cgi/webservice.pl?\
    {"params":{"site_list":"' + str(sitesString) + '",\
    "start_time":"' + str(parmStartTime) + '000000",\
    "varfrom":' + str(parmVarFrom) + ',\
    "interval":"' + str(parmInterval) + '",\
    "varto":"'+ str(parmVarTo)+'",\
    "datasource":"' + str(parmDataSource) + ' ",\
    "end_time":"' + str(parmEndTime) + '000000",\
    "data_type":"' + str(parmDataType) + '","multiplier":"1"},\
    "function":"get_ts_traces","version":"2"}"))'
    proxies={} # populate with proxy settings
    requestString=requestString.replace(" ", "")
    r = requests.get(requestString
                    )
    return r
#-------------------------------------------------------------------------------------------------------------------------------
def Extract_Data (state, data):
    if int(data['error_num']) > 0:
        print( data['error_msg'])
    else:
        for sample in data['_return']['traces']:
            count = 0
            this_site=sample['site']
            this_sitename=sample['site_details']['name']
            this_variable=sample['varto_details']['variable']
            this_name=sample['varto_details']['name']
            this_units=sample['varto_details']['units']
            for obs in sample['trace']:
                if int(obs['q']) < 999:
                    count = count + 1
                    year=str(obs['t'])[0:4]
                    month=str(obs['t'])[4:6]
                    day=str(obs['t'])[6:8]
                    obsdate = datetime.date(int(year), int(month),int(day))
                    objRow = [state,this_site,'WATER',obsdate,obs['v'],obs['q']]
                    lstObservation.append(objRow)
        #print (this_sitename + str(' - ') + this_variable + str(' - ') + this_name + str(' - ') + this_units + str(' - ') + str(count));
#-------------------------------------------------------------------------------------------------------------------------------

def split(input_list, parts):  #splits a into parts of a simmlar size
    return [input_list[i::parts] for i in range(parts)]

lstObservation = []

#Returning todays date for real time data
today = str(date.today().strftime('%Y%m%d'))


def gaugePull(gauge_list, start_time_user, end_time_user, varfrom = "", varto = "", interval = "day", datatype = "mean"):
    
    nsw_sitelist, qld_sitelist, vic_sitelist, JUNK = state_sorter(gauge_list)
    
    callStartTime = start_time_user
    callEndTime = end_time_user
    callVarFrom = varfrom
    callVarTo = varto
    callInterval = interval 
    callDataType = datatype
    
    if nsw_sitelist:
        #Make call to the state API for NSW
        callDataSource = 'CP' #published
        callState = 'NSW'
        parts = len(nsw_sitelist) # break up nsw calls into this many parts, one part per gauge
        n = 0
        
        for sub_nsw_sitelist in tqdm(split(nsw_sitelist,parts), desc='loading NSW gauge data',
                                     position = 0, leave = False, 
                                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',):
            n+=1
            if sub_nsw_sitelist:
                callSites = sub_nsw_sitelist                                
                returnedJson = CallStateAPI(callState, callSites, callStartTime, callEndTime, callDataSource, callVarFrom, callVarTo, callInterval, callDataType)
#                 print(callState, callSites, callStartTime, callEndTime, callDataSource)
                if returnedJson.ok:
                    Extract_Data(callState, returnedJson.json())
#                     print("NSW loaded "+ str(n) + " of " + str(parts) + " parts" )
                else:
                    print("NSW failed, status code:" + str(returnedJson.status_code))

#-----------------------------------
#Make call to the state API for VIC
    if vic_sitelist:
        callDataSource = 'PUBLISH'
        callState = 'VIC'
        parts = len(vic_sitelist) # break up Vic calls into this many parts
        n = 0
        
        for sub_vic_sitelist in tqdm(split(vic_sitelist,parts), desc='loading Vic gauge data', 
                                     position = 0, leave = False,
                                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',):
            n+=1
            if sub_vic_sitelist:
                callSites = sub_vic_sitelist                                
                returnedJson = CallStateAPI(callState, callSites, callStartTime, callEndTime, callDataSource, callVarFrom, callVarTo, callInterval, callDataType)
                if returnedJson.ok:
                    Extract_Data(callState, returnedJson.json())
#                     print("VIC loaded "+ str(n) + " of " + str(parts) + " parts" )
                else:
                    print("VIC failed, status code:" + srt(returnedJson.status_code))

#-----------------------------------
    if qld_sitelist:
        #Make call to the state API for QLD
        callDataSource = 'AT'
        callState = 'QLD'
        callSites = qld_sitelist
        returnedJson = CallStateAPI(callState, callSites, callStartTime, callEndTime, callDataSource, callVarFrom, callVarTo, callInterval, callDataType)
        if returnedJson.ok:
            Extract_Data(callState, returnedJson.json())
#             print("QLD loaded")
        else:
            print("QLD failed, status code:" + str(returnedJson.status_code))
#-----------------------------------------------------------------------------------------------------
# Create the dataframe
    cols = ["DATASOURCEID", "SITEID", "SUBJECTID", "DATETIME", "VALUE", "QUALITYCODE"]
    FlowDataFrame = pd.DataFrame(data = lstObservation, columns = cols)  

#   FlowDataFrame = sqlContext.createDataFrame(data = lstObservation, schema = new_schema)
    return FlowDataFrame