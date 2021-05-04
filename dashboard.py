import ipywidgets as widgets
from IPython.display import display
import traitlets
import itertools
import pandas as pd
import data_inputs, ewr_evaluation
import copy
from tkinter import Tk, filedialog
from datetime import datetime

catchments_gauges = data_inputs.catchments_gauges_dict()
climate_file = widgets.Dropdown(
    options=[('Standard - 1911 to 2018 climate categorisation'), ('NSW 10,000 year climate sequence')],
    value='Standard - 1911 to 2018 climate categorisation',
    description='',
    style= {'description_width':'initial'}
)
################################################################################################
# Selecting the model files: 
class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = "Select Files"
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)
        
    @staticmethod
    def select_files(b):
        global file_names
        global select_benchmark
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        try:
            # Create Tk root
            root = Tk()
            # Hide the main window
            root.withdraw()
            # Raise the root to the top of all windows.
            root.call('wm', 'attributes', '.', '-topmost', True)
            # List of selected fileswill be set to b.value
            b.files = filedialog.askopenfilename(multiple=True)
            b.description = "Files Selected"
            b.icon = "check-square-o"
            b.style.button_color = "lightgreen"
                    # Section to allow user to select benchmark scenario: 
            with benchmark_output:
                file_names = []
                for file in load_model_files.files: 
                    #Get the name of the file:
                    full_name = file.split('/')
                    file_name = full_name[-1].split('.csv')[0]
                    file_names.append(file_name)
                select_benchmark = widgets.Select(
                    options = file_names,
                    description='Select Benchmark scenario:',
                    disabled=False,
                    rows=len(file_names),
                    layout=widgets.Layout(width="100%"),
                    style= {'description_width':'initial'}
                )   
                display(widgets.HBox([select_benchmark]))
        except:
            print('no file selected')
            pass

def update_benchmark_choices():
    '''saves the names of the file names loaded into the dashboard
    to the select_benchmark widget'''
    global file_names
    select_benchmark.options = file_names
    
benchmark_output = widgets.Output(layout={'border': '1px solid black'})
benchmark_output.observe(update_benchmark_choices, 'value')


or_text = widgets.HBox([widgets.Label(value="~     OR     ~", 
                                      layout=widgets.Layout(width="100%"),
                                      style= {'description_width':'initial'})])
fileLocationHeader = widgets.HBox([widgets.Label(value="Provide file path and names:", 
                                      layout=widgets.Layout(width="100%"),
                                      style= {'description_width':'initial'})])

path_box_1 = widgets.Text(
    value='',
    placeholder='Enter/path/to/first/file',
    description='Output file one name:',
    disabled=False,
    style= {'description_width':'initial'}
)

path_box_2 = widgets.Text(
    value='',
    placeholder='Enter/path/to/second/file',
    description='Output file two name:',
    disabled=False,
    style= {'description_width':'initial'}
)
path_box_3 = widgets.Text(
    value='',
    placeholder='Enter/path/to/third/file',
    description='Output file three name:',
    disabled=False,
    style= {'description_width':'initial'}
)
path_box_4 = widgets.Text(
    value='',
    placeholder='Enter/path/to/fourth/file',
    description='Output file four name:',
    disabled=False,
    style= {'description_width':'initial'}
)
path_box_5 = widgets.Text(
    value='',
    placeholder='Enter/path/to/fifth/file',
    description='Output file five name:',
    disabled=False,
    style= {'description_width':'initial'}
)


#------------------Tolerance widgets-----------------------#
# Scenario testing:
min_threshold_allowance_s = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Allowance applied to min threshold (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

max_threshold_allowance_s = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Allowance applied to max threshold (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

duration_allowance_s = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Allowance applied to min duration (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

drawdown_allowance_s = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Allowance applied to max drawdown rate (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

# Observed flows:

min_threshold_allowance_o = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Allowance applied to min threshold (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

max_threshold_allowance_o = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Allowance applied to max threshold (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

duration_allowance_o = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Allowance applied to min duration (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

drawdown_allowance_o = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Allowance applied to max drawdown rate (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

###############################################################################################        
# Model results display in the dasboard:

def catchment_checker(catchment):
    '''Takes in a catchment name
    Allocates the list of gauges in this catchment
    Returns this list '''

    gauge_list = list(catchments_gauges[catchment].keys())
    
    return gauge_list   
    
def view(x=''):
    '''Takes in the user selection (entire basin or a catchment), if entire basin, 
    just shows the data summary, if a catchment is selected, a table is generated 
    with the subset of the data summary table containing these locations'''
    
    if x=='All': 
        return display(data_summary_s.style)
    
    return display((data_summary_s.loc[data_summary_s.index.isin(catchment_checker(x),
                                                                 level='location')]).style)

def analysis_selection_matching(user_selections):
    ''' Match boolean user inputs to options for analysis. If frequency is selected, the program   
    adds in the target frequency. Returns the list of requests '''
    
    analysis_choices = ['Years with events', 'Frequency', 'Max dry', 'Years since last event',
                       'Number of events', 'Average event length', 'Average events per year',
                        'Average time between events', 'Average days above low flow',
                        'Average CtF days per year', 'Average length CtF spells']
    list_of_requests =  list(itertools.compress(analysis_choices, user_selections))
    
    for i in range(len(list_of_requests)):
        if list_of_requests[i] == 'Frequency':
            list_of_requests.insert(i+1, 'Target frequency')
            
    return list_of_requests

def get_file_names(loaded_files):
    '''Take in the file location strings from the users loaded files,
    return dictionary with file location string mapped to file name'''
    
    file_locations = {}
    for file in loaded_files:
        full_name = file.split('/')
        name_exclude_extension = full_name[-1].split('.csv')[0]
        file_locations[str(name_exclude_extension)] = file
        
    return file_locations

def get_locations_from_scenarios(data_summary):
    '''Ingest a summary of results, look at the locations analysed, 
    return a list of catchments included in the analysis'''
    
    location_list = set(data_summary_s.index.get_level_values(0))
    catchments_gauges_subset = copy.deepcopy(catchments_gauges)
    for catch in catchments_gauges.keys():
        for site in catchments_gauges[catch]:
            if site not in location_list:
                del catchments_gauges_subset[catch][site]
            else:
                continue        
    items = ['All']+list(catchments_gauges_subset.keys())   
    
    return items
    
def on_model_button_clicked(b):
    '''Run the scenario testing program'''
    
    with model_run_output:
        b.style.button_color='lightgreen'
        global raw_data_s, data_summary_s, select_benchmark
        # Check which box has data:
        if load_model_files.files != []:
            modelFiles = load_model_files.files
        else:
            modelFiles = []
            if path_box_1.value != '':
                modelFiles.append(path_box_1.value)
            if path_box_2.value != '':
                modelFiles.append(path_box_2.value)
            if path_box_3.value != '':
                modelFiles.append(path_box_3.value)
            if path_box_4.value != '':
                modelFiles.append(path_box_4.value)
            if path_box_5.value != '':
                modelFiles.append(path_box_5.value)
              
        # Get file names and their system locations
        file_locations = get_file_names(modelFiles)
        # Get user analysis requests:
        user_requests=[years_events_s.value, freq_events_s.value, max_dry_s.value, time_since_s.value,
                      numEvents_s.value, avLengthEvents_s.value, avNumEvents_s.value, 
                      avDaysBetween_s.value, avLowFlowDays_s.value, avCtfDaysPerYear_s.value,
                      avLenCtfSpells_s.value]
        list_of_requests = analysis_selection_matching(user_requests)
        
        # Get ewr tables:
        ewr_data, see_notes_ewrs, undefined_ewrs, noThresh_df,\
        no_duration, DSF_ewrs = data_inputs.get_ewr_table()
        
        # Get tolerance:
        minThreshold_tolerance = (100 - min_threshold_allowance_s.value)/100
        maxThreshold_tolerance = (100 + max_threshold_allowance_s.value)/100
        duration_tolerance = (100 - duration_allowance_s.value)/100
        drawdown_tolerance = (100 - drawdown_allowance_s.value)/100

        allowanceDict ={'minThreshold': minThreshold_tolerance, 'maxThreshold': maxThreshold_tolerance, 
                        'duration': duration_tolerance, 'drawdownTolerance': drawdown_tolerance}
        
        # Get bigmod metadata:
        bigmod_metadata = data_inputs.get_bigmod_codes()
        # Run the requested analysis on the loaded scenarios:
        raw_data_s, data_summary_s =ewr_evaluation.scenario_handler(file_locations,
                                                                    list_of_requests,
                                                                    ewr_data,
                                                                    model_format_type.value,
                                                                    bigmod_metadata,
                                                                    allowanceDict,
                                                                    climate_file.value
                                                                   )
        items = get_locations_from_scenarios(data_summary_s)              
        w = widgets.Select(options=items)
        #show the results of the selected catchment:
        out = widgets.interactive(view, x=w) 
        display(out)

###############################################################################################        
########### Observed output section #########

def get_gauges_to_pull():
    ''' Get gauges to pull from state portal based on user selection,
    return the list of gauges '''
    
    sitesList = list(sites.value)
    gauges_list = list()
    for i in sitesList:
        for j in catchments_gauges:
            for k, v in catchments_gauges[j].items():
                if i == v:
                    gauges_list.append(k)
                    
    return gauges_list

def get_user_dates():
    '''Retrieve start and end dates from the widgets, 
    convert these to compatible format for API call,
    return the dates'''
    
    start_year = str(start_date.value.year)
    start_month = str(start_date.value.month)
    start_day = str(start_date.value.day)
    if len(start_month) == 1:
        start_month = '0' + start_month
    if len(start_day) == 1:
        start_day = '0' + start_day

    start_time_user = start_year + start_month + start_day

    end_year = str(end_date.value.year)
    end_month = str(end_date.value.month)
    end_day = str(end_date.value.day)
    if len(end_month) == 1:
        end_month = '0'+ end_month
    if len(end_day) == 1:
        end_day = '0' + end_day

    end_time_user = end_year + end_month + end_day 
    input_params_o = {'start time': start_time_user, 'end time': end_time_user}
    
    return input_params_o
    
def on_gauge_button_clicked(b):
    '''Run the realtime program'''
    with gauge_run_output:
        b.style.button_color='lightgreen'
        global raw_data_o, results_summary_o
        # Get gauge list:
        gauges_list = get_gauges_to_pull()
        # Get user requests for analysis:
        user_requests = [years_events_o.value, freq_events_o.value, max_dry_o.value, time_since_o.value,
                        numEvents_o.value, avLengthEvents_o.value, avNumEvents_o.value, 
                        avDaysBetween_o.value, avLowFlowDays_o.value, avCtfDaysPerYear_o.value,
                        avLenCtfSpells_o.value] 
        input_params_o = get_user_dates()
        list_of_request_o = analysis_selection_matching(user_requests)
        # Get ewr tables:
        ewr_data, see_notes_ewrs, undefined_ewrs, noThresh_df,\
        no_duration, DSF_ewrs = data_inputs.get_ewr_table()
        
        # Get tolerances:
        minThreshold_tolerance = (100 - min_threshold_allowance_s.value)/100
        maxThreshold_tolerance = (100 + max_threshold_allowance_s.value)/100
        duration_tolerance = (100 - duration_allowance_s.value)/100
        drawdown_tolerance = (100 - drawdown_allowance_s.value)/100

        allowanceDict ={'minThreshold': minThreshold_tolerance, 'maxThreshold': maxThreshold_tolerance, 
                        'duration': duration_tolerance, 'drawdownTolerance': drawdown_tolerance}
        
        raw_data_o, results_summary_o = ewr_evaluation.realtime_handler(gauges_list,
                                                                        list_of_request_o,
                                                                        input_params_o,
                                                                        ewr_data,
                                                                        allowanceDict,
                                                                        'Standard - 1911 to 2018 climate categorisation') # hardcode in the climate cats for the observed flows
        display(results_summary_o.style)

###############################################################################################        
def getMetadata():
    '''Run this function to get the metadata from the model run.
    Returns a pandas dataframe with the relevant metadata'''
    todaysDate = str(datetime.today().strftime('%Y-%m-%d')) 

    metadata = pd.DataFrame(columns = ['Name', 'Details'])
    nameList = ['date run', 'database accessed on', 'duration tolerance applied', 'minimum threshold tolerance applied', 'maximum threshold tolerance applied', 
                'maximum drawdown rate tolerance applied']
    metadataList = [todaysDate, todaysDate, duration_allowance_s.value, min_threshold_allowance_s.value, max_threshold_allowance_s .value, drawdown_allowance_s.value]
    metadata['Name'] = nameList
    metadata['Details'] = metadataList
    
    return metadata

def getPlanningUnitInfo():
    '''Run this function to get the planning unit MDBA ID and equivilent planning unit name as specified in the LTWP'''
    ewr_data, see_notes_ewrs, undefined_ewrs, noThresh_df, no_duration, DSF_ewrs = data_inputs.get_ewr_table()
        
    planningUnits = ewr_data.groupby(['PlanningUnitID', 'PlanningUnitName']).size().reset_index().drop([0], axis=1) 
    
    return planningUnits
    
def model_output_button_clicked(b):
    '''Output the scenario testing to excel'''
    with model_output:
        b.style.button_color='lightgreen'
        global data_summary_s
        global raw_data_s
        
        # Get the metadata from the model run:
        metadata_df = getMetadata()
        planningUnitData = getPlanningUnitInfo()
        model_file_name = file_name_s.value # Gettng the user file name
        
        # Define the gauge and scenario lists to be used:
        location_list_1 = set(data_summary_s.index.get_level_values(0)) # Get the gauges / Repeated line
        pu_list_1 = set(data_summary_s.index.get_level_values(1)) # Get the gauges / Repeated line
        model_scenario_list = list(data_summary_s.columns.levels[0]) # Get scenarios / Repeated?
        
        writer = pd.ExcelWriter('Output_files/' + model_file_name + '.xlsx', engine='xlsxwriter')
        metadata_df.to_excel(writer, sheet_name='Metadata')
        planningUnitData.to_excel(writer, sheet_name='Planning unit metadata')
        data_summary_s.to_excel(writer, sheet_name='Data_summary')
        get_index = data_summary_s.reset_index().set_index(['location', 'planning unit']).index.unique()
        for locPU in get_index:
            temp_df = pd.DataFrame()
            for model_scenario in model_scenario_list:
                if temp_df.empty == True:
                    temp_df = raw_data_s[model_scenario][locPU[0]][locPU[1]].copy(deep=True)
                    temp_df.columns = pd.MultiIndex.from_product([[str(model_scenario)],temp_df.columns])
                else:
                    df_to_add = pd.DataFrame()
                    df_to_add = raw_data_s[model_scenario][locPU[0]][locPU[1]].copy(deep=True)
                    df_to_add.columns = \
                    pd.MultiIndex.from_product([[str(model_scenario)],df_to_add.columns])
                    temp_df = pd.concat([temp_df, df_to_add], axis = 1)
                temp_df.to_excel(writer, sheet_name=str(locPU))   

        writer.save()

###############################################################################################   

def gauge_output_button_clicked(b):
    '''Output the realtime analysis to excel'''
    
    with gauge_output:
        b.style.button_color='lightgreen'
        global results_summary_o
        global raw_data_o
        
        realtime_fileName = file_name_o.value # Gettng the user file name
        # Define the gauge and scenario lists to be used:
        location_list_1 = set(results_summary_o.index.get_level_values(0)) # Get the gauges / Repeated line
        
        writer = pd.ExcelWriter('Output_files/' + realtime_fileName + '.xlsx', engine='xlsxwriter')
        results_summary_o.to_excel(writer, sheet_name='Gauge_data_summary') 
        for location in location_list_1:
            raw_data_o['gauge data'][location].to_excel(writer, sheet_name=location)   
        writer.save()
        
###############################################################################################
def change_catchment(*args):
    '''Updates the sites displayed based on the user selection of the catchment'''
    
    sites.options=list(catchments_gauges[catchment.value].values())
    
###############################################################################################
#################################### Widgets list #############################################
###############################################################################################

#################################### Gauge/real-time ##########################################
#Input parameters:
start_date = widgets.DatePicker(description='Start date:', disabled=False)
end_date = widgets.DatePicker(description='End date',disabled=False)
catchment = widgets.Select(options=list(catchments_gauges.keys()),
                           description='Catchment: (select one)',
                           rows=len(catchments_gauges.keys()),
                           layout=widgets.Layout(width="100%"),
                           style= {'description_width':'initial'})
sites = widgets.SelectMultiple(options=catchments_gauges[catchment.value].values(),
                               description='Sites: (shift to select multiple)',
                               rows=len(catchments_gauges[catchment.value].values()),
                               layout=widgets.Layout(width="100%"),
                               style= {'description_width':'initial'}) 
catchment.observe(change_catchment, 'value')
years_events_o = widgets.Checkbox(value=True,description='Years with events',disabled=False,indent=False)
freq_events_o = widgets.Checkbox(value=True,description='Frequency',disabled=False,indent=False)
max_dry_o = widgets.Checkbox(value=True,description='Max dry',disabled=False,indent=False)
time_since_o = widgets.Checkbox(value=True,description='Last event',disabled=False,indent=False)
numEvents_o = widgets.Checkbox(value=True,description='Number of events',disabled=False,indent=False)
avLengthEvents_o = widgets.Checkbox(value=True,description='Average event length',
                                    disabled=False,indent=False)
avNumEvents_o = widgets.Checkbox(value=True,description='Average events per year',
                                    disabled=False,indent=False)
avEventDaysPerYear_o = widgets.Checkbox(value=True,description='Average events days per year',
                                    disabled=False,indent=False)
avDaysBetween_o = widgets.Checkbox(value=True,description='Average time between events',
                                    disabled=False,indent=False)
avLowFlowDays_o = widgets.Checkbox(value=True,description='Average days above low flow',
                                    disabled=False,indent=False)
avCtfDaysPerYear_o = widgets.Checkbox(value=True,description='Average CtF days per year',
                                    disabled=False,indent=False)
avLenCtfSpells_o = widgets.Checkbox(value=True,description='Average length CtF spells',
                                    disabled=False,indent=False)

file_name_o = widgets.Text(
    value='Real time ewr evaluation',
    placeholder='Enter file name',
    description='Output file name:',
    disabled=False,
    style= {'description_width':'initial'}
)

#Run the analysis:
gauge_run_button = widgets.Button(description="Run gauge program")
gauge_run_output = widgets.Output(layout={'border': '1px solid black'}) 

#Output the analysis results:
gauge_output_button = widgets.Button(description="Output to Excel", style= {'description_width':'initial'})
gauge_output = widgets.Output()

#On click functionality:
gauge_run_button.on_click(on_gauge_button_clicked)   
gauge_output_button.on_click(gauge_output_button_clicked)

threshold_allowance_o = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Threshold allowance (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

duration_allowance_o = widgets.BoundedIntText(
    value=0,
    min=0,
    max=20,
    step=1,
    description='Duration allowance (%)',
    disabled=False,
    style= {'description_width':'initial'}
)

##################################### Scenario testing #######################################
load_model_files = SelectFilesButton()

# Type of analysis:
years_events_s = widgets.Checkbox(value=True,description='Years with events',disabled=False,indent=False)
freq_events_s = widgets.Checkbox(value=True,description='Frequency',disabled=False,indent=False)
max_dry_s = widgets.Checkbox(value=True,description='Max dry',disabled=False,indent=False)
time_since_s = widgets.Checkbox(value=True,description='Last event',disabled=False,indent=False)
numEvents_s = widgets.Checkbox(value=True,description='Number of events',disabled=False,indent=False)
avLengthEvents_s = widgets.Checkbox(value=True,description='Average event length',
                                    disabled=False,indent=False)
avNumEvents_s = widgets.Checkbox(value=True,description='Average events per year',
                                   disabled=False,indent=False)
avEventDaysPerYear_s = widgets.Checkbox(value=True,description='Average event days per year',
                                    disabled=False,indent=False)
avDaysBetween_s = widgets.Checkbox(value=True,description='Average time between events',
                                    disabled=False,indent=False)
avLowFlowDays_s = widgets.Checkbox(value=True,description='Average days above low flow',
                                    disabled=False,indent=False)
avCtfDaysPerYear_s = widgets.Checkbox(value=True,description='Average CtF days per year',
                                    disabled=False,indent=False)
avLenCtfSpells_s = widgets.Checkbox(value=True,description='Average length CtF spells',
                                    disabled=False,indent=False)

file_name_s = widgets.Text(
    value='Scenario_test',
    placeholder='Enter file name',
    description='Output file name:',
    disabled=False,
    style= {'description_width':'initial'}
)

#Run the analysis:
model_run_button = widgets.Button(description="Run program")
model_run_output = widgets.Output(layout={'border': '1px solid black'}) 

#Output the analysis results:
model_output_button = widgets.Button(description="Output to Excel")
model_output = widgets.Output(layout={'border': '1px solid black'}, style= {'description_width':'initial'})

#On click functionality:
model_run_button.on_click(on_model_button_clicked)
model_output_button.on_click(model_output_button_clicked)

model_format_type = widgets.Dropdown(
    options=[('Source'), ('Bigmod'), ('IQQM'), ('NSW 10,000 years')],
    value='Bigmod',
    description='',
    style= {'description_width':'initial'}
)

climate_file = widgets.Dropdown(
    options=[('Standard - 1911 to 2018 climate categorisation'), ('NSW 10,000 year climate sequence')],
    value='Standard - 1911 to 2018 climate categorisation',
    description='',
    style= {'description_width':'initial'}
)

##################################### Dashbaord labels #######################################
# Widget labels for the scenario testing:
gauge_inputs_title = widgets.HBox([widgets.Label(value="Configure inputs for real-world EWR checks",
                                                 layout=widgets.Layout(width="100%"),
                                                 style= {'description_width':'initial'})])
model_inputs_title = widgets.HBox([widgets.Label(value="Configure inputs for scenario testing",
                                                 layout=widgets.Layout(width="100%"),
                                                 style= {'description_width':'initial'})])
analysis_selection_s = widgets.HBox([widgets.Label(value="4. Select the type of analysis")])
model_selection = widgets.HBox([widgets.Label(value="1. Upload model files")])
model_format_header = widgets.HBox([widgets.Label(value="2. Select format of timeseries data",
                                                  layout=widgets.Layout(width="100%"),
                                                  style= {'description_width':'initial'})])
climate_format_header = widgets.HBox([widgets.Label(value="3. Select climate sequence to load",
                                                  layout=widgets.Layout(width="100%"),
                                                  style= {'description_width':'initial'})])
allowance_header_s = widgets.HBox([widgets.Label(value="5. Enter an allowance",
                                                  layout=widgets.Layout(width="100%"),
                                                  style= {'description_width':'initial'})])
run_header_s = widgets.HBox([widgets.Label(value="6. Enter output file name and run",
                                                  layout=widgets.Layout(width="100%"),
                                                  style= {'description_width':'initial'})])
# Widget labels for the observed flows:
date_selection = widgets.HBox([widgets.Label(value="1. Select date range of interest")])
cs_selection = widgets.HBox([widgets.Label(value="2. Select the catchment (1) and sites (1+)")])
analysis_selection_o = widgets.HBox([widgets.Label(value="3. Select the type of analysis")])
allowance_header_o = widgets.HBox([widgets.Label(value="4. Enter an allowance",
                                                  layout=widgets.Layout(width="100%"),
                                                  style= {'description_width':'initial'})])
run_header_o = widgets.HBox([widgets.Label(value="5. Enter output file name and run",
                                                  layout=widgets.Layout(width="100%"),
                                                  style= {'description_width':'initial'})])
# Widget labels for both:
justLine = widgets.HBox([widgets.Label(value="________________________________________________________________________________", 
                                      layout=widgets.Layout(width="100%"),
                                      style= {'description_width':'initial'})])

##################################### Grouping together #######################################
gauge_input_widgets = widgets.VBox([gauge_inputs_title, justLine, date_selection, widgets.HBox([start_date, end_date]), 
                                     justLine, cs_selection, widgets.HBox([catchment, sites]),
                                     justLine, analysis_selection_o, years_events_o, freq_events_o, 
                                     max_dry_o, time_since_o, numEvents_o, avLengthEvents_o, 
                                     avNumEvents_o, avLowFlowDays_o, avCtfDaysPerYear_o, avLenCtfSpells_o,
                                     justLine, allowance_header_o, widgets.HBox([min_threshold_allowance_o, max_threshold_allowance_o, duration_allowance_o, drawdown_allowance_o]),
                                     justLine, run_header_o, widgets.HBox([file_name_o, gauge_run_button])])

model_input_widgets= widgets.VBox([model_inputs_title, justLine, model_selection, 
                                   widgets.VBox([load_model_files, or_text, fileLocationHeader, path_box_1, path_box_2, path_box_3, path_box_4, path_box_5]),
                                   widgets.VBox([justLine, model_format_header, model_format_type]),
                                   widgets.VBox([justLine, climate_format_header, climate_file]),
                                   justLine, analysis_selection_s, years_events_s, freq_events_s, 
                                   max_dry_s, time_since_s, numEvents_s, avLengthEvents_s,
                                   avNumEvents_s, avLowFlowDays_s, avCtfDaysPerYear_s, avLenCtfSpells_s,
                                   justLine, allowance_header_s, widgets.HBox([min_threshold_allowance_s, max_threshold_allowance_s, duration_allowance_s, drawdown_allowance_s]),
                                   justLine, run_header_s, widgets.HBox([file_name_s, model_run_button])])

model_output_widgets = widgets.HBox([model_output_button, model_output])
# Then group them together:    
tab_gauge = widgets.Tab()
tab_gauge.children = [gauge_input_widgets, gauge_run_output, gauge_output_button]
tab_gauge.set_title(0, 'Input')
tab_gauge.set_title(1, 'Results dashboard')
tab_gauge.set_title(2, 'Output results') 
    
tab_model = widgets.Tab()
tab_model.children = [model_input_widgets, model_run_output, model_output_widgets]    # model_output_button
tab_model.set_title(0, 'Input')
tab_model.set_title(1, 'Results dashboard')
tab_model.set_title(2, 'Output results') 

ewr_dashboard = widgets.Tab()
ewr_dashboard.children = [tab_gauge, tab_model]
ewr_dashboard.set_title(0, 'Observed')
ewr_dashboard.set_title(1, 'Scenario testing')