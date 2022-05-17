from ipywidgets import Button, Text, VBox, HBox, Layout, Dropdown, Checkbox, \
    DatePicker, Select, SelectMultiple, Tab, BoundedFloatText, Label, Output, interactive
from IPython.display import display
import traitlets
import itertools
import pandas as pd
import copy
from tkinter import Tk, filedialog
from datetime import datetime, date
import xlsxwriter

from . import data_inputs, scenario_handling, observed_handling

#-------------------------------------------------------------------------------------------------#
#-----------------------------------Tab 1 - Observed flow widgets---------------------------------#
#-------------------------------------------------------------------------------------------------#

# Input tab widgets--------------------------------------------------------------------------------
# Label widgets:
defaults_label = {'style': {'description_width':'initial'}}
gauge_inputs_title = Label(value="Configure inputs for observed EWRs", **defaults_label)
date_selection = Label(value="1. Select date range of interest", **defaults_label)
cs_selection = Label(value="2. Select the catchment (1) and sites (1+)", **defaults_label)
allowance_header_o = Label(value="3. Enter an allowance", **defaults_label)
run_header_o = Label(value="4. Enter output file name and run", **defaults_label)

# Date selection widgets:
start_date = DatePicker(description='Start date:', value = date(2014,1,1), disabled=False)
end_date = DatePicker(description='End date', value = date(2020,1,1), disabled=False)

# Catchment and site selection widgets:
defaults_sites = {'layout': Layout(width="100%"), 'style': {'description_width':'initial'}}


def change_catchment(*args):
    '''Updates the sites displayed based on the user selection of the catchment'''
    catchments_gauges = data_inputs.map_gauge_to_catchment()
    sites.options=list(catchments_gauges[catchment.value].values())
    
catchments_gauges = data_inputs.map_gauge_to_catchment()

catchment = Select(options=list(catchments_gauges.keys()), 
                           description='Catchment: (select one)',
                           rows=len(catchments_gauges.keys()), **defaults_sites)

sites = SelectMultiple(options=catchments_gauges[catchment.value].values(),
                               description='Sites: (shift to select multiple)',
                               rows=len(catchments_gauges[catchment.value].values()),
                               **defaults_sites)
catchment.observe(change_catchment, 'value')

# Apply allowance to EWR indicators:
defaults = {'value': 0, 'min': 0, 'max': 20, 'step': 0.1, 'disabled': False,
            'style': {'description_width':'initial'}}
min_threshold_allowance_o = BoundedFloatText(description='Allowance applied to min threshold (%)', **defaults)
max_threshold_allowance_o = BoundedFloatText(description='Allowance applied to max threshold (%)', **defaults)
duration_allowance_o = BoundedFloatText(description='Allowance applied to min duration (%)', **defaults)
drawdown_allowance_o = BoundedFloatText(description='Allowance applied to max drawdown rate (%)', **defaults)

# Output file name:
file_name_o = Text(value='Observed flow EWRs', placeholder='Enter file name', 
                           description='Output file name:', disabled=False, 
                           style= {'description_width':'initial'})
# Results tab--------------------------------------------------------------------------------------
def get_gauges_to_pull():
    '''Convert sites selected to gauge numbers'''
    catchments_gauges = data_inputs.map_gauge_to_catchment()
    gauges = []
    for gauge, name in catchments_gauges[catchment.value].items():
        if name in sites.value:
            gauges.append(gauge)
    return gauges
    
def on_gauge_button_clicked(b):
    '''Run the realtime program'''
    with gauge_run_output:
        b.style.button_color='lightgreen'
        tab_gauge.selected_index = 1
        global raw_data_o, results_summary_o
        raw_data_o, results_summary_o = None, None
        # Get gauge list:
        gauges = get_gauges_to_pull()

        # Retrieve and convert dates:
#         dates = {'start_date': str(start_date.value).replace('-',''), 
#                  'end_date': str(end_date.value).replace('-','')}
        dates = {'start_date': start_date.value, 
                 'end_date': end_date.value}
        # Get allowances:
        MINT = (100 - min_threshold_allowance_s.value)/100
        MAXT = (100 + max_threshold_allowance_s.value)/100
        DUR = (100 - duration_allowance_s.value)/100
        DRAW = (100 - drawdown_allowance_s.value)/100
        allow ={'minThreshold': MINT, 'maxThreshold': MAXT, 'duration': DUR, 'drawdown': DRAW}
        # hardcode in the climate cats for the observed flows:
        raw_data_o, results_summary_o = observed_handling.observed_handler(gauges, dates, allow,
                                                                           'Standard - 1911 to 2018 climate categorisation') 
        display(results_summary_o.style)

def gauge_output_button_clicked(b):
    '''Output the realtime analysis to excel'''
    
    with gauge_output:
        b.style.button_color='lightgreen'
        global results_summary_o
        global raw_data_o
        
        realtime_fileName = file_name_o.value # Gettng the user file name        
        writer = pd.ExcelWriter('Output_files/' + realtime_fileName + '.xlsx', engine='xlsxwriter')
        results_summary_o.to_excel(writer, sheet_name='Gauge_data_summary') 
        PU_items = data_inputs.get_planning_unit_info()
        PU_items.to_excel(writer, sheet_name='Planning unit metadata')
        
        get_index = results_summary_o.reset_index().set_index(['gauge', 'planning unit']).index.unique()
        for locPU in get_index:
            temp_df = pd.DataFrame()
            if temp_df.empty == True:
                temp_df = raw_data_o['observed'][locPU[0]][locPU[1]].copy(deep=True)
                temp_df.columns = pd.MultiIndex.from_product([[str('observed')],temp_df.columns])
            else:
                df_to_add = pd.DataFrame()
                df_to_add = raw_data_o['observed'][locPU[0]][locPU[1]].copy(deep=True)
                df_to_add.columns = \
                pd.MultiIndex.from_product([[str('observed')],df_to_add.columns])        
                temp_df = pd.concat([temp_df, df_to_add], axis = 1)
                
            PU_code = PU_items['PlanningUnitID'].loc[PU_items[PU_items['PlanningUnitName'] == locPU[1]].index[0]]
            temp_df.to_excel(writer, sheet_name=str(PU_code))                
                
        writer.save()
#Run the analysis (from input tab):
gauge_run_button = Button(description="Run gauge program")
gauge_run_output = Output(layout={'border': '1px solid black'})
gauge_run_button.on_click(on_gauge_button_clicked)
# Output results tab-------------------------------------------------------------------------------
default_output = {'style': {'description_width':'initial'}}
gauge_output_button = Button(description='Output to Excel', **default_output)
gauge_output = Output(layout={'border': '1px solid black', **default_output})
gauge_output_button.on_click(gauge_output_button_clicked)
#-------------------------------------------------------------------------------------------------#
#-----------------------------------Tab 2 - Scenario testing widgets------------------------------#
#-------------------------------------------------------------------------------------------------#

# Input tab----------------------------------------------------------------------------------------
#Labels:
defaults_label = {'style': {'description_width':'initial'}}

model_inputs_title = Label(value="Configure inputs for scenario testing", **defaults_label)
model_selection = Label(value="1. Upload model files", **defaults_label)
model_format_header = Label(value="2. Select format of timeseries data", **defaults_label)
climate_format_header = Label(value="3. Select climate sequence to load", **defaults_label)
allowance_header_s = Label(value="4. Enter an allowance", **defaults_label)
run_header_s = Label(value="5. Enter output file name and run", **defaults_label)
fileLocationHeader = Label(value="Provide file path and names:", **defaults_label)
# Local file upload:
class SelectFilesButton(Button):
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
        except:
            print('no file selected')
            pass
load_model_files = SelectFilesButton()        
# Remote file upload:
box_defaults = {'value':'', 'placeholder':'Enter/path/to/file', 'disabled': False, 
                'style': {'description_width':'initial'}}
path_box_1 = Text(description='Scenario file one name:', **box_defaults)
path_box_2 = Text(description='Scenario file two name:', **box_defaults)
path_box_3 = Text(description='Scenario file three name:', **box_defaults)
path_box_4 = Text(description='Scenario file four name:', **box_defaults)
path_box_5 = Text(description='Scenario file five name:', **box_defaults)

#Allowance widgets:
default = {'value': 0, 'min': 0, 'max': 20, 'step': 0.1, 'disabled': False,
            'style': {'description_width':'initial'}}
min_threshold_allowance_s = BoundedFloatText(description='Allowance applied to min threshold (%)', **defaults)
max_threshold_allowance_s = BoundedFloatText(description='Allowance applied to max threshold (%)', **defaults)
duration_allowance_s = BoundedFloatText(description='Allowance applied to min duration (%)', **defaults)
drawdown_allowance_s = BoundedFloatText(description='Allowance applied to max drawdown rate (%)', **defaults)

# Model format widget:
model_format_type = Dropdown(
    options=[('Bigmod - MDBA'), ('IQQM - NSW 10,000 years'), ('Source - NSW (res.csv)')],
    value='Bigmod - MDBA',
    description='',
    style= {'description_width':'initial'})
# Climate file widget:
climate_file = Dropdown(
    options=[('Standard - 1911 to 2018 climate categorisation'), ('NSW 10,000 year climate sequence')],
    value='Standard - 1911 to 2018 climate categorisation',
    description='',
    style= {'description_width':'initial'})
# Output file name:
file_name_s = Text(
    value='Scenario_test',
    placeholder='Enter file name',
    description='Output file name:',
    disabled=False,
    style= {'description_width':'initial'})
# Results display tab------------------------------------------------------------------------------

def catchment_checker(catchment):
    '''Pass catchment name, returns list of gauges in this catchment'''
    catchments_gauges = data_inputs.map_gauge_to_catchment()
    return list(catchments_gauges[catchment].keys()) 
    
def view(x=''):
    '''Shows either all results, or results restricted to user catchment selection'''
    if x=='All': 
        return display(data_summary_s.style)
    return display((data_summary_s.loc[data_summary_s.index.isin(catchment_checker(x), level='gauge')]).style)

def get_locations_from_scenarios(data_summary):
    '''Ingest a summary of results, look at the locations analysed, 
    return a list of catchments included in the analysis'''
    catchments_gauges = data_inputs.map_gauge_to_catchment()
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

def get_file_names(loaded_files):
    '''Take in the file location strings from the users loaded files,
    return dictionary with file location string mapped to file name'''
    
    file_locations = {}
    for file in loaded_files:
        full_name = file.split('/')
        name_exclude_extension = full_name[-1].split('.csv')[0]
        file_locations[str(name_exclude_extension)] = file
        
    return file_locations

def on_model_button_clicked(b):
    '''Run the scenario testing program'''
    
    with model_run_output:
        b.style.button_color='lightgreen'
        tab_model.selected_index = 1
        global raw_data_s, data_summary_s, select_benchmark
        # Get file names and their system locations
        if load_model_files.files != []:
            modelFiles = load_model_files.files
        else:
            modelFiles = []
            if path_box_1.value != '':
                PB1 = path_box_1.value.strip()
                modelFiles.append(PB1)
            if path_box_2.value != '':
                PB2 = path_box_2.value.strip()
                modelFiles.append(PB2)
            if path_box_3.value != '':
                PB3 = path_box_3.value.strip()
                modelFiles.append(PB3)
            if path_box_4.value != '':
                PB4 = path_box_4.value.strip()
                modelFiles.append(PB4)
            if path_box_5.value != '':
                PB5 = path_box_5.value.strip()
                modelFiles.append(PB5)
        file_locations = get_file_names(modelFiles)
        # Get tolerance:
        minThreshold_tolerance = (100 - min_threshold_allowance_s.value)/100
        maxThreshold_tolerance = (100 + max_threshold_allowance_s.value)/100
        duration_tolerance = (100 - duration_allowance_s.value)/100
        drawdown_tolerance = (100 - drawdown_allowance_s.value)/100

        allowanceDict ={'minThreshold': minThreshold_tolerance, 'maxThreshold': maxThreshold_tolerance, 
                        'duration': duration_tolerance, 'drawdown': drawdown_tolerance}
        
        # Run the requested analysis on the loaded scenarios:
        raw_data_s, data_summary_s = scenario_handling.scenario_handler(file_locations,
                                                                        model_format_type.value,
                                                                        allowanceDict,
                                                                        climate_file.value)
        items = get_locations_from_scenarios(data_summary_s)              
        w = Select(options=items)
        #show the results of the selected catchment:
        out = interactive(view, x=w) 
        display(out)

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

def model_output_button_clicked(b):
    '''Output the scenario testing to excel'''
    with model_output:
        b.style.button_color='lightgreen'
        global data_summary_s
        global raw_data_s
        
        # Get the metadata from the model run:
        metadata_df = getMetadata()
        PU_items = data_inputs.get_planning_unit_info()
        model_file_name = file_name_s.value # Gettng the user file name
        
        model_scenario_list = list(data_summary_s.columns.levels[0]) # Get scenarios / Repeated?
        
        writer = pd.ExcelWriter('Output_files/' + model_file_name + '.xlsx', engine='xlsxwriter')
        metadata_df.to_excel(writer, sheet_name='Metadata')
        PU_items.to_excel(writer, sheet_name='Planning unit metadata')
        data_summary_s.to_excel(writer, sheet_name='Data_summary')
        get_index = data_summary_s.reset_index().set_index(['gauge', 'planning unit']).index.unique()
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
                PU_code = PU_items['PlanningUnitID'].loc[PU_items[PU_items['PlanningUnitName'] == locPU[1]].index[0]]
                sheet_name = str(locPU[0]) + '-' + str(PU_code)
                temp_df.to_excel(writer, sheet_name=sheet_name)

        writer.save()
#Run analysis button (from input tab):
model_run_button = Button(description="Run program")
model_run_output = Output(layout={'border': '1px solid black'}) 
model_run_button.on_click(on_model_button_clicked)
#Output results tab--------------------------------------------------------------------------------
model_output_button = Button(description="Output to Excel")
model_output = Output(layout={'border': '1px solid black'}, style= {'description_width':'initial'})
model_output_button.on_click(model_output_button_clicked)
#-------------------------------------------------------------------------------------------------#
#--------------------------------------Widgets used by both tabs----------------------------------#
#-------------------------------------------------------------------------------------------------#
justLine = HBox([Label(value="________________________________________________________________________________", 
                       layout=Layout(width="100%"),
                       style = {'description_width':'initial'})])
                       
or_text = HBox([Label(value="~     OR     ~", 
                                      layout=Layout(width="100%"),
                                      style= {'description_width':'initial'})])
#-------------------------------------------------------------------------------------------------#
#-------------------------------------------Aggregate widgets-------------------------------------#
#-------------------------------------------------------------------------------------------------#
gauge_input_widgets = VBox([gauge_inputs_title, justLine, date_selection, HBox([start_date, end_date]), 
                                     justLine, cs_selection, HBox([catchment, sites]),
                                     justLine, allowance_header_o, HBox([min_threshold_allowance_o, max_threshold_allowance_o, 
                                                                         duration_allowance_o, drawdown_allowance_o]),
                                     justLine, run_header_o, HBox([file_name_o, gauge_run_button])])

model_input_widgets= VBox([model_inputs_title, justLine, model_selection, 
                                   VBox([load_model_files, or_text, fileLocationHeader, path_box_1, path_box_2, path_box_3, path_box_4, path_box_5]),
                                   VBox([justLine, model_format_header, model_format_type]),
                                   VBox([justLine, climate_format_header, climate_file]),
                                   justLine, allowance_header_s, HBox([min_threshold_allowance_s, max_threshold_allowance_s, duration_allowance_s, drawdown_allowance_s]),
                                   justLine, run_header_s, HBox([file_name_s, model_run_button])])

model_output_widgets = HBox([model_output_button, model_output])
gauge_output_widgets = HBox([gauge_output_button, gauge_output])
# Then group them together:    
tab_gauge = Tab()
tab_gauge.children = [gauge_input_widgets, gauge_run_output, gauge_output_widgets]
tab_gauge.set_title(0, 'Input')
tab_gauge.set_title(1, 'Results dashboard')
tab_gauge.set_title(2, 'Output results') 
    
tab_model = Tab()
tab_model.children = [model_input_widgets, model_run_output, model_output_widgets]
tab_model.set_title(0, 'Input')
tab_model.set_title(1, 'Results dashboard')
tab_model.set_title(2, 'Output results') 

ewr_dashboard = Tab()
ewr_dashboard.children = [tab_gauge, tab_model]
ewr_dashboard.set_title(0, 'Observed')
ewr_dashboard.set_title(1, 'Scenario testing')