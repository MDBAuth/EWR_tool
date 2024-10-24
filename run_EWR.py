import os
from py_ewr.scenario_handling import ScenarioHandler
import pandas as pd
def main():
    scenarios = {'Scenario1': ['C:\GIT_stuff\EWR_test_data\with_flow_gaugesbig.bmd.csv']} # insert the file you need to run through the EWR tool
    model_format = 'Bigmod - MDBA'
    os.chdir('C:\GIT_stuff\EWR_test_data') #changing the folder to create the results folder :- change as necessary
    os.mkdir('EWR_results') # create a folder to save the EWR outputs within the prefered folder :- change as necessary
    os.chdir('EWR_results')
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
