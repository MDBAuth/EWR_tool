#USER INPUT REQUIRED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Minimum 1 scenario and 1 related file required
scenarios = {'Observed': ['/dbfs/mnt/project-data/Basin_Strategy_and_Knowledge/Applied_Science/Eco_Hydrology/ESLT/Zeta/Victoria_EWRs/NCCMA/NCCMA_flow.csv']}

model_format = 'Standard time-series'

parameter_sheet="https://raw.githubusercontent.com/MDBAuth/EWR_tool/refs/heads/dev_NCCMA/py_ewr/parameter_metadata/parameter_sheet_NCCMA.csv"

output_path="/dbfs/mnt/project-data/Basin_Strategy_and_Knowledge/Applied_Science/Eco_Hydrology/ESLT/Zeta/Victoria_EWRs/NCCMA/EWR_output/"
# END USER INPUT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


from py_ewr.scenario_handling import ScenarioHandler
import pandas as pd



def run_ewr_scenario_and_save_results(scenarios, model_format,parameter_sheet, output_path):
    # ewr_results_dict = {}
    # yearly_results_dict = {}
    # all_events_dict = {}
    # all_interEvents_dict = {}
    # all_successful_Events_dict = {}
    # all_successful_interEvents_dict = {}

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
                                    model_format = model_format, parameter_sheet=parameter_sheet)

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
        ewr_results.to_csv(os.path.join(output_path, scenario_name + "_ewr_results.csv"), index=False)
        yearly_ewr_results.to_csv(os.path.join(output_path, scenario_name + "_yearly_ewr_results.csv"), index=False)
        all_events.to_csv(os.path.join(output_path,scenario_name +  "_all_events.csv"), index=False)
        all_interEvents.to_csv(os.path.join(output_path, scenario_name + "_all_interEvents.csv"), index=False)
        all_successful_Events.to_csv(os.path.join(output_path, scenario_name + "_all_successful_events.csv"), index=False)
        all_successful_interEvents.to_csv(os.path.join(output_path,scenario_name + "_all_successful_interEvents.csv"), index=False)
        print(f"Results saved to: {output_path}")

        # # Save the final tables to the dictionaries:   
        # ewr_results_dict[scenario_name] = ewr_results
        # yearly_results_dict[scenario_name] = yearly_ewr_results
        # all_events_dict[scenario_name] = all_events_dict
        # all_interEvents_dict[scenario_name] = all_interEvents
        # all_successful_Events_dict[scenario_name] = all_successful_Events
        # all_successful_interEvents_dict[scenario_name] = all_successful_interEvents

run_ewr_scenario_and_save_results(scenarios, model_format,parameter_sheet, output_path)
