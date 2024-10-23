import pandas as pd
import multiprocessing as mp
import os

from py_ewr.scenario_handling import ScenarioHandler

#####--------------------- DEFINE EWR TOOL RUN FUCTION -------------------------#####

def run_EWR_tool(
        scenario_name,
        scenario_files,
        model_format
    ):

    ewr_results = pd.DataFrame()
    yearly_ewr_results = pd.DataFrame()
    all_events = pd.DataFrame()
    all_interEvents = pd.DataFrame()
    all_successful_Events = pd.DataFrame()
    all_successful_interEvents = pd.DataFrame()

    for file in scenario_files:
        # Running the EWR tool:
        ewr_sh = ScenarioHandler(scenario_file=file, model_format=model_format)

        # Return each table and stitch the different files of the same scenario together:
        # Table 1: Summarised EWR results for the entire timeseries
        temp_ewr_results = ewr_sh.get_ewr_results()
        ewr_results = pd.concat([ewr_results, temp_ewr_results], axis=0)
        # Table 2: Summarised EWR results, aggregated to water years:
        temp_yearly_ewr_results = ewr_sh.get_yearly_ewr_results()
        yearly_ewr_results = pd.concat(
            [yearly_ewr_results, temp_yearly_ewr_results], axis=0
        )
        # Table 3: All events details regardless of duration
        temp_all_events = ewr_sh.get_all_events()
        all_events = pd.concat([all_events, temp_all_events], axis=0)
        # Table 4: Inverse of Table 3 showing the interevent periods
        temp_all_interEvents = ewr_sh.get_all_interEvents()
        all_interEvents = pd.concat([all_interEvents, temp_all_interEvents], axis=0)
        # Table 5: All events details that also meet the duration requirement:
        temp_all_successfulEvents = ewr_sh.get_all_successful_events()
        all_successful_Events = pd.concat(
            [all_successful_Events, temp_all_successfulEvents], axis=0
        )
        # Table 6: Inverse of Table 5 showing the interevent periods:
        temp_all_successful_interEvents = ewr_sh.get_all_successful_interEvents()
        all_successful_interEvents = pd.concat(
            [all_successful_interEvents, temp_all_successful_interEvents], axis=0
        )

    ewr_results.to_csv(scenario_name + "_all_results.csv")
    yearly_ewr_results.to_csv(scenario_name + "_yearly_ewr_results.csv")
    all_events.to_csv(scenario_name + "_all_events.csv")
    all_interEvents.to_csv(scenario_name + "_all_interevents.csv")
    all_successful_Events.to_csv(scenario_name + "_all_successful_Events.csv")
    all_successful_interEvents.to_csv(scenario_name + "_all_successful_interEvents.csv")


#####----------------------- RUN EWR_TOOL IN PARALLEL --------------------------#####

# get the number of cores
num_processes = os.cpu_count()

if __name__ == "__main__":

    # OPTIONAL USER INPUT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # define number of cores here manually:
    # default is all available cores (num_processes) replace with intger eg. processes = 4
    
    pool = mp.Pool(processes = num_processes)

    # END OPTIONAL USER INPUT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # USER INPUT REQUIRED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Minimum 1 scenario and 1 related file required
    scenarios = {
        "Scenario1": ["file/location/1", "file/location/2", "file/location/3"],
        "Scenario2": ["file/location/1", "file/location/2", "file/location/3"],
    }

    # model  format
    model_format = "Bigmod - MDBA"

    # END USER INPUT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # argument list for the run_EWR tool function
    args_list = [
        (scenario_name, scenario_files, model_format)
        for scenario_name, scenario_files in scenarios.items()
    ]

    pool.starmap(run_EWR_tool, args_list)
    pool.close()
    pool.join()



