from py_ewr import scenario_handling

# Testing the netcdf format:
# Input params
scenarios =  ['unit_testing_files/ex_tasker.nc']
model_format = 'IQQM - netcdf'
allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
climate = 'NSW 10,000 year climate sequence'

# Pass to the class

ewr_sh = scenario_handling.ScenarioHandler(scenarios, model_format, allowance, climate)

ewr_sh.process_scenarios()

# Testing the iqqm csv format:
# Input params
scenarios =  ['unit_testing_files/murray_IQQM_df_wp.csv']
model_format = 'IQQM - NSW 10,000 years'
allowance = {'minThreshold': 1.0, 'maxThreshold': 1.0, 'duration': 1.0, 'drawdown': 1.0}
climate = 'NSW 10,000 year climate sequence'

# Pass to the class

ewr_sh = scenario_handling.ScenarioHandler(scenarios, model_format, allowance, climate)

ewr_sh.process_scenarios()