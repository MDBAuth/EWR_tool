### BASIC OVERVIEW ####
 1. eliminate bloat where possible
    - not changing any logic of calculations 
 3. employ numpy arrays were possible
     - vectorise operations where possible (may not be for more complicated EWRs)
  4. Dask dataframes?
     - in tool parallelisation by way of changing the class of the dataframes to something that can easily interface compute clusters and on device cpu and memory to be parallelised
     - https://docs.dask.org/en/stable/dataframe.html
  5. cython / c 
     - move some guts:
       - probably the calc bits not sure? 
