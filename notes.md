### BASIC OVERVIEW ####
 1. Eliminate bloat where possible
    - not changing any logic of calculations 
 3. Employ numpy arrays were possible
    - vectorise operations where possible (may not be for more complicated EWRs)
 4. Parallel processing/threadpool executor?
 5. Dask dataframes?
    - in tool parallelisation by way of changing the class of the dataframes to something that can easily interface compute clusters and on device cpu and memory to be parallelised
    - https://docs.dask.org/en/stable/dataframe.html
 5. Ray?
 6. Numba?
 7. Cython / c 
    - move some guts:
      - probably the calc bits not sure? 
