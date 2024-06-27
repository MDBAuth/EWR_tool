import xarray as xr
from pandas import DataFrame as Dataframe


def read_netcdf_as_dataframe(netcdf_path: str) -> Dataframe:
    dataset = xr.open_dataset(netcdf_path, engine='netcdf4')
    df = dataset.to_dataframe()
    dataset.close()

    return df


def save_dataframe_as_netcdf(df, output_path: str) -> None:
    # Convert DataFrame to Xarray Dataset
    ds = xr.Dataset.from_dataframe(df)
    
    # Modify variable names to ensure they are valid for NetCDF
    for var_name in ds.variables:
        new_var_name = var_name.replace(" ", "_")  # Replace spaces with underscores
        new_var_name = ''.join(c for c in new_var_name if c.isalnum() or c == "_")  # Remove non-alphanumeric characters
        ds = ds.rename({var_name: new_var_name})

    # Save the modified Xarray Dataset as a NetCDF file
    ds.to_netcdf(output_path)