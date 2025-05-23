from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py_ewr",
    version="2.3.7",
    author="Martin Job",
    author_email="Martin.Job@mdba.gov.au",
    description="Environmental Water Requirement calculator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MDBAuth/EWR_tool",
    project_urls={
        "Bug Tracker": "https://github.com/MDBAuth/EWR_tool/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Framework :: Pytest',
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "pandas>2",#==2.0.3",
        "requests>2",#==2.25.1",
        "mdba-gauge-getter==0.5.1",
        "cachetools>5",#==5.2.0",
        "xarray",#==2023.09.0",
        "h5py",#==3.12.1",
        "netCDF4",#==1.6.4",
        "numpy<2"
    ],
    package_data={'': ["model_metadata/*.csv", "parameter_metadata/*.csv","parameter_metadata/*.json"]},
)
