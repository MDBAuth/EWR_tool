from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py_ewr",
    version="0.0.4",
    author="Martin Job",
    author_email="Martin.Job@mdba.gov.au",
    description="NSW Environmental Water Requirement calculator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MDBAuth/EWR_tool",
    project_urls={
        "Bug Tracker": "https://github.com/MDBAuth/EWR_tool/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "ipython==8.3.0",
        "ipywidgets==7.7.0",
        "pandas==1.4.2",
        "requests==2.25.1",
        "tqdm==4.64.0",
        "traitlets==5.2.0",
        "mdba-gauge-getter==0.2",
    ],
    package_data={'': ['climate_data/*.csv',"model_metadata/*.csv"]},
    python_requires=">=3.6",
)
