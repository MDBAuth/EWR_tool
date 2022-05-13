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
        "ipython==7.16.1",
        "ipywidgets==7.6.3",
        "numpy==1.19.2",
        "pandas==1.1.3",
        "requests==2.25.1",
        "tqdm==4.59.0",
        "traitlets==4.3.3",
        "xlsxwriter==3.0.1",
        "mdba-gauge-getter==0.2",
    ],
    python_requires=">=3.6",
)
