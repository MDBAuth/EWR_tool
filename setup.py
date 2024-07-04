from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py_ewr",
    version="2.1.9",
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Framework :: Pytest',
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "ipython==8.8.0",
        "ipywidgets==7.7.0",
        "pandas==2.0.3",
        "requests==2.25.1",
        "tqdm==4.64.0",
        "mdba-gauge-getter==0.5.1",
        "cachetools==5.2.0",
        "numpy<2"
    ],
    package_data={'': ["model_metadata/*.csv", "parameter_metadata/*.csv","parameter_metadata/*.json"]},
)
