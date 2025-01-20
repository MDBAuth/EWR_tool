import pandas as pd
from pathlib import Path
import os

BASE_PATH = Path(__file__).resolve().parents[1]


def get_relevant_EWRs(
    EWR_df,
    relevance_df,
    EWR_join=["LTWPShortName", "Gauge", "Code"],
    relevance_join=["LTWPShortName", "gauge", "ewr_code"],
):
    """Takes in the EWR dataframe, and a data frame with a column representing
    if an EWR is relevant to a project.

    The default is to assume that a project is relevant

    Args:
        EWR_df (pd.DataFrame): Dataframe of EWRs
        relevance_df (pd.DataFrame): Dataframe with identifying which EWRs are relevant.
            Must contain a column headed "relevant", where 1 == TRUE
        EWR_join (list): List of column names to join on for the EWR_df (foreign key)
        relevance_join (list): List of column names to join on, corresponding to the EWR_df (foreign key)

    Results:
        EWRs (pd.DataFrame): Dataframe containing only those EWRs that are relevant
    """
    # Merge dataframes on desired columns
    merged_df = pd.merge(
        EWR_df, relevance_df, left_on=EWR_join, right_on=relevance_join, how="left"
    )
    print("step1:", merged_df)

    # Filter dataframe for relevant EWRs. Assume that if there is no entry, the EWR is relevant.
    relevant_EWRs = merged_df[(merged_df["relevant"] != 0)]
    print("step 2:", relevant_EWRs)

    # Tidy up and remove the relevance column
    relevant_EWRs = relevant_EWRs.drop("relevant", axis=1)
    print("step3:", relevant_EWRs)
    return relevant_EWRs


def find_duplicates(df, primary_key):
    """
    Used for sanity check of the primary key, to ensure there aren't duplicate rows where there
    shouldn't be (there are rip).
    There used to be a functionality in get_relevant_ewrs to remove duplicates, but I've taken that away
    as it's not relevant
    """

    df_copy = df.loc[:]
    df_copy["duplicate_rows"] = df_copy.duplicated(subset=primary_key)
    duplicates = df_copy[df_copy["duplicate_rows"] == True]
    print(duplicates)


if __name__ == "__main__":
    # Set up paths
    base_path = os.path.join(BASE_PATH, "py_ewr/parameter_metadata")
    parameter_sheet = os.path.join(base_path, "parameter_sheet.csv")
    ewr_relevance = os.path.join(base_path, "ewr_relevance.csv")
    small_relevance = os.path.join(base_path, "small_relevance.csv")

    # What column headings we should join on. One-to-one relationship
    # EWR_join = ['PlanningUnitName','LTWPShortName','Gauge', 'Code'] # todo NOTE there are duplicates in this version
    # relevance_join = ['PlanningUnitName','LTWPShortName','gauge','Code']
    EWR_join = ["Gauge"]
    relevance_join = ["gauge"]

    # Load data
    EWR_df = pd.read_csv(
        parameter_sheet,
        usecols=[
            "PlanningUnitID",
            "PlanningUnitName",
            "LTWPShortName",
            "CompliancePoint/Node",
            "Gauge",
            "Code",
            "StartMonth",
            "EndMonth",
            "TargetFrequency",
            "TargetFrequencyMin",
            "TargetFrequencyMax",
            "EventsPerYear",
            "Duration",
            "MinSpell",
            "FlowThresholdMin",
            "FlowThresholdMax",
            "MaxInter-event",
            "WithinEventGapTolerance",
            "WeirpoolGauge",
            "FlowLevelVolume",
            "LevelThresholdMin",
            "LevelThresholdMax",
            "VolumeThreshold",
            "DrawdownRate",
            "AccumulationPeriod",
            "Multigauge",
            "MaxSpell",
            "TriggerDay",
            "TriggerMonth",
            "DrawDownRateWeek",
        ],
        dtype="str",
        encoding="cp1252",
    )
    relevance_df = pd.read_csv(
        ewr_relevance,
        usecols=["PlanningUnitName", "LTWPShortName", "gauge", "Code", "relevant"],
        dtype={
            "PlanningUnitName": "str",
            "LTWPShortName": "str",
            "Gauge": "str",
            "Code": "str",
            "relevant": "float",
        },
        #    dtype='str',
        encoding="utf-8-sig",
    )
    small_relevance_df = pd.read_csv(
        small_relevance,
        usecols=["gauge", "relevant"],
        dtype={"gauge": "str", "relevant": "float"},
        encoding="utf-8-sig",
    )

    get_relevant_EWRs(EWR_df, small_relevance_df, EWR_join, relevance_join)
