"""Compute effect of 2000 US dollar tuition subsidy predicted by different
model specifications.

"""
import itertools
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _load_pickle
from src.library.housekeeping import _save_to_pickle


def compute_subsidy_effect_on_experience(data_without_subsidy, data_with_subsidy):
    """Compute effect of subsidy subsidy on occupational experience (i.e.
        education, occupation A and occupation B).

    Args:
        data_without_subsidy (list of dataset): List of dataset, without subsidy
            subsidy.
        data_with_subsidy (list of dataset): List of dataset, with subsidy
            subsidy. Must have the same length of `data_without_subsidy`.
        mom_func (func): Function to compute moment of interest.

    Returns:
        dict

    """
    effectDict = {"Experience_Edu": {}, "Experience_A": {}, "Experience_B": {}}

    for i, mom in enumerate(["Experience_Edu", "Experience_A", "Experience_B"]):

        subsidy_effect = compute_subsidy_effect(
            data_without_subsidy, data_with_subsidy, mom
        )

        mean = {"mean": subsidy_effect.mean(axis=0)}
        std = {"std": subsidy_effect.std(axis=0)}

        effectDict[mom].update(mean)
        effectDict[mom].update(std)

    return effectDict


def compute_subsidy_effect(data_without_subsidy, data_with_subsidy, moment):
    """Compute effect of subsidy subsidy on a certain moment (i.e. years of education).

    Args:
        data_without_subsidy (list of dataset): List of dataset, without subsidy
            subsidy.
        data_with_subsidy (list of dataset): List of dataset, with subsidy
            subsidy. Must have the same length of `data_without_subsidy`.
        moment (func): Moment of interest.

    Returns:
        numpy.ndarray

    """
    momDict = {}

    keys = ["mom_without_subsidy", "mom_with_subsidy"]

    for data, key in zip([data_without_subsidy, data_with_subsidy], keys):
        momDict[key] = np.array([df[(moment, "mean")] for df in data])

    effect = momDict["mom_with_subsidy"] - momDict["mom_without_subsidy"]

    return effect


def get_predicted_effect(dataDict, mom):

    res = {key: {} for key in dataDict.keys()}

    for key, data in dataDict.items():
        no_sub = np.array([df.iloc[39][(mom, "mean")] for df in data["no_sub"]])
        sub = np.array([df.iloc[39][(mom, "mean")] for df in data["sub"]])

        effect = sub - no_sub

        res[key]["Mean"] = effect.mean()
        res[key]["Median"] = np.median(effect)
        res[key]["SD"] = effect.std()
        res[key]["Min."] = effect.min()
        res[key]["Max."] = effect.max()

    return res


if __name__ == "__main__":

    data = {"true": {}, "miss_exp": {}, "miss_1": {}, "miss_2": {}, "miss_3": {}}
    subsidy_effect_per_period = {}
    for model in data.keys():
        data[model]["no_sub"] = _load_pickle(
            ppj("OUT_DATA", "counterfactual_data", f"data_{model}.pickle")
        )
        data[model]["sub"] = _load_pickle(
            ppj("OUT_DATA", "counterfactual_data", f"data_{model}_subsidy.pickle")
        )

        subsidy_effect_per_period[model] = compute_subsidy_effect_on_experience(
            data[model]["no_sub"], data[model]["sub"]
        )
    subsidy_effect_average = get_predicted_effect(data, "Experience_Edu")

    subsidy_effect_distr = compute_subsidy_effect(
        data["true"]["no_sub"], data["true"]["sub"], "Experience_Edu"
    )
    subsidy_effect_distr = (
        pd.DataFrame(subsidy_effect_distr).melt().rename(columns={"variable": "group"})
    )
    subsidy_effect_distr = subsidy_effect_distr[subsidy_effect_distr["group"] > 0]

    # save results
    subsidy_effect_distr.to_csv(
        ppj("OUT_ANALYSIS", "counterfactual_analysis", "subsidy_effect_distr.csv")
    )
    for subsidy_effect, suffix in zip(
        [subsidy_effect_per_period, subsidy_effect_average], ["per_period", "average"]
    ):
        _save_to_pickle(
            subsidy_effect,
            ppj(
                "OUT_ANALYSIS",
                "counterfactual_analysis",
                f"subsidy_effect_{suffix}.pickle",
            ),
        )
