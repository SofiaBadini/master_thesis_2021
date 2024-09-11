"""Given observed moments and weighting matrix in `OUT_ANALYSIS`, "msm_estimation",
generate values of Method of Simulated moment criterion function for different
value of selected parameters, keeping all other parameters fixed.

The goal is to study whether the resulting univariate distributions (in particular,
those of the time-preference patameters) have a minimum around the true parameter
value and are reasonably smooth.

"""
import itertools

import numpy as np
import pandas as pd
import respy as rp
import yaml
from tqdm import tqdm

from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _load_pickle
from src.library.housekeeping import _save_to_pickle
from src.library.housekeeping import _temporary_working_directory

from src.library.compute_moments import calc_very_restricted_choice_probabilities
from src.library.compute_moments import calc_restricted_choice_probabilities
from src.library.compute_moments import calc_unrestricted_choice_probabilities
from src.library.compute_moments import calc_very_restricted_wage_distribution
from src.library.compute_moments import calc_restricted_wage_distribution
from src.library.compute_moments import calc_unrestricted_wage_distribution
from src.library.compute_moments import _replace_nans
from src.library.compute_moments import _replace_nans

from src.analysis.calc_moments import calc_moments


def get_univariate_distribution(params, params_base, crit_func, steps):
    """Get values of criterion function for different values of selected
    parameters, keeping all other `params` fixed.

    Args:
        params (pd.DataFrame): Model parameters.
        params_base (pd.DataFrame): Dataframe of parameters whose values will be
            varied to compute the criterion function. Need to have column "upper"
            and "lower" specifying respectively the maximum and minimum
            parameter's value for which the criterion function is computed.
        crit_func (func): Crtierion function.
        steps (numpy.ndarray): Step size. Determine the number of parameters'
            values for which the criterion function is evaluated.

    Returns:
        dict

    """
    resultsDict = dict()

    for index in tqdm(params_base.index):

        upper, lower = params_base.loc[index][["upper", "lower"]]
        grid = np.linspace(lower, upper, steps)

        results = list()

        for value in tqdm(grid):
            params_ = params.copy()
            params_.loc[index, "value"] = value
            fval = crit_func(params_)
            result = {"x": value, "y": fval}
            results.append(result)

        resultsDict[index] = pd.DataFrame.from_dict(results)

    return resultsDict


if __name__ == "__main__":

    for s in ["exp", "hyp"]:

        # load params
        params = pd.read_csv(
            ppj("IN_MODEL_SPECS", f"params_{s}.csv"),
            sep=";",
            index_col=["category", "name"],
        )
        params["value"] = params["value"].astype(float)

        # load options
        with open(ppj("IN_MODEL_SPECS", f"options_{s}.yaml")) as options:
            options = yaml.safe_load(options)

        # get empirical moments
        empirical_moments = _load_pickle(
            ppj("OUT_ANALYSIS", "msm_estimation", f"moments_{s}.pickle")
        )

        # get weighting matrix
        weighting_matrix = _load_pickle(
            ppj("OUT_ANALYSIS", "msm_estimation", f"weighting_matrix_{s}.pickle")
        )

        # import `params_base`
        params_base = pd.read_csv(
            ppj("IN_MODEL_SPECS", f"params_base_{s}.csv"),
            sep=",",
            index_col=["category", "name"],
        )

        params_base["value"] = params_base["value"].astype(float)

        params_base_local = params_base[params_base["grid"] == "local"]
        params_base_global = params_base[params_base["grid"] == "global"]

        steps_local = 41
        steps_global = 81

        with _temporary_working_directory(snippet=f"univariate_{s}"):

            # get criterion function
            weighted_sum_squared_errors = rp.get_moment_errors_func(
                params=params,
                options=options,
                calc_moments=calc_moments[s],
                replace_nans=_replace_nans,
                empirical_moments=empirical_moments,
                weighting_matrix=weighting_matrix,
            )

            for params_base, steps, suffix in zip(
                [params_base_global, params_base_local],
                [steps_global, steps_local],
                ["global", "local"],
            ):

                # get criterion results, local
                results = get_univariate_distribution(
                    params=params,
                    params_base=params_base,
                    crit_func=weighted_sum_squared_errors,
                    steps=steps,
                )

                _save_to_pickle(
                    results,
                    ppj(
                        "OUT_ANALYSIS",
                        "univariate_distr_data",
                        f"univariate_distr_{s}_{suffix}.pickle",
                    ),
                )
