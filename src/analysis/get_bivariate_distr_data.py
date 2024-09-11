"""Given observed moments and weighting matrix in `OUT_ANALYSIS`, "msm_estimation",
generate values of Method of Simulated Moments criterion function for combinations
of discount factor and present bias values.

The goal is to study the bivariate distribution of the time preference parameters
around the combination of true parameter values.

"""
import itertools

import numpy as np
import pandas as pd
import respy as rp
import yaml
from tqdm import tqdm

from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _load_pickle
from src.library.housekeeping import _temporary_working_directory

from src.library.compute_moments import calc_very_restricted_choice_probabilities
from src.library.compute_moments import calc_restricted_choice_probabilities
from src.library.compute_moments import calc_unrestricted_choice_probabilities
from src.library.compute_moments import calc_very_restricted_wage_distribution
from src.library.compute_moments import calc_restricted_wage_distribution
from src.library.compute_moments import calc_unrestricted_wage_distribution
from src.library.compute_moments import _replace_nans

from src.analysis.calc_moments import calc_moments


def get_bivariate_distribution(params, crit_func, grid_delta, grid_beta):
    """Compute value of criterion function for differen value fo discount factor
    and present bias parameter.

    Args:
        params (pd.DataFrame): DataFrame containing model parameters.
        crit_func (dict): Dictionary containing model options.
        grid_delta (np.array): Values of discount factor.
        grid_beta (np.array): Values of present-bias parameter.

    Returns:
        pd.DataFrame

    """
    results = []

    for beta, delta in tqdm(itertools.product(grid_beta, grid_delta)):
        params_ = params.copy()
        params_.loc[("beta", "beta"), "value"] = beta
        params_.loc[("delta", "delta"), "value"] = delta
        val = crit_func(params_)
        result = {"beta": beta, "delta": delta, "val": val}
        results.append(result)

    return pd.DataFrame.from_dict(results)


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

        with _temporary_working_directory(snippet="heatmap"):

            # get criterion function
            weighted_sum_squared_errors = rp.get_moment_errors_func(
                params=params,
                options=options,
                calc_moments=calc_moments[s],
                replace_nans=_replace_nans,
                empirical_moments=empirical_moments,
                weighting_matrix=weighting_matrix,
            )

            # get bivariate distribution results
            results = get_bivariate_distribution(
                crit_func=weighted_sum_squared_errors,
                params=params,
                grid_delta=np.arange(0.945, 0.9625, 0.0025),
                grid_beta=np.arange(0.75, 1.05, 0.01),
            )

            results.to_csv(
                ppj("OUT_ANALYSIS", "bivariate_distr_data", "bivariate_distr.csv")
            )
