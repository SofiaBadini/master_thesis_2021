"""Compute moments and weighting matrix for empirical datasets."""

import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _save_to_pickle

from src.library.compute_moments import calc_very_restricted_choice_probabilities
from src.library.compute_moments import calc_restricted_choice_probabilities
from src.library.compute_moments import calc_unrestricted_choice_probabilities
from src.library.compute_moments import calc_very_restricted_wage_distribution
from src.library.compute_moments import calc_restricted_wage_distribution
from src.library.compute_moments import calc_unrestricted_wage_distribution
from src.library.compute_moments import get_weighting_matrix
from src.library.compute_moments import _replace_nans

from src.analysis.calc_moments import calc_moments


if __name__ == "__main__":

    for s in ["exp", "hyp"]:

        # load observed data
        df = pd.read_pickle(ppj("OUT_DATA", "main_datasets", f"df_{s}.pickle"))

        # compute empirical moments
        empirical_moments = {
            "Choice Probabilities Restricted": _replace_nans(
                calc_restricted_choice_probabilities(df)
            ),
            "Choice Probabilities Unrestricted": _replace_nans(
                calc_unrestricted_choice_probabilities(df)
            ),
            "Wage Distribution Restricted": _replace_nans(
                calc_restricted_wage_distribution(df)
            ),
            "Wage Distribution Unrestricted": _replace_nans(
                calc_unrestricted_wage_distribution(df)
            ),
        }

        if s == "hyp":
            empirical_moments["Choice Probabilities Very Restricted"] = _replace_nans(
                calc_very_restricted_choice_probabilities(df)
            )
            empirical_moments["Wage Distribution Very Restricted"] = _replace_nans(
                calc_very_restricted_wage_distribution(df)
            )

        # compute weighting matrix
        weighting_matrix = get_weighting_matrix(
            data=df,
            empirical_moments=empirical_moments,
            calc_moments=calc_moments[s],
            n_bootstrap_samples=250,
            n_observations_per_sample=5000,
        )

        _save_to_pickle(
            weighting_matrix,
            ppj("OUT_ANALYSIS", "msm_estimation", f"weighting_matrix_{s}.pickle"),
        )
        _save_to_pickle(
            empirical_moments,
            ppj("OUT_ANALYSIS", "msm_estimation", f"moments_{s}.pickle"),
        )
