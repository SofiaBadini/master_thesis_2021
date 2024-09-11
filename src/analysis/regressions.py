"""Regressions to check effect of restrictions."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _save_to_pickle


if __name__ == "__main__":

    for s in ["exp", "hyp"]:

        df = pd.read_pickle(ppj("OUT_DATA", "main_datasets", f"df_{s}.pickle"))

        df_restricted = df.query("Period == 0")
        restriction = np.where(df_restricted["Policy"] == "restricted", 1, 0)

        if s is "hyp":
            restriction2 = np.where(df_restricted["Policy"] == "veryrestricted", 1, 0)
            restriction = np.column_stack((restriction, restriction2))

        restriction = sm.add_constant(restriction)

        dep_vars = [
            df.query("Period == @period")["Experience_Edu"] for period in [9, 39]
        ]

        fitted_models = [
            sm.OLS(endog=dep_var, exog=restriction, missing="drop").fit(cov_type="HC1")
            for dep_var in dep_vars
        ]

        _save_to_pickle(
            fitted_models,
            ppj("OUT_ANALYSIS", "regression_results", f"fitted_models_{s}.pickle"),
        )
