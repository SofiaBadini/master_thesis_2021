"""Generate tables. In particular: Summary statistics of empirical moments (p. 20),
choice restrictions regression results (p. 21), counterfactual predictions (p. 32).

"""

import json
import pandas as pd
import estimagic.visualization.estimation_table as et

from collections import namedtuple
from statsmodels.iolib.summary2 import summary_col

from src.final.metadata.dictionaries import title_dict

from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _load_pickle


def get_moments_tables(moments, df, names_dict, policies, labels):

    tables = {}

    choices = [moments[f"Choice Probabilities {label}"].mean() for label in labels]
    wages = [moments[f"Wage Distribution {label}"].mean() for label in labels]
    nobs = [
        df.query("Policy == @policy and Period == 0").shape[0] for policy in policies
    ]

    table_choices = pd.concat(
        [
            pd.DataFrame(
                data=choices, index=names_dict["index_choices"][: len(policies)]
            )
            .rename(columns=names_dict["column_choices"])
            .T,
            pd.DataFrame(data=nobs, index=names_dict["index_choices"][: len(policies)])
            .rename(columns={0: "Observations"})
            .T,
        ]
    )

    table_wages = pd.concat(
        [
            pd.DataFrame(data=wages, index=names_dict["index_wages"][: len(policies)])
            .rename(columns=names_dict["column_wages"])
            .T,
            pd.DataFrame(data=nobs, index=names_dict["index_wages"][: len(policies)])
            .rename(columns={0: "Observations"})
            .T,
        ]
    )

    tables["choices"] = table_choices
    tables["wages"] = table_wages

    return tables


if __name__ == "__main__":

    regDict = {"exp": {}, "hyp": {}}
    momDict = {"exp": {}, "hyp": {}}

    namedtuplee = namedtuple("namedtuplee", "params info")

    with open(ppj("IN_FINAL", "metadata", "names_dict.json"), "r") as f:
        names_dict = json.load(f)

    policiesDict = {
        "exp": ["unrestricted", "restricted"],
        "hyp": ["unrestricted", "restricted", "veryrestricted"],
    }
    labelsDict = {
        "exp": ["Unrestricted", "Restricted"],
        "hyp": ["Unrestricted", "Restricted", "Very Restricted"],
    }

    for s in ["exp", "hyp"]:

        # moment tables, generate and save iteratively
        moments = _load_pickle(
            ppj("OUT_ANALYSIS", "msm_estimation", f"moments_{s}.pickle")
        )
        df = pd.read_pickle(ppj("OUT_DATA", "main_datasets", f"df_{s}.pickle"))

        tables = get_moments_tables(
            moments, df, names_dict, policiesDict[s], labelsDict[s]
        )
        momDict[s] = tables

        # regression tables, create dictionary
        fitted_models = _load_pickle(
            ppj("OUT_ANALYSIS", "regression_results", f"fitted_models_{s}.pickle")
        )

        for i, fitted_model in enumerate(fitted_models):
            regDict[s][i] = namedtuplee(
                params=et._extract_params_from_sm(fitted_model),
                info={**et._extract_info_from_sm(fitted_model)},
            )

    # moment tables, save from dictionary
    for moment in ["choices", "wages"]:
        tables = pd.concat(
            [momDict["exp"][moment], momDict["hyp"][moment]],
            axis=1,
            keys=["Exponential discounting", "Hyperbolic discounting"],
        )
        tables.to_latex(
            ppj("OUT_TABLES", f"{moment}.tex"),
            float_format="{:0.2f}".format,
            escape=False,
        )

    # regression tables, save from dictionary
    res_tex = et.estimation_table(
        [regDict["exp"][0], regDict["hyp"][0], regDict["exp"][1], regDict["hyp"][1],],
        return_type="latex",
        custom_model_names={
            r"Years of education ($t=10$)": [0, 1],
            r"Years of education ($t=40$)": [2, 3],
        },
        custom_col_names=["Exponential", "Hyperbolic", "Exponential", "Hyperbolic"],
        custom_param_names={
            0: "Restriction",
            "x1": "Restriction",
            "x2": "Double restriction",
            "const": "Intercept",
        },
        left_decimals=4,
        alignment_warning=False,
        siunitx_warning=False,
    )

    with open(ppj("OUT_TABLES", "regressions.tex"), "w") as f:
        f.write(res_tex)

    # counterfactual predictions table
    counterfactuals = _load_pickle(
        ppj("OUT_ANALYSIS", "counterfactual_analysis", "subsidy_effect_average.pickle")
    )
    counterfactuals = pd.DataFrame.from_dict(counterfactuals).rename(columns=title_dict)
    counterfactuals.T.to_latex(
        ppj("OUT_TABLES", "counterfactuals.tex"),
        escape=False,
        float_format="{:0.3f}".format,
    )
