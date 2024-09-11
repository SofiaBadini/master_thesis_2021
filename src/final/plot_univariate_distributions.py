"""Plot criterion values univariate distribution for selected parameters."""
import pandas as pd
import seaborn as sns
import json

from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _load_pickle

from src.final.metadata.dictionaries import title_dict_univariate
from src.final.visualize import plot_univariate_distribution

# Set specs for plot style
sns.set_context("paper", font_scale=1.5)
sns.set_style("white")


if __name__ == "__main__":

    # load data
    for s in ["exp", "hyp"]:

        # load `params_base`
        params_base = pd.read_csv(
            ppj("IN_MODEL_SPECS", f"params_base_{s}.csv"),
            sep=",",
            index_col=["category", "name"],
        )

        params_base["value"] = params_base["value"].astype(float)

        for suffix in ["local", "global"]:

            # get `params_base`
            _params_base = params_base[params_base["grid"] == suffix]

            # load data
            dataDict = _load_pickle(
                ppj(
                    "OUT_ANALYSIS",
                    "univariate_distr_data",
                    f"univariate_distr_{s}_{suffix}.pickle",
                )
            )

            if suffix == "local":

                # Occupation A
                _temp = {k: dataDict[k] for k in dataDict.keys() if k[0] == "wage_a"}
                fig = plot_univariate_distribution(
                    _temp, title_dict_univariate, _params_base
                )
                fig.savefig(
                    ppj("OUT_FIGURES", "univariate_distributions", f"occ_a_{s}.png"),
                    dpi=350,
                    bbox_inches="tight",
                )

                # Occupation B
                _temp = {k: dataDict[k] for k in dataDict.keys() if k[0] == "wage_b"}
                fig = plot_univariate_distribution(
                    _temp, title_dict_univariate, _params_base, adjustment=True
                )
                fig.savefig(
                    ppj("OUT_FIGURES", "univariate_distributions", f"occ_b_{s}.png"),
                    dpi=350,
                    bbox_inches="tight",
                )

                # Shocks
                _temp = {
                    k: dataDict[k] for k in dataDict.keys() if k[0] == "shocks_sdcorr"
                }
                fig = plot_univariate_distribution(
                    _temp,
                    title_dict_univariate,
                    _params_base,
                    adjustment=True,
                    pad=0.15,
                )
                fig.savefig(
                    ppj(
                        "OUT_FIGURES",
                        "univariate_distributions",
                        f"shocks_sdcorr_{s}.png",
                    ),
                    dpi=350,
                    bbox_inches="tight",
                )

            if s == "exp":

                _temp = {("delta", "delta"): dataDict[("delta", "delta")]}
                fig = plot_univariate_distribution(
                    _temp, title_dict_univariate, _params_base, set_title=False
                )
                fig.savefig(
                    ppj(
                        "OUT_FIGURES",
                        "univariate_distributions",
                        f"time_preferences_{s}_{suffix}.png",
                    ),
                    dpi=350,
                    bbox_inches="tight",
                )

            elif s == "hyp":
                _temp = {
                    k: dataDict[k]
                    for k in (("delta", "delta"), ("beta", "beta"))
                    if k in dataDict
                }
                fig = plot_univariate_distribution(
                    _temp, title_dict_univariate, _params_base
                )
                fig.savefig(
                    ppj(
                        "OUT_FIGURES",
                        "univariate_distributions",
                        f"time_preferences_{s}_{suffix}.png",
                    ),
                    dpi=350,
                    bbox_inches="tight",
                )
