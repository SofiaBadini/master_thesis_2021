"""Plot counterfactual predictions."""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.final.visualize import plot_counterfactual_predictions
from src.final.visualize import plot_average_predicted_choices
from src.final.visualize import plot_ridgeplot

from src.final.metadata.dictionaries import style_dict_subsidy
from src.final.metadata.dictionaries import style_dict_life_cycle
from src.final.metadata.dictionaries import title_dict
from src.final.metadata.dictionaries import moments_dict

from src.library.housekeeping import _load_pickle

if __name__ == "__main__":

    # true effect on years of education (distribution)
    subsidy_effect_distr = pd.read_csv(
        ppj("OUT_ANALYSIS", "counterfactual_analysis", "subsidy_effect_distr.csv")
    )

    fig = plot_ridgeplot(
        df=subsidy_effect_distr,
        groups=list(range(1, 40)),
        value_col="value",
        xlim=(-0.05, 2),
        figsize=(16, 10),
        col_palette=None,
        bw=0.015,
        kernel="epanechnikov",
        first_group_is_complete=False,
        ticker_value=0.25,
        hspace=0.2,
    )

    fig.savefig(
        ppj("OUT_FIGURES", "counterfactual", "ridgeplot.png"),
        dpi=350,
        bbox_inches="tight",
    )

    # comparison of predicted effects
    data = _load_pickle(
        ppj(
            "OUT_ANALYSIS",
            "counterfactual_analysis",
            "subsidy_effect_per_period.pickle",
        )
    )

    data1 = {
        key: value
        for key, value in data.items()
        if key in ["true", "miss_exp", "miss_1"]
    }
    data2 = {
        key: value for key, value in data.items() if key in ["true", "miss_2", "miss_3"]
    }

    for i, d in enumerate([data1, data2]):
        fig = plot_counterfactual_predictions(
            data_dict=d,
            style_dict=style_dict_subsidy,
            title_dict=title_dict,
            mom="Experience_Edu",
            ylabel="Years of Education",
        )
        fig.savefig(
            ppj("OUT_FIGURES", "counterfactual", f"prediction_{i+1}.png"),
            bbox_inches="tight",
            dpi=350,
        )

    # life cycle pattern
    data = {"true": {}, "miss_exp": {}, "miss_1": {}, "miss_2": {}, "miss_3": {}}
    for model in data.keys():
        data[model] = _load_pickle(
            ppj("OUT_DATA", "counterfactual_data", f"data_{model}.pickle")
        )

    data1 = {
        "true": {"data": data["true"], "title": title_dict["true"]},
        "miss_exp": {"data": data["miss_exp"], "title": title_dict["miss_exp"],},
    }

    data2 = {
        "true": {"data": data["true"], "title": title_dict["true"]},
        "miss_1": {"data": data["miss_1"], "title": title_dict["miss_1"]},
        "miss_2": {"data": data["miss_2"], "title": title_dict["miss_2"]},
        "miss_3": {"data": data["miss_3"], "title": title_dict["miss_3"]},
    }

    df_emp = _load_pickle(ppj("OUT_DATA", "main_datasets", "df_hyp.pickle"))

    for i, d in enumerate([data1, data2]):
        fig = plot_average_predicted_choices(
            df_emp=df_emp,
            data_dict=d,
            moments_dict=moments_dict,
            style_dict=style_dict_life_cycle,
        )
        fig.savefig(
            ppj("OUT_FIGURES", "counterfactual", f"fit_{i+1}.png"),
            bbox_inches="tight",
            dpi=350,
        )
