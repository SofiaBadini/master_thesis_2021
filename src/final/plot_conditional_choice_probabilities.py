"""Plot conditional choice probabilities in main datasets."""

from bld.project_paths import project_paths_join as ppj

from src.final.metadata.dictionaries import color_dict
from src.final.metadata.dictionaries import label_dict

from src.final.visualize import compare_choice_probabilities
from src.library.housekeeping import _load_pickle


for s in ["exp", "hyp"]:

    # load dataset
    df = _load_pickle(ppj("OUT_DATA", "main_datasets", f"df_{s}.pickle"))

    kwargs = (
        {"policy_dict": {"unrestricted": "Unr", "restricted": "R"}}
        if s is "exp"
        else {
            "policy_dict": {
                "unrestricted": "Unr",
                "restricted": "R",
                "veryrestricted": "VR",
            }
        }
    )

    fig_choice_probabilities = compare_choice_probabilities(
        df=df, color_dict=color_dict, label_dict=label_dict, **kwargs
    )
    fig_choice_probabilities.savefig(
        ppj("OUT_FIGURES", "choice_probabilities", f"choice_probabilities_{s}.png"),
        bbox_inches="tight",
        dpi=350,
    )
