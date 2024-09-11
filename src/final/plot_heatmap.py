"""Plot heatmap criterion."""
import pandas as pd
import seaborn as sns
import numpy as np
from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _load_pickle

from src.final.visualize import plot_heatmap3d


if __name__ == "__main__":

    # load load
    data = pd.read_csv(
        ppj("OUT_ANALYSIS", "bivariate_distr_data", "bivariate_distr.csv"),
    )

    fig = plot_heatmap3d(data)
    fig.savefig(
        ppj("OUT_FIGURES", "heatmap", "heatmap.png"), dpi=350, bbox_inches="tight"
    )
