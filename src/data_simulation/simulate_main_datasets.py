"""Simulate main datasets for the two different specifications of the model
based on :cite:`KeaneWolpin1994` (KW94).

"""
import pandas as pd
import respy as rp
import yaml

from bld.project_paths import project_paths_join as ppj
from src.library.housekeeping import _temporary_working_directory

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

        # change seeds to simulated "observed" data
        options["solution_seed"] = 0
        options["simulation_seed"] = 1000

        with _temporary_working_directory(snippet=s):
            # simulate main datasets
            simulate = rp.get_simulate_func(params, options)
            df = simulate(params)
            df.to_pickle(ppj("OUT_DATA", "main_datasets", f"df_{s}.pickle"))
