"""Auxiliary dictionaries."""

policy_dict = {"unrestricted": "Unr", "restricted": "R", "veryrestricted": "VR"}
color_dict = {"a": "#eb760a", "b": "#43aa8b", "edu": "#f9c74f", "home": "#577590"}
label_dict = {"a": "Occ. A", "b": "Occ. B", "edu": "Education", "home": "Home"}
title_dict = {
    "true": r"True ($\beta = 0.8, \delta = 0.95$)",
    "miss_exp": r"Exponential ($\beta = 1, \delta = 0.938$)",
    "miss_1": r"Global minimum ($\beta = 0.83, \delta = 0.948$)",
    "miss_2": r"Misspecified ($\beta = 0.86, \delta = 0.946$)",
    "miss_3": r"Misspecified ($\beta = 0.78, \delta = 0.952$)",
}
style_dict_subsidy = {
    "true": {"line": {"linestyle": "-"}, "fill": {"facecolor": "grey", "alpha": 0.35}},
    "miss_exp": {
        "line": {"linestyle": "-", "marker": "D", "mfc": "white"},
        "fill": {
            "hatch": "//",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.3,
        },
    },
    "miss_1": {
        "line": {"linestyle": "-", "marker": "D", "mfc": "black"},
        "fill": {
            "hatch": "\\",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.3,
        },
    },
    "miss_2": {
        "line": {"linestyle": "-", "marker": "o", "mfc": "black"},
        "fill": {
            "hatch": "//",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.3,
        },
    },
    "miss_3": {
        "line": {"linestyle": "-", "marker": "x", "markersize": 10},
        "fill": {
            "hatch": "\\",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.3,
        },
    },
}
style_dict_life_cycle = {
    "true": {"linestyle": "-", "color": "black", "linewidth": 2},
    "miss_exp": {"linestyle": "-", "color": "indianred", "alpha": 0.8, "linewidth": 2},
    "miss_1": {"linestyle": "-", "color": "#66c2a5", "alpha": 0.8, "linewidth": 1.75},
    "miss_2": {"linestyle": "-", "color": "#fc8d62", "alpha": 0.8, "linewidth": 1.5},
    "miss_3": {"linestyle": "-", "color": "#9da0cb", "alpha": 0.8, "linewidth": 1.5},
}
title_dict_univariate = {
    ("delta", "delta"): {"title": "Discount Factor", "label": r"$\delta$"},
    ("beta", "beta"): {"title": "Present Bias", "label": r"$\beta$"},
    ("wage_a", "exp_edu"): {"title": "Return to Education", "label": r"$\gamma^E_A$",},
    ("wage_a", "exp_a"): {
        "title": "Return to Experience in A",
        "label": r"$\gamma^A_A$",
    },
    ("wage_b", "exp_edu"): {"title": "Return to Education", "label": r"$\gamma^E_B$",},
    ("wage_b", "exp_a"): {
        "title": "Return to Experience in A",
        "label": r"$\gamma^A_B$",
    },
    ("wage_b", "exp_b"): {
        "title": "Return to Experience in B",
        "label": r"$\gamma^B_B$",
    },
    ("shocks_sdcorr", "sd_a"): {
        "title": "Standard deviation, A",
        "label": r"$\sigma_A$",
    },
    ("shocks_sdcorr", "sd_b"): {
        "title": "Standard deviation, B",
        "label": r"$\sigma_B$",
    },
    ("shocks_sdcorr", "sd_edu"): {
        "title": "Standard deviation, Education",
        "label": r"$\sigma_E$",
    },
    ("shocks_sdcorr", "sd_home"): {
        "title": "Standard deviation, Home",
        "label": r"$\sigma_H$",
    },
    ("shocks_sdcorr", "corr_b_a"): {
        "title": "Correlation between A and B",
        "label": r"$\rho_{A,B}$",
    },
    ("shocks_sdcorr", "corr_home_edu"): {
        "title": "Correlation between Home and Education",
        "label": r"$\rho_{H,E}$",
    },
}
