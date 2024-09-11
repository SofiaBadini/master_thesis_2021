"""Visualization module."""
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

from src.library.compute_moments import calc_choice_probabilities
from src.library.compute_moments import calc_wage_distribution


def compare_choice_probabilities(df, policy_dict, color_dict, label_dict):
    """Plot choice probabilities, comparing behavior of restricted and
    unrestricted agents.

    Args:
        df (pd.DataFrame): Dataframe with choice probabilities.
        policy_dict (dict): Dictionary, from policies to policies.
        color_dict (dict): Dictionary, from choices to colors.
        label_dict (dict): Dictionary, from choices to labels.

    Return:
        Matplotlib figure.

    """
    sns.set_context("paper", font_scale=2)
    sns.set_style("white")
    fig = plt.figure(figsize=(15, 5))

    # policies
    policies = list(policy_dict.keys())
    n_policies = len(policies)

    # specify grid
    height_ratios = [1 / n_policies] * n_policies
    gs = fig.add_gridspec(
        n_policies, 2, height_ratios=height_ratios, width_ratios=[0.5, 1]
    )

    # conditional choices axis
    axs = [fig.add_subplot(gs[i, 0]) for i in range(0, n_policies)]

    # unconditional choices axis
    ax_main = fig.add_subplot(gs[0:, 1:2])

    for ax, policy in zip(axs, policies):

        df.query("Policy == @policy").groupby("Period").Choice.value_counts(
            normalize=True
        ).unstack().plot.bar(
            stacked=True, rot=0, legend=False, width=1.0, color=color_dict, ax=ax
        )

        # decluttering
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        # set policy name
        ax.set_title(
            policy_dict[policy], pad=20, x=-0.095, y=0, weight="bold", fontsize=16
        )

        # set dashed line at t = 4, where agents are restricted
        if policy == "restricted":
            ax.axvline(x=3.5, color="black", linestyle="dashed", linewidth=2)
        if policy == "veryrestricted":
            ax.axvline(x=3.5, color="black", linestyle="dashed", linewidth=2)
            ax.axvline(x=1.5, color="black", linestyle="dashed", linewidth=2)

    plt.suptitle("Conditional Choice Probabilities", x=0.255, y=1.07, fontsize=18)

    # main axis (unconditional choice probabilities)
    df.groupby("Period").Choice.value_counts(normalize=True).unstack().plot.bar(
        stacked=True, rot=90, legend=False, color=color_dict, width=0.75, ax=ax_main,
    )
    ax_main.yaxis.tick_right()
    ax_main.tick_params(right=False, bottom=False)
    ax_main.set_title("Unconditional Choice Probabilities", y=1.18, fontsize=18)
    ax_main.set_xlabel("Period", x=0.96, labelpad=20, fontsize=18)

    # legend
    labels = list(label_dict.values())
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0, -0.3),
        ncol=len(labels),
        labels=labels,
        frameon=False,
    )

    # annotate time preference parameter
    delta = df["Discount_Rate"][0][0]
    beta = df["Present_Bias"][0][0]
    plt.gcf().text(
        0.125, 0.915, f"δ = {delta}, β = {beta}", fontstyle="oblique", fontsize=18
    )

    fig.subplots_adjust(wspace=0.025, hspace=0.05)

    return fig


def plot_univariate_distribution(
    results_dict, title_dict, params_base, adjustment=False, pad=0.1, set_title=True
):
    """Plot parameters' univariate distribution.

        Args:
            results_dict (dict): Dictionary of results, where parameters' names are
                the keys and values of criterion function are the values.
            title_dict (dict): Dictionary mapping parameters to plot title.
            params_base (pd.DataFrame): DataFrame of parameters to be plotted.
            adjustment (bool): Whether to adjust the plot yticklabels. Default
                is False.
            pad (float): Where to position the new yticklabel, if `adjustment`
                is True. Default is 0.1.
            set_title (bool): Whether to set title for the plot. Default is True.

    """

    # Set specs for plot style
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")

    n_params = len(results_dict.keys())  # parameters belonging to the group
    n_rows = [round(n_params / 2) if n_params > 1 else 1][0]  # rows the plots will have

    if n_rows == 1:
        height = 4
    elif n_rows == 2:
        height = 9
    elif n_rows == 3:
        height = 15

    fig, axs = plt.subplots(n_rows, 2, figsize=(17.5, height))
    axs = axs.flatten()

    ymins = []
    ymaxs = []

    for (key, df), ax in zip(results_dict.items(), axs):

        x = df["x"].values
        y = df["y"].values

        ax.plot(x, y, color="black", linewidth=2)
        title = title_dict[key]["title"]
        xaxis_label = title_dict[key]["label"]
        ax.set_xlabel(f"{xaxis_label}", labelpad=7.5)
        ax.grid(axis="y")

        if set_title is True:
            ax.set_title(f"{title}", y=1.065, fontsize=16)

        true_value = params_base.loc[key, "value"]
        ax.axvline(
            true_value,
            color="#A9A9A9",
            linestyle="--",
            linewidth=3,
            label="True value",
        )

        local_minima = df[df["y"] == np.min(y)]["x"].values

        for i, local_minimum in enumerate(local_minima):
            kwargs = {"label": "Local minimum"} if i == 0 else {}
            ax.axvline(
                local_minimum, color="#A9A9A9", linestyle="--", linewidth=1.5, **kwargs,
            )

        ax.legend(frameon=False)
        ax.ticklabel_format(axis="y", style="sci", useMathText=True, scilimits=(2, 2))
        ax.set_yticklabels = [str(i) for i in np.linspace(np.min(y), np.max(y), 2)]
        ax.set_yticks(np.linspace(np.min(y), np.max(y), 2))

        if adjustment == True:
            keys = [
                ("wage_b", "exp_a"),
                ("shocks_sdcorr", "sd_edu"),
                ("shocks_sdcorr", "sd_home"),
                ("shocks_sdcorr", "corr_home_edu"),
            ]
            if key in keys:
                ax.set_yticklabels = [np.min(y)]
                ax.set_yticks([np.min(y)])
                annot = np.max(y) / 100
                ax.annotate(
                    annot.round(decimals=2),
                    (-0.0945, pad),
                    xycoords="axes fraction",
                    fontsize=13,
                )

        ymin, ymax = ax.get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)

    for ax in axs:
        ax.set_ylim(min(ymins), max(ymaxs))

    if n_rows * 2 > n_params:
        fig.delaxes(axs[-1])

    fig.subplots_adjust(hspace=0.5, wspace=0.125)

    return fig


def _get_custom_cmaps():
    """Generate customized non-linear color map and sequential cmap. """
    steps = [0, 0.25, 0.5, 0.8, 1]
    hexcolors = [
        # sequential
        ["#CBC0D3", "#D5CCDB", "#DFD9E4", "#EAE5ED", "#EFECF1"],
        # segmented
        ["#444572", "#577590", "#43aa8b", "#f9c74f", "#FAE450"],
    ]

    cmaps = []
    for hexcolor in hexcolors:
        colors = list(zip(steps, hexcolor))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            name="custom_cmap", colors=colors
        )
        cmaps.append(cmap)

    return cmaps


def get_custom_cmap():
    """Generate custom cmap."""
    steps = [0, 0.05, 0.2, 0.5, 1]
    hexcolors = ["#444572", "#577590", "#43aa8b", "#f9c74f", "#FAE450"]
    colors = list(zip(steps, hexcolors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name="custom_cmap", colors=colors
    )

    return cmap


def plot_heatmap2d(df, threshold):
    """Plot 2D heatmap from criterion values for combinations of different
    discount factor and present bias values.

    Args:
        df (pd.DataFrame): Data. Need to have three columns: "beta", "delta" and
            "val".
        threshold (float): Value of criterion that makes colormap switch.
        title (string): Title of plot.

    Returns:
        Matplotlib figure.

    """
    # Set specs for plot style
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")

    fig = plt.figure(figsize=(7.5, 8.5))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[0.965, 0.035], width_ratios=[0.625, 0.385]
    )
    ax = fig.add_subplot(gs[0:1, 0:])
    cax2 = fig.add_subplot(gs[1:, 0:1])
    cax1 = fig.add_subplot(gs[1:, 1:])

    cmaps = _get_custom_cmaps()

    # prepare data
    df = df.round({"beta": 3, "delta": 3})
    df_pivot = df.pivot_table(values="val", index="beta", columns="delta")
    max_ = np.max(df["val"])
    min_ = np.min(df["val"])

    # plot heatmap
    for i, cmap in enumerate(cmaps):
        kwargs = (
            {"mask": df_pivot > threshold, "vmin": min_, "vmax": threshold}
            if i == 1
            else {"mask": df_pivot < threshold, "vmin": threshold, "vmax": max_}
        )
        ax = sns.heatmap(df_pivot, cmap=cmap, cbar=False, ax=ax, **kwargs)
        ax.set_yticklabels(
            ax.get_yticklabels(), rotation=0, horizontalalignment="right"
        )

    for i, cax in enumerate([cax1, cax2]):
        cax.grid(False)
        ticks_cmap1 = [t for t in np.linspace(min_, threshold, 6)][:-1]
        ticks_cmap2 = [t for t in np.linspace(threshold, max_, 4)]
        kwargs = {"ticks": ticks_cmap1} if i == 1 else {"ticks": ticks_cmap2}
        format_ = matplotlib.ticker.ScalarFormatter(useOffset=False, useMathText=True)
        format_.set_powerlimits((3, 3))
        cb = colorbar(
            ax.get_children()[i],
            cax=cax,
            orientation="horizontal",
            drawedges=False,
            format=format_,
            **kwargs,
        )
        cb.ax.tick_params(labelsize=12)
        cb.ax.tick_params(size=0)
        for _, spine in cax.spines.items():
            spine.set_visible(False)

    cax2.xaxis.offsetText.set_visible(False)
    cax2.set_xlabel("Criterion value", labelpad=12, x=0.2)
    ax.set_xlabel(r"$\delta$", fontsize=16, labelpad=15)
    ax.set_ylabel(r"$\beta$", fontsize=16, rotation=0, labelpad=20)

    fig.axes[0].invert_yaxis()
    fig.subplots_adjust(wspace=0, hspace=0.45)

    return fig


def plot_heatmap3d(df_heatmap):
    """Plot 3D heatmap."""
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")

    fig = plt.figure(figsize=(25, 18))
    ax = plt.axes(projection="3d")
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # axes
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.w_zaxis.grid(visible=False)

    # labels
    ax.yaxis.labelpad = 30
    ax.xaxis.labelpad = 50
    ax.zaxis.labelpad = 30

    ax.tick_params(axis="z", pad=15)
    ax.tick_params(axis="x", pad=20)

    # data
    df_pivot = df_heatmap.pivot_table(values="val", index="beta", columns="delta")

    # coordinates
    x = df_heatmap.delta.drop_duplicates().values
    y = df_heatmap.beta.drop_duplicates().values
    x, y = np.meshgrid(x, y)

    z = df_pivot.values
    ax.set_zticks([])

    # colormap
    colormap = get_custom_cmap()
    surf = ax.plot_surface(x, y, z, cmap=colormap)
    format_ = matplotlib.ticker.ScalarFormatter(useOffset=False, useMathText=True)
    format_.set_powerlimits((3, 3))
    colorbar(
        surf,
        ax=ax,
        shrink=0.7,
        aspect=10,
        ticks=[10_000] + list(np.linspace(20_000, 140_000, 7)),
        format=format_,
    )

    # labels' names
    ax.set_xlabel(r"$\delta$", fontsize=20, rotation=270)
    ax.set_ylabel(r"$\beta$", fontsize=20)
    ax.set_zlabel("Criterion", fontsize=20, rotation=360)

    # disable automatic rotation
    ax.zaxis.set_rotate_label(False)

    ax.view_init(50, 10)

    return fig


def plot_counterfactual_predictions(data_dict, style_dict, title_dict, mom, ylabel):
    """Plot average predicted effect of tuition subsidy in each period, plus
    simulation noise, on specified moment.

    Args:
        data_dict (dict): Dictionary of data.
        style_dict (dict): Dictionary to style graphical elements of the plot.
        title_dict (dict): Dictionary of titles.
        mom (str): Moment to plot. Can be "A", "B", or "Edu".
        ylabel (str): Label of y-axis.

    Returns:
        Matplotlib figure.

    """
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=(15, 7.5))

    for key, data in data_dict.items():

        data = data[mom]

        ax.plot(
            data["mean"],
            color="black",
            linewidth=2,
            label=" ",
            **style_dict[key]["line"],
        )

        fill = ax.fill_between(
            range(0, len(data["mean"])),
            np.add(data["mean"], data["std"]),
            np.subtract(data["mean"], data["std"]),
            label=title_dict[key],
            **style_dict[key]["fill"],
        )

    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_xlabel("Period", labelpad=10)

    handles, labels = ax.get_legend_handles_labels()
    handles_positions = [[0, 1, 2], [3, 4, 5]]
    bbox_to_anchor = [(0.6, 0.3), (0.95, 0.3)]

    for i, title in enumerate(["Mean", "SD"]):
        kwargs = {"labelspacing": 0.875} if title == "Mean" else {"handletextpad": 1.75}
        legend = plt.legend(
            handles=list(handles[j] for j in handles_positions[i]),
            ncol=1,
            bbox_to_anchor=bbox_to_anchor[i],
            title=title,
            frameon=False,
            **kwargs,
        )
        legend._legend_box.align = "left"
        plt.gca().add_artist(legend)

    return fig


def plot_average_predicted_choices(
    df_emp, data_dict, moments_dict, style_dict, plot_observed=True
):
    """Plot moments of interest for list of pd.DataFrame.

    Args:
        df_emp (pd.DataFrame): Empirical dataset.
        data_dict (dict): Dictionary of data.
        moments_dict (dict): Dictionary of moments to plot.
        style_dict (dict): Dictionary to style graphical elements of the plot.
        plot_observed (bool): Whether the moment of the empirical datasets should
            be visualized in the plot.

    Return:
        Matplotlib figure.

    """
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")

    nrows = math.ceil(len(moments_dict) / 2)
    ncols = math.floor(len(moments_dict) / 2)
    kwargs = {"figsize": (15, 10)} if len(moments_dict) == 4 else {"figsize": (15, 15)}

    fig, axs = plt.subplots(nrows, ncols, **kwargs)
    axs = axs.flatten()

    for ax, moment in zip(axs, moments_dict.keys()):

        temp_dict = {}

        for key, value in data_dict.items():

            value = value["data"]
            qois = [df[moment] for df in value]
            temp_dict[key] = pd.concat(qois, axis=1).mean(axis=1)

        # plot average prediction
        for key in temp_dict.keys():
            ax.plot(temp_dict[key], label=data_dict[key]["title"], **style_dict[key])

        # plot observed conditional choice probabilities
        if plot_observed is True:
            ax.plot(
                calc_choice_probabilities(df_emp.query("Policy == 'unrestricted'"))[
                    moment
                ],
                linewidth=1.5,
                color="black",
                linestyle="--",
                alpha=0.75,
                label="Observed (unrestricted)",
            )

        # styling
        ax.set_title(moments_dict[moment], fontsize=16, fontweight="heavy")
        ax.set_xlim(xmin=-1)

    # adjust spacing
    fig.subplots_adjust(hspace=0.25, wspace=0.125)
    # add legend
    kwargs = (
        {"ncol": len(data_dict) + 1, "bbox_to_anchor": (0, -0.3)}
        if len(data_dict) < 4
        else {"ncol": 2, "bbox_to_anchor": (0, -0.4)}
    )
    plt.legend(loc="lower center", frameon=False, **kwargs)

    return fig


def plot_ridgeplot(
    df,
    groups,
    value_col,
    xlim=(-0.2, 1),
    figsize=(16, 10),
    col_palette=None,
    bw=0.015,
    kernel="epanechnikov",
    first_group_is_complete=True,
    ticker_value=0.25,
    hspace=-0.25,
):
    """Create overlapping densities plot.

    Args:
        df (pd.DataFrame): Data frame containing a value column (name passed as
            argument), and a group column (must be called "group"), that classifies each
            row to a single group. Note that the group must be categorical.
        groups (list): List of unique group names contained in df["group"]. The order of
            this list determines the order of the subplots.
        value_col (str): Name of value column.
        xlim (tuple): x-axis limits.
        figsize (tuple): Figure size.
        col_palette (list): Color palette. If None, defaults to
            itertools.cycle(seaborn.cubehelix_palette(10, rot=-.3, light=.7)).
        bw (float): Bandwidth for the density estimation kernel.
        first_group_is_complete (bool): If after getting the unique ordered groups the
            first group represents the complete sample.
        ticker_value (float): Float passed to matplotlib.ticker.MultipleLocator.
        hspace (float): Float passed to grid_spec.GridSpec().update.

    Returns:
        fig (matplotlib.figure.Figure): The ridge plot figure.

    """
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    n_groups = len(groups)
    n_obs = len(df) / 2 if first_group_is_complete else len(df)

    gs = grid_spec.GridSpec(n_groups, 1)
    fig = plt.figure(figsize=figsize)

    kde = KernelDensity(kernel=kernel, bandwidth=bw)

    for i, g in enumerate(groups):
        edgecol = "black"
        facecol = "lightgrey"
        fontcol = "black"
        hlinecol = "white"
        hlinelw = 0
        fontweight = None

        # create plotting data
        data = df.query("group==@g")[value_col].to_numpy()
        freq = (len(data) / n_obs) * 100

        # train kernel density estimator and evaluate on grid
        kde.fit(data[:, None])
        x = np.linspace(xlim[0], xlim[1], 500)
        log_prob = kde.score_samples(x[:, None])
        density = np.exp(log_prob)

        # creating new axes object for each group
        ax = fig.add_subplot(gs[i : i + 1, 0:])  # noqa: E203

        # actual plotting
        ax.fill_between(x, density, 0, edgecolor=edgecol, facecolor=facecol, lw=1)
        ax.axhline(y=0, lw=hlinelw, clip_on=False, color=hlinecol)

        # set xlim
        ax.set_xlim(*xlim)

        # make background transparent
        rect = ax.patch
        rect.set_alpha(0)

        # remove axis ticks, labels and borders
        ax.set_yticklabels([])
        ax.set_yticks([])

        if i != n_groups - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_value))

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax.spines[s].set_visible(False)

        # add group text, percentages (this need changing depending on the application)
        fontweight = fontweight
        ax.text(-0.1, 1.5, g, fontweight=fontweight, ha="right", color=fontcol)

        if i == n_groups - 1:
            ax.set_xlabel("Predicted increase in years of education", labelpad=10)
            ax.set_ylabel("Period", labelpad=50, y=25)

    gs.update(hspace=hspace)

    return fig
