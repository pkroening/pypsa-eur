import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
plt.style.use("bmh")


def _load_and_clean(file_path, n_header, n_index):
    df = pd.read_csv(
        file_path, index_col=list(range(n_index)), header=list(range(n_header))
    )

    # Handle empty column values from cost optimal solution
    for i, columns in enumerate(df.columns.levels):
        columns_new = ["" if "Unnamed" in c else c for c in columns.tolist()]
        df = df.rename(columns=dict(zip(columns.tolist(), columns_new)), level=i)

    return df


def _get_levels(df):
    return (
        df.columns.get_level_values(level=0).unique(),  # clusters
        df.columns.get_level_values(level=1).unique(),  # opts
        df.columns.get_level_values(level=2).unique(),  # sector_opts
        df.columns.get_level_values(level=3).unique(),  # planning_horizons
        df.columns.get_level_values(level=4).unique(),  # objectives
        df.columns.get_level_values(level=5).unique(),  # slacks
    )


def _unique_components(df, region):
    """Yield each (component, carrier) once, for rows located in `region`."""
    plotted = set()
    for row in df.index:
        component, location, carrier = row[-3:]
        if location.startswith(region) and (component, carrier) not in plotted:
            plotted.add((component, carrier))
            yield component, carrier


def _row_subplots(n_rows):
    fig, axes = plt.subplots(n_rows, figsize=(10, 5), sharex=True)
    return fig, np.atleast_1d(axes)


def _finalize(fig, axes, y_max, title, ylabel, save_dir, filename):
    fig.suptitle(title)
    fig.supxlabel("Time")
    fig.supylabel(ylabel)
    for ax in axes.flatten():
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.1)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{filename}")
    plt.close(fig)


def plot_capacities(file_path, n_header, region: tuple, save_path):
    df = _load_and_clean(file_path, n_header, n_index=3)

    clusters, opts, sector_opts, planning_horizons, objectives, slacks = _get_levels(df)

    # PLotting strings
    region_str = ",".join(region)
    prop = file_path.split("nodal_")[1].removesuffix(".csv")

    # Iterate
    idx = pd.IndexSlice
    for component, carrier in _unique_components(df, region):
        for cluster in clusters:
            for opt in opts:
                for sector_opt in sector_opts:

                    # One figure to compare slacks/objective function in each subfigure
                    # -1 due to the default being plottet everywhere
                    fig_sla, axes_sla = _row_subplots(len(slacks) - 1)
                    fig_obj, axes_obj = _row_subplots(len(objectives) - 1)

                    # Initialize max val for scaling
                    y_max = 0

                    for alt_obj in enumerate(objectives):

                        for slack in enumerate(slacks):

                            # Get optimization type
                            default = bool(not alt_obj[1] and not slack[1])
                            mga = bool(alt_obj[1] and slack[1])

                            if default or mga:

                                # Get x
                                x = planning_horizons

                                # Get y
                                y = df.loc[
                                    idx[component, :, carrier],
                                    idx[cluster, opt, sector_opt, x, alt_obj[1], slack[1]]
                                ]
                                y = y[y.index.get_level_values(level="location").str.startswith(region)].sum(axis="index")

                                # Update y_max
                                y_max = max(y_max, y.max())

                                # Plot
                                if default:
                                    for ax in axes_sla.flatten():
                                        ax.plot(x, y, marker="x", color="black", label="cost optimal")
                                    for ax in axes_obj.flatten():
                                        ax.plot(x, y, marker="x", color="black", label="cost optimal")
                                elif mga:
                                    axes_sla[slack[0]].set_ylabel(f"s={float(slack[1]):.0%}")
                                    axes_sla[slack[0]].plot(x, y, marker="x", label=alt_obj[1])

                                    axes_obj[alt_obj[0]].set_ylabel(alt_obj[1])
                                    axes_obj[alt_obj[0]].plot(x, y, marker="x", label=slack[1])

                    # Save file
                    filename = f"{cluster}_{opt}_{sector_opt}-{component}_{carrier}_{region_str}.svg"
                    for subdir, view_fig, view_axes in [
                        ("comp_obj", fig_sla, axes_sla),
                        ("comp_slack", fig_obj, axes_obj)
                    ]:
                        path = f"{save_path}pathways/{prop}/{subdir}"
                        _finalize(view_fig, view_axes, y_max, region_str, prop, path, filename)


def plot_costs(file_path, n_header, region: tuple, save_path):
    df = _load_and_clean(file_path, n_header, n_index=4)

    clusters, opts, sector_opts, planning_horizons, objectives, slacks = _get_levels(df)

    # PLotting strings
    region_str = ",".join(region)
    prop = file_path.split("nodal_")[1].removesuffix(".csv")

    # Iterate
    idx = pd.IndexSlice
    for component, carrier in _unique_components(df, region):
        for cluster in clusters:
            for opt in opts:
                for sector_opt in sector_opts:

                    # One figure to compare slacks/objective function in each subfigure
                    # -1 due to the default being plottet everywhere
                    fig_cap_sla, axes_cap_sla = _row_subplots(len(slacks) - 1)
                    fig_mar_sla, axes_mar_sla = _row_subplots(len(slacks) - 1)
                    fig_tot_sla, axes_tot_sla = _row_subplots(len(slacks) - 1)

                    fig_cap_obj, axes_cap_obj = _row_subplots(len(objectives) - 1)
                    fig_mar_obj, axes_mar_obj = _row_subplots(len(objectives) - 1)
                    fig_tot_obj, axes_tot_obj = _row_subplots(len(objectives) - 1)

                    # Initialize max val for scaling
                    y_cap_max = 0
                    y_mar_max = 0
                    y_tot_max = 0

                    for alt_obj in enumerate(objectives):

                        for slack in enumerate(slacks):
                            default = bool(not alt_obj[1] and not slack[1])
                            mga = bool(alt_obj[1] and slack[1])

                            if default or mga:
                                # Get x
                                x = planning_horizons

                                # Get y (try-except because some components might not have these costs)
                                try:
                                    y_cap = df.loc[
                                        idx["capital", component, :, carrier],
                                        idx[cluster, opt, sector_opt, x, alt_obj[1], slack[1]]
                                    ]
                                    y_cap = y_cap[y_cap.index.get_level_values(level="location").str.startswith(region)].sum(axis="index")
                                except KeyError:
                                    y_cap = np.zeros_like(x)

                                try:
                                    y_mar = df.loc[
                                        idx["marginal", component, :, carrier],
                                        idx[cluster, opt, sector_opt, x, alt_obj[1], slack[1]]
                                    ]
                                    y_mar = y_mar[y_mar.index.get_level_values(level="location").str.startswith(region)].sum(axis="index")
                                except KeyError:
                                    y_mar = np.zeros_like(x)
                                y_tot = y_cap + y_mar

                                # Update y_max
                                y_cap_max = max(y_cap_max, y_cap.max())
                                y_mar_max = max(y_mar_max, y_mar.max())
                                y_tot_max = max(y_tot_max, y_tot.max())

                                if default:
                                    for axes, y in [(axes_cap_sla, y_cap), (axes_mar_sla, y_mar), (axes_tot_sla, y_tot)]:
                                        for ax in axes.flatten():
                                            ax.plot(x, y, marker="x", color="black", label="cost optimal")
                                    for axes, y in [(axes_cap_obj, y_cap), (axes_mar_obj, y_mar), (axes_tot_obj, y_tot)]:
                                        for ax in axes.flatten():
                                            ax.plot(x, y, marker="x", color="black", label="cost optimal")
                                elif mga:
                                    for axes, y in [(axes_cap_sla, y_cap), (axes_mar_sla, y_mar), (axes_tot_sla, y_tot)]:
                                        axes[slack[0]].set_ylabel(f"s={float(slack[1]):.0%}")
                                        axes[slack[0]].plot(x, y, marker="x", label=alt_obj[1])
                                    for axes, y in [(axes_cap_obj, y_cap), (axes_mar_obj, y_mar), (axes_tot_obj, y_tot)]:
                                        axes[alt_obj[0]].set_ylabel(alt_obj[1])
                                        axes[alt_obj[0]].plot(x, y, marker="x", label=f"s={float(slack[1]):.0%}")

                    filename = f"{cluster}_{opt}_{sector_opt}-{component}_{carrier}_{region_str}.svg"
                    for subdir, fig, axes, y_max, n in [
                        ("comp_obj", fig_cap_sla, axes_cap_sla, y_cap_max, "capital"),
                        ("comp_obj", fig_mar_sla, axes_mar_sla, y_mar_max, "marginal"),
                        ("comp_obj", fig_tot_sla, axes_tot_sla, y_tot_max, "total"),
                        ("comp_slack", fig_cap_obj, axes_cap_obj, y_cap_max, "capital"),
                        ("comp_slack", fig_mar_obj, axes_mar_obj, y_mar_max, "marginal"),
                        ("comp_slack", fig_tot_obj, axes_tot_obj, y_tot_max, "total"),
                    ]:
                        path = f"{save_path}pathways/{prop}/{n}/{subdir}"
                        _finalize(fig, axes, y_max, region_str, f"{n} {prop}", path, filename)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_pathways",
            configfiles="config/test/config.myopic-mga.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Currently, breaks for default (intendet)
    region = tuple(snakemake.params.mga.get("region", None))

    n_header = 6

    save_path = snakemake.params.save_path


    plot_capacities(
        file_path=snakemake.input.nodal_capacities,
        n_header=n_header,
        region=region,
        save_path=save_path,
    )

    plot_costs(
        file_path=snakemake.input.nodal_costs,
        n_header=n_header,
        region=region,
        save_path=save_path,
    )
