import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
plt.style.use("bmh")


def plot_capacities(file_path, n_header, region: tuple, save_path):
    # Load data
    df = pd.read_csv(
        file_path, index_col=list(range(3)), header=list(range(n_header))
    )

    # Handle empty columns values from cost optimal solution
    for i, columns in enumerate(df.columns.levels):
        columns_new = columns.tolist()
        for j, row in enumerate(columns_new):
            if "Unnamed" in row:
                columns_new[j] = ""
        df = df.rename(columns=dict(zip(columns.tolist(), columns_new)), level=i)

    # Get multiindex values
    clusters = df.columns.get_level_values(level=0).unique()
    opts = df.columns.get_level_values(level=1).unique()
    sector_opts = df.columns.get_level_values(level=2).unique()
    planing_horizons = df.columns.get_level_values(level=3).unique()
    objectives = df.columns.get_level_values(level=4).unique()
    slacks = df.columns.get_level_values(level=5).unique()

    # PLotting strings
    region_str = ""
    for reg in region:
        if region_str:
            region_str += f",{reg}"
        else:
            region_str = reg
    property = file_path.split("nodal_")[1].removesuffix(".csv")

    # Iterate
    idx = pd.IndexSlice
    plotted = []
    for component, location, carrier in df.index:
        if (component, carrier) not in plotted and location.startswith(region):
            for cluster in clusters:
                for opt in opts:
                    for sector_opt in sector_opts:

                        # Plot
                        fig, axes = plt.subplots(len(slacks)-1, figsize=(10, 5), sharex=True)
                        axes = np.atleast_1d(axes)

                        y_max = 0
                        for alt_obj in enumerate(objectives):

                            for slack in enumerate(slacks):
                                default = bool(not alt_obj[1] and not slack[1])
                                mga = bool(alt_obj[1] and slack[1])

                                if default or mga:
                                    x = planing_horizons
                                    y = df.loc[
                                        idx[component, :, carrier],
                                        idx[cluster, opt, sector_opt, x, alt_obj[1], slack[1]]
                                    ]
                                    y = y[y.index.get_level_values(level="location").str.startswith(region)].sum(axis="index")
                                    y_max = max(y_max, y.max())

                                    if default:
                                        for ax in axes.flatten():
                                            ax.plot(x, y, marker="x", color="black", label="cost optimimal")
                                    elif mga:
                                        axes[slack[0]].set_ylabel(f"s={float(slack[1]):.0%}")
                                        axes[slack[0]].plot(x, y, marker="x", label=alt_obj[1])

                        fig.suptitle(region_str)
                        fig.supxlabel("Time")
                        fig.supylabel(property)
                        for ax in axes.flatten():
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                            ax.set_ylim(0, y_max*1.1)
                        path = f"{save_path}pathways/{property}"
                        os.makedirs(path, exist_ok=True)
                        fig.savefig(f"{path}/{cluster}_{opt}_{sector_opt}-{component}_{carrier}_{region_str}.svg")
                        plt.close("all")
                        plotted.append((component, carrier))


def plot_costs(file_path, n_header, region: tuple, save_path):
    # Load data
    df = pd.read_csv(
        file_path, index_col=list(range(4)), header=list(range(n_header))
    )

    # Handle empty columns values from cost optimal solution
    for i, columns in enumerate(df.columns.levels):
        columns_new = columns.tolist()
        for j, row in enumerate(columns_new):
            if "Unnamed" in row:
                columns_new[j] = ""
        df = df.rename(columns=dict(zip(columns.tolist(), columns_new)), level=i)

    # Get multiindex values
    clusters = df.columns.get_level_values(level=0).unique()
    opts = df.columns.get_level_values(level=1).unique()
    sector_opts = df.columns.get_level_values(level=2).unique()
    planing_horizons = df.columns.get_level_values(level=3).unique()
    objectives = df.columns.get_level_values(level=4).unique()
    slacks = df.columns.get_level_values(level=5).unique()

    # PLotting strings
    region_str = ""
    for reg in region:
        if region_str:
            region_str += f",{reg}"
        else:
            region_str = reg
    property = file_path.split("nodal_")[1].removesuffix(".csv")

    # Iterate
    idx = pd.IndexSlice
    plotted = []
    for cost_type, component, location, carrier in df.index:
        if location.startswith(region):
            if (component, carrier) not in plotted:
                for cluster in clusters:
                    for opt in opts:
                        for sector_opt in sector_opts:

                            # Plot
                            fig_cap, axes_cap = plt.subplots(len(slacks)-1, figsize=(10, 5), sharex=True)
                            fig_mar, axes_mar = plt.subplots(len(slacks)-1, figsize=(10, 5), sharex=True)
                            fig_tot, axes_tot = plt.subplots(len(slacks)-1, figsize=(10, 5), sharex=True)
                            axes_cap = np.atleast_1d(axes_cap)
                            axes_mar = np.atleast_1d(axes_mar)
                            axes_tot = np.atleast_1d(axes_tot)

                            y_cap_max = 0
                            y_mar_max = 0
                            y_tot_max = 0

                            for alt_obj in enumerate(objectives):

                                for slack in enumerate(slacks):
                                    default = bool(not alt_obj[1] and not slack[1])
                                    mga = bool(alt_obj[1] and slack[1])

                                    if default or mga:
                                        x = planing_horizons
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

                                        y_cap_max = max(y_cap_max, y_cap.max())
                                        y_mar_max = max(y_mar_max, y_mar.max())
                                        y_tot_max = max(y_tot_max, y_tot.max())

                                        if default:
                                            for axes, y in [(axes_cap, y_cap), (axes_mar, y_mar), (axes_tot, y_tot)]:
                                                for ax in axes.flatten():
                                                    ax.plot(x, y, marker="x", color="black", label="cost optimimal")
                                        elif mga:
                                            for axes, y in [(axes_cap, y_cap), (axes_mar, y_mar), (axes_tot, y_tot)]:
                                                axes[slack[0]].set_ylabel(f"s={float(slack[1]):.0%}")
                                                axes[slack[0]].plot(x, y, marker="x", label=alt_obj[1])

                            for fig, axes, y_max, n in [
                                (fig_cap, axes_cap, y_cap_max, "capital"),
                                (fig_mar, axes_mar, y_mar_max, "marginal"),
                                (fig_tot, axes_tot, y_tot_max, "total")
                            ]:
                                fig.suptitle(region_str)
                                fig.supxlabel("Time")
                                fig.supylabel(f"{n} {property}")

                                for ax in axes.flatten():
                                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                    if y_max > 0:
                                        ax.set_ylim(0, y_max*1.1)

                                path = f"{save_path}pathways/{property}/{n}"
                                os.makedirs(path, exist_ok=True)
                                fig.savefig(f"{path}/{cluster}_{opt}_{sector_opt}-{component}_{carrier}_{region_str}.svg")

                            plt.close("all")
                            plotted.append((component, carrier))




if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_pathways",
            configfiles="config/config.myopic-mga.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

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
