import logging

import matplotlib.pyplot as plt
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
plt.style.use("bmh")


def plot_capacities(smk, n_header):
    # Get region
    region = tuple(smk.params.mga.get("region", None))
    region_str = ""
    for reg in region:
        if region_str:
            region_str += f",{reg}"
        else:
            region_str = reg

    # Load data
    df = pd.read_csv(
        smk.input.nodal_capacities, index_col=list(range(3)), header=list(range(n_header))
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
                        y_max = 0

                        for alt_obj in enumerate(objectives):

                            for slack in enumerate(slacks):
                                default = bool(not alt_obj[1] and not slack[1])
                                mga = bool(alt_obj[1] and slack[1])

                                if default or mga:
                                    x = planing_horizons
                                    y = df.loc[
                                        idx[component, :, carrier],
                                        idx[cluster, opt, sector_opts, x, alt_obj[1], slack[1]]
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
                        fig.supylabel("Capacities")
                        for ax in axes.flatten():
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                            ax.set_ylim(0, y_max*1.1)
                        fig.savefig(f"results/test-sector-myopic-mga/pathways/capacities/regional/{cluster}_{opt}_{sector_opt}-{component}_{carrier}_{region_str}.svg")
                        plt.close()
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

    n_header = 6

    plot_capacities(snakemake, n_header)

