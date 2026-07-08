import logging

import matplotlib.pyplot as plt
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
plt.style.use("bmh")


def plot_capacities(smk, n_header):
    df = pd.read_csv(
        smk.input.nodal_capacities, index_col=list(range(3)), header=list(range(n_header))
    )

    # columns = df.columns
    # for col in df.columns:
    #     for c in col:
    #         if c.beginswith("Unnamed"):
    #             c = ""
    for i, columns in enumerate(df.columns.levels):
        columns_new = columns.tolist()
        for j, row in enumerate(columns_new):
            if "Unnamed" in row:
                columns_new[j] = ""
        df = df.rename(columns=dict(zip(columns.tolist(), columns_new)), level=i)

    clusters = df.columns.get_level_values(level=0).unique()
    opts = df.columns.get_level_values(level=1).unique()
    sector_opts = df.columns.get_level_values(level=2).unique()
    planing_horizons = df.columns.get_level_values(level=3).unique()
    objectives = df.columns.get_level_values(level=4).unique()
    slacks = df.columns.get_level_values(level=5).unique()

    region = tuple(smk.params.mga.get("region", None))
    region_str = ""
    for reg in region:
        if region_str:
            region_str += f",{reg}"
        else:
            region_str = reg

    idx = pd.IndexSlice
    done = []
    for component, location, carrier in df.index:
        if (component, carrier) not in done and location.startswith(region):
            for cluster in clusters:
                for opt in opts:
                    for sector_opt in sector_opts:

                        fig, axes = plt.subplots(len(objectives)-1, len(slacks)-1, figsize=(12, 8), sharex=True, sharey=True)

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

                                    if default:
                                        for ax in axes.flatten():
                                            ax.plot(x, y, marker="x", color="black", label="cost optimimal")
                                    elif mga:
                                        axes[-1, slack[0]].set_xlabel(slack[1])
                                        axes[alt_obj[0], 0].set_ylabel(alt_obj[1])
                                        for ax in axes[alt_obj[0],:]:
                                            ax.plot(x, y, marker="x", label=f"{alt_obj[1]} - {slack[1]}")
                                        for ax in axes[:,slack[0]]:
                                            ax.plot(x, y, marker="x", label=f"{alt_obj[1]} - {slack[1]}")

                        fig.suptitle(region_str)
                        # fig.supxlabel("Time")
                        fig.supylabel("Capacities")
                        for ax in axes.flatten():
                            ax.legend()
                        fig.savefig(f"results/test-sector-myopic-mga/pathways/capacities/nodal/{cluster}_{opt}_{sector_opt}-{component}_{carrier}_{region_str}.svg")
                        plt.close()
                        done.append((component, carrier))




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

