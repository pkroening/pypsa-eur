# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Create summary CSV files for all scenario runs including costs, capacities,
capacity factors, curtailment, energy balances, prices and other metrics.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, set_scenario_config
from make_summary import (
    OUTPUTS,
    assign_carriers,
    assign_locations,
    # accessed by globals()["calculate_" + output]
    calculate_capacities,  # noqa: F401
    calculate_capacity_factors,  # noqa: F401
    calculate_costs,  # noqa: F401
    calculate_curtailment,  # noqa: F401
    calculate_energy,  # noqa: F401
    calculate_energy_balance,  # noqa: F401
    calculate_market_values,  # noqa: F401
    calculate_metrics,  # noqa: F401
    calculate_nodal_capacities,  # noqa: F401
    calculate_nodal_capacity_factors,  # noqa: F401
    calculate_nodal_costs,  # noqa: F401
    calculate_nodal_energy_balance,  # noqa: F401
    calculate_prices,  # noqa: F401
    calculate_weighted_prices,  # noqa: F401
)

idx = pd.IndexSlice
logger = logging.getLogger(__name__)
opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}


def calculate_cumulative_cost(costs, planning_horizons):
    cumulative_cost = pd.DataFrame(
        index=costs.sum().index,
        columns=pd.Series(data=np.arange(0, 0.1, 0.01), name="social discount rate"),
    )

    # discount cost and express them in money value of planning_horizons[0]
    for r in cumulative_cost.columns:
        cumulative_cost[r] = [
            costs.sum()[index] / ((1 + r) ** (index[3] - planning_horizons[0]))
            for index in cumulative_cost.index
        ]

    # integrate cost throughout the transition path
    for r in cumulative_cost.columns:
        for cluster in cumulative_cost.index.get_level_values(level=0).unique():
            for opts in cumulative_cost.index.get_level_values(level=1).unique():
                for sector_opts in cumulative_cost.index.get_level_values(level=2).unique():
                    for alt_obj in cumulative_cost.index.get_level_values(level=4).unique():
                        for slack in cumulative_cost.index.get_level_values(level=5).unique():
                            # Not all cases
                            default = bool(not alt_obj and not slack)
                            mga = bool(alt_obj and slack)
                            if mga or default:
                                cumulative_cost.loc[(cluster, opts, sector_opts, "cumulative cost", alt_obj, slack), r] = np.trapezoid(
                                    x=planning_horizons,
                                    y=cumulative_cost.loc[idx[cluster, opts, sector_opts, planning_horizons, alt_obj, slack], r].values,
                                )

    return cumulative_cost

def make_summaries(networks_dict: dict) -> dict[str, pd.DataFrame]:
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "opt", "sector_opt", "planning_horizon", "alternative_objectives", "slack"],
    )

    df_dict = {output: pd.DataFrame(columns=columns, dtype=float) for output in OUTPUTS}
    for label, filename in networks_dict.items():
        logger.info(f"Make summary for scenario {label}, using {filename}")

        n = pypsa.Network(filename)

        assign_carriers(n)
        assign_locations(n)

        for output in OUTPUTS:
            df_dict[output][label] = globals()["calculate_" + output](n)

    return df_dict


def to_csv(df_dict: dict[str, pd.DataFrame]):
    for key, df in df_dict.items():
        df.to_csv(snakemake.output[key])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_summary_mga",
            configfiles="config/test/config.myopic-mga.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    pypsa.set_option("params.statistics.nice_names", False)
    pypsa.set_option("params.statistics.drop_zero", False)

    # Networks MGA
    networks_dict = {
        (cluster, opt, sector_opt, planning_horizon, alternative_objectives, slack): "results/"
        + snakemake.params.RDIR
        + f"/networks/base_s_{cluster}_{opt}_{sector_opt}_{planning_horizon}_{alternative_objectives}_{slack}.nc"
        for cluster in snakemake.params.scenario["clusters"]
        for opt in snakemake.params.scenario["opts"]
        for sector_opt in snakemake.params.scenario["sector_opts"]
        for planning_horizon in snakemake.params.scenario["planning_horizons"]
        for alternative_objectives in snakemake.params.mga["alternative_objectives"].keys()
        for slack in snakemake.params.mga["slack"]
    }
    # Networks default
    networks_dict.update(
        {
            (cluster, opt, sector_opt, planning_horizon, "", ""): "results/"
            + snakemake.params.RDIR
            + f"/networks/base_s_{cluster}_{opt}_{sector_opt}_{planning_horizon}.nc"
            for cluster in snakemake.params.scenario["clusters"]
            for opt in snakemake.params.scenario["opts"]
            for sector_opt in snakemake.params.scenario["sector_opts"]
            for planning_horizon in snakemake.params.scenario["planning_horizons"]
        }
    )


    df_dict = make_summaries(networks_dict)

    df_dict["metrics"].loc["total costs"] = df_dict["costs"].sum()

    to_csv(df_dict)

    if snakemake.params.foresight == "myopic":
        cumulative_cost = calculate_cumulative_cost(df_dict["costs"], snakemake.params.scenario["planning_horizons"])
        cumulative_cost.to_csv(
            "results/" + snakemake.params.RDIR + "csvs/cumulative_cost.csv"
        )
