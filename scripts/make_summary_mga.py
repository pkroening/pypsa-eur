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
from _helpers import configure_logging, load_costs, set_scenario_config
from make_summary import (
    assign_carriers,
    assign_locations,
    # accessed by globals()["calculate_" + output]
    calculate_capacities,
    calculate_capacity_factors,
    calculate_costs,
    calculate_curtailment,
    calculate_energy,
    calculate_energy_balance,
    calculate_market_values,
    calculate_metrics,
    calculate_nodal_capacities,
    calculate_nodal_capacity_factors,
    calculate_nodal_costs,
    calculate_nodal_energy_balance,
    calculate_prices,
    calculate_weighted_prices,
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
            costs.sum()[index] / ((1 + r) ** (index[-1] - planning_horizons[0]))
            for index in cumulative_cost.index
        ]

    # integrate cost throughout the transition path
    for r in cumulative_cost.columns:
        for cluster in cumulative_cost.index.get_level_values(level=0).unique():
            for sector_opts in cumulative_cost.index.get_level_values(
                level=1
            ).unique():
                for sense in cumulative_cost.index.get_level_values(
                    level=3
                ).unique():
                    for slack in cumulative_cost.index.get_level_values(
                        level=4
                    ).unique():
                        cumulative_cost.loc[
                            (
                                cluster,
                                sector_opts,
                                "cumulative cost",
                                sense,
                                slack,
                            ),
                            r,
                        ] = np.trapezoid(
                            cumulative_cost.loc[
                                idx[
                                    cluster,
                                    sector_opts,
                                    planning_horizons,
                                    sense,
                                    slack,
                                ],
                                r,
                            ].values,
                            x=planning_horizons,
                        )

    return cumulative_cost

def make_summaries(networks_dict):
    outputs = [
        "capacities",
        "capacity_factors",
        "costs",
        "curtailment",
        "energy",
        "market_values",
        "metrics",
        "nodal_capacities",
        "nodal_capacity_factors",
        "nodal_costs",
        "nodal_energy_balance",
        "prices",
        "energy_balance",
        "weighted_prices",
    ]

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "opt", "planning_horizon", "sense", "slack"],
    )

    df = {output: pd.DataFrame(columns=columns, dtype=float) for output in outputs}
    for label, filename in networks_dict.items():
        logger.info(f"Make summary for scenario {label}, using {filename}")

        n = pypsa.Network(filename)

        assign_carriers(n)
        assign_locations(n)

        for output in outputs:
            df[output][label] = globals()["calculate_" + output](n)

    return df


def to_csv(df):
    for key in df:
        df[key].to_csv(snakemake.output[key])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("make_summary")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    networks_dict = {
        (cluster, opt + sector_opt, planning_horizon, sense, slack): "results/"
        + snakemake.params.RDIR
        + f"/networks/base_s_{cluster}_{opt}_{sector_opt}_{planning_horizon}_{sense}{slack}.nc"
        for cluster in snakemake.params.scenario["clusters"]
        for opt in snakemake.params.scenario["opts"]
        for sector_opt in snakemake.params.scenario["sector_opts"]
        for planning_horizon in snakemake.params.scenario["planning_horizons"]
        for sense in ["min", "max"]
        for slack in snakemake.params.scenario["slack"]
    }

    costs_db = load_costs(snakemake.input.costs)

    df = make_summaries(networks_dict)

    df["metrics"].loc["total costs"] = df["costs"].sum()

    to_csv(df)

    if snakemake.params.foresight == "myopic":
        cumulative_cost = calculate_cumulative_cost(df["costs"], snakemake.params.scenario["planning_horizons"])
        cumulative_cost.to_csv(
            "results/" + snakemake.params.RDIR + "csvs/cumulative_cost.csv"
        )
