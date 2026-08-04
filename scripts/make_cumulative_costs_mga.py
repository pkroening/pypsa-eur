# SPDX-FileCopyrightText: Peter Kröning
#
# SPDX-License-Identifier: MIT
"""
Calculate cumulative costs for near optimal myopic foresight scenarios.
"""

import logging

import numpy as np
import pandas as pd

try:
    from numpy import trapezoid
except ImportError:
    # before numpy 2.0
    from numpy import trapz as trapezoid

from scripts._helpers import configure_logging, set_scenario_config

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


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
                for sector_opts in cumulative_cost.index.get_level_values(
                    level=2
                ).unique():
                    for alt_obj in cumulative_cost.index.get_level_values(
                        level=4
                    ).unique():
                        for slack in cumulative_cost.index.get_level_values(
                            level=5
                        ).unique():
                            # Not all cases
                            default = bool(not alt_obj and not slack)
                            mga = bool(alt_obj and slack)
                            if mga or default:
                                cumulative_cost.loc[
                                    (
                                        cluster,
                                        opts,
                                        sector_opts,
                                        "cumulative cost",
                                        alt_obj,
                                        slack,
                                    ),
                                    r,
                                ] = trapezoid(
                                    x=planning_horizons,
                                    y=cumulative_cost.loc[
                                        idx[
                                            cluster,
                                            opts,
                                            sector_opts,
                                            planning_horizons,
                                            alt_obj,
                                            slack,
                                        ],
                                        r,
                                    ].values,
                                )

    return cumulative_cost


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_cumulative_costs_mga",
            configfiles="config/test/config.myopic-mga.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    costs = pd.read_csv(
        snakemake.input.costs, index_col=[0, 1, 2], header=list(range(6))
    )

    # clean multiindex: handle empty scenario levels & make planning horizons numeric
    names = costs.columns.names
    costs.columns = pd.MultiIndex.from_tuples(
        [
            tuple(
                pd.to_numeric(value)
                if name == "planning_horizon"
                else ("" if str(value).startswith("Unnamed:") else value)
                for name, value in zip(names, column)
            )
            for column in costs.columns
        ],
        names=names,
    )

    planning_horizons = snakemake.params.scenario["planning_horizons"]

    cumulative_cost = calculate_cumulative_cost(costs, planning_horizons)
    cumulative_cost.to_csv(snakemake.output[0])
