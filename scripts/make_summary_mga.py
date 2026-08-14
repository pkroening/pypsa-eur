# SPDX-FileCopyrightText: : Peter Kröning and Contributors to <https://github.com/koen-vg/eu-hydrogen>
#
# SPDX-License-Identifier: MIT
"""
Create summary CSV files for all scenario runs including costs, capacities,
capacity factors, curtailment, energy balances, prices and other metrics.
"""

import logging

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

logger = logging.getLogger(__name__)
opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}


def make_summaries(networks_dict: dict) -> dict[str, pd.DataFrame]:
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=[
            "cluster",
            "opt",
            "sector_opt",
            "planning_horizon",
            "alternative_objectives",
            "slack",
        ],
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
        (
            cluster,
            opt,
            sector_opt,
            planning_horizon,
            alternative_objectives,
            slack,
        ): "results/"
        + snakemake.params.RDIR
        + f"/networks/base_s_{cluster}_{opt}_{sector_opt}_{planning_horizon}_{alternative_objectives}_{slack}.nc"
        for cluster in snakemake.params.scenario["clusters"]
        for opt in snakemake.params.scenario["opts"]
        for sector_opt in snakemake.params.scenario["sector_opts"]
        for planning_horizon in snakemake.params.scenario["planning_horizons"]
        for alternative_objectives in snakemake.params.mga["alternative_objectives"]
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
