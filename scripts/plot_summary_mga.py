# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur> and <https://github.com/koen-vg/eu-hydrogen>, edited by Peter Kröning
#
# SPDX-License-Identifier: MIT
"""
Creates plots from summary CSV files.
"""

import logging

import matplotlib.pyplot as plt

from scripts._helpers import configure_logging, set_scenario_config
from scripts.plot_summary import (
    plot_balances,
    plot_carbon_budget_distribution,
    plot_costs,
    plot_energy,
)

logger = logging.getLogger(__name__)
plt.style.use("bmh")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_summary_mga",
            configfiles="config/test/config.myopic-mga.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n_header = 6

    plot_costs(snakemake, n_header)

    plot_energy(snakemake, n_header)

    plot_balances(snakemake, n_header)

    co2_budget = snakemake.params["co2_budget"]
    if (
        isinstance(co2_budget, str) and co2_budget.startswith("cb")
    ) or snakemake.params["foresight"] == "perfect":
        options = snakemake.params.sector
        plot_carbon_budget_distribution(snakemake.input.eurostat, options)
