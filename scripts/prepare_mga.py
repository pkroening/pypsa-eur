# SPDX-FileCopyrightText: Peter Kröning and Contributors to <https://github.com/koen-vg/eu-hydrogen>
#
# SPDX-License-Identifier: MIT
import logging

import pandas as pd
import pypsa
from linopy import merge
from pypsa.descriptors import nominal_attrs

from scripts.prepare_mga_regional import (
    get_cross_border_components,
    split_df_by_region,
    split_expression_by_region,
)

logger = logging.getLogger(__name__)
pypsa.network.power_flow.logger.setLevel(logging.WARNING)


def parse_optimization_sense(
    alternative_objective: str,
) -> int:
    """
    Parse the optimization sense to -1 or +1

    Parameters
    ----------
    sense: str | int
        Optimization sense of alternate objective function

    Returns
    -------
    sense : int
        Optimization sense of alternate objective function
    """
    sense = alternative_objective.split(sep="-")[0]
    if (isinstance(sense, str) and sense.startswith("min")) or (
        isinstance(sense, int) and sense > 0
    ):
        sense = +1
    elif (isinstance(sense, str) and sense.startswith("max")) or (
        isinstance(sense, int) and sense < 0
    ):
        sense = -1
    else:
        raise ValueError(f"Could not parse optimization sense {sense}")

    return sense


def set_mga_objective(
    n: pypsa.Network,
    mga_config: dict,
    alternative_objective: str,
):
    """
    Set the new objective function for modelling to generate alternatives.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized

    mga_config : dict
        Configuration for mga study
    sense : str | int
        Optimization sense of alternate objective function
    """
    # Get linopy model
    m = n.model

    # Parse sense
    sense = parse_optimization_sense(alternative_objective)

    # Build objective function
    expr_config = mga_config["alternative_objectives"][alternative_objective]
    weights = {}

    if "import" in alternative_objective:
        region = mga_config.get("region", None)
        if not region:
            raise ValueError(
                "For optimization of cross border components a region has to be defined in the mga config."
            )
        cross_border_components = get_cross_border_components(region, n)

    else:
        cross_border_components = False

    static = expr_config["weights"].get("static", {})
    for component in static:
        vars = {}
        for var in static[component]:
            w = pd.Series(0, index=n.components[component].static.index)
            for carrier, const in static[component][var].items():
                mask = (
                    n.components[component].static.carrier == carrier
                ) & n.components[component].static.p_nom_extendable
                w.loc[mask] = const
            vars[var] = w
        weights[component] = vars
    varying = expr_config["weights"].get("varying", {})
    for component in varying:
        static = n.components[component].static
        vars = {}
        for var in varying[component]:
            w = pd.DataFrame(0, columns=static.index, index=n.snapshots)

            if cross_border_components:
                sign = cross_border_components[component]
                carriers = varying[component][var]
                if carriers:
                    carrier_weights = pd.Series(0, index=static.index)
                    for carrier, const in carriers.items():
                        carrier_weights[static["carrier"] == carrier] = const
                    sign = sign * carrier_weights
                w = w.add(sign, axis=1)
            else:
                for carrier, const in varying[component][var].items():
                    mask = (
                        static["carrier"] == carrier
                    )  # TODO: add regional for "tech"? -> probably does weird operation then
                    w.loc[:, mask] = const

            w = w.multiply(n.snapshot_weightings.objective, axis=0)
            vars[var] = w
        weights[component] = vars

    new_expr = []
    for component, attrs in weights.items():
        for attr, coeffs in attrs.items():
            if isinstance(coeffs, dict):
                coeffs = pd.Series(coeffs)
            if attr == nominal_attrs[component] and isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(n.get_extendable_i(component))
                coeffs.index.name = ""
            elif isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(index=n.components[component].static.index)
            elif isinstance(coeffs, pd.DataFrame):
                coeffs = coeffs.reindex(
                    columns=n.components[component].static.index, index=n.snapshots
                )
            new_expr.append(m[f"{component}-{attr}"] * coeffs * sense)

    if "import" in alternative_objective:
        flows = merge(new_expr)
        imports = m.add_variables(name="imports", coords=flows.coords, lower=0)

        name = "imports_sign"
        if name not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                name=name,
                type=name,
            )
        m.add_constraints(
            flows - imports <= 0,
            name=f"GlobalConstraint-{name}",
        )
        new_obj = imports.sum()

    else:
        new_obj = merge(new_expr)

    m.objective = new_obj

    # Save meta data
    n.meta["sense"] = sense


def calculate_slack_myopic(
    slack_nom: float,
    slack_initial_fraction: float,
    current_horizon: int,
    planning_horizons: list[int],
) -> float:
    """
    Calculate slack for modelling to generate alternatives objective constraint

    We gradually increase slack from an initial fraction of the nominal slack at the first planning horizon linearly to the full slack at the last planning horizon.
    This is helpful to avoid infeasible optimization problems and/or bad system designs, where the model would lean heavily in one technology in early optimization horizons.

    Parameters
    ----------
    slack_nom : float
        The nomial slack of the objective
    slack_initial_fraction : float
        The fraction by which is the nominal slack reduced in the first horizon
    current_horizon : int
        The current planning horizon year
    planning_horizons : list[int]
        All planing horizons for myopic foresight

    Returns
    -------
    float
        slack for the current horizon
    """
    planning_horizon_frac = (current_horizon - min(planning_horizons)) / (
        max(planning_horizons) - min(planning_horizons)
    )
    slack = slack_nom * (
        slack_initial_fraction + (1 - slack_initial_fraction) * planning_horizon_frac
    )
    logger.info(f"Slack for horizon {current_horizon}: {slack}")

    return slack


def set_mga_constraint(
    n: pypsa.Network,
    snakemake,
):
    """
    Set constraint for former objective
    """
    # Get linopy model and former objective function
    m = n.model
    obj_func = m.objective.expression
    obj_func_has_constant = obj_func.has_constant

    # Slack and bound
    slack = float(snakemake.wildcards.slack)
    if snakemake.params.foresight == "myopic":
        current_horizon = int(snakemake.wildcards.planning_horizons)
        planning_horizons = snakemake.params.planning_horizons
        slack_initial_fraction = snakemake.params.mga.get("slack_initial_fraction", 1.0)
        slack = calculate_slack_myopic(
            slack, slack_initial_fraction, current_horizon, planning_horizons
        )

    def calc_bound(
        capex: pd.DataFrame, capex_installed: pd.DataFrame, opex: pd.DataFrame
    ):
        obj_opt = capex.sum() + opex.sum()
        slack_abs = obj_opt * slack
        obj_bound = obj_opt + slack_abs
        if not obj_func_has_constant:
            # since there are no constant terms in the original objective function
            obj_bound = obj_bound - capex_installed.sum()
        return obj_bound

    # Get cost-optimal network and values
    n_opt = pypsa.Network(snakemake.input.network_opt)
    capex = n_opt.statistics.capex(groupby="country", groupby_method="sum")
    capex_installed = n_opt.statistics.installed_capex(
        groupby="country", groupby_method="sum"
    )
    opex = n_opt.statistics.opex(groupby="country", groupby_method="sum")
    del n_opt

    # Check whether to split by region
    region = snakemake.params.mga.get("region", None)
    if region:
        capex_in, capex_out = split_df_by_region(capex, region)
        capex_installed_in, capex_installed_out = split_df_by_region(
            capex_installed, region
        )
        opex_in, opex_out = split_df_by_region(opex, region)

        obj_bound_in_region = calc_bound(capex_in, capex_installed_in, opex_in)
        obj_bound_out_region = calc_bound(capex_out, capex_installed_out, opex_out)

        n.meta["obj_bound"] = (obj_bound_in_region, obj_bound_out_region)  # meta data

        obj_func_in, obj_func_out = split_expression_by_region(n, obj_func, region)

        for c_name, expr, bound in [
            ("near_opt_bound_in_region", obj_func_in, obj_bound_in_region),
            ("near_opt_bound_out_region", obj_func_out, obj_bound_out_region),
        ]:
            if c_name not in n.global_constraints.index:
                n.add(
                    "GlobalConstraint",
                    name=c_name,
                    type=c_name,
                    sense="<=",
                    constant=bound,
                )
            m.add_constraints(expr <= bound, name=f"GlobalConstraint-{c_name}")

    else:
        obj_bound = calc_bound(capex, capex_installed, opex)
        n.meta["obj_bound"] = obj_bound  # meta data

        # Add globalconstraint object so dual variable can be registered (if it doesn't already exist)
        c_name = "near_opt_bound"
        if c_name not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                name=c_name,
                type=c_name,
                sense="<=",
                constant=obj_bound,
            )
        m.add_constraints(
            obj_func <= obj_bound,
            name=f"GlobalConstraint-{c_name}",
        )


def prepare_mga(
    n: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    snakemake,
):
    """
    Prepare the network for modelling to generate alternatives by restricting costs and introducing a new objective method.

    Parameters
    ----------
    n: pypsa.Network
        Network to be optimized
    snapshots : pd.DatetimeIndex
        Unused. Kept for needed form of custom_extra_functionality
        The snapshots of the network
    snakemake
    """
    mga_config = snakemake.params.mga
    alternative_objective = snakemake.wildcards.alternative_objectives

    set_mga_constraint(n, snakemake)
    set_mga_objective(n, mga_config, alternative_objective)
