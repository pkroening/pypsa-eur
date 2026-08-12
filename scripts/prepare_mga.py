# SPDX-FileCopyrightText: Peter Kröning and Contributors to <https://github.com/koen-vg/eu-hydrogen>
#
# SPDX-License-Identifier: MIT
import logging

import pandas as pd
import pypsa
from linopy import LinearExpression, merge
from pypsa.descriptors import nominal_attrs
from pypsa.statistics import groupers

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


def build_cost_expression(
    n: pypsa.Network,
    cost_type: str,
) -> LinearExpression:
    """
    Build the capital or operational expenditures of the network as a linear expression.

    The counterpart of `n.statistics.capex`/`n.statistics.opex`, grouped by country in
    the same way. Capex of non-extendable components only enters as a constant term and
    is dropped, see `non_extendable_capex`.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    cost_type : str
        Either "capex" or "opex"

    Returns
    -------
    LinearExpression
        Expenditures grouped by component and country
    """
    return getattr(n.optimize.expressions, cost_type)(groupby="country").reset_const()


def non_extendable_capex(n: pypsa.Network) -> pd.Series:
    """
    Calculate the capital expenditures of non-extendable components, grouped by country.

    This is the share of `n.statistics.capex` that is constant during optimization and
    therefore not part of `build_cost_expression`.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized

    Returns
    -------
    pd.Series
        Capital expenditures per country
    """
    capex = []
    for component, attr in nominal_attrs.items():
        static = n.components[component].static
        if static.empty:
            continue
        fixed = static[~static[f"{attr}_extendable"]]
        port = "" if "bus" in static.columns else "0"
        country = groupers.country(n, component, port=port)[fixed.index]
        capex.append((fixed[attr] * fixed["capital_cost"]).groupby(country).sum())

    return pd.concat(capex)


def build_cost_objective(
    n: pypsa.Network,
    mga_config: dict,
    cost_type: str,
    sense: int,
) -> LinearExpression:
    """
    Build an objective function for capital or operational expenditures.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    mga_config : dict
        Configuration for mga study
    cost_type : str
        Either "capex" or "opex"
    sense : int
        Optimization sense of alternate objective function

    Returns
    -------
    LinearExpression
        Alternate objective function
    """
    expr = build_cost_expression(n, cost_type)

    region = mga_config.get("region", None)
    if region:
        new_obj, _ = split_expression_by_region(expr, region)
    else:
        new_obj = expr.sum()

    return new_obj * sense


def build_weighted_objective(
    n: pypsa.Network,
    mga_config: dict,
    alternative_objective: str,
    sense: int,
) -> LinearExpression:
    """
    Build an objective function from the carrier weights given in the mga config.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    mga_config : dict
        Configuration for mga study
    alternative_objective : str
        Name of the alternate objective function
    sense : int
        Optimization sense of alternate objective function

    Returns
    -------
    LinearExpression
        Alternate objective function
    """
    # Get linopy model
    m = n.model

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

    static_weights = expr_config["weights"].get("static", {})
    for component, attrs in static_weights.items():
        static = n.components[component].static
        vars = {}
        for var, carriers in attrs.items():
            w = pd.Series(0.0, index=static.index)
            for carrier, const in carriers.items():
                w[(static["carrier"] == carrier) & static["p_nom_extendable"]] = const
            vars[var] = w
        weights[component] = vars

    varying_weights = expr_config["weights"].get("varying", {})
    for component, attrs in varying_weights.items():
        static = n.components[component].static
        vars = {}
        for var, carriers in attrs.items():
            if cross_border_components:
                w = cross_border_components[component].astype(float)
                if carriers:
                    carrier_weights = pd.Series(0.0, index=static.index)
                    for carrier, const in carriers.items():
                        carrier_weights[static["carrier"] == carrier] = const
                    w = w * carrier_weights
            else:
                w = pd.Series(0.0, index=static.index)
                for carrier, const in carriers.items():
                    # TODO: add regional for "tech"? -> probably does weird operation then
                    w[static["carrier"] == carrier] = const
            vars[var] = w
        weights[component] = vars

    new_expr = []
    for component, attrs in weights.items():
        for attr, coeffs in attrs.items():
            variable = m[f"{component}-{attr}"]
            expr = variable * coeffs.reindex(variable.indexes["name"], fill_value=0)
            if "snapshot" in variable.dims:
                expr = expr * n.snapshot_weightings.objective
            new_expr.append(expr * sense)

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

    return new_obj


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
    alternative_objective : str
        Name of the alternate objective function
    """
    # Parse sense and objective
    sense = parse_optimization_sense(alternative_objective)
    cost_type = alternative_objective.split(sep="-", maxsplit=1)[-1]

    # Build objective function
    if cost_type in ("capex", "opex"):
        new_obj = build_cost_objective(n, mga_config, cost_type, sense)
    else:
        new_obj = build_weighted_objective(n, mga_config, alternative_objective, sense)

    n.model.objective = new_obj

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
    # Get linopy model
    m = n.model

    # Slack
    slack = float(snakemake.wildcards.slack)
    if snakemake.params.foresight == "myopic":
        current_horizon = int(snakemake.wildcards.planning_horizons)
        planning_horizons = snakemake.params.planning_horizons
        slack_initial_fraction = snakemake.params.mga.get("slack_initial_fraction", 1.0)
        slack = calculate_slack_myopic(
            slack, slack_initial_fraction, current_horizon, planning_horizons
        )

    def calc_bound(capex: pd.Series, opex: pd.Series, capex_const: pd.Series) -> float:
        # The expressions cover extendable capacities only, so the constant capex of the
        # non-extendable components is subtracted from the bound instead
        return (1 + slack) * (capex.sum() + opex.sum()) - capex_const.sum()

    # Get cost-optimal network and values
    n_opt = pypsa.Network(snakemake.input.network_opt)
    capex = n_opt.statistics.capex(groupby="country", groupby_method="sum")
    opex = n_opt.statistics.opex(groupby="country", groupby_method="sum")
    del n_opt

    # Costs of the network to be optimized, grouped like the statistics above
    capex_expr = build_cost_expression(n, "capex")
    opex_expr = build_cost_expression(n, "opex")
    capex_const = non_extendable_capex(n)

    # Check whether to split by region
    region = snakemake.params.mga.get("region", None)
    if region:
        capex_in, capex_out = split_df_by_region(capex, region)
        opex_in, opex_out = split_df_by_region(opex, region)
        capex_const_in, capex_const_out = split_df_by_region(capex_const, region)
        capex_expr_in, capex_expr_out = split_expression_by_region(capex_expr, region)
        opex_expr_in, opex_expr_out = split_expression_by_region(opex_expr, region)

        constraints = {
            "near_opt_bound_in_region": (
                capex_expr_in + opex_expr_in,
                calc_bound(capex_in, opex_in, capex_const_in),
            ),
            "near_opt_bound_out_region": (
                capex_expr_out + opex_expr_out,
                calc_bound(capex_out, opex_out, capex_const_out),
            ),
        }
    else:
        constraints = {
            "near_opt_bound": (
                capex_expr.sum() + opex_expr.sum(),
                calc_bound(capex, opex, capex_const),
            )
        }

    for c_name, (expr, obj_bound) in constraints.items():
        # Add globalconstraint object so dual variable can be registered (if it doesn't already exist)
        if c_name not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                name=c_name,
                type=c_name,
                sense="<=",
                constant=obj_bound,
            )
        m.add_constraints(expr <= obj_bound, name=f"GlobalConstraint-{c_name}")

    # Save meta data
    obj_bounds = [bound for _, bound in constraints.values()]
    n.meta["obj_bound"] = tuple(obj_bounds) if region else obj_bounds[0]


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
