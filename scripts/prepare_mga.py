# SPDX-FileCopyrightText: Peter Kröning and Contributors to <https://github.com/koen-vg/eu-hydrogen>
#
# SPDX-License-Identifier: MIT
import logging
from collections import defaultdict

import pandas as pd
import pypsa
from linopy import LinearExpression, merge
from pypsa.descriptors import nominal_attrs

from scripts.prepare_mga_regional import (
    country_grouper,
    get_cross_border_components,
    split_df_by_region,
    split_expression_by_region,
)

logger = logging.getLogger(__name__)
pypsa.network.power_flow.logger.setLevel(logging.WARNING)


def parse_optimization_sense(alternative_objective: str) -> int:
    """
    Parse the optimization sense of an alternate objective function.

    Parameters
    ----------
    alternative_objective : str
        Name of the alternate objective function, e.g. "min-capex"

    Returns
    -------
    int
        +1 for minimization, -1 for maximization
    """
    sense = alternative_objective.split(sep="-")[0]
    if sense.startswith("min"):
        return +1
    if sense.startswith("max"):
        return -1
    raise ValueError(f"Could not parse optimization sense {sense}")


def build_cost_expression(
    n: pypsa.Network,
    cost_type: str,
) -> LinearExpression:
    """
    Build the capital or operational expenditures of the network as a linear expression.

    The counterpart of `n.statistics.capex`/`n.statistics.opex`, grouped by country in
    the same way, see `country_grouper`. Capex of non-extendable components only enters
    as a constant term and is dropped, see `non_extendable_capex`.

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
    return getattr(n.optimize.expressions, cost_type)(
        groupby=country_grouper
    ).reset_const()


def non_extendable_capex(n: pypsa.Network) -> pd.Series:
    """
    Calculate the capital expenditures of non-extendable components, grouped by country.

    This is the share of `n.statistics.capex` that is constant during optimization and
    therefore not part of `build_cost_expression`. It has to be calculated here rather
    than read off the expression: `n.optimize.expressions.capex` aligns the fixed
    capacities with the extendable variables, whose indexes are disjoint, so the
    constant silently drops out for every component class that has extendable members.

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
        country = country_grouper(n, component)[fixed.index]
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


def carrier_weights(static: pd.DataFrame, carriers: dict | None) -> pd.Series:
    """
    Spread the carrier weights of the mga config over the components.

    Parameters
    ----------
    static : pd.DataFrame
        Static data of the component class
    carriers : dict | None
        Weight per carrier, or None to weight nothing

    Returns
    -------
    pd.Series
        Weight per component
    """
    weights = pd.Series(0.0, index=static.index)
    for carrier, const in (carriers or {}).items():
        weights[static["carrier"] == carrier] = const

    return weights


def build_weighted_objective(
    n: pypsa.Network,
    mga_config: dict,
    alternative_objective: str,
    sense: int,
) -> LinearExpression:
    """
    Build an objective function from the carrier weights given in the mga config.

    Import objectives weight the connectors crossing the region border by their
    direction, everything else weights components by their carrier.

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
    m = n.model
    weights_config = mga_config["alternative_objectives"][alternative_objective][
        "weights"
    ]

    minimize_imports = "import" in alternative_objective
    cross_border = None
    if minimize_imports:
        if sense < 0:
            raise NotImplementedError(
                "Maximizing imports is not supported: the positive part taken below "
                "only bounds the flows from above when the objective is minimized."
            )
        region = mga_config.get("region", None)
        if not region:
            raise ValueError(
                "For optimization of cross border components a region has to be defined in the mga config."
            )
        cross_border = get_cross_border_components(region, n)

    weights = defaultdict(dict)

    for component, attrs in weights_config.get("static", {}).items():
        static = n.components[component].static
        extendable = static[f"{nominal_attrs[component]}_extendable"]
        for attr, carriers in attrs.items():
            weights[component][attr] = carrier_weights(static, carriers).where(
                extendable, 0.0
            )

    for component, attrs in weights_config.get("varying", {}).items():
        static = n.components[component].static
        for attr, carriers in attrs.items():
            w = carrier_weights(static, carriers)
            if cross_border is None:
                pass
            elif component in cross_border:
                w = cross_border[component].astype(float) * (w if carriers else 1.0)
            else:
                # Assets rather than connectors, like the primary energy each country
                # buys for itself, are an import wherever they sit inside the region
                w = w.where(country_grouper(n, component).isin(region), 0.0)
            weights[component][attr] = w

    terms = []
    for component, attrs in weights.items():
        for attr, w in attrs.items():
            variable = m[f"{component}-{attr}"]
            expr = variable * w.reindex(variable.indexes["name"], fill_value=0)
            if "snapshot" in variable.dims:
                expr = expr * n.snapshot_weightings.objective
            terms.append(expr * sense)

    new_obj = merge(terms)

    if minimize_imports:
        # Count only the positive part of every flow, so that exports on one
        # connector cannot offset imports on another
        imports = m.add_variables(name="imports", coords=new_obj.coords, lower=0)
        m.add_constraints(new_obj - imports <= 0, name="imports_sign")
        new_obj = imports.sum()

    return new_obj


def set_mga_objective(
    n: pypsa.Network,
    mga_config: dict,
    alternative_objective: str,
) -> None:
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
    sense = parse_optimization_sense(alternative_objective)
    cost_type = alternative_objective.split(sep="-", maxsplit=1)[-1]

    if cost_type in ("capex", "opex"):
        n.model.objective = build_cost_objective(n, mga_config, cost_type, sense)
    else:
        n.model.objective = build_weighted_objective(
            n, mga_config, alternative_objective, sense
        )


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


def assert_no_capex_out_of_region(
    n: pypsa.Network,
    capex_expr_out: LinearExpression,
) -> None:
    """
    Check that no capital cost is left to be decided outside of the region.

    `prepare_mga_regional` fixes every capacity outside the region, so the near-optimal
    bound there only has to cover the operational costs. Components without a country
    are not fixed, but their cost is expected to be zero, see `stranded_costs`.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    capex_expr_out : LinearExpression
        Out-of-region part of the capital expenditures

    Raises
    ------
    RuntimeError
        If an extendable component out of the region carries a capital cost
    """
    # `flat` drops the terms with a zero coefficient and the empty slots of the array
    terms = capex_expr_out.flat
    if terms.empty:
        return

    positions = n.model.variables.get_label_position(terms["vars"].to_numpy())
    raise RuntimeError(
        "Capacities outside the region are expected to be fixed at their cost-optimal "
        f"value, but these still carry a capital cost: {positions}"
    )


def set_mga_constraint(
    n: pypsa.Network,
    snakemake,
) -> None:
    """
    Constrain the system costs to stay within the slack around the cost optimum.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    snakemake
    """
    m = n.model

    slack = float(snakemake.wildcards.slack)
    if snakemake.params.foresight == "myopic":
        slack = calculate_slack_myopic(
            slack,
            snakemake.params.mga.get("slack_initial_fraction", 1.0),
            int(snakemake.wildcards.planning_horizons),
            snakemake.params.planning_horizons,
        )

    def calc_bound(capex: pd.Series, opex: pd.Series, capex_const: pd.Series) -> float:
        # The expressions cover extendable capacities only, so the constant capex of the
        # non-extendable components is subtracted from the bound instead
        return (1 + slack) * (capex.sum() + opex.sum()) - capex_const.sum()

    # Get cost-optimal network and values
    n_opt = pypsa.Network(snakemake.input.network_opt)
    capex = n_opt.statistics.capex(groupby=country_grouper, groupby_method="sum")
    opex = n_opt.statistics.opex(groupby=country_grouper, groupby_method="sum")

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

        if not capex_expr_out.flat.empty:
            raise RuntimeError(
                "Decion variables outside the region are expected to not carry capital cost."
            )

        constraints = {
            "near_opt_bound_in_region": (
                capex_expr_in + opex_expr_in,
                calc_bound(capex_in, opex_in, capex_const_in),
            ),
            # decision variables just operation -> slack base operation
            "near_opt_bound_out_region": (
                opex_expr_out,
                (1 + slack) * opex_out.sum(),
            ),
        }
    else:
        constraints = {
            "near_opt_bound": (
                capex_expr.sum() + opex_expr.sum(),
                calc_bound(capex, opex, capex_const),
            )
        }

    del n_opt

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


def prepare_mga(
    n: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    snakemake,
) -> None:
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
    set_mga_constraint(n, snakemake)
    set_mga_objective(
        n, snakemake.params.mga, snakemake.wildcards.alternative_objectives
    )
