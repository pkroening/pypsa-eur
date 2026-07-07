# SPDX-FileCopyrightText: : 2026 - Peter Kröning
#
# SPDX-License-Identifier: MIT
import logging

import pandas as pd
import pypsa
from linopy import LinearExpression, QuadraticExpression, merge
from pypsa.descriptors import nominal_attrs

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
    if (
        isinstance(sense, str) and sense.startswith("min") or
        isinstance(sense, int) and sense > 0
    ):
        sense = +1
    elif (
        isinstance(sense, str) and sense.startswith("max") or
        isinstance(sense, int) and sense < 0
    ):
        sense = -1
    else:
        raise ValueError(f"Could not parse optimization sense {sense}")

    return sense

def set_mga_objective(
        n : pypsa.Network,
        mga_config : dict,
        alternative_objective : str,
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
    obj_config = mga_config["alternative_objectives"][alternative_objective]
    weights = {}
    static = obj_config["weights"].get("static", {})
    for component in static:
        vars = {}
        for var in static[component]:
            w = pd.Series(0, index=n.components[component].static.index)
            for carrier, const in static[component][var].items():
                mask = (n.components[component].static.carrier == carrier) & n.components[component].static.p_nom_extendable
                w.loc[mask] = const
            vars[var] = w
        weights[component] = vars
    varying = obj_config["weights"].get("varying", {})
    for component in varying:
        static = n.components[component].static
        vars = {}
        for var in varying[component]:
            w = pd.DataFrame(0, columns=static.index, index=n.snapshots)
            for carrier, const in varying[component][var].items():
                mask = static["carrier"] == carrier    # TODO: add regional for "tech"?
                w.loc[:, mask] = const
            w = w.multiply(n.snapshot_weightings.objective, axis=0)
            vars[var] = w
        weights[component] = vars

    objective = []
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
                coeffs = coeffs.reindex(columns=n.components[component].static.index, index=n.snapshots)
            objective.append(m[f"{c}-{attr}"] * coeffs * sense)

    m.objective = merge(objective)

    # Save meta data
    n.meta["sense"] = sense


def calculate_slack_myopic(
        slack_nom : float,
        slack_initial_fraction : float,
        current_horizon : int,
        planning_horizons : list[int],
    ) -> float:
    """
    Calculate slack for modelling to generate alternatives objective constraint

    We gradually increase slack from an initial fraction of the nominal slack at the first planning horizon linearly to the full slack at the last planning horizon.
    This is helpfull to avoid infeasible optimization problems and/or bad system designs, where the model would lean heavily in one technology in early optimizaton horizons.

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

def get_objective_bound(snakemake) -> float:
    """
    Calculate objective bound for mga constraint.
    """
    # Calculate slack
    slack_nom = float(snakemake.wildcards.slack)
    if snakemake.params.foresight == "myopic":
        current_horizon = int(snakemake.wildcards.planning_horizons)
        planning_horizons = snakemake.params.planning_horizons
        slack_initial_fraction = snakemake.params.mga.get("slack_initial_fraction", 1.0)
        slack = calculate_slack_myopic(
            slack_nom,
            slack_initial_fraction,
            current_horizon,
            planning_horizons
        )
    else:
        slack = slack_nom

    # Get optimal objective value
    n_opt = pypsa.Network(snakemake.input.network_opt)
    capex = n_opt.statistics.capex(groupby="country", groupby_method="sum")
    capex_installed = n_opt.statistics.installed_capex(groupby="country", groupby_method="sum")
    opex = n_opt.statistics.opex(groupby="country", groupby_method="sum")

    region = snakemake.params.mga.get("region", None)
    if region:
        capex = capex.loc[pd.IndexSlice[:, region]]
        capex_installed = capex_installed.loc[pd.IndexSlice[:, region]]

    obj_opt = capex.sum() + opex.sum()
    obj_bound = obj_opt * (1+slack) - capex_installed.sum()

    del n_opt

    return obj_bound


def set_mga_constraint(
        n : pypsa.Network,
        snakemake,
    ):
    """
    Set constraint for former objective
    """
    # Get linopy model
    m = n.model

    # Get former objective function
    obj_func = m.objective
    if not isinstance(obj_func, (LinearExpression, QuadraticExpression)):
        obj_func = obj_func.expression

    # Get objective bound
    obj_bound = get_objective_bound(snakemake)
    n.meta["obj_bound"] = obj_bound     # meta data

    # Add globalconstraint object so dual variable can be registered (if it doesn't already exist)
    name = "mga_bound"
    if name not in n.global_constraints.index:
        n.add(
            "GlobalConstraint",
            name=name,
            type=name,
            # carrier_attribute="",
            sense="<=",
            constant=obj_bound,
        )
    m.add_constraints(
        obj_func <= obj_bound,
        name=f"GlobalConstraint-{name}",
    )


def prepare_mga(
        n : pypsa.Network,
        snapshots : pd.DatetimeIndex,
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

