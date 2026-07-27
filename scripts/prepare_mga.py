# SPDX-FileCopyrightText: : 2026 - Peter Kröning
#
# SPDX-License-Identifier: MIT
import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from linopy import LinearExpression, merge
from pypsa.descriptors import nominal_attrs

from scripts.prepare_mga_regional import (
    get_buses_of_regions,
    get_cross_border_components,
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
    expr_config = mga_config["alternative_objectives"][alternative_objective]
    weights = {}

    cross_border_components = False
    if "import" in alternative_objective:
        region = mga_config.get("region", None)
        if not region:
            raise ValueError("For optimization of cross border components a region has to be defined in the mga config.")
        cross_border_components = get_cross_border_components(region, n)

    static = expr_config["weights"].get("static", {})
    for component in static:
        vars = {}
        for var in static[component]:
            w = pd.Series(0, index=n.components[component].static.index)
            for carrier, const in static[component][var].items():
                mask = (n.components[component].static.carrier == carrier) & n.components[component].static.p_nom_extendable
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
                w = w.add(sign, axis=1)
            else:
                for carrier, const in varying[component][var].items():
                    mask = static["carrier"] == carrier    # TODO: add regional for "tech"?
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
                coeffs = coeffs.reindex(columns=n.components[component].static.index, index=n.snapshots)
            new_expr.append(m[f"{component}-{attr}"] * coeffs * sense)

    if "import" in alternative_objective:
        flows = merge(new_expr)
        imports = m.add_variables(
            name="imports",
            coords=flows.coords,
            lower=0
        )

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

def set_mga_constraint(
        n : pypsa.Network,
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
            slack,
            slack_initial_fraction,
            current_horizon,
            planning_horizons
        )

    def calc_bound(capex: pd.DataFrame, capex_installed: pd.DataFrame, opex: pd.DataFrame):
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
    capex_installed = n_opt.statistics.installed_capex(groupby="country", groupby_method="sum")
    opex = n_opt.statistics.opex(groupby="country", groupby_method="sum")

    # Check wether to split by region
    region = snakemake.params.mga.get("region", None)
    if region:

        def split_region_df(df: pd.DataFrame, region: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
            country = df.index.get_level_values("country")
            in_mask = country.isin(region)
            return df.loc[in_mask], df.loc[~in_mask]

        def split_region_expression(expr: LinearExpression, region: list[str]) -> tuple[LinearExpression, LinearExpression]:
            """
            Split a linear expression into the parts that belong to variables of components
            inside and outside of the region.

            Since PyPSA sums out the component index ("name") when assembling the objective,
            the split cannot be done via `.sel`/coordinates on `expr` itself. Instead, every
            variable label occurring in `expr` is traced back to the component/asset it
            belongs to (via `model.variables`) and attributed to the region based on all of
            that component's buses. Components fully inside/outside the region (the common
            case for single-bus components like Generator, Load, ...) are attributed fully to
            one side; components crossing the region border (e.g. a Line or Link with one bus
            inside and one outside) are split 50/50 between the two expressions.

            Note: terms that cannot be attributed to a component with a bus (e.g. the
            aggregated `objective_constant` variable, present if `include_objective_constant`
            was used) are counted as outside the region, since they cannot be disaggregated
            further.

            Parameters
            ----------
            expr : linopy.LinearExpression
                Expression to split, e.g. the (former) objective function.
            region : list[str]
                Countries being part of the region.

            Returns
            -------
            tuple[LinearExpression, LinearExpression]
                Expression restricted to the terms inside, and outside of the region.
            """
            buses_in, buses_out, buses_neither = get_buses_of_regions(region=region, n=n, eu_assignment="out_region")
            cross_border_components = get_cross_border_components(region, n)

            # Determine, for every variable label appearing in the model, the fraction of
            # its term attributed to the region: 1.0/0.0 based on the component's primary
            # bus, unless it is a connector crossing the region border (per
            # get_cross_border_components), in which case it is split 50/50.
            in_region_frac_by_label = []
            for var_name, variable in m.variables.items():
                # Get component corresponding to variable
                comp = var_name.split("-")[0]
                comp = n.components[comp]

                # Get primary bus
                bus0 = "bus" if "bus" in comp.static.columns else "bus0"

                # Get labels
                labels = variable.labels
                labels_coords_name = labels.coords["name"]

                frac_in = comp.static[bus0].reindex(labels_coords_name.values).isin(buses_in).astype(float)

                cb = cross_border_components.get(comp.name, None)
                if cb is not None:
                    frac_in[cb.reindex(labels_coords_name.values) != 0] = 0.5

                frac_in = xr.DataArray(frac_in.to_numpy(), coords={"name": labels_coords_name}, dims=["name"])
                frac_in, labels = xr.broadcast(frac_in, labels)

                flat_labels = labels.values.ravel()
                mask = flat_labels != -1
                in_region_frac_by_label.append(
                    pd.Series(frac_in.values.ravel()[mask], index=flat_labels[mask])
                )

            in_region_frac_by_label = pd.concat(in_region_frac_by_label)
            in_region_frac_by_label = in_region_frac_by_label[~in_region_frac_by_label.index.duplicated()]

            flat = expr.flat
            unattributed = ~flat["vars"].isin(in_region_frac_by_label.index)
            if unattributed.any():
                logger.warning(
                    f"{unattributed.sum()} term(s) of the expression could not be attributed "
                    "to a component with a bus and are counted as outside the region."
                )
            in_region_frac = flat["vars"].map(in_region_frac_by_label).fillna(0.0).to_numpy()
            coeffs = flat["coeffs"].to_numpy()
            vars = flat["vars"].to_numpy()

            def build_expr(coeffs: np.ndarray) -> LinearExpression:
                keep = coeffs != 0
                data = xr.Dataset(
                    {
                        "coeffs": ("_term", coeffs[keep]),
                        "vars": ("_term", vars[keep]),
                    }
                )
                return LinearExpression(data, m)

            return build_expr(coeffs * in_region_frac), build_expr(coeffs * (1 - in_region_frac))

        capex_in, capex_out = split_region_df(capex, region)
        capex_installed_in, capex_installed_out = split_region_df(capex_installed, region)
        opex_in, opex_out = split_region_df(opex, region)

        obj_bound_in_region = calc_bound(capex_in, capex_installed_in, opex_in)
        obj_bound_out_region = calc_bound(capex_out, capex_installed_out, opex_out)

        n.meta["obj_bound"] = (obj_bound_in_region, obj_bound_out_region)   # meta data

        obj_func_in, obj_func_out = split_region_expression(obj_func, region)

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
                            constant=bound
                        )
                    m.add_constraints(
                        expr <= bound,
                        name=f"GlobalConstraint-{c_name}"
                    )

    else:
        obj_bound = calc_bound(capex, capex_installed, opex)
        n.meta["obj_bound"] = obj_bound     # meta data

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

    del n_opt

    print(m)

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
