import logging
import os

import numpy as np
import pandas as pd
import pypsa
from _benchmark import memory_logger
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from linopy import LinearExpression, QuadraticExpression, merge
from pypsa.descriptors import nominal_attrs
from solve_network import (
    extra_functionality,
    prepare_network,
)

logger = logging.getLogger(__name__)
pypsa.network.power_flow.logger.setLevel(logging.WARNING)


def optimize_mga_fixed_bound(
    n : pypsa.Network,
    obj_bound : float,
    weights : dict,
    sense : str ="min",
    model_kwargs : dict={},
    **kwargs,
):
    """
    Run modelling-to-generate-alternatives (MGA) on network to find
    near-optimal solutions. Modification of
    pypsa.optimize.abstract.optimize_mga with direct cost bound
    argument.

    Parameters
    ----------
    n : pypsa.Network
    obj_bound : float
        Right hand side on total system cost constraint.
    weights : dict-like
        Weights for alternate objective function. The default is None, which
        minimizes generation capacity. The weights dictionary should be keyed
        with the component and variable (see ``pypsa/variables.csv``), followed
        by a float, dict, pd.Series or pd.DataFrame for the coefficients of the
        objective function. Examples:

        >>> {"Generator": {"p_nom": 1}}
        >>> {"Generator": {"p_nom": pd.Series(1, index=n.generators.index)}}
        >>> {"Generator": {"p_nom": {"gas": 1, "coal": 2}}}
        >>> {"Generator": {"p": pd.Series(1, index=n.generators.index)}
        >>> {"Generator": {"p": pd.DataFrame(1, columns=n.generators.index, index=n.snapshots)}

        Weights for non-extendable components are ignored. The dictionary does
        not need to provide weights for all extendable components.
    sense : str|int
        Optimization sense of alternate objective function. Defaults to 'min'.
        Can also be 'max'.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or
        `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    status : str
        The status of the optimization, either "ok" or one of the codes listed
        in https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html
    condition : str
        The termination condition of the optimization, either
        "optimal" or one of the codes listed in
        https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html
    """
    if weights is None:
        weights = dict(Generator=dict(p_nom=pd.Series(1, index=n.generators.index)))

    # create basic model
    m = n.optimize.create_model(
        include_objective_constant=False,
        **model_kwargs,
    )

    # build budget constraint
    objective = m.objective
    if not isinstance(objective, (LinearExpression, QuadraticExpression)):
        objective = objective.expression

    name = "total_system_cost"
    # Add globalconstraint object so dual variable can be registered (if it doesn't already exist)
    if name not in n.global_constraints.index:
        n.add(
            "GlobalConstraint",
            name=name,
            type="budget",
            carrier_attribute="",
            sense="<=",
            constant=obj_bound,
        )

    # Add constraint to model
    m.add_constraints(
        objective <= obj_bound,
        name=f"GlobalConstraint-{name}",
    )

    # parse optimization sense
    if (
        isinstance(sense, str)
        and sense.startswith("min")
        or isinstance(sense, int)
        and sense > 0
    ):
        sense = 1
    elif (
        isinstance(sense, str)
        and sense.startswith("max")
        or isinstance(sense, int)
        and sense < 0
    ):
        sense = -1
    else:
        raise ValueError(f"Could not parse optimization sense {sense}")

    # build alternate objective
    objective = []
    for c, attrs in weights.items():
        for attr, coeffs in attrs.items():
            if isinstance(coeffs, dict):
                coeffs = pd.Series(coeffs)
            if attr == nominal_attrs[c] and isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(n.get_extendable_i(c))
                coeffs.index.name = ""
            elif isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(index=n.components[c].static.index)
            elif isinstance(coeffs, pd.DataFrame):
                coeffs = coeffs.reindex(columns=n.components[c].static.index, index=n.snapshots)
            objective.append(m[f"{c}-{attr}"] * coeffs * sense)

    m.objective = merge(objective)

    status, condition = n.optimize.solve_model(**kwargs)

    # write MGA coefficients into metadata
    n.meta["obj_bound"] = obj_bound
    n.meta["sense"] = sense

    def convert_to_dict(obj) -> dict:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    n.meta["weights"] = convert_to_dict(weights)

    return status, condition


def prepare_solver_options(solving : dict) -> tuple[dict]:
    # The following solver setup follows that of `solve_network` in `solve_network.py`:
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs = {}

    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    kwargs["io_api"] = cf_solving.get("io_api", None)

    model_kwargs = {}
    model_kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    model_kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    if "model_options" in solving:
        model_kwargs = model_kwargs | solving["model_options"]

        if "solver_dir" in model_kwargs and "$" in model_kwargs["solver_dir"]:
            # Resolve env var as path
            model_kwargs["solver_dir"] = os.path.expandvars(model_kwargs["solver_dir"])
            logger.info(f"Set solver_dir to {model_kwargs['solver_dir']}")

    return kwargs, model_kwargs


def near_opt(
    n : pypsa.Network,
    config,
    params,
    solving,
    near_opt_config,
    current_horizon,
    sense,
    cost_bound,
):
    kwargs, model_kwargs = prepare_solver_options(solving)
    cf_solving = solving["options"]

    n.config = config
    n.params = params

    weights = {}
    static = near_opt_config["weights"].get("static", {})
    for c in static:
        vars = {}
        for v in static[c]:
            w = pd.Series(0, index=n.components[c].static.index)
            for carrier, const in static[c][v].items():
                w.loc[(n.components[c].static.carrier == carrier) & n.components[c].static.p_nom_extendable] = const
            vars[v] = w
        weights[c] = vars
    varying = near_opt_config["weights"].get("varying", {})
    for c in varying:
        vars = {}
        for v in varying[c]:
            w = pd.DataFrame(0, columns=n.components[c].static.index, index=n.snapshots)
            for carrier, const in varying[c][v].items():
                w.loc[:, n.components[c].static.carrier == carrier] = const
            w = w.multiply(n.snapshot_weightings.objective, axis=0)
            vars[v] = w
        weights[c] = vars

    status, condition = optimize_mga_fixed_bound(
        n,
        cost_bound,
        weights=weights,
        sense=sense,
        model_kwargs=model_kwargs,
        **kwargs,
    )
    # TODO: save Status

    if status == "ok":
        print("Solved successfully")
        n.meta["near_opt_status"] = "success"

        return n

    elif (
        (status == "warning")
        and (condition == "other")
        and ("numeric" not in solving["solver"]["options"])
    ):
        # It's possible that the model just needs to be solved with increased numerical focus
        logger.warning(
            "Solving possibly failed due to numerical trouble. "
            "Trying again with increased numerical focus."
        )
        solving["solver"]["options"] = "gurobi-numeric-focus"
        n = near_opt(
            n,
            config,
            params,
            solving,
            near_opt_config,
            current_horizon,
            sense,
            cost_bound,
        )

        return n

    elif "infeasible" in condition:
        # First, try to solve to optimality instead, in case the model
        # was infeasible because of the objective bound.
        logger.warning("Cost bound too tight! Solved to cost-optimality instead.")
        del model_kwargs["transmission_losses"]
        del model_kwargs["linearized_unit_commitment"]
        kwargs["model_kwargs"] = model_kwargs
        status, condition = n.optimize(**kwargs)

        if status == "ok":
            logger.info("Cost-optimisation successful")
            n.meta["near_opt_status"] = "too_expensive"
            n.meta["opt_system_cost"] = (
                n.statistics.installed_capex().sum() + n.objective
            )
            return n

        elif "infeasible" in condition:
            if cf_solving.get("print_infeasibilities", True):
                labels = n.model.compute_infeasibilities()
                logger.info(f"Labels:\n{labels}")
                n.model.print_infeasibilities()

    raise RuntimeError(
        f"Solve with condition status {status} and condition {condition}"
    )


def calculate_slack(
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


def get_region_buses(
        region : list[str],
        n : pypsa.Network,
    ) -> tuple[pd.Index]:
    mask = n.buses.index.str.startswith("EU")
    mask += n.buses.index.str.contains("atmosphere")
    for country in region:
        mask += (n.buses["country"] == country)
    buses_inside = n.buses[mask].index
    buses_outside = n.buses[~mask].index
    return buses_inside, buses_outside


def prepare_regional_network(
        region: str,
        n_mga: pypsa.Network,
        n_opt: pypsa.Network,
    ):
    """
    Merge the optimal network with the free region to explore alternatives.

    Parameters
    ----------
    region : str | list[str]
        Geographical region, where capacities of components shall be expanded in the mga
    n_mga: pypsa.Network
        Network to be prepared for the mga optimization
    n_opt: pypsa.Network
        Network of the optimal solution
    """

    # Get buses outside of region
    if not n_mga.buses.index.equals(n_opt.buses.index):
        raise IndexError("The buses of the cost optimized network and the network for mga differ unexpectedly.")
    buses_in, buses_out = get_region_buses(region=region, n=n_mga)

    ## Merge networks
    components_to_skip = ["LineType"]
    for c_mga, c_opt in zip(n_mga.components, n_opt.components):
        assert c_mga.name == c_opt.name
        if c_mga.name in components_to_skip:
            continue

        # Get components inside and outside of region
        if c_mga.name == "Bus":
            out_region = buses_out
            in_region_opt = buses_in
        else:
            bus_col = [c for c in c_mga.static.columns if "bus" in c]

            in_region = c_mga.static[bus_col].isin(buses_in)
            out_region = in_region[~in_region.any(axis="columns")].index

            in_region_opt = c_opt.static[bus_col].isin(buses_in)
            in_region_opt = in_region_opt[in_region_opt.any(axis="columns")].index

        if out_region.intersection(in_region_opt).any():
            raise RuntimeError(f"There are components both inside and outside of the region: {out_region.intersection(in_region_opt).to_list()}")

        # Replace values outisde of region
        n_mga.remove(c_mga.name, out_region)
        n_opt.remove(c_opt.name, in_region_opt)
    n_mga.merge(n_opt, components_to_skip=components_to_skip, inplace=True, with_time=False)

    ## Disable extentable components outside of region
    for c_mga in n_mga.components:
        # Get extendable attribute columns
        attributes = [
            column[:-len("_nom_extendable")] for column in c_mga.static.columns
            if "_nom_extendable" in column
        ]

        # Get components outside of region
        bus_col = [c for c in c_mga.static.columns if "bus" in c]
        in_region = c_mga.static[bus_col].isin(buses_in)
        out_region = in_region[~in_region.any(axis="columns")].index

        # Disable components
        c_mga.static.loc[
            out_region,
            [f"{attr}_nom_extendable" for attr in attributes],
        ] = False


def get_optimal_value(
        region : None | list[str],
        n_opt : pypsa.Network,
    ) -> float:
    """
    Get optimal objective value (for region).

    Parameters
    ----------
    region : str | list[str]
        Geographical region, where capacities of components shall be expanded in the mga
    n_opt: pypsa.Network
        Network of the optimal solution

    Returns
    -------
    float
        Base value for the mga near optimality constraint
    """
    if region:
        # TODO: calulation using objective
        capex = n_opt.statistics.capex(groupby="country", groupby_method="sum").loc[pd.IndexSlice[:, region]].sum()
        opex = n_opt.statistics.opex(groupby="country", groupby_method="sum").sum()

        obj_base = capex + opex

    else:
        # TODO
        obj_base = n_opt.objective

    return obj_base

def prepare_network_regional_mga(
        n : pypsa.Network,
    ):
    """
    Todo: content
    """
    region = snakemake.params.near_opt.get("region", None)
    if isinstance(region, str):
        region = [region]

    # Load optimal network
    n_opt = pypsa.Network(snakemake.input.network_opt)

    # Get optimal value of objective and calculate slack
    obj_base = get_optimal_value(region, n_opt)
    slack = calculate_slack(
        slack_nom=float(snakemake.wildcards.slack),
        slack_initial_fraction=snakemake.params.near_opt.get("slack_initial_fraction", 1.0),
        current_horizon=int(current_horizon),
        planning_horizons=planning_horizons
    )
    obj_bound = obj_base*(1+slack)

    # Prepare regional network
    if region:
        prepare_regional_network(region, n, n_opt)



    # Add near optimality constraint
    # TODO

    # Remove optimal network
    del n_opt

    return obj_bound


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "near_opt_myopic",
            configfiles="config/test/config.myopic-mga.yaml",
            clusters="5",
            opts="",
            sector_opts="",
            planning_horizons="2045",
            sense="max",
            slack=0.1,
        )

    # Set up logging, configuration, etc. (compare `solve_network.py`)
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]
    cf_solving = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    # Load network
    n = pypsa.Network(snakemake.input.network)
    current_horizon = snakemake.wildcards.planning_horizons
    planning_horizons = snakemake.params.planning_horizons

    # Prepare network
    prepare_network(
        n=n,
        solve_opts=solve_opts,
        foresight="myopic",
        planning_horizons=current_horizon,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth", None),
        rolling_horizon=False,
    )
    print(type(snakemake))
    obj_bound = prepare_network_regional_mga(
        n=n,
    )

    # Solve network
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None),
        interval=getattr(snakemake.config.get("solving", {}), "mem_logging_frequency", 30),
    ) as mem:
        m = near_opt(
            n,
            snakemake.config,
            snakemake.params,
            snakemake.params.solving,
            snakemake.params.near_opt,
            current_horizon,
            snakemake.wildcards.sense,
            obj_bound,
        )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    # Save solved network
    m.export_to_netcdf(snakemake.output[0])
