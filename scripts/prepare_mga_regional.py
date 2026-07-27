# SPDX-FileCopyrightText: : 2026 - Peter Kröning
#
# SPDX-License-Identifier: MIT
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from linopy import LinearExpression


# Regional utils
def get_buses_of_regions(
        n : pypsa.Network,
        region : list[str],
        eu_assignment : None | str = None,
    ) -> tuple[pd.Index]:
    """
    Get list of buses being in- or outside of region.

    Parameters
    ----------
    region : list[str]
        Countries beeing part of region
    n : pypsa.Network
        The PyPSA network instance
    eu_assignment : None | str
        Weather to assign the eu buses to the region, outside, both or none of them

    Returns
    -------
    tuple[pd.Index]
        buses inside, buses outside
    """
    # Get masks
    mask_eu = (n.buses["location"] == "EU")

    mask_inside = pd.Series(index=n.buses.index, data=False)
    for country in region:
        mask_inside += (n.buses["country"] == country)


    mask_outside = ~mask_inside & ~mask_eu

    # Assign eu
    if eu_assignment == "in_region" or eu_assignment == "both":
        mask_inside += mask_eu
    if eu_assignment == "out_region" or eu_assignment == "both":
        mask_outside += mask_eu

    # Get buses
    buses_inside = n.buses[mask_inside].index
    buses_outside = n.buses[mask_outside].index
    buses_neither = n.buses[~mask_inside & ~mask_outside].index
    return buses_inside, buses_outside, buses_neither

def get_cross_border_components(
        region: list[str],
        n: pypsa.Network,
        ignore_c02: bool = True,
    ) -> dict[str, pd.Series | None]:
    buses_inside, buses_outside, buses_neither = get_buses_of_regions(region=region, n=n, eu_assignment="out_region")
    # ignore atmosphere and co2 flows
    if ignore_c02:
        for pat in ("atmosphere", "co2"):
            buses_inside = buses_inside.append(
                n.buses[n.buses.index.str.contains(pat)].index
            )
        buses_inside = buses_inside.drop_duplicates()

    cross_border_components = dict()
    connectors = ["Line", "Link"]
    for component in n.components:

        bus_col = [col for col in component.static.columns if "bus" in col]
        if component.name in connectors:
            # Get bools w
            in_region = component.static[bus_col].isin(buses_inside)
            out_region = component.static[bus_col].isin(buses_outside)

            # Reference bus
            b0 = "bus0"

            # Get mask
            ignore = (in_region == out_region)
            flow_in = in_region.gt(in_region[b0], axis=0) & ~ignore
            flow_out = in_region.lt(in_region[b0], axis=0) & ~ignore

            # Get signs
            cross_border = pd.Series(0, index=in_region.index)
            cross_border[flow_in.any(axis="columns")] = +1
            cross_border[flow_out.any(axis="columns")] = -1

        elif len(bus_col) <= 1:
            continue

        else:
            raise ValueError(f"Got unexpected component that might have multiple buses: {component}")

        cross_border_components[component.name] = cross_border

    return cross_border_components

def get_variable_region_mapping(
        n : pypsa.Network,
        region: list[str]
    ) -> pd.Series:

    buses_in, buses_out, buses_neither = get_buses_of_regions(region=region, n=n, eu_assignment="out_region")
    cross_border_components = get_cross_border_components(region, n)

    # Determine variable label region mapping
    in_region_by_label = []
    for var_name, variable in n.model.variables.items():
        # Get component corresponding to variable
        comp = var_name.split("-")[0]
        comp = n.components[comp]

        # Get region information
        bus0 = "bus" if "bus" in comp.static.columns else "bus0"
        in_region = comp.static[bus0].isin(buses_in).astype(float)

        cbc = cross_border_components.get(comp.name, None)
        if cbc is not None:
            in_region[cbc != 0] = 0.5

        # Get labels
        labels = variable.labels

        in_region = xr.DataArray(
            in_region.reindex(labels.coords["name"].values).to_numpy(),
            coords={"name": labels.coords["name"]},
            dims=["name"],
        )
        in_region, labels = xr.broadcast(in_region, labels)

        flat_labels = labels.values.flatten()
        mask = (flat_labels != -1)
        in_region_by_label.append(
            pd.Series(in_region.values.ravel()[mask], index=flat_labels[mask])
        )

    return pd.concat(in_region_by_label)

def split_expression_by_region(
        n : pypsa.Network,
        expr: LinearExpression,
        region: list[str]
    ) -> tuple[LinearExpression, LinearExpression]:
    # Convert expression
    expr_df = expr.flat

    # Get variables mapping
    in_region = get_variable_region_mapping(n, region)

    # Check if all variables are in mapping
    not_mapped = ~expr_df["vars"].isin(in_region.index)
    if not_mapped.any():
        raise ValueError("Some term(s) of the expression are not in regional mapping.")

    # Build expressions
    variables = expr_df["vars"].to_numpy()
    coefficients = expr_df["coeffs"].to_numpy()
    region_share = expr_df["vars"].map(in_region).fillna(0.0).to_numpy()
    def build_expr(coeffs: np.ndarray) -> LinearExpression:
        keep = (coeffs != 0)
        data = xr.Dataset(
            {
                "coeffs": ("_term", coeffs[keep]),
                "vars": ("_term", variables[keep]),
            }
        )
        return LinearExpression(data, n.model)

    return build_expr(coefficients * region_share), build_expr(coefficients * (1 - region_share))


def split_df_by_region(df: pd.DataFrame, region: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    country = df.index.get_level_values("country")
    in_mask = country.isin(region)
    return df.loc[in_mask], df.loc[~in_mask]


# Prepare network
def prepare_mga_regional(
        n : pypsa.Network,
        snakemake,
    ):
    """
    Prepare the network for a regional modelling to generate alternatives by merging with regions of the optimal network, which should not be expanded.

    Parameters
    ----------
    n: pypsa.Network
        Network to be optimized
    snakemake
    """
    # Get region
    region = snakemake.params.mga.get("region", None)
    buses_in, buses_out, buses_neither = get_buses_of_regions(region=region, n=n, eu_assignment="in_region")

    # Load optimal network
    n_opt = pypsa.Network(snakemake.input.network_opt)
    if not n.buses.index.equals(n_opt.buses.index):
        raise IndexError("The buses of the cost optimized network and the network for mga differ unexpectedly.")

    ## Merge networks
    components_to_skip = n.standard_type_components
    components_to_skip.update({"Carrier", "Global Constraints"})
    for comp, comp_opt in zip(n.components, n_opt.components):
        if comp.name != comp_opt.name:
            raise ValueError("While iterating through the component classes of the networks, different are reached.")

        # TODO: remove
        # comp.static.sort_index().to_csv(f"dev/nw_df/static/{comp.name}-pre_regio.csv")
        # comp_opt.static.sort_index().to_csv(f"dev/nw_df/static/{comp.name}-opt.csv")
        # for _prop, _df in comp_opt.dynamic.items():
        #     _df.to_csv(f"dev/nw_df/dynamic/{comp.name}_{_prop}-opt.csv")


        # Skip some components
        if comp.name in components_to_skip:
            continue

        # Disable extension and fix optimal value
        attributes = [
            col[:-len("_extendable")] for col in comp.static.columns
            if "_nom_extendable" in col
        ]
        comp_opt.static[attributes] = comp_opt.static[[f"{attr}_opt" for attr in attributes]]
        comp_opt.static[[f"{attr}_extendable" for attr in attributes]] = False

        # Get components inside and outside of region
        if comp.name == "Bus":
            out_region = buses_out
            in_region_opt = buses_in
        else:
            bus_col = [c for c in comp.static.columns if "bus" in c]

            in_region = comp.static[bus_col].isin(buses_in)
            out_region = in_region[~in_region.any(axis="columns")].index

            in_region_opt = comp_opt.static[bus_col].isin(buses_in)
            in_region_opt = in_region_opt[in_region_opt.any(axis="columns")].index

        if out_region.intersection(in_region_opt).any():
            raise RuntimeError(f"There are components both inside and outside of the region: {out_region.intersection(in_region_opt).to_list()}")

        # Remove components
        n.remove(comp.name, out_region)
        n_opt.remove(comp_opt.name, in_region_opt)
    n.merge(n_opt, components_to_skip=components_to_skip, inplace=True, with_time=False)

    # Delete optimal network
    del n_opt

    # TODO: remove
    # for comp in n.components:
    #     comp.static.sort_index().to_csv(f"dev/nw_df/static/{comp.name}-post_regio.csv")
