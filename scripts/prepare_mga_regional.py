# SPDX-FileCopyrightText: : 2026 - Peter Kröning
#
# SPDX-License-Identifier: MIT
import pandas as pd
import pypsa


def get_cross_border_components(
        region: list[str],
        n: pypsa.Network
    ) -> dict[str, pd.Index]:
    """
    Get components with flows across the region border

    Parameters
    ----------
    region : list[str]
        Countries beeing part of region
    n : pypsa.Network
        The PyPSA network instance

    Returns
    -------
    dict[str, pd.Index]
        {component name, index of cross boader components}
    """
    buses_in, buses_out, buses_global = get_buses_of_regions(region=region, n=n, with_global="inside")

    cross_border_components = {}
    for comp in n.components:
        static  = comp.static

        bus_col = [col for col in static.columns if "bus" in col]
        if len(bus_col) <= 1:
            cross_border_components[comp.name] = pd.Series(index=static.index)
            continue

        in_region = static[bus_col].isin(buses_in).any(axis="columns")
        out_region = static[bus_col].isin(buses_out).any(axis="columns")

        cross_border = in_region == out_region

        cross_border_components[comp.name] = cross_border

    return cross_border_components

def get_buses_of_regions(
        n : pypsa.Network,
        region : list[str],
        with_global : None | str = None,
    ) -> tuple[pd.Index]:
    """
    Get list of buses being in- or outside of region.

    Parameters
    ----------
    region : list[str]
        Countries beeing part of region
    n : pypsa.Network
        The PyPSA network instance
    with_global : None | str
        Weather to include the global buses to the region, outside or none of them

    Returns
    -------
    tuple[pd.Index]
        buses inside, buses outside
    """
    mask_glob = pd.Series(index=n.buses.index, data=False)
    mask_glob += n.buses.index.str.startswith("EU")
    mask_glob += n.buses.index.str.contains("atmosphere")

    mask_inside = pd.Series(index=n.buses.index, data=False)
    for country in region:
        mask_inside += (n.buses["country"] == country)

    mask_outside = ~mask_inside&~mask_glob

    if with_global == "inside":
        mask_inside += mask_glob
    elif with_global == "outside":
        mask_outside += mask_glob

    buses_inside = n.buses[mask_inside].index
    buses_outside = n.buses[mask_outside].index
    buses_global = n.buses[mask_glob].index
    return buses_inside, buses_outside, buses_global


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
    buses_in, buses_out, buses_global = get_buses_of_regions(region=region, n=n, with_global="inside")

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
