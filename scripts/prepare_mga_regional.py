# SPDX-FileCopyrightText: : 2026 - Peter Kröning
#
# SPDX-License-Identifier: MIT
import pandas as pd
import pypsa


def get_cross_border_components(
        region: list[str],
        n: pypsa.Network
    ) -> dict[str, pd.Series | None]:
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
    dict[str, pd.Series]
        {component name, cross boader components with sign}
    """
    buses_inside, buses_outside, buses_neither = get_buses_of_regions(region=region, n=n, eu_assignment="out_region")

    cross_border_components = dict()
    connectors = ["Line", "Link"]
    for component in n.components:

        bus_col = [col for col in component.static.columns if "bus" in col]
        if component.name in connectors:
            in_region = component.static[bus_col].isin(buses_inside)
            out_region = component.static[bus_col].isin(buses_outside)

            b0 = "bus0"
            bus_col.remove(b0)
            cross_border = pd.DataFrame(
                index=in_region.index,
                columns=[
                    f"Flow: {b0} -> {b}"
                    for b in bus_col
                ],
                data=0
            )
            for b in bus_col:
                # Get flow directions
                flow = f"Flow: {b0} -> {b}"
                cross_border.loc[in_region[b0] < in_region[b], flow] = +1
                cross_border.loc[in_region[b0] > in_region[b], flow] = -1
                cross_border.loc[in_region[b] == out_region[b], flow] = 0

            # Check
            has_pos = (cross_border == 1).any(axis=1)
            has_neg = (cross_border == -1).any(axis=1)
            if (has_pos & has_neg).any():
                raise ValueError(f"Component {component.name} has flows from {b0} to both inside and outside the given region.")
            cross_border = pd.Series(0, index=cross_border.index)
            cross_border[has_pos] = 1
            cross_border[has_neg] = -1

        elif len(bus_col) <= 1:
            continue

        else:
            raise ValueError(f"Got unexpected component that might have multiple buses: {component}")

        cross_border_components[component.name] = cross_border

    return cross_border_components

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
