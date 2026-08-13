# SPDX-FileCopyrightText: Peter Kröning and Contributors to <https://github.com/koen-vg/eu-hydrogen>
#
# SPDX-License-Identifier: MIT
import logging

import numpy as np
import pandas as pd
import pypsa
from linopy import LinearExpression

from scripts._helpers import sanitize_custom_columns

logger = logging.getLogger(__name__)

# Headroom of the interface links, far above any national commodity flow but
# small enough to keep the constraint matrix well scaled
INTERFACE_P_NOM = 1e7


def get_buses_of_regions(
    n: pypsa.Network,
    region: list[str],
    eu_assignment: str | None = None,
) -> tuple[pd.Index, pd.Index]:
    """
    Get the buses in- and outside of the region.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    region : list[str]
        Countries being part of region
    eu_assignment : str | None
        Whether to assign the EU-wide buses to "in_region", "out_region", "both"
        or, if None, to neither

    Returns
    -------
    tuple[pd.Index, pd.Index]
        Buses inside, buses outside of region
    """
    eu = n.buses["location"] == "EU"
    inside = n.buses["country"].isin(region) & ~eu
    outside = ~inside & ~eu

    if eu_assignment in ("in_region", "both"):
        inside |= eu
    if eu_assignment in ("out_region", "both"):
        outside |= eu

    return n.buses.index[inside], n.buses.index[outside]


def get_cross_border_components(
    region: list[str],
    n: pypsa.Network,
    ignore_co2: bool = True,
) -> dict[str, pd.Series]:
    """
    Sign the connectors crossing the region border: +1 into, -1 out of the region.

    The sign is relative to `bus0`, so that multiplying it with the flow variable
    yields a positive value for an import and a negative one for an export.

    Parameters
    ----------
    region : list[str]
        Countries being part of region
    n : pypsa.Network
        Network to be optimized
    ignore_co2 : bool
        Treat co2 and atmosphere buses as being inside the region, so that
        emissions are not counted as a traded commodity

    Returns
    -------
    dict[str, pd.Series]
        Sign per component, for "Line" and "Link"
    """
    if not n.transformers.empty:
        raise NotImplementedError("Cross-border transformers are not accounted for.")

    buses_inside, buses_outside = get_buses_of_regions(
        n=n, region=region, eu_assignment="out_region"
    )
    if ignore_co2:
        buses_inside = buses_inside.union(
            n.buses.index[n.buses.index.str.contains("co2|atmosphere")]
        )

    cross_border = {}
    for c in ("Line", "Link"):
        static = n.components[c].static
        bus_cols = [col for col in static.columns if col.startswith("bus")]
        inside = static[bus_cols].isin(buses_inside)
        outside = static[bus_cols].isin(buses_outside)

        # Unused ports fall in neither set, co2 buses in both; skip both cases
        crossing = inside ^ outside
        into = (inside.gt(inside["bus0"], axis=0) & crossing).any(axis="columns")
        out_of = (inside.lt(inside["bus0"], axis=0) & crossing).any(axis="columns")

        cross_border[c] = (
            pd.Series(0, index=static.index).mask(into, 1).mask(out_of, -1)
        )

    return cross_border


def country_grouper(
    n: pypsa.Network,
    c: str,
    port: str = "",
    nice_names: bool = False,
) -> pd.Series:
    """
    Group components by the country of their first port that has one.

    Drop-in replacement for `pypsa.statistics.groupers.country`, usable as the
    `groupby` argument of both `n.statistics` and `n.optimize.expressions`.
    The pypsa grouper reads a single port, which leaves everything anchored on an
    EU-wide commodity bus without a country: conventional generation is modelled
    as a link from the EU fuel bus, so its costs would fall outside every region.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    c : str
        Name of the component class
    port : str
        Unused. Kept for the signature the pypsa statistics accessors call with
    nice_names : bool
        Unused. Kept for the signature the pypsa statistics accessors call with

    Returns
    -------
    pd.Series
        Country per component
    """
    static = n.components[c].static
    country = pd.Series("", index=static.index)
    for bus in sorted(col for col in static.columns if col.startswith("bus")):
        country = country.where(
            country != "", static[bus].map(n.buses.country).fillna("")
        )

    return country.rename("country")


def national_name(name: str, country: str) -> str:
    """
    Name of the national copy of an EU-wide bus or asset.

    Parameters
    ----------
    name : str
        Name of the EU-wide component, e.g. "EU oil primary"
    country : str
        Country the copy belongs to

    Returns
    -------
    str
        Name of the copy, e.g. "DE oil primary"
    """
    return f"{country} {name.removeprefix('EU ')}"


def pool_supply(
    n: pypsa.Network,
    eu_bus: str,
    eu_buses: pd.Index,
    max_depth: int = 3,
) -> tuple[list[str], pd.Index]:
    """
    Find the buses and links making up the supply side of an EU-wide commodity pool.

    Walks upstream from the pool bus over the links delivering into it from another
    EU-wide bus, as `EU oil` is refined from `EU oil primary`.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    eu_bus : str
        The EU-wide commodity bus
    eu_buses : pd.Index
        The EU-wide buses that may be part of a supply chain
    max_depth : int
        How many conversion steps to follow upstream

    Returns
    -------
    tuple[list[str], pd.Index]
        Buses of the pool, starting with `eu_bus`, and the links between them
    """
    buses = [eu_bus]
    frontier = [eu_bus]
    for _ in range(max_depth):
        feeding = n.links[
            n.links["bus1"].isin(frontier) & n.links["bus0"].isin(eu_buses)
        ]
        upstream = [b for b in feeding["bus0"].unique() if b not in buses]
        if not upstream:
            break
        buses += upstream
        frontier = upstream

    links = n.links.index[n.links["bus0"].isin(buses) & n.links["bus1"].isin(buses)]

    return buses, links


def bus_attrs(n: pypsa.Network, bus: str) -> dict:
    """
    Read the attributes a national copy of a bus inherits.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    bus : str
        Name of the bus to copy from

    Returns
    -------
    dict
        Carrier, unit and coordinates of the bus
    """
    source = n.buses.loc[bus]

    return {attr: source[attr] for attr in ("carrier", "unit", "x", "y")}


def dangling_buses(n: pypsa.Network, candidates: dict) -> list[str]:
    """
    Find bus names that components refer to but that do not exist in the network.

    A planning horizon starts from a freshly pooled network, while the assets carried
    over from the previous one already point at their national buses.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    candidates : dict
        Bus names to look for

    Returns
    -------
    list[str]
        The referenced names that are missing
    """
    referenced = set()
    for c in ("Link", "Generator", "Store", "StorageUnit"):
        static = n.components[c].static
        cols = [col for col in static.columns if col.startswith("bus")]
        if static.empty or not cols:
            continue
        referenced |= set(pd.unique(static[cols].to_numpy().ravel()))

    return sorted(referenced & set(candidates) - set(n.buses.index))


def copy_to_country(
    n: pypsa.Network,
    component: str,
    names: pd.Index,
    country: str,
    bus_map: dict[str, str],
    share: float = 1.0,
    overrides: dict | None = None,
) -> None:
    """
    Copy EU-wide components to a country, pointing them at its own buses.

    Components that already exist, because a previous planning horizon carried them
    over, are left untouched.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized, modified in place
    component : str
        Name of the component class
    names : pd.Index
        Components to copy
    country : str
        Country to copy them to
    bus_map : dict[str, str]
        Where each EU-wide bus of the chain sits in that country
    share : float
        Fraction of already built capacity the country gets, so that capacity is
        split rather than duplicated
    overrides : dict | None
        Attributes to set on the copies
    """
    static = n.components[component].static.reindex(names)
    if static.empty:
        return

    copies = static.copy()
    for col in [c for c in copies.columns if c.startswith("bus")]:
        copies[col] = copies[col].map(bus_map).fillna(copies[col])
    for col, value in (overrides or {}).items():
        copies[col] = value

    # Capacity already built for the whole pool is split, not handed to everyone
    for attr in ("p_nom", "e_nom"):
        if f"{attr}_extendable" in copies.columns:
            copies.loc[~copies[f"{attr}_extendable"], attr] *= share

    copies.index = pd.Index([national_name(name, country) for name in copies.index])

    copies = copies[~copies.index.isin(n.components[component].static.index)]
    if not copies.empty:
        n.add(component, copies.index, **copies)


def regionalise_eu_buses(
    n: pypsa.Network,
    carriers: list[str],
) -> None:
    """
    Give every country its own copy of the EU-wide commodity chains.

    PyPSA-Eur pools commodities like oil or coal on a single EU-wide bus that every
    consumer draws from directly, and buys the primary energy there on behalf of the
    whole model. Its cost then belongs to no country, which leaves it outside every
    region of the mga cost split.

    Each country therefore gets its own supply chain: the primary energy generator,
    the conversion links above it and the commodity store, all pointing at a national
    commodity bus. Supply is unlimited at a fixed price and carries no capital cost,
    so replicating it changes no cost; it only moves each euro to the country that
    spends it, where `country_grouper` can find it.

    The EU-wide bus is kept as a pure trade hub, reached over one free bidirectional
    interface link per country, for the commodities that some country produces:
    synthetic oil, methanol and ammonia have to be able to cross a border. For the
    rest, coal, lignite and uranium, the only source is the supply every country now
    has, so the pool is dropped rather than left as a way to obscure an import.

    Idempotent, so it can run again on a network that carries the national buses
    over from a previous planning horizon.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized, modified in place
    carriers : list[str]
        Carriers of the EU-wide commodity buses to nationalise
    """
    commodity_buses = n.buses[
        (n.buses["location"] == "EU") & n.buses["carrier"].isin(carriers)
    ]
    missing = set(carriers) - set(commodity_buses["carrier"])
    if missing:
        logger.warning(f"No EU-wide bus for carrier(s) {sorted(missing)}, skipping.")

    # co2 is accounted for globally and never part of a commodity chain
    eu_buses = n.buses.index[
        (n.buses["location"] == "EU") & ~n.buses.index.str.contains("co2|atmosphere")
    ]
    bus_cols = [col for col in n.links.columns if col.startswith("bus")]
    countries = [c for c in n.buses["country"].unique() if c]

    for eu_bus, bus in commodity_buses.iterrows():
        interface = f"{bus.carrier} interface"
        if interface not in n.carriers.index:
            n.add("Carrier", interface, nice_name=interface)

        pool_buses, pool_links = pool_supply(n, eu_bus, eu_buses)

        # A previous horizon carries its assets over already nationalised, but not the
        # buses they point at. Create those first so the references resolve again.
        national = {national_name(b, c): (b, c) for b in pool_buses for c in countries}
        for missing_bus in dangling_buses(n, national):
            source, name = national[missing_bus]
            n.add(
                "Bus", missing_bus, country=name, location=name, **bus_attrs(n, source)
            )

        # Links on the pool, excluding its own supply chain and the interfaces of an
        # earlier call
        on_pool = n.links[bus_cols].isin([eu_bus]).any(axis="columns")
        connected = n.links.index[on_pool & (n.links["carrier"] != interface)]
        connected = connected.difference(pool_links)
        country = country_grouper(n, "Link")[connected]

        # Keep the pool as a trade hub only where a country delivers into it
        out_ports = [col for col in bus_cols if col != "bus0"]
        delivers = n.links.loc[connected, out_ports].isin([eu_bus]).any(axis="columns")
        traded = bool((delivers & (country != "")).any())

        # Links without any national port are the pooled supply, they are copied below
        country = country[country != ""]
        groups = dict(list(country.groupby(country)))

        # A country may only have assets carried over, without a fresh link on the pool
        present = sorted(
            set(groups)
            | {c for c in countries if national_name(eu_bus, c) in n.buses.index}
        )
        share = 1 / len(present) if present else 1.0

        for name in present:
            group = groups.get(name, pd.Series(dtype=object))
            national_bus = national_name(eu_bus, name)
            if national_bus not in n.buses.index:
                n.add(
                    "Bus",
                    national_bus,
                    country=name,
                    location=name,
                    **bus_attrs(n, eu_bus),
                )

            if traded:
                link = f"{national_bus} interface"
                if link not in n.links.index:
                    n.add(
                        "Link",
                        link,
                        bus0=eu_bus,
                        bus1=national_bus,
                        carrier=interface,
                        p_nom=INTERFACE_P_NOM,
                        p_min_pu=-1,
                        lifetime=np.inf,
                    )

            # Move the national side of every connected link onto the national bus
            ports = n.links.loc[group.index, bus_cols]
            n.links.loc[group.index, bus_cols] = ports.mask(
                ports == eu_bus, national_bus
            )

            # Give the country the supply chain that used to serve the whole model
            bus_map = {b: national_name(b, name) for b in pool_buses}
            copy_to_country(
                n,
                "Bus",
                pd.Index(pool_buses[1:]),
                name,
                bus_map,
                overrides={"country": name, "location": name},
            )
            for c in ("Generator", "Store"):
                static = n.components[c].static
                on_pool = static.index[static["bus"].isin(pool_buses)]
                copy_to_country(n, c, on_pool, name, bus_map, share)
            copy_to_country(n, "Link", pool_links, name, bus_map, share)

        # Drop the pooled supply now that every country has its own
        n.remove("Link", pool_links)
        for c in ("Generator", "Store"):
            static = n.components[c].static
            n.remove(c, static.index[static["bus"].isin(pool_buses)])
        n.remove("Bus", pool_buses[1:] if traded else pool_buses)

        logger.info(
            f"{eu_bus}: nationalised over {country.nunique()} country/-ies, "
            f"{'kept as trade hub' if traded else 'pool dropped'}."
        )

    # The interface links leave pypsa-eur's own link columns unset, and `reversed`
    # is used as a boolean mask in add_lossy_bidirectional_link_constraints
    sanitize_custom_columns(n)

    stranded = stranded_costs(n)
    if stranded:
        logger.warning(
            f"Cost-bearing assets on a bus without a country, their cost falls "
            f"outside every region of the mga split: {stranded}"
        )


def stranded_costs(n: pypsa.Network) -> list[str]:
    """
    Find cost-bearing components that belong to no country.

    Their cost is left out of the regional mga cost split, so this should stay empty
    once the EU-wide commodity chains are nationalised.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized

    Returns
    -------
    list[str]
        Names of the components
    """
    stranded = []
    for c in ("Generator", "Link", "Store", "StorageUnit"):
        static = n.components[c].static
        if static.empty:
            continue
        costly = (static["capital_cost"] != 0) | (static["marginal_cost"] != 0)
        stranded += static.index[(country_grouper(n, c) == "") & costly].to_list()

    return stranded


def split_df_by_region(
    df: pd.DataFrame, region: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split statistics grouped by country into their in- and out-of-region part.

    Parameters
    ----------
    df : pd.DataFrame
        Values with a "country" index level, as returned by the `n.statistics`
        accessor with `groupby=country_grouper`
    region : list[str]
        Countries being part of region

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Values inside, values outside of region
    """
    in_region = df.index.get_level_values("country").isin(region)

    return df.loc[in_region], df.loc[~in_region]


def split_expression_by_region(
    expr: LinearExpression, region: list[str]
) -> tuple[LinearExpression, LinearExpression]:
    """
    Split an expression grouped by country into its in- and out-of-region part.

    Uses the same country assignment as `split_df_by_region` does for the
    corresponding `n.statistics` values.

    Parameters
    ----------
    expr : LinearExpression
        Expression with a ("component", "country") group index, as returned by
        the `n.optimize.expressions` accessor with `groupby=country_grouper`
    region : list[str]
        Countries being part of region

    Returns
    -------
    tuple[LinearExpression, LinearExpression]
        Expression inside, expression outside of region
    """
    groups = expr.indexes["group"]
    in_region = groups.get_level_values("country").isin(region)

    return (
        expr.sel(group=groups[in_region]).sum(),
        expr.sel(group=groups[~in_region]).sum(),
    )


def prepare_mga_regional(
    n: pypsa.Network,
    snakemake,
) -> None:
    """
    Replace everything outside the region by the cost-optimal network.

    Only the region is free to deviate from the cost optimum; outside of it the
    cost-optimal capacities are fixed and may merely be redispatched.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized, modified in place
    snakemake
    """
    region = snakemake.params.mga["region"]
    buses_in, buses_out = get_buses_of_regions(
        n=n, region=region, eu_assignment="in_region"
    )

    n_opt = pypsa.Network(snakemake.input.network_opt)
    if set(n.buses.index) != set(n_opt.buses.index):
        raise IndexError(
            "The buses of the cost optimized network and the network for mga differ: "
            f"{sorted(set(n.buses.index) ^ set(n_opt.buses.index))}"
        )

    # Carriers and global constraints are network-wide, so keep the ones of `n`;
    # sub networks are rebuilt from the topology anyway
    components_to_skip = set(n.standard_type_components) | {
        "Carrier",
        "GlobalConstraint",
        "SubNetwork",
    }

    # Components are attributed like their costs are, so that the part of the network
    # taken from `n_opt` matches the part of the cost bound taken from it. Assets on an
    # EU-wide bus belong to no country and are shared, so they stay with `n`. Buses have
    # no country of their own and are split directly. Assigned up front because the
    # removals below drop the buses the country of a component is read from.
    names = [c.name for c in n.components if c.name not in components_to_skip]
    partition = {}
    for name in names:
        if name == "Bus":
            partition[name] = (buses_out, buses_in)
            continue
        country, country_opt = country_grouper(n, name), country_grouper(n_opt, name)
        partition[name] = (
            n.components[name].static.index[~country.isin(region) & (country != "")],
            n_opt.components[name].static.index[
                country_opt.isin(region) | (country_opt == "")
            ],
        )

    for name in names:
        comp_opt = n_opt.components[name]

        # Disable extension and fix optimal value
        attrs = [
            col.removesuffix("_extendable")
            for col in comp_opt.static.columns
            if col.endswith("_nom_extendable")
        ]
        comp_opt.static[attrs] = comp_opt.static[[f"{attr}_opt" for attr in attrs]]
        comp_opt.static[[f"{attr}_extendable" for attr in attrs]] = False

        out_region, in_region_opt = partition[name]
        overlap = out_region.intersection(in_region_opt)
        if not overlap.empty:
            raise RuntimeError(
                f"There are {name}s both inside and outside of the region: {overlap.to_list()}"
            )

        n.remove(name, out_region)
        n_opt.remove(name, in_region_opt)

    n.merge(n_opt, components_to_skip=components_to_skip, inplace=True)
