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


def regionalise_eu_buses(
    n: pypsa.Network,
    carriers: list[str],
) -> None:
    """
    Route flows between EU-wide commodity buses and national assets over one bus per country.

    PyPSA-Eur pools commodities like oil or coal on a single EU bus that every
    consumer draws from directly, so each consumer link is its own border crossing
    and its flow is measured in whatever sits at bus0. Inserting a national bus per
    commodity and country leaves exactly one crossing per commodity, carrying the
    commodity itself in both directions.

    Supply (generators and stores) stays on the EU bus, so the commodity remains a
    European pool. The interface links are free and bidirectional, which makes this
    a pure reformulation: the EU-wide balance decomposes into the national balances
    plus the interface flows, leaving the cost optimum unchanged.

    Idempotent, so it can run again on a network that carries the national buses
    over from a previous planning horizon.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized, modified in place
    carriers : list[str]
        Carriers of the EU-wide commodity buses to route nationally
    """
    eu_buses = n.buses[
        (n.buses["location"] == "EU") & n.buses["carrier"].isin(carriers)
    ]
    missing = set(carriers) - set(eu_buses["carrier"])
    if missing:
        logger.warning(f"No EU-wide bus for carrier(s) {sorted(missing)}, skipping.")

    bus_cols = [col for col in n.links.columns if col.startswith("bus")]
    countries = [c for c in n.buses["country"].unique() if c]

    for eu_bus, bus in eu_buses.iterrows():
        interface = f"{bus.carrier} interface"
        if interface not in n.carriers.index:
            n.add("Carrier", interface, nice_name=interface)

        # A previous horizon carries its links over already routed, but not the
        # buses they point at. Send those back to the EU bus and re-route below.
        national = [f"{country} {bus.carrier}" for country in countries]
        ports = n.links[bus_cols]
        dangling = ports.isin(national) & ~ports.isin(n.buses.index)
        n.links[bus_cols] = ports.mask(dangling, eu_bus)

        # Links on the EU bus, excluding the interfaces of an earlier call
        on_eu_bus = n.links[bus_cols].isin([eu_bus]).any(axis="columns")
        connected = n.links.index[on_eu_bus & (n.links["carrier"] != interface)]

        # Links without any national port stay on the EU bus, they are pooled supply
        country = country_grouper(n, "Link")[connected]
        country = country[country != ""]

        for name, group in country.groupby(country):
            national_bus = f"{name} {bus.carrier}"
            if national_bus not in n.buses.index:
                n.add(
                    "Bus",
                    national_bus,
                    carrier=bus.carrier,
                    unit=bus.unit,
                    country=name,
                    location=name,
                    x=bus.x,
                    y=bus.y,
                )

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

        logger.info(
            f"Routed {len(country)} link(s) on {eu_bus} "
            f"over {country.nunique()} national bus(es)."
        )

    # The interface links leave pypsa-eur's own link columns unset, and `reversed`
    # is used as a boolean mask in add_lossy_bidirectional_link_constraints
    sanitize_custom_columns(n)


def get_pool_prices(n: pypsa.Network, carriers: list[str]) -> dict[str, float]:
    """
    Unit cost of the supply feeding each EU-wide commodity pool.

    Only commodity bought from outside the model carries a cost here: a generator on
    the pool bus, or one upstream of an unattributed conversion link feeding it, as
    for oil which is refined from `EU oil primary`. Commodities that are merely
    produced nationally, like methanol and ammonia, get a price of zero because their
    cost is already charged to the producing country through their production links.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized, with the pools already routed nationally
    carriers : list[str]
        Carriers of the EU-wide commodity buses, see `regionalise_eu_buses`

    Returns
    -------
    dict[str, float]
        Price per carrier
    """
    eu_buses = n.buses[
        (n.buses["location"] == "EU") & n.buses["carrier"].isin(carriers)
    ]
    link_country = country_grouper(n, "Link")

    prices = {}
    for eu_bus, bus in eu_buses.iterrows():
        gens = n.generators[n.generators["bus"] == eu_bus]
        if not gens.empty:
            prices[bus.carrier] = gens["marginal_cost"].mean()
            continue

        # Supply arriving over a link that belongs to no country, priced per unit of
        # commodity delivered to the pool
        price = 0.0
        feeding = n.links[(n.links["bus1"] == eu_bus) & (link_country == "")]
        for _, link in feeding.iterrows():
            upstream = n.generators[n.generators["bus"] == link["bus0"]]
            if not upstream.empty:
                price = (
                    upstream["marginal_cost"].mean() + link["marginal_cost"]
                ) / link["efficiency"]
        prices[bus.carrier] = price

    logger.info(
        f"Commodity pool prices: { {k: round(v, 3) for k, v in prices.items()} }"
    )

    return prices


def interface_price_coefficients(
    n: pypsa.Network, prices: dict[str, float], countries: list[str]
) -> pd.Series:
    """
    Price the interface links of the given countries, everything else zero.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    prices : dict[str, float]
        Price per carrier, see `get_pool_prices`
    countries : list[str]
        Countries whose draw from the pools is to be priced

    Returns
    -------
    pd.Series
        Price per link
    """
    carrier = n.links["carrier"]
    interfaces = carrier.str.endswith(" interface")
    country = country_grouper(n, "Link")

    coeffs = pd.Series(0.0, index=n.links.index)
    selected = interfaces & country.isin(countries)
    coeffs[selected] = (
        carrier[selected].str.removesuffix(" interface").map(prices).fillna(0.0)
    )

    return coeffs


def pool_cost_expression(
    n: pypsa.Network, prices: dict[str, float], countries: list[str]
) -> LinearExpression:
    """
    Build the cost of the commodity the given countries draw from the EU-wide pools.

    The pools are a European commodity market that belongs to no country, so their
    supply cost is attributed to whoever draws it: every country reaches a pool over
    exactly one interface link per commodity, see `regionalise_eu_buses`, and pays
    the pool price for what flows over it. Exporting into a pool yields a credit, so
    domestic production nets against the draw just as it does for the flows.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    prices : dict[str, float]
        Price per carrier, see `get_pool_prices`
    countries : list[str]
        Countries whose draw from the pools is to be priced

    Returns
    -------
    LinearExpression
        Cost of the commodity drawn by the given countries
    """
    p = n.model["Link-p"]
    coeffs = interface_price_coefficients(n, prices, countries)
    expr = p * coeffs.reindex(p.indexes["name"], fill_value=0.0)

    return (expr * n.snapshot_weightings.objective).sum()


def pool_cost_value(
    n: pypsa.Network, prices: dict[str, float], countries: list[str]
) -> float:
    """
    Calculate the counterpart of `pool_cost_expression` for a solved network.

    Parameters
    ----------
    n : pypsa.Network
        Solved network
    prices : dict[str, float]
        Price per carrier, see `get_pool_prices`
    countries : list[str]
        Countries whose draw from the pools is to be priced

    Returns
    -------
    float
        Cost of the commodity drawn by the given countries
    """
    coeffs = interface_price_coefficients(n, prices, countries)
    coeffs = coeffs[coeffs != 0]
    flows = n.links_t.p0.reindex(columns=coeffs.index, fill_value=0.0)

    return float(
        flows.mul(coeffs, axis="columns")
        .mul(n.snapshot_weightings.objective, axis="index")
        .to_numpy()
        .sum()
    )


def get_countries(n: pypsa.Network, region: list[str]) -> tuple[list[str], list[str]]:
    """
    Get the countries in- and outside of the region.

    Parameters
    ----------
    n : pypsa.Network
        Network to be optimized
    region : list[str]
        Countries being part of region

    Returns
    -------
    tuple[list[str], list[str]]
        Countries inside, countries outside of region
    """
    countries = sorted(c for c in n.buses["country"].unique() if c)

    return [c for c in countries if c in region], [
        c for c in countries if c not in region
    ]


def split_df_by_region(
    df: pd.DataFrame, region: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split statistics grouped by country into their in- and out-of-region part.

    Components that belong to no country are the EU-wide commodity pools. They are
    dropped here and attributed to the countries drawing from them instead, see
    `pool_cost_value`.

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
    country = df.index.get_level_values("country")
    in_region = country.isin(region)

    return df.loc[in_region], df.loc[~in_region & (country != "")]


def split_expression_by_region(
    expr: LinearExpression, region: list[str]
) -> tuple[LinearExpression, LinearExpression]:
    """
    Split an expression grouped by country into its in- and out-of-region part.

    Uses the same country assignment as `split_df_by_region` does for the
    corresponding `n.statistics` values, so the EU-wide commodity pools are dropped
    here as well and attributed via `pool_cost_expression` instead.

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
    countries = groups.get_level_values("country")
    in_region = countries.isin(region)
    out_region = ~in_region & (countries != "")
    return (
        expr.sel(group=groups[in_region]).sum(),
        expr.sel(group=groups[out_region]).sum(),
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
