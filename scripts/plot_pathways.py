# SPDX-FileCopyrightText: Peter Kröning
#
# SPDX-License-Identifier: MIT

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
plt.style.use("bmh")


def _load_and_clean(file_path, n_header, n_index):
    df = pd.read_csv(
        file_path, index_col=list(range(n_index)), header=list(range(n_header))
    )

    # Handle empty column values from cost optimal solution
    for i, columns in enumerate(df.columns.levels):
        columns_new = ["" if "Unnamed" in c else c for c in columns.tolist()]
        df = df.rename(columns=dict(zip(columns.tolist(), columns_new)), level=i)

    return df


def _get_levels(df):
    return (
        df.columns.get_level_values(level=0).unique(),  # clusters
        df.columns.get_level_values(level=1).unique(),  # opts
        df.columns.get_level_values(level=2).unique(),  # sector_opts
        df.columns.get_level_values(level=3).unique(),  # planning_horizons
        df.columns.get_level_values(level=4).unique(),  # objectives
        df.columns.get_level_values(level=5).unique(),  # slacks
    )


def _unique_components(df, region):
    """Yield each (component, carrier) once, for rows located in `region`."""
    plotted = set()
    for row in df.index:
        component, location, carrier = row[-3:]
        if location.startswith(region) and (component, carrier) not in plotted:
            plotted.add((component, carrier))
            yield component, carrier


def _row_subplots(n_rows):
    fig, axes = plt.subplots(n_rows, figsize=(10, 5), sharex=True, layout="constrained")
    return fig, np.atleast_1d(axes)


def _finalize(fig, axes, y_max, title, ylabel, save_dir, filename, legend_kwargs=None):
    legend_kwargs = legend_kwargs or {"loc": "center left", "bbox_to_anchor": (1, 0.5)}
    fig.suptitle(title)
    fig.supxlabel("Time")
    fig.supylabel(ylabel)
    for ax in axes.flatten():
        ax.legend(**legend_kwargs)
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.1)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{filename}")
    plt.close(fig)


def plot_capacities(file_path, n_header, region: tuple, save_path):
    df = _load_and_clean(file_path, n_header, n_index=3)

    clusters, opts, sector_opts, planning_horizons, objectives, slacks = _get_levels(df)

    # Plotting
    region_str = ",".join(region)
    prop = file_path.split("nodal_")[1].removesuffix(".csv")
    slack_range = max([float(s) for s in slacks if s])

    # Iterate
    idx = pd.IndexSlice
    for cluster in clusters:
        for opt in opts:
            for sector_opt in sector_opts:
                for component, carrier in _unique_components(df, region):
                    # One figure to compare slacks/objective function in each subfigure
                    # -1 due to the default being plottet everywhere
                    fig_sla, axes_sla = _row_subplots(len(slacks) - 1)
                    fig_obj, axes_obj = _row_subplots(len(objectives) - 1)

                    # Initialize max val for scaling
                    y_max = 0

                    for alt_obj in enumerate(objectives):
                        for slack in enumerate(slacks):
                            # Get optimization type
                            default = bool(not alt_obj[1] and not slack[1])
                            mga = bool(alt_obj[1] and slack[1])

                            if default or mga:
                                # Get x
                                x = planning_horizons

                                # Get y
                                y = df.loc[
                                    idx[component, :, carrier],
                                    idx[
                                        cluster,
                                        opt,
                                        sector_opt,
                                        x,
                                        alt_obj[1],
                                        slack[1],
                                    ],
                                ]
                                y = y[
                                    y.index.get_level_values(
                                        level="location"
                                    ).str.startswith(region)
                                ].sum(axis="index")

                                # Update y_max
                                y_max = max(y_max, y.max())

                                # Plot
                                if default:
                                    for ax in axes_sla.flatten():
                                        ax.plot(
                                            x,
                                            y,
                                            marker="x",
                                            color="black",
                                            label="cost optimal",
                                            zorder=2.01,
                                        )
                                    for ax in axes_obj.flatten():
                                        ax.plot(
                                            x,
                                            y,
                                            marker="x",
                                            color="black",
                                            label="cost optimal",
                                            zorder=2.01,
                                        )
                                elif mga:
                                    axes_sla[slack[0]].set_ylabel(
                                        f"s={float(slack[1]):.0%}"
                                    )
                                    axes_sla[slack[0]].plot(
                                        x, y, marker="x", label=alt_obj[1]
                                    )

                                    # Darker/more opaque and on top the closer the slack is to cost optimal
                                    slack_val = float(slack[1])
                                    color = plt.cm.Oranges(
                                        0.85 - 0.5 * slack_val / slack_range
                                    )
                                    zorder = 2 + 0.01 * 0.9 * (
                                        1 - slack_val / slack_range
                                    )

                                    axes_obj[alt_obj[0]].set_ylabel(alt_obj[1])
                                    axes_obj[alt_obj[0]].fill_between(
                                        x, 0, y, color=color, alpha=0.4, zorder=zorder
                                    )
                                    axes_obj[alt_obj[0]].plot(
                                        x,
                                        y,
                                        marker="x",
                                        color=color,
                                        label=f"s={slack_val:.0%}",
                                        zorder=zorder,
                                    )

                    # Save file
                    filename = f"{cluster}_{opt}_{sector_opt}-{component}_{carrier}_{region_str}.png"
                    for subdir, view_fig, view_axes in [
                        ("comp_obj", fig_sla, axes_sla),
                        ("comp_slack", fig_obj, axes_obj),
                    ]:
                        path = f"{save_path}pathways/{prop}/{subdir}"
                        _finalize(
                            view_fig, view_axes, y_max, region_str, prop, path, filename
                        )


def plot_costs(file_path, n_header, region: tuple, save_path):
    df = _load_and_clean(file_path, n_header, n_index=4)

    clusters, opts, sector_opts, planning_horizons, objectives, slacks = _get_levels(df)

    # Plotting strings
    region_str = ",".join(region)
    prop = file_path.split("nodal_")[1].removesuffix(".csv")
    slack_range = max([float(s) for s in slacks if s])

    # Fixed component order/colors so the same component always gets the same color across stacked figures
    components = list(_unique_components(df, region))
    stacked_labels = [f"{component} {carrier}" for component, carrier in components]
    stacked_colors = plt.cm.tab20(np.linspace(0, 1, len(components)))

    # Iterate
    idx = pd.IndexSlice
    for cluster in clusters:
        for opt in opts:
            for sector_opt in sector_opts:
                # Per-component cost trajectories, keyed by (alt_obj, slack) scenario, for the stacked figures below
                stack_cap = {}
                stack_mar = {}
                stack_tot = {}

                for component, carrier in components:
                    # One figure to compare slacks/objective function in each subfigure
                    # -1 due to the default being plottet everywhere
                    fig_cap_sla, axes_cap_sla = _row_subplots(len(slacks) - 1)
                    fig_mar_sla, axes_mar_sla = _row_subplots(len(slacks) - 1)
                    fig_tot_sla, axes_tot_sla = _row_subplots(len(slacks) - 1)

                    fig_cap_obj, axes_cap_obj = _row_subplots(len(objectives) - 1)
                    fig_mar_obj, axes_mar_obj = _row_subplots(len(objectives) - 1)
                    fig_tot_obj, axes_tot_obj = _row_subplots(len(objectives) - 1)

                    # Initialize max val for scaling
                    y_cap_max = 0
                    y_mar_max = 0
                    y_tot_max = 0

                    for alt_obj in enumerate(objectives):
                        for slack in enumerate(slacks):
                            default = bool(not alt_obj[1] and not slack[1])
                            mga = bool(alt_obj[1] and slack[1])

                            if default or mga:
                                # Get x
                                x = planning_horizons

                                # Get y (try-except because some components might not have these costs)
                                try:
                                    y_cap = df.loc[
                                        idx["capital", component, :, carrier],
                                        idx[
                                            cluster,
                                            opt,
                                            sector_opt,
                                            x,
                                            alt_obj[1],
                                            slack[1],
                                        ],
                                    ]
                                    y_cap = y_cap[
                                        y_cap.index.get_level_values(
                                            level="location"
                                        ).str.startswith(region)
                                    ].sum(axis="index")
                                except KeyError:
                                    y_cap = np.zeros(len(x))

                                try:
                                    y_mar = df.loc[
                                        idx["marginal", component, :, carrier],
                                        idx[
                                            cluster,
                                            opt,
                                            sector_opt,
                                            x,
                                            alt_obj[1],
                                            slack[1],
                                        ],
                                    ]
                                    y_mar = y_mar[
                                        y_mar.index.get_level_values(
                                            level="location"
                                        ).str.startswith(region)
                                    ].sum(axis="index")
                                except KeyError:
                                    y_mar = np.zeros(len(x))
                                y_tot = y_cap + y_mar

                                # Update y_max
                                y_cap_max = max(y_cap_max, y_cap.max())
                                y_mar_max = max(y_mar_max, y_mar.max())
                                y_tot_max = max(y_tot_max, y_tot.max())

                                # Append trajectories for the stacked figures, keyed by scenario
                                scenario_key = (alt_obj[1], slack[1])
                                stack_cap.setdefault(scenario_key, []).append(
                                    np.asarray(y_cap, dtype=float)
                                )
                                stack_mar.setdefault(scenario_key, []).append(
                                    np.asarray(y_mar, dtype=float)
                                )
                                stack_tot.setdefault(scenario_key, []).append(
                                    np.asarray(y_tot, dtype=float)
                                )

                                if default:
                                    for axes, y in [
                                        (axes_cap_sla, y_cap),
                                        (axes_mar_sla, y_mar),
                                        (axes_tot_sla, y_tot),
                                    ]:
                                        for ax in axes.flatten():
                                            ax.plot(
                                                x,
                                                y,
                                                marker="x",
                                                color="black",
                                                label="cost optimal",
                                                zorder=2.01,
                                            )
                                    for axes, y in [
                                        (axes_cap_obj, y_cap),
                                        (axes_mar_obj, y_mar),
                                        (axes_tot_obj, y_tot),
                                    ]:
                                        for ax in axes.flatten():
                                            ax.plot(
                                                x,
                                                y,
                                                marker="x",
                                                color="black",
                                                label="cost optimal",
                                                zorder=2.01,
                                            )
                                elif mga:
                                    for axes, y in [
                                        (axes_cap_sla, y_cap),
                                        (axes_mar_sla, y_mar),
                                        (axes_tot_sla, y_tot),
                                    ]:
                                        axes[slack[0]].set_ylabel(
                                            f"s={float(slack[1]):.0%}"
                                        )
                                        axes[slack[0]].plot(
                                            x, y, marker="x", label=alt_obj[1]
                                        )

                                    # Darker/more opaque and on top the closer the slack is to cost optimal
                                    slack_val = float(slack[1])
                                    color = plt.cm.Oranges(
                                        0.85 - 0.5 * slack_val / slack_range
                                    )
                                    zorder = 2 + 0.01 * 0.9 * (
                                        1 - slack_val / slack_range
                                    )

                                    for axes, y in [
                                        (axes_cap_obj, y_cap),
                                        (axes_mar_obj, y_mar),
                                        (axes_tot_obj, y_tot),
                                    ]:
                                        axes[alt_obj[0]].set_ylabel(alt_obj[1])
                                        axes[alt_obj[0]].fill_between(
                                            x,
                                            0,
                                            y,
                                            color=color,
                                            alpha=0.4,
                                            zorder=zorder,
                                        )
                                        axes[alt_obj[0]].plot(
                                            x,
                                            y,
                                            marker="x",
                                            color=color,
                                            label=f"s={slack_val:.0%}",
                                            zorder=zorder,
                                        )

                    filename = f"{cluster}_{opt}_{sector_opt}-{component}_{carrier}_{region_str}.png"
                    for subdir, fig, axes, y_max, n in [
                        ("comp_obj", fig_cap_sla, axes_cap_sla, y_cap_max, "capital"),
                        ("comp_obj", fig_mar_sla, axes_mar_sla, y_mar_max, "marginal"),
                        ("comp_obj", fig_tot_sla, axes_tot_sla, y_tot_max, "total"),
                        ("comp_slack", fig_cap_obj, axes_cap_obj, y_cap_max, "capital"),
                        (
                            "comp_slack",
                            fig_mar_obj,
                            axes_mar_obj,
                            y_mar_max,
                            "marginal",
                        ),
                        ("comp_slack", fig_tot_obj, axes_tot_obj, y_tot_max, "total"),
                    ]:
                        path = f"{save_path}pathways/{prop}/{n}/{subdir}"
                        _finalize(
                            fig, axes, y_max, region_str, f"{n} {prop}", path, filename
                        )

                # Stacked cost breakdown across all components, one figure per slack x objective scenario
                for alt_obj in enumerate(objectives):
                    for slack in enumerate(slacks):
                        default = bool(not alt_obj[1] and not slack[1])
                        mga = bool(alt_obj[1] and slack[1])

                        if not (default or mga):
                            continue

                        scenario_key = (alt_obj[1], slack[1])
                        y_cap_by_comp = stack_cap[scenario_key]
                        y_mar_by_comp = stack_mar[scenario_key]
                        y_tot_by_comp = stack_tot[scenario_key]

                        scenario_label = alt_obj[1] if alt_obj[1] else "cost-optimal"
                        slack_label = f"s{slack[1]}" if slack[1] else "s0"
                        title = f"{region_str} - {scenario_label} {slack_label}"
                        filename = f"{cluster}_{opt}_{sector_opt}-{scenario_label}_{slack_label}_{region_str}.png"

                        x = planning_horizons
                        for name, ys in [
                            ("capital", y_cap_by_comp),
                            ("marginal", y_mar_by_comp),
                            ("total", y_tot_by_comp),
                        ]:
                            fig, ax = plt.subplots(
                                figsize=(16, 9), layout="constrained"
                            )
                            ax.stackplot(
                                x, *ys, labels=stacked_labels, colors=stacked_colors
                            )
                            y_max = np.sum(ys, axis=0).max()
                            path = f"{save_path}pathways/{prop}/{name}/stacked"
                            legend_kwargs = {
                                "loc": "upper center",
                                "bbox_to_anchor": (0.5, -0.15),
                                "ncol": 6,
                                "fontsize": "small",
                            }
                            _finalize(
                                fig,
                                np.atleast_1d(ax),
                                y_max,
                                title,
                                f"{name} {prop}",
                                path,
                                filename,
                                legend_kwargs=legend_kwargs,
                            )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_pathways",
            configfiles="config/test/config.myopic-mga.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Currently, breaks for default (intended)
    region = tuple(snakemake.params.mga.get("region", None))

    n_header = 6

    save_path = snakemake.params.save_path

    plot_capacities(
        file_path=snakemake.input.nodal_capacities,
        n_header=n_header,
        region=region,
        save_path=save_path,
    )

    plot_costs(
        file_path=snakemake.input.nodal_costs,
        n_header=n_header,
        region=region,
        save_path=save_path,
    )
