# SPDX-FileCopyrightText: Peter Kröning
#
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
plt.style.use("bmh")

SCENARIO_LEVELS = ["cluster", "opt", "sector_opt", "alternative_objectives", "slack"]
DEFAULT_LEGEND = {"loc": "center left", "bbox_to_anchor": (1, 0.5)}
STACKED_LEGEND = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, -0.15),
    "ncol": 6,
    "fontsize": "small",
}


def _clean_columns(columns: pd.MultiIndex) -> pd.MultiIndex:
    """Empty header cells of the cost optimal solution are read as 'Unnamed: ...'."""
    return pd.MultiIndex.from_tuples(
        [
            tuple("" if "Unnamed" in level else level for level in col)
            for col in columns
        ],
        names=columns.names,
    )


def _region_totals(
    file_path: str, n_header: int, n_index: int, region: tuple, levels: list[str]
) -> pd.DataFrame:
    """Sum the rows located in `region` per `levels`."""
    df = pd.read_csv(
        file_path,
        index_col=list(range(n_index)),
        header=list(range(n_header)),
    )
    totals = (
        df[df.index.get_level_values("location").str.startswith(region)]
        .groupby(level=levels, sort=False)
        .sum()
    )
    totals.columns = _clean_columns(totals.columns)

    # Technologies absent from the region would only yield flat zero lines
    return totals[(totals != 0).any(axis="columns")]


def _by_scenario(
    totals: pd.DataFrame, horizons: list[str]
) -> dict[tuple, pd.DataFrame]:
    """Split the columns into one technology x horizon frame per scenario."""
    return {
        key: sub.droplevel(SCENARIO_LEVELS).reindex(horizons).T
        for key, sub in totals.T.groupby(level=SCENARIO_LEVELS, sort=False)
    }


def _row_subplots(n_rows: int):
    fig, axes = plt.subplots(n_rows, figsize=(10, 5), sharex=True, layout="constrained")
    return fig, np.atleast_1d(axes)


def _finalize(
    fig, axes, y_max, title, ylabel, save_dir, filename, legend=DEFAULT_LEGEND
):
    fig.suptitle(title)
    fig.supxlabel("Time")
    fig.supylabel(ylabel)
    for ax in axes:
        ax.legend(**legend)
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.1)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / filename)
    plt.close(fig)


def _plot_technology(
    x,
    y_default,
    y_mga,
    objectives,
    slacks,
    slack_range,
    title,
    ylabel,
    save_dir,
    filename,
):
    """One figure comparing objectives per slack, one comparing slacks per objective."""
    fig_obj, axes_obj = _row_subplots(len(slacks))
    fig_slack, axes_slack = _row_subplots(len(objectives))

    y_max = max(
        (y.max() for y in (*y_mga.values(), y_default) if y is not None), default=0
    )

    for ax, slack in zip(axes_obj, slacks):
        ax.set_ylabel(f"s={float(slack):.0%}")
    for ax, objective in zip(axes_slack, objectives):
        ax.set_ylabel(objective)

    for (objective, slack), y in y_mga.items():
        axes_obj[slacks.index(slack)].plot(x, y, marker="x", label=objective)

        # Darker/more opaque and on top the closer the slack is to cost optimal
        share = float(slack) / slack_range
        color = plt.cm.Oranges(0.85 - 0.5 * share)
        zorder = 2 + 0.009 * (1 - share)

        ax = axes_slack[objectives.index(objective)]
        ax.fill_between(x, 0, y, color=color, alpha=0.4, zorder=zorder)
        ax.plot(
            x,
            y,
            marker="x",
            color=color,
            label=f"s={float(slack):.0%}",
            zorder=zorder,
        )

    if y_default is not None:
        for ax in (*axes_obj, *axes_slack):
            ax.plot(
                x,
                y_default,
                marker="x",
                color="black",
                label="cost optimal",
                zorder=2.01,
            )

    _finalize(fig_obj, axes_obj, y_max, title, ylabel, f"{save_dir}/comp_obj", filename)
    _finalize(
        fig_slack, axes_slack, y_max, title, ylabel, f"{save_dir}/comp_slack", filename
    )


def _plot_stacked(x, frame, labels, colors, title, ylabel, save_dir, filename):
    """Stacked breakdown over all technologies of a single scenario."""
    values = frame.to_numpy()
    fig, ax = plt.subplots(figsize=(16, 9), layout="constrained")
    ax.stackplot(x, values, labels=labels, colors=colors)
    y_max = values.sum(axis=0).max() if len(values) else 0
    _finalize(
        fig, [ax], y_max, title, ylabel, save_dir, filename, legend=STACKED_LEGEND
    )


def _plot_pathways(
    quantities: dict[str, pd.DataFrame],
    prop: str,
    region_str: str,
    save_path: str,
    stacked: bool = False,
):
    """Plot the near optimal pathways of every quantity, keyed by its subdirectory."""
    columns = next(iter(quantities.values())).columns
    runs = columns.droplevel(
        ["planning_horizon", "alternative_objectives", "slack"]
    ).unique()
    horizons = list(columns.get_level_values("planning_horizon").unique())
    objectives = [
        o for o in columns.get_level_values("alternative_objectives").unique() if o
    ]
    slacks = [s for s in columns.get_level_values("slack").unique() if s]

    if not objectives or not slacks:
        logger.warning(f"No near optimal solutions found for {prop}, skipping plots.")
        return
    slack_range = max(float(s) for s in slacks)

    for label, totals in quantities.items():
        scenarios = _by_scenario(totals, horizons)
        save_dir = f"{save_path}pathways/" + "/".join(filter(None, (prop, label)))
        ylabel = " ".join(filter(None, (label, prop)))
        stack_labels = [" ".join(row) for row in totals.index]
        stack_colors = plt.cm.tab20(np.linspace(0, 1, len(totals)))

        for run in runs:
            run_str = "_".join(run)
            default = scenarios.get((*run, "", ""))
            mga = {
                (objective, slack): scenarios[(*run, objective, slack)]
                for objective in objectives
                for slack in slacks
                if (*run, objective, slack) in scenarios
            }

            y_default = None if default is None else default.to_numpy()
            y_mga = {key: frame.to_numpy() for key, frame in mga.items()}

            for i, row in enumerate(totals.index):
                _plot_technology(
                    horizons,
                    None if y_default is None else y_default[i],
                    {key: y[i] for key, y in y_mga.items()},
                    objectives,
                    slacks,
                    slack_range,
                    region_str,
                    ylabel,
                    save_dir,
                    f"{run_str}-{'_'.join(row)}_{region_str}.png",
                )

            if not stacked:
                continue

            stacks = {} if default is None else {("cost-optimal", "s0"): default}
            stacks.update(
                {
                    (objective, f"s{slack}"): frame
                    for (objective, slack), frame in mga.items()
                }
            )
            for (scenario, slack_label), frame in stacks.items():
                _plot_stacked(
                    horizons,
                    frame,
                    stack_labels,
                    stack_colors,
                    f"{region_str} - {scenario} {slack_label}",
                    ylabel,
                    f"{save_dir}/stacked",
                    f"{run_str}-{scenario}_{slack_label}_{region_str}.png",
                )


def plot_capacities(file_path: str, n_header: int, region: tuple, save_path: str):
    totals = _region_totals(
        file_path, n_header, 3, region, levels=["component", "carrier"]
    )
    _plot_pathways({"": totals}, "capacities", ",".join(region), save_path)


def plot_costs(file_path: str, n_header: int, region: tuple, save_path: str):
    totals = _region_totals(
        file_path, n_header, 4, region, levels=["cost", "component", "carrier"]
    )

    # Not every technology has both cost types
    rows = totals.droplevel("cost").index.unique()
    capital, marginal = (
        totals[totals.index.get_level_values("cost") == cost]
        .droplevel("cost")
        .reindex(rows, fill_value=0.0)
        for cost in ("capital", "marginal")
    )

    _plot_pathways(
        {"capital": capital, "marginal": marginal, "total": capital + marginal},
        "costs",
        ",".join(region),
        save_path,
        stacked=True,
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
