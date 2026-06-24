# -*- coding: utf-8 -*-
"""
NEE Flux Analysis - plotting the full dataset predictions and SHAP values
Uses full dataset prediction file from stage 2 (NEE_HPT_Za*.py)
@author: Isabella Rosenberg Jørgensen
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SITE_IDS = ["ZaF", "ZaH"]


def _top_shap_cols(df, shap_cols, top_n=None):
	if top_n is None:
		return list(shap_cols)

	filtered = [c for c in shap_cols if c.replace("SHAP_", "") not in {"sin_doy", "cos_doy"}]
	mean_abs = df[filtered].abs().mean().sort_values(ascending=False)
	return list(mean_abs.index[:top_n])


def _feature_color_map(feature_cols):
	color_map = {}
	for col in feature_cols:
		feature = col.replace("SHAP_", "")
		if feature.startswith("TA"):
			color_map[col] = "#d73027"
		elif feature.startswith("TS"):
			color_map[col] = "#b35806"
		elif feature.startswith("SWC"):
			color_map[col] = "#4575b4"
		elif feature.startswith("RG"):
			color_map[col] = "#fdae61"
		elif feature.startswith("P_lag") or feature == "precipitation_rate":
			color_map[col] = "#1b9e77"
		elif "Snow" in feature or feature in ["DSSM", "D_SNOW"]:
			color_map[col] = "#80b1d3"
		elif feature in ["VPD_f", "RH"]:
			color_map[col] = "#984ea3"
		elif "NDVI" in feature:
			color_map[col] = "#4daf4a"
		elif "sin_doy" in feature:
			color_map[col] = "#999999"
		elif "cos_doy" in feature:
			color_map[col] = "#666666"
		else:
			color_map[col] = "#6c6c6c"
	return color_map


def plot_full_dataset_observed_predicted_line(df_full, plots_dir, site_id):
	series_df = df_full[["Date", "Observed_NEE", "Predicted_NEE"]].dropna().sort_values("Date")

	fig, ax = plt.subplots(figsize=(13, 6))
	ax.plot(series_df["Date"], series_df["Observed_NEE"], color="black", linewidth=1.0, label="Observed NEE")
	ax.plot(series_df["Date"], series_df["Predicted_NEE"], color="tab:blue", linewidth=1.0, alpha=0.9, label="Predicted NEE")
	ax.set_xlabel("Date")
	ax.set_ylabel("NEE")
	ax.set_title(f"Full Dataset ({site_id}): Observed vs Predicted NEE")
	ax.grid(True, alpha=0.25)
	ax.legend()

	plt.tight_layout()
	fig.savefig(plots_dir / f"plot_full_dataset_observed_predicted_line_{site_id}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_weekly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, site_ids, top_n=None):
	data_by_site = {}
	all_cols = set()
	master_periods = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		tmp = df.copy()
		tmp["WeekPeriod"] = tmp["Date"].dt.to_period("W")
		weekly_shap = tmp.groupby("WeekPeriod")[shap_cols].mean().sort_index()

		nee_col = "Predicted_NEE" if "Predicted_NEE" in tmp.columns else "Observed_NEE"
		weekly_nee = tmp.groupby("WeekPeriod")[nee_col].mean().sort_index()

		valid_periods = [
			p for p in weekly_shap.index
			if p.start_time.month in [6, 7, 8, 9, 10]
		]

		weekly_shap = weekly_shap.loc[valid_periods]
		weekly_nee = weekly_nee.loc[valid_periods]

		master_periods.update(weekly_shap.index.tolist())

		data_by_site[site] = {"shap": weekly_shap, "nee": weekly_nee}

	all_cols = sorted(all_cols)
	master_periods = sorted(master_periods)
	if not master_periods:
		raise ValueError("No weekly periods found for full-dataset continuous plot.")

	bar_step = 1.03
	bar_width = 0.78

	master_periods = [
		p for p in master_periods
		if p.start_time.month in [6, 7, 8, 9, 10]
	]

	x_vals = np.arange(len(master_periods)) * bar_step
	x_index = pd.Index(master_periods)

	x_years = np.array([period.year for period in master_periods])
	x_months = np.array([period.start_time.month for period in master_periods])

	month_tick_positions = []
	month_tick_labels = []

	unique_months = sorted(
		{(p.year, p.start_time.month) for p in master_periods}
	)

	for year, month in unique_months:

		if month not in [6, 7, 8, 9, 10]:
			continue

		indices = [
			i for i, p in enumerate(master_periods)
			if p.year == year and p.start_time.month == month
		]

		if indices:
			midpoint = np.mean(indices) * bar_step
			month_tick_positions.append(midpoint)

			month_tick_labels.append(
				pd.Timestamp(year=year, month=month, day=1).strftime("%b")
			)

	year_boundaries = [
		(index - 0.5) * bar_step
		for index in range(1, len(master_periods))
		if x_years[index] != x_years[index - 1]
	]

	year_label_positions = []

	for year in sorted(set(x_years.tolist())):
		year_positions = np.where(x_years == year)[0]

		if len(year_positions):
			year_label_positions.append(
				(year_positions.mean() * bar_step, str(year))
			)

	color_map = _feature_color_map(all_cols)

	fig, axes = plt.subplots(len(site_ids), 1, figsize=(10, 6.5), sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	for ax, site in zip(axes, site_ids):
		shap_vals = data_by_site[site]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(x_index).fillna(0.0)
		nee_vals = data_by_site[site]["nee"].reindex(x_index)

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			values = shap_vals[col].values
			pos = np.clip(values, 0, None)
			neg = np.clip(values, None, 0)
			ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[col])
			ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[col])
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		ax.plot(x_vals, nee_vals.values, color="black", linewidth=1.2)
		ax.axhline(y=0, color="black", linewidth=1.0, alpha=0.9)
		ax.set_title(f"{site}", y=1.10, pad=1)
		ax.set_ylabel("NEE / Mean SHAP")
		ax.grid(True, axis="y", alpha=0.2)

		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		nee_clean = nee_vals.dropna()
		y_max = max(pos_sum.max() if len(pos_sum) else 0.0, nee_clean.max() if len(nee_clean) else 0.0) * 1.05
		y_min = min(neg_sum.min() if len(neg_sum) else 0.0, nee_clean.min() if len(nee_clean) else 0.0, 0.0) * 1.05
		if np.isclose(y_min, y_max):
			buffer = 1.0 if np.isclose(y_min, 0.0) else max(1e-6, abs(y_min) * 0.05)
			y_min -= buffer
			y_max += buffer
		ax.set_ylim(y_min, y_max)
		ax.set_xlim(-0.5 * bar_step, (len(x_vals) - 0.5) * bar_step)
		for boundary in year_boundaries:
			ax.axvline(x=boundary, color="black", linewidth=0.8, alpha=0.6)

	axes[-1].set_xticks(month_tick_positions)
	axes[-1].set_xticklabels(
		month_tick_labels,
		rotation=0,
		ha="center",
		fontsize=10
	)
	for x_pos, year_label in year_label_positions:
		axes[0].text(x_pos, 1.03, year_label, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [col.replace("SHAP_", "") for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")
	fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(0.93, 0.5), fontsize=8)

	title_suffix = f"Top {top_n} Features" if top_n is not None else "All Features"
	fig.suptitle(f"Weekly SHAP with NEE, Full Dataset ({title_suffix})", y=0.98)
	plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	filename = "plot_weekly_stacked_shap_two_sites_full_dataset_continuous.png" if top_n is None else f"plot_weekly_stacked_shap_two_sites_full_dataset_continuous_top{top_n}.png"
	fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
	plt.close(fig)


def main():
	base_dir = Path(__file__).resolve().parent
	plots_dir = base_dir / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)

	for site_id in SITE_IDS:
		full_dataset_path = base_dir / "output" / site_id / f"NEE_HPT_{site_id}" / "full_dataset_predictions_shap_values.csv"
		if not full_dataset_path.exists():
			raise FileNotFoundError(f"Missing full-dataset SHAP + prediction file: {full_dataset_path}")
		df_full = pd.read_csv(full_dataset_path, parse_dates=["Date"])
		plot_full_dataset_observed_predicted_line(df_full, plots_dir, site_id)

	plot_weekly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS)
	plot_weekly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, top_n=8)


if __name__ == "__main__":
	main()