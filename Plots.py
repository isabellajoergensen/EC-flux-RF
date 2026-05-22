# Loads per-row SHAP values from output/<site_id>/shap_values.csv.
# Creates multiple plot types summarizing SHAP importance across years/months.
# Always runs for both ZaF and ZaH.

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SITE_IDS = ["ZaF", "ZaH"]

def _mean_abs_by_group(df, group_col, shap_cols):
	grouped = df.groupby(group_col)[shap_cols].apply(lambda x: x.abs().mean())
	return grouped


def _top_shap_cols(df, shap_cols, exclude=None, top_n=10):
	exclude = exclude or set()
	filtered = [c for c in shap_cols if c.replace("SHAP_", "") not in exclude]
	mean_abs = df[filtered].abs().mean().sort_values(ascending=False)
	return list(mean_abs.index[:top_n]), mean_abs


def plot_yearly_mean_abs_bar(df, shap_cols, plots_dir, top_n=5):
	top_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)
	yearly = _mean_abs_by_group(df, "Year", top_cols)
	years = yearly.index.tolist()

	fig, ax = plt.subplots(figsize=(12, 6))
	width = 0.8 / len(top_cols)
	x = np.arange(len(years))

	for i, col in enumerate(top_cols):
		ax.bar(x + i * width, yearly[col].values, width=width, label=col.replace("SHAP_", ""))

	ax.set_xticks(x + width * (len(top_cols) - 1) / 2)
	ax.set_xticklabels(years, rotation=0)
	ax.set_ylabel("Mean |SHAP|")
	ax.set_title("Yearly Mean |SHAP| for Top Features")
	ax.legend(ncol=2, fontsize=9)
	ax.grid(True, axis="y", alpha=0.3)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_yearly_mean_abs_shap_bar.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_year_feature_heatmap(df, shap_cols, plots_dir, top_n=10):
	top_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)
	yearly = _mean_abs_by_group(df, "Year", top_cols)

	fig, ax = plt.subplots(figsize=(10, 6))
	im = ax.imshow(yearly.T.values, aspect="auto", cmap="viridis")

	ax.set_yticks(np.arange(len(top_cols)))
	ax.set_yticklabels([c.replace("SHAP_", "") for c in top_cols])
	ax.set_xticks(np.arange(len(yearly.index)))
	ax.set_xticklabels(yearly.index.tolist())
	ax.set_xlabel("Year")
	ax.set_title("Mean |SHAP| by Year and Feature")

	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("Mean |SHAP|")

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_year_feature_heatmap.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_month_feature_heatmap(df, shap_cols, plots_dir, top_n=10):
	top_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)
	monthly = _mean_abs_by_group(df, "Month", top_cols)
	months = list(range(1, 13))
	monthly = monthly.reindex(months)

	fig, ax = plt.subplots(figsize=(10, 6))
	im = ax.imshow(monthly.T.values, aspect="auto", cmap="magma")

	ax.set_yticks(np.arange(len(top_cols)))
	ax.set_yticklabels([c.replace("SHAP_", "") for c in top_cols])
	ax.set_xticks(np.arange(len(months)))
	ax.set_xticklabels(months)
	ax.set_xlabel("Month")
	ax.set_title("Mean |SHAP| by Month and Feature")

	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("Mean |SHAP|")

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_month_feature_heatmap.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_yearly_beeswarm(df, shap_cols, plots_dir, top_n=6):
	top_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)
	years = sorted(df["Year"].unique())
	ncols = min(3, len(years))
	nrows = int(math.ceil(len(years) / ncols))

	fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
	axes = np.array(axes).reshape(-1)

	for ax, year in zip(axes, years):
		subset = df[df["Year"] == year]
		ax.axvline(0, color="black", lw=1, alpha=0.6)

		for i, col in enumerate(top_cols):
			vals = subset[col].values
			jitter = np.random.uniform(-0.25, 0.25, size=len(vals))
			y = i + jitter
			ax.scatter(vals, y, s=6, alpha=0.3)

		ax.set_yticks(range(len(top_cols)))
		ax.set_yticklabels([c.replace("SHAP_", "") for c in top_cols])
		ax.set_title(f"Year {year}")
		ax.set_xlabel("SHAP value")
		ax.grid(True, axis="x", alpha=0.2)

	for ax in axes[len(years):]:
		ax.axis("off")

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_yearly_beeswarm.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_monthly_violin_top_feature(df, shap_cols, plots_dir):
	top_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=1)
	top_col = top_cols[0]
	data = []
	positions = []
	for m in range(1, 13):
		vals = df[df["Month"] == m][top_col].values
		vals = vals[np.isfinite(vals)]
		if len(vals) >= 2:
			data.append(vals)
			positions.append(m)

	fig, ax = plt.subplots(figsize=(10, 5))
	if data:
		ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
	ax.set_xticks(np.arange(1, 13))
	ax.set_xticklabels(range(1, 13))
	ax.set_xlabel("Month")
	ax.set_ylabel("SHAP value")
	ax.set_title(f"Monthly SHAP Distribution for {top_col.replace('SHAP_', '')}")
	ax.grid(True, axis="y", alpha=0.3)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_monthly_violin_top_feature.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_monthly_stack_area(df, shap_cols, plots_dir, top_n=6):
	top_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)
	df = df.copy()
	df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()

	monthly = df.groupby("YearMonth")[top_cols].apply(lambda x: x.abs().mean())
	monthly = monthly.sort_index()

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.stackplot(
		monthly.index,
		[monthly[col].values for col in top_cols],
		labels=[c.replace("SHAP_", "") for c in top_cols],
		alpha=0.8,
	)
	ax.set_xlabel("Month")
	ax.set_ylabel("Mean |SHAP|")
	ax.set_title("Monthly Mean |SHAP| (Stacked Area)")
	ax.legend(loc="upper left", ncol=2, fontsize=8)
	ax.grid(True, axis="y", alpha=0.2)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_monthly_stack_area.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_monthly_mean_sd_ribbons(df, shap_cols, plots_dir, top_n=6):
	top_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)

	monthly = df.groupby("Month")[top_cols].apply(lambda x: x.abs())
	monthly_mean = monthly.groupby(level=0).mean()
	monthly_std = monthly.groupby(level=0).std()

	months = list(range(1, 13))
	monthly_mean = monthly_mean.reindex(months)
	monthly_std = monthly_std.reindex(months)

	fig, ax = plt.subplots(figsize=(12, 6))
	for col in top_cols:
		mean_vals = monthly_mean[col].values
		std_vals = monthly_std[col].values
		ax.plot(months, mean_vals, label=col.replace("SHAP_", ""))
		ax.fill_between(
			months,
			mean_vals - std_vals,
			mean_vals + std_vals,
			alpha=0.2,
		)

	ax.set_xticks(months)
	ax.set_xlabel("Month")
	ax.set_ylabel("Mean |SHAP|")
	ax.set_title("Monthly Mean |SHAP| with SD Ribbons")
	ax.legend(ncol=2, fontsize=8)
	ax.grid(True, axis="y", alpha=0.2)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_monthly_mean_sd_ribbons.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_yearly_mean_sd_bars(df, shap_cols, plots_dir, top_n=6):
	top_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)

	yearly = df.groupby("Year")[top_cols].apply(lambda x: x.abs())
	yearly_mean = yearly.groupby(level=0).mean()
	yearly_std = yearly.groupby(level=0).std()
	years = yearly_mean.index.tolist()

	fig, ax = plt.subplots(figsize=(12, 6))
	width = 0.8 / len(top_cols)
	x = np.arange(len(years))

	for i, col in enumerate(top_cols):
		means = yearly_mean[col].values
		stds = yearly_std[col].values
		ax.bar(
			x + i * width,
			means,
			width=width,
			yerr=stds,
			capsize=3,
			label=col.replace("SHAP_", ""),
		)

	ax.set_xticks(x + width * (len(top_cols) - 1) / 2)
	ax.set_xticklabels(years, rotation=0)
	ax.set_ylabel("Mean |SHAP|")
	ax.set_title("Yearly Mean |SHAP| with SD Error Bars")
	ax.legend(ncol=2, fontsize=8)
	ax.grid(True, axis="y", alpha=0.2)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_yearly_mean_sd_bars.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_weekly_stacked_bars_with_nee(df_shap, df_nee, shap_cols, plots_dir):
	all_cols = list(shap_cols)
	colors = plt.cm.tab20(np.linspace(0, 1, len(all_cols)))
	color_map = {col: colors[i] for i, col in enumerate(all_cols)}

	shap_weekly = df_shap.copy()
	shap_weekly["Week"] = shap_weekly["Date"].dt.to_period("W").dt.start_time
	weekly_shap = shap_weekly.groupby("Week")[all_cols].mean()
	weekly_shap = weekly_shap.sort_index()

	nee_weekly = df_nee.copy()
	nee_weekly["Week"] = nee_weekly["Date"].dt.to_period("W").dt.start_time
	weekly_nee = nee_weekly.groupby("Week")["Observed_NEE"].mean().sort_index()

	common_weeks = weekly_shap.index.intersection(weekly_nee.index)
	weekly_shap = weekly_shap.loc[common_weeks]
	weekly_nee = weekly_nee.loc[common_weeks]

	fig, ax = plt.subplots(figsize=(13, 6))
	bottom_pos = np.zeros(len(weekly_shap))
	bottom_neg = np.zeros(len(weekly_shap))
	for col in all_cols:
		vals = weekly_shap[col].values
		pos = np.clip(vals, 0, None)
		neg = np.clip(vals, None, 0)
		color = color_map[col]
		ax.bar(weekly_shap.index, pos, bottom=bottom_pos, label=col.replace("SHAP_", ""), width=5, color=color)
		ax.bar(weekly_shap.index, neg, bottom=bottom_neg, width=5, color=color)
		bottom_pos = bottom_pos + pos
		bottom_neg = bottom_neg + neg

	ax.set_xlabel("Week")
	ax.set_ylabel("NEE / Mean |SHAP|")
	ax.set_title("Weekly Mean |SHAP| (Stacked Bars) with NEE")
	ax.grid(True, axis="y", alpha=0.2)
	ax.legend(loc="upper left", ncol=2, fontsize=8)

	ax.plot(weekly_nee.index, weekly_nee.values, color="black", linewidth=1.5, label="Observed NEE")

	stack_max = bottom_pos.max() if len(weekly_shap) else 0.0
	stack_min = bottom_neg.min() if len(weekly_shap) else 0.0
	nee_min = weekly_nee.min() if len(weekly_nee) else 0.0
	nee_max = weekly_nee.max() if len(weekly_nee) else 0.0
	y_max = max(stack_max, nee_max) * 1.05
	y_min = min(stack_min, nee_min, 0.0) * 1.05
	ax.set_ylim(y_min, y_max)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_weekly_stacked_shap_with_nee.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_daily_stacked_bars_with_nee(df_shap, df_nee, shap_cols, plots_dir):
	all_cols = list(shap_cols)
	colors = plt.cm.tab20(np.linspace(0, 1, len(all_cols)))
	color_map = {col: colors[i] for i, col in enumerate(all_cols)}

	shap_daily = df_shap.copy()
	shap_daily["Day"] = shap_daily["Date"].dt.floor("D")
	daily_shap = shap_daily.groupby("Day")[all_cols].mean().sort_index()

	nee_daily = df_nee.copy()
	nee_daily["Day"] = nee_daily["Date"].dt.floor("D")
	daily_nee = nee_daily.groupby("Day")["Observed_NEE"].mean().sort_index()

	common_days = daily_shap.index.intersection(daily_nee.index)
	daily_shap = daily_shap.loc[common_days]
	daily_nee = daily_nee.loc[common_days]

	fig, ax = plt.subplots(figsize=(13, 6))
	bottom_pos = np.zeros(len(daily_shap))
	bottom_neg = np.zeros(len(daily_shap))
	for col in all_cols:
		vals = daily_shap[col].values
		pos = np.clip(vals, 0, None)
		neg = np.clip(vals, None, 0)
		color = color_map[col]
		ax.bar(daily_shap.index, pos, bottom=bottom_pos, label=col.replace("SHAP_", ""), width=0.9, color=color)
		ax.bar(daily_shap.index, neg, bottom=bottom_neg, width=0.9, color=color)
		bottom_pos = bottom_pos + pos
		bottom_neg = bottom_neg + neg

	ax.set_xlabel("Day")
	ax.set_ylabel("NEE / Mean SHAP")
	ax.set_title("Daily Mean SHAP (Stacked Bars) with NEE")
	ax.grid(True, axis="y", alpha=0.2)
	ax.legend(loc="upper left", ncol=2, fontsize=8)

	ax.plot(daily_nee.index, daily_nee.values, color="black", linewidth=1.0, label="Observed NEE")

	stack_max = bottom_pos.max() if len(daily_shap) else 0.0
	stack_min = bottom_neg.min() if len(daily_shap) else 0.0
	nee_min = daily_nee.min() if len(daily_nee) else 0.0
	nee_max = daily_nee.max() if len(daily_nee) else 0.0
	y_max = max(stack_max, nee_max) * 1.05
	y_min = min(stack_min, nee_min, 0.0) * 1.05
	ax.set_ylim(y_min, y_max)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_daily_stacked_shap_with_nee.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_biweekly_stacked_bars_with_nee(df_shap, df_nee, shap_cols, plots_dir):
	all_cols = list(shap_cols)
	colors = plt.cm.tab20(np.linspace(0, 1, len(all_cols)))
	color_map = {col: colors[i] for i, col in enumerate(all_cols)}

	shap_biweekly = df_shap.copy()
	shap_biweekly["BiWeek"] = shap_biweekly["Date"].dt.to_period("2W").dt.start_time
	biweekly_shap = shap_biweekly.groupby("BiWeek")[all_cols].mean().sort_index()

	nee_biweekly = df_nee.copy()
	nee_biweekly["BiWeek"] = nee_biweekly["Date"].dt.to_period("2W").dt.start_time
	biweekly_nee = nee_biweekly.groupby("BiWeek")["Observed_NEE"].mean().sort_index()

	common_weeks = biweekly_shap.index.intersection(biweekly_nee.index)
	biweekly_shap = biweekly_shap.loc[common_weeks]
	biweekly_nee = biweekly_nee.loc[common_weeks]

	fig, ax = plt.subplots(figsize=(13, 6))
	bottom_pos = np.zeros(len(biweekly_shap))
	bottom_neg = np.zeros(len(biweekly_shap))
	for col in all_cols:
		vals = biweekly_shap[col].values
		pos = np.clip(vals, 0, None)
		neg = np.clip(vals, None, 0)
		color = color_map[col]
		ax.bar(biweekly_shap.index, pos, bottom=bottom_pos, label=col.replace("SHAP_", ""), width=10, color=color)
		ax.bar(biweekly_shap.index, neg, bottom=bottom_neg, width=10, color=color)
		bottom_pos = bottom_pos + pos
		bottom_neg = bottom_neg + neg

	ax.set_xlabel("Bi-Week")
	ax.set_ylabel("NEE / Mean SHAP")
	ax.set_title("Bi-Weekly Mean SHAP (Stacked Bars) with NEE")
	ax.grid(True, axis="y", alpha=0.2)
	ax.legend(loc="upper left", ncol=2, fontsize=8)

	ax.plot(biweekly_nee.index, biweekly_nee.values, color="black", linewidth=1.2, label="Observed NEE")

	stack_max = bottom_pos.max() if len(biweekly_shap) else 0.0
	stack_min = bottom_neg.min() if len(biweekly_shap) else 0.0
	nee_min = biweekly_nee.min() if len(biweekly_nee) else 0.0
	nee_max = biweekly_nee.max() if len(biweekly_nee) else 0.0
	y_max = max(stack_max, nee_max) * 1.05
	y_min = min(stack_min, nee_min, 0.0) * 1.05
	ax.set_ylim(y_min, y_max)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_biweekly_stacked_shap_with_nee.png", dpi=300, bbox_inches="tight")
	plt.close(fig)



def plot_stacked_bars_two_sites(
	base_dir,
	plots_dir,
	site_ids,
	period,
	period_label,
	bar_width,
	filename,
	top_n=None,
	shap_filename="shap_values.csv",
	nee_filename="test_predictions.csv",
):
	data_by_site = {}
	all_cols = set()

	for site in site_ids:
		shap_path = base_dir / "output" / site / shap_filename
		nee_path = base_dir / "output" / site / nee_filename
		if not shap_path.exists():
			raise FileNotFoundError(f"Missing SHAP file: {shap_path}")
		if not nee_path.exists():
			raise FileNotFoundError(f"Missing NEE file: {nee_path}")

		shap_df = pd.read_csv(shap_path, parse_dates=["Date"])
		nee_df = pd.read_csv(nee_path, parse_dates=["Date"])
		shap_cols = [c for c in shap_df.columns if c.startswith("SHAP_")]
		if top_n is not None:
			shap_cols, _ = _top_shap_cols(shap_df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)
		all_cols.update(shap_cols)

		shap_period = shap_df.copy()
		shap_period["Period"] = shap_period["Date"].dt.to_period(period).dt.start_time
		period_shap = shap_period.groupby("Period")[shap_cols].mean().sort_index()

		nee_period = nee_df.copy()
		nee_period["Period"] = nee_period["Date"].dt.to_period(period).dt.start_time
		period_nee = nee_period.groupby("Period")["Observed_NEE"].mean().sort_index()

		common = period_shap.index.intersection(period_nee.index)
		period_shap = period_shap.loc[common]
		period_nee = period_nee.loc[common]

		data_by_site[site] = {"shap": period_shap, "nee": period_nee, "cols": shap_cols}

	all_cols = sorted(all_cols)
	colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(all_cols))))
	color_map = {col: colors[i] for i, col in enumerate(all_cols)}

	fig, axes = plt.subplots(len(site_ids), 1, figsize=(14, 8), sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	for ax, site in zip(axes, site_ids):
		site_data = data_by_site[site]
		shap_vals = site_data["shap"].reindex(columns=all_cols).fillna(0.0)
		nee_vals = site_data["nee"]

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			vals = shap_vals[col].values
			pos = np.clip(vals, 0, None)
			neg = np.clip(vals, None, 0)
			color = color_map[col]
			ax.bar(shap_vals.index, pos, bottom=bottom_pos, width=bar_width, color=color)
			ax.bar(shap_vals.index, neg, bottom=bottom_neg, width=bar_width, color=color)
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		ax.plot(nee_vals.index, nee_vals.values, color="black", linewidth=1.2)
		ax.set_title(f"{period_label} SHAP with NEE ({site})")
		ax.set_xlabel(period_label)
		ax.grid(True, axis="y", alpha=0.2)
		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		stack_max = pos_sum.max() if len(pos_sum) else 0.0
		stack_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_min = nee_vals.min() if len(nee_vals) else 0.0
		nee_max = nee_vals.max() if len(nee_vals) else 0.0
		y_max = max(stack_max, nee_max) * 1.05
		y_min = min(stack_min, nee_min, 0.0) * 1.05
		ax.set_ylim(y_min, y_max)

	axes[0].set_ylabel("NEE / Mean SHAP")
	legend_handles = [
		plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols
	]
	legend_labels = [col.replace("SHAP_", "") for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Observed NEE")
	fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

	plt.tight_layout()
	fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_stacked_bars_two_sites_by_year_may_nov(
	base_dir,
	plots_dir,
	site_ids,
	period,
	period_label,
	bar_width,
	filename,
	include_years=None,
	top_n=None,
	shap_filename="shap_values.csv",
	nee_filename="test_predictions.csv",
):
	data_by_site = {}
	all_cols = set()
	years_union = set()

	for site in site_ids:
		shap_path = base_dir / "output" / site / shap_filename
		nee_path = base_dir / "output" / site / nee_filename
		if not shap_path.exists():
			raise FileNotFoundError(f"Missing SHAP file: {shap_path}")
		if not nee_path.exists():
			raise FileNotFoundError(f"Missing NEE file: {nee_path}")

		shap_df = pd.read_csv(shap_path, parse_dates=["Date"])
		nee_df = pd.read_csv(nee_path, parse_dates=["Date"])

		shap_df = shap_df[shap_df["Date"].dt.month.between(6, 9)].copy()
		nee_df = nee_df[nee_df["Date"].dt.month.between(6, 9)].copy()

		shap_cols = [c for c in shap_df.columns if c.startswith("SHAP_")]
		if top_n is not None:
			shap_cols, _ = _top_shap_cols(shap_df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)
		all_cols.update(shap_cols)

		shap_df["Period"] = shap_df["Date"].dt.to_period(period).dt.start_time
		nee_df["Period"] = nee_df["Date"].dt.to_period(period).dt.start_time

		period_shap = shap_df.groupby("Period")[shap_cols].mean().sort_index()
		nee_col = "Predicted_NEE" if "Predicted_NEE" in nee_df.columns else "Observed_NEE"
		period_nee = nee_df.groupby("Period")[nee_col].mean().sort_index()

		common = period_shap.index.intersection(period_nee.index)
		period_shap = period_shap.loc[common]
		period_nee = period_nee.loc[common]

		years = sorted(period_shap.index.year.unique().tolist())
		years_union.update(years)
		data_by_site[site] = {"shap": period_shap, "nee": period_nee, "years": years}

	all_cols = sorted(all_cols)
	if include_years is not None:
		years_union.update(include_years)
	years_union = sorted(years_union)
	if not years_union:
		raise ValueError("No Jun-Sep data found for requested full-dataset combined plot.")

	colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(all_cols))))
	color_map = {col: colors[i] for i, col in enumerate(all_cols)}

	fig, axes = plt.subplots(
		len(site_ids),
		len(years_union),
		figsize=(4.5 * len(years_union), 4.0 * len(site_ids)),
		sharex=False,
		sharey=False,
	)
	if len(site_ids) == 1 and len(years_union) == 1:
		axes = np.array([[axes]])
	elif len(site_ids) == 1:
		axes = np.array([axes])
	elif len(years_union) == 1:
		axes = np.array([[ax] for ax in axes])

	for row, site in enumerate(site_ids):
		shap_all = data_by_site[site]["shap"].reindex(columns=all_cols).fillna(0.0)
		nee_all = data_by_site[site]["nee"]

		# Keep y-scale fixed per site across all year panels.
		site_y_min = np.inf
		site_y_max = -np.inf
		for year in years_union:
			year_mask = shap_all.index.year == year
			shap_year = shap_all.loc[year_mask]
			nee_year = nee_all.loc[nee_all.index.year == year]
			if shap_year.empty or nee_year.empty:
				continue
			pos_sum = np.clip(shap_year.values, 0, None).sum(axis=1)
			neg_sum = np.clip(shap_year.values, None, 0).sum(axis=1)
			stack_max = pos_sum.max() if len(pos_sum) else 0.0
			stack_min = neg_sum.min() if len(neg_sum) else 0.0
			nee_min = nee_year.min() if len(nee_year) else 0.0
			nee_max = nee_year.max() if len(nee_year) else 0.0
			site_y_max = max(site_y_max, stack_max, nee_max)
			site_y_min = min(site_y_min, stack_min, nee_min, 0.0)

		if not np.isfinite(site_y_min) or not np.isfinite(site_y_max):
			site_y_min, site_y_max = -1.0, 1.0
		elif site_y_min == site_y_max:
			site_y_min -= 1.0
			site_y_max += 1.0
		else:
			site_y_min *= 1.05
			site_y_max *= 1.05

		for col_idx, year in enumerate(years_union):
			ax = axes[row, col_idx]
			year_mask = shap_all.index.year == year
			shap_vals = shap_all.loc[year_mask]
			nee_vals = nee_all.loc[nee_all.index.year == year]

			if shap_vals.empty or nee_vals.empty:
				ax.axis("off")
				continue

			bottom_pos = np.zeros(len(shap_vals))
			bottom_neg = np.zeros(len(shap_vals))
			for feature in all_cols:
				vals = shap_vals[feature].values
				pos = np.clip(vals, 0, None)
				neg = np.clip(vals, None, 0)
				color = color_map[feature]
				ax.bar(shap_vals.index, pos, bottom=bottom_pos, width=bar_width, color=color)
				ax.bar(shap_vals.index, neg, bottom=bottom_neg, width=bar_width, color=color)
				bottom_pos = bottom_pos + pos
				bottom_neg = bottom_neg + neg

			ax.plot(nee_vals.index, nee_vals.values, color="black", linewidth=1.1)
			ax.set_title(f"{site} - {year}")
			if row == len(site_ids) - 1:
				ax.set_xlabel("Month")

			if col_idx == 0:
				ax.set_ylabel("NEE / Mean SHAP")
				ax.tick_params(axis="y", labelleft=True)
			else:
				ax.set_ylabel("")
				ax.tick_params(axis="y", labelleft=False)
			ax.axhline(y=0, color="black", linewidth=1.2, alpha=0.9)
			ax.grid(True, axis="y", alpha=0.2)

			x_start = pd.Timestamp(year=year, month=5, day=1)
			x_end = pd.Timestamp(year=year, month=10, day=31)
			ax.set_xlim(x_start, x_end)
			label_months = [6, 7, 8, 9]
			month_ticks = [pd.Timestamp(year=year, month=m, day=1) for m in label_months]
			month_labels = [pd.Timestamp(year=year, month=m, day=1).strftime("%b") for m in label_months]
			ax.set_xticks(month_ticks)
			ax.set_xticklabels(month_labels)
			ax.tick_params(axis="x", labelrotation=0)

			ax.set_ylim(site_y_min, site_y_max)

	legend_handles = [
		plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols
	]
	legend_labels = [col.replace("SHAP_", "") for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")
	fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

	fig.suptitle(f"{period_label} SHAP with NEE (Data: Jun-Sep, Axis: May-Oct, by Year)", y=1.02)
	plt.tight_layout(rect=[0, 0, 0.88, 0.98])
	fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_biweekly_stacked_bars_two_sites(
	base_dir,
	plots_dir,
	site_ids,
	top_n=None,
	suffix="",
	shap_filename="shap_values.csv",
	nee_filename="test_predictions.csv",
):
	plot_stacked_bars_two_sites(
		base_dir,
		plots_dir,
		site_ids,
		period="2W",
		period_label="Bi-Week",
		bar_width=10,
		filename=f"plot_biweekly_stacked_shap_two_sites{suffix}.png",
		top_n=top_n,
		shap_filename=shap_filename,
		nee_filename=nee_filename,
	)


def plot_weekly_stacked_bars_two_sites(
	base_dir,
	plots_dir,
	site_ids,
	top_n=None,
	suffix="",
	shap_filename="shap_values.csv",
	nee_filename="test_predictions.csv",
):
	plot_stacked_bars_two_sites(
		base_dir,
		plots_dir,
		site_ids,
		period="W",
		period_label="Week",
		bar_width=5,
		filename=f"plot_weekly_stacked_shap_two_sites{suffix}.png",
		top_n=top_n,
		shap_filename=shap_filename,
		nee_filename=nee_filename,
	)


def plot_daily_stacked_bars_two_sites(
	base_dir,
	plots_dir,
	site_ids,
	top_n=None,
	suffix="",
	shap_filename="shap_values.csv",
	nee_filename="test_predictions.csv",
):
	plot_stacked_bars_two_sites(
		base_dir,
		plots_dir,
		site_ids,
		period="D",
		period_label="Day",
		bar_width=0.9,
		filename=f"plot_daily_stacked_shap_two_sites{suffix}.png",
		top_n=top_n,
		shap_filename=shap_filename,
		nee_filename=nee_filename,
	)


def plot_monthly_stacked_bars_two_sites(
	base_dir,
	plots_dir,
	site_ids,
	top_n=None,
	suffix="",
	shap_filename="shap_values.csv",
	nee_filename="test_predictions.csv",
):
	plot_stacked_bars_two_sites(
		base_dir,
		plots_dir,
		site_ids,
		period="M",
		period_label="Month",
		bar_width=20,
		filename=f"plot_monthly_stacked_shap_two_sites{suffix}.png",
		top_n=top_n,
		shap_filename=shap_filename,
		nee_filename=nee_filename,
	)


def plot_weekly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, site_ids, top_n=None):
	data_by_site = {}
	all_cols = set()
	master_periods = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		if top_n is not None:
			shap_cols, _ = _top_shap_cols(df, shap_cols, exclude={"sin_doy", "cos_doy"}, top_n=top_n)
		all_cols.update(shap_cols)

		tmp = df.copy()
		tmp["WeekPeriod"] = tmp["Date"].dt.to_period("W")
		weekly_shap = tmp.groupby("WeekPeriod")[shap_cols].mean().sort_index()
		nee_col = "Predicted_NEE" if "Predicted_NEE" in tmp.columns else "Observed_NEE"
		weekly_nee = tmp.groupby("WeekPeriod")[nee_col].mean().sort_index()
		master_periods.update(weekly_shap.index.tolist())

		data_by_site[site] = {"shap": weekly_shap, "nee": weekly_nee}

	all_cols = sorted(all_cols)
	master_periods = sorted(master_periods)
	if not master_periods:
		raise ValueError("No weekly periods found for full-dataset continuous plot.")

	bar_step = 1.03
	bar_width = 0.78
	x_vals = np.arange(len(master_periods)) * bar_step
	x_index = pd.Index(master_periods)
	x_years = np.array([p.year for p in master_periods])
	x_months = np.array([p.start_time.month for p in master_periods])
	month_tick_positions = []
	month_tick_labels = []
	for i in range(len(master_periods)):
		if i == 0 or x_months[i] != x_months[i - 1] or x_years[i] != x_years[i - 1]:
			if x_months[i] in {7, 8, 9}:
				month_tick_positions.append(i * bar_step)
				month_tick_labels.append(master_periods[i].start_time.strftime("%b"))

	year_boundaries = []
	for i in range(1, len(master_periods)):
		if x_years[i] != x_years[i - 1]:
			year_boundaries.append((i - 0.5) * bar_step)

	year_label_positions = []
	for year in sorted(set(x_years.tolist())):
		year_positions = np.where(x_years == year)[0]
		if len(year_positions):
			year_label_positions.append((year_positions.mean() * bar_step, str(year)))

	colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(all_cols))))
	color_map = {col: colors[i] for i, col in enumerate(all_cols)}

	fig, axes = plt.subplots(len(site_ids), 1, figsize=(10, 6.5), sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	for ax, site in zip(axes, site_ids):
		shap_vals = data_by_site[site]["shap"].reindex(columns=all_cols).fillna(0.0)
		nee_vals = data_by_site[site]["nee"]

		if shap_vals.empty:
			ax.axis("off")
			continue

		# Discontinuous axis: only keep observed weekly periods across all years/sites.
		shap_vals = shap_vals.reindex(x_index).fillna(0.0)
		nee_vals = nee_vals.reindex(x_index)  # keep NaN so NEE line breaks at no-data periods

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			vals = shap_vals[col].values
			pos = np.clip(vals, 0, None)
			neg = np.clip(vals, None, 0)
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
		stack_max = pos_sum.max() if len(pos_sum) else 0.0
		stack_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_clean = nee_vals.dropna()
		nee_min = nee_clean.min() if len(nee_clean) else 0.0
		nee_max = nee_clean.max() if len(nee_clean) else 0.0
		y_max = max(stack_max, nee_max) * 1.05
		y_min = min(stack_min, nee_min, 0.0) * 1.05
		if np.isclose(y_min, y_max):
			buffer = 1.0 if np.isclose(y_min, 0.0) else max(1e-6, abs(y_min) * 0.05)
			y_min -= buffer
			y_max += buffer
		ax.set_ylim(y_min, y_max)
		ax.set_xlim(-0.5 * bar_step, (len(x_vals) - 0.5) * bar_step)

		for boundary in year_boundaries:
			ax.axvline(x=boundary, color="black", linewidth=0.8, alpha=0.6)

	axes[-1].set_xticks(month_tick_positions)
	axes[-1].set_xticklabels(month_tick_labels)

	# Add year labels above top panel to indicate the year sections.
	for x_pos, year_label in year_label_positions:
		axes[0].text(x_pos, 1.03, year_label, transform=axes[0].get_xaxis_transform(),
					ha="center", va="bottom", fontsize=9)

	legend_handles = [
		plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols
	]
	legend_labels = [col.replace("SHAP_", "") for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")
	fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(0.93, 0.5), fontsize=8)

	title_suffix = f"Top {top_n} Features" if top_n is not None else "All Features"
	fig.suptitle(f"Weekly SHAP with NEE, Full Dataset ({title_suffix})", y=0.98)
	plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	filename = "plot_weekly_stacked_shap_two_sites_full_dataset_continuous.png"
	if top_n is not None:
		filename = f"plot_weekly_stacked_shap_two_sites_full_dataset_continuous_top{top_n}.png"
	fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_monthly_stacked_bars_with_nee(df_shap, df_nee, shap_cols, plots_dir):
	all_cols = list(shap_cols)
	colors = plt.cm.tab20(np.linspace(0, 1, len(all_cols)))
	color_map = {col: colors[i] for i, col in enumerate(all_cols)}

	shap_monthly = df_shap.copy()
	shap_monthly["MonthStart"] = shap_monthly["Date"].dt.to_period("M").dt.start_time
	monthly_shap = shap_monthly.groupby("MonthStart")[all_cols].mean().sort_index()

	nee_monthly = df_nee.copy()
	nee_monthly["MonthStart"] = nee_monthly["Date"].dt.to_period("M").dt.start_time
	monthly_nee = nee_monthly.groupby("MonthStart")["Observed_NEE"].mean().sort_index()

	common_months = monthly_shap.index.intersection(monthly_nee.index)
	monthly_shap = monthly_shap.loc[common_months]
	monthly_nee = monthly_nee.loc[common_months]

	fig, ax = plt.subplots(figsize=(13, 6))
	bottom_pos = np.zeros(len(monthly_shap))
	bottom_neg = np.zeros(len(monthly_shap))
	for col in all_cols:
		vals = monthly_shap[col].values
		pos = np.clip(vals, 0, None)
		neg = np.clip(vals, None, 0)
		color = color_map[col]
		ax.bar(monthly_shap.index, pos, bottom=bottom_pos, label=col.replace("SHAP_", ""), width=20, color=color)
		ax.bar(monthly_shap.index, neg, bottom=bottom_neg, width=20, color=color)
		bottom_pos = bottom_pos + pos
		bottom_neg = bottom_neg + neg

	ax.set_xlabel("Month")
	ax.set_ylabel("NEE / Mean SHAP")
	ax.set_title("Monthly Mean SHAP (Stacked Bars) with NEE")
	ax.grid(True, axis="y", alpha=0.2)
	ax.legend(loc="upper left", ncol=2, fontsize=8)

	ax.plot(monthly_nee.index, monthly_nee.values, color="black", linewidth=1.2, label="Observed NEE")

	stack_max = bottom_pos.max() if len(monthly_shap) else 0.0
	stack_min = bottom_neg.min() if len(monthly_shap) else 0.0
	nee_min = monthly_nee.min() if len(monthly_nee) else 0.0
	nee_max = monthly_nee.max() if len(monthly_nee) else 0.0
	y_max = max(stack_max, nee_max) * 1.05
	y_min = min(stack_min, nee_min, 0.0) * 1.05
	ax.set_ylim(y_min, y_max)

	plt.tight_layout()
	fig.savefig(plots_dir / "plot_monthly_stacked_shap_with_nee.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


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


def plot_monthly_mean_shap_stacked_with_nee(df_full, shap_cols, plots_dir, site_id):
	all_cols = list(shap_cols)
	colors = plt.cm.tab20(np.linspace(0, 1, len(all_cols)))
	color_map = {col: colors[i] for i, col in enumerate(all_cols)}

	monthly_shap = df_full.groupby("Month")[all_cols].mean().reindex(range(1, 13)).fillna(0.0)
	monthly_nee = df_full.groupby("Month")["Observed_NEE"].mean().reindex(range(1, 13))

	fig, ax = plt.subplots(figsize=(13, 6))
	bottom_pos = np.zeros(len(monthly_shap))
	bottom_neg = np.zeros(len(monthly_shap))
	for col in all_cols:
		vals = monthly_shap[col].values
		pos = np.clip(vals, 0, None)
		neg = np.clip(vals, None, 0)
		color = color_map[col]
		ax.bar(monthly_shap.index, pos, bottom=bottom_pos, label=col.replace("SHAP_", ""), width=0.8, color=color)
		ax.bar(monthly_shap.index, neg, bottom=bottom_neg, width=0.8, color=color)
		bottom_pos = bottom_pos + pos
		bottom_neg = bottom_neg + neg

	ax.plot(monthly_nee.index, monthly_nee.values, color="black", linewidth=1.8, label="Observed NEE")
	ax.set_xticks(range(1, 13))
	ax.set_xlabel("Month")
	ax.set_ylabel("Mean SHAP / Mean NEE")
	ax.set_title(f"Monthly Mean SHAP (Stacked) with Mean Observed NEE ({site_id})")
	ax.grid(True, axis="y", alpha=0.25)

	stack_max = bottom_pos.max() if len(monthly_shap) else 0.0
	stack_min = bottom_neg.min() if len(monthly_shap) else 0.0
	nee_min = monthly_nee.min() if len(monthly_nee) else 0.0
	nee_max = monthly_nee.max() if len(monthly_nee) else 0.0
	y_max = max(stack_max, nee_max) * 1.05
	y_min = min(stack_min, nee_min, 0.0) * 1.05
	ax.set_ylim(y_min, y_max)
	ax.legend(loc="upper left", ncol=2, fontsize=8)

	plt.tight_layout()
	fig.savefig(plots_dir / f"plot_monthly_mean_shap_stacked_with_nee_full_dataset_{site_id}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def main():
	base_dir = Path(__file__).resolve().parent
	full_dataset_plots_dir = base_dir / "plots" / "full_dataset_predictions"
	combined_dir = base_dir / "plots" / "combined"
	full_dataset_plots_dir.mkdir(parents=True, exist_ok=True)
	combined_dir.mkdir(parents=True, exist_ok=True)

	for site_id in SITE_IDS:
		output_dir = base_dir / "output" / site_id
		plots_dir = base_dir / "plots" / site_id
		plots_dir.mkdir(parents=True, exist_ok=True)

		shap_path = output_dir / "shap_values.csv"
		if not shap_path.exists():
			raise FileNotFoundError(f"Missing SHAP file: {shap_path}")

		nee_path = output_dir / "test_predictions.csv"
		if not nee_path.exists():
			raise FileNotFoundError(f"Missing NEE file: {nee_path}")

		df = pd.read_csv(shap_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		nee_df = pd.read_csv(nee_path, parse_dates=["Date"])

		full_dataset_path = output_dir / "full_dataset_predictions_shap_values.csv"
		if not full_dataset_path.exists():
			raise FileNotFoundError(f"Missing full-dataset SHAP + prediction file: {full_dataset_path}")
		df_full = pd.read_csv(full_dataset_path, parse_dates=["Date"])
		full_shap_cols = [c for c in df_full.columns if c.startswith("SHAP_")]

		plot_yearly_mean_abs_bar(df, shap_cols, plots_dir)
		plot_year_feature_heatmap(df, shap_cols, plots_dir)
		plot_month_feature_heatmap(df, shap_cols, plots_dir)
		plot_yearly_beeswarm(df, shap_cols, plots_dir)
		plot_monthly_violin_top_feature(df, shap_cols, plots_dir)
		plot_monthly_stack_area(df, shap_cols, plots_dir)
		plot_monthly_mean_sd_ribbons(df, shap_cols, plots_dir)
		plot_yearly_mean_sd_bars(df, shap_cols, plots_dir)
		plot_weekly_stacked_bars_with_nee(df, nee_df, shap_cols, plots_dir)
		plot_daily_stacked_bars_with_nee(df, nee_df, shap_cols, plots_dir)
		plot_biweekly_stacked_bars_with_nee(df, nee_df, shap_cols, plots_dir)
		plot_monthly_stacked_bars_with_nee(df, nee_df, shap_cols, plots_dir)
		plot_full_dataset_observed_predicted_line(df_full, full_dataset_plots_dir, site_id)
		plot_monthly_mean_shap_stacked_with_nee(df_full, full_shap_cols, full_dataset_plots_dir, site_id)

	plot_biweekly_stacked_bars_two_sites(base_dir, combined_dir, ["ZaF", "ZaH"])
	plot_weekly_stacked_bars_two_sites(base_dir, combined_dir, ["ZaF", "ZaH"])
	plot_daily_stacked_bars_two_sites(base_dir, combined_dir, ["ZaF", "ZaH"])
	plot_monthly_stacked_bars_two_sites(base_dir, combined_dir, ["ZaF", "ZaH"])
	plot_biweekly_stacked_bars_two_sites(base_dir, combined_dir, ["ZaF", "ZaH"], top_n=8, suffix="_top8")
	plot_weekly_stacked_bars_two_sites(base_dir, combined_dir, ["ZaF", "ZaH"], top_n=8, suffix="_top8")
	plot_daily_stacked_bars_two_sites(base_dir, combined_dir, ["ZaF", "ZaH"], top_n=8, suffix="_top8")
	plot_monthly_stacked_bars_two_sites(base_dir, combined_dir, ["ZaF", "ZaH"], top_n=8, suffix="_top8")

	# Full-dataset combined stacked SHAP+NEE plots (May-Nov, separated by year)
	plot_stacked_bars_two_sites_by_year_may_nov(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		period="2W",
		period_label="Bi-Week",
		bar_width=10,
		filename="plot_biweekly_stacked_shap_two_sites_full_dataset.png",
		include_years=list(range(2018, 2026)),
		shap_filename="full_dataset_predictions_shap_values.csv",
		nee_filename="full_dataset_predictions_shap_values.csv",
	)

	# Full-dataset weekly combined timeline (all years in one continuous plot)
	plot_weekly_stacked_bars_two_sites_full_dataset_continuous(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
	)
	plot_weekly_stacked_bars_two_sites_full_dataset_continuous(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		top_n=8,
	)
	plot_stacked_bars_two_sites_by_year_may_nov(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		period="W",
		period_label="Week",
		bar_width=5,
		filename="plot_weekly_stacked_shap_two_sites_full_dataset.png",
		include_years=list(range(2018, 2026)),
		shap_filename="full_dataset_predictions_shap_values.csv",
		nee_filename="full_dataset_predictions_shap_values.csv",
	)
	plot_stacked_bars_two_sites_by_year_may_nov(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		period="D",
		period_label="Day",
		bar_width=0.9,
		filename="plot_daily_stacked_shap_two_sites_full_dataset.png",
		include_years=list(range(2018, 2026)),
		shap_filename="full_dataset_predictions_shap_values.csv",
		nee_filename="full_dataset_predictions_shap_values.csv",
	)
	plot_stacked_bars_two_sites_by_year_may_nov(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		period="M",
		period_label="Month",
		bar_width=20,
		filename="plot_monthly_stacked_shap_two_sites_full_dataset.png",
		include_years=list(range(2018, 2026)),
		shap_filename="full_dataset_predictions_shap_values.csv",
		nee_filename="full_dataset_predictions_shap_values.csv",
	)

	plot_stacked_bars_two_sites_by_year_may_nov(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		period="2W",
		period_label="Bi-Week",
		bar_width=10,
		include_years=list(range(2018, 2026)),
		top_n=8,
		filename="plot_biweekly_stacked_shap_two_sites_full_dataset_top8.png",
		shap_filename="full_dataset_predictions_shap_values.csv",
		nee_filename="full_dataset_predictions_shap_values.csv",
	)
	plot_stacked_bars_two_sites_by_year_may_nov(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		period="W",
		period_label="Week",
		bar_width=5,
		include_years=list(range(2018, 2026)),
		top_n=8,
		filename="plot_weekly_stacked_shap_two_sites_full_dataset_top8.png",
		shap_filename="full_dataset_predictions_shap_values.csv",
		nee_filename="full_dataset_predictions_shap_values.csv",
	)
	plot_stacked_bars_two_sites_by_year_may_nov(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		period="D",
		period_label="Day",
		bar_width=0.9,
		include_years=list(range(2018, 2026)),
		top_n=8,
		filename="plot_daily_stacked_shap_two_sites_full_dataset_top8.png",
		shap_filename="full_dataset_predictions_shap_values.csv",
		nee_filename="full_dataset_predictions_shap_values.csv",
	)
	plot_stacked_bars_two_sites_by_year_may_nov(
		base_dir,
		full_dataset_plots_dir,
		["ZaF", "ZaH"],
		period="M",
		period_label="Month",
		bar_width=20,
		include_years=list(range(2018, 2026)),
		top_n=8,
		filename="plot_monthly_stacked_shap_two_sites_full_dataset_top8.png",
		shap_filename="full_dataset_predictions_shap_values.csv",
		nee_filename="full_dataset_predictions_shap_values.csv",
	)


if __name__ == "__main__":
	main()
