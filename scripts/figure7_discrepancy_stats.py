#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2025-10-01 Wed 03:58 PM
## Comment : Discrepancy stats
##
## ------------------------------
import sys
import json
import re
from pathlib import Path
from itertools import product as prod
from string import ascii_letters as letters
import argparse

import warnings
#warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import violinplot, putils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
version = "7.3"

aris = [10, 100]

xv_length = 10

marginal_ref = "GEV"
copula_ref = "Gumbel"
cens_prob_thresh_obs_ref = 30
cens_prob_thresh_cov_ref = 30
prior_ref = "default"
timeperiod_ref = "ALL"
clip_to_obs_ref = False

metric_names = [f"obs_ari{ari}_postpred_logbias" for ari in aris]

output_name = "covarfit"

awidth = 11
aheight = 4
fdpi = 300
show_count = False

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
basename = source_file.stem
froot = source_file.parent.parent

fdata = froot / "data"
fout = froot / "outputs" / f"metrics_v{version}"
fimg = froot / "images" / basename
fimg.mkdir(exist_ok=True, parents=True)
for f in fimg.glob("*.png"):
    f.unlink()

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")
fs = fdata / "stations.csv"
stations, _ = csv.read_csv(fs, index_col="STATIONID",
                           dtype={"STATIONID": str})

dur = stations.filter(regex="MAX_AMS_DUR", axis=1).squeeze()

obs_type = "obs"
fr = fout / f"{output_name}_metrics_v{version}.csv"
metrics, _ = csv.read_csv(fr, dtype={"stationid": str})

def reformat(cov):
    cov = cov.str.replace("_MEA.*", "", regex=True)
    return cov.str.replace(".*_", "", regex=True).str.lower()
metrics.loc[:, "covariate"] = reformat(metrics.covariate)

fr = fout / f"{output_name}_metrics_equiv_v{version}.csv"
metrics_equiv, _ = csv.read_csv(fr, dtype={"stationid": str})
metrics_equiv.loc[:, "covariate"] = reformat(metrics_equiv.covariate)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

covariates = ["awral"]
xv_lengths = [xv_length]

marginal_names = metrics.marginal_name.unique()
cens_obs = metrics.cens_prob_thresh_obs.unique()
cens_cov = metrics.cens_prob_thresh_cov.unique()
copula_names = metrics.copula_name.unique()

# option loop
archive = {}
for marginal, copula, cov, censo, censc \
        in prod(marginal_names,
                copula_names,
                covariates,
                cens_obs,
                cens_cov):

    if marginal != marginal_ref and \
            (censo != cens_prob_thresh_obs_ref \
                or censc != cens_prob_thresh_cov_ref):
        continue

    index = metrics.cens_prob_thresh_obs == censo
    index &= metrics.prior == prior_ref
    index &= metrics.timeperiod == timeperiod_ref
    index &= metrics.obs_type == obs_type
    index &= metrics.marginal_name == marginal
    index &= metrics.clip_to_obs == clip_to_obs_ref
    index &= metrics.stationid.isin(stations.index)

    i1 = metrics.copula_name == copula
    i1 &= metrics.cens_prob_thresh_cov == censc
    i1 &= metrics.covariate.isin(["none", cov])

    i2 = metrics.copula_name == copula_ref
    i2 &= metrics.cens_prob_thresh_cov == cens_prob_thresh_cov_ref
    i2 &= metrics.covariate == "none"
    index &= i1 | i2

    me = metrics.loc[index]

    nval = len(me)
    if nval == 0:
        continue

    ieq = metrics_equiv.cens_prob_thresh_obs == censo
    ieq &= metrics_equiv.prior == prior_ref
    ieq &= metrics_equiv.timeperiod == timeperiod_ref
    ieq &= metrics_equiv.marginal_name == marginal
    ieq &= metrics_equiv.clip_to_obs == clip_to_obs_ref

    excluded = marginal != marginal_ref\
            or censo != cens_prob_thresh_obs_ref\
            or censc != cens_prob_thresh_cov_ref\
            or copula != copula_ref
    if ieq.sum() == 0 or excluded:
        equiv = None
    else:
        equiv = metrics_equiv.loc[ieq]

    LOGGER.info("")
    LOGGER.info(f"Plotting {marginal}-{copula}-{cov},"\
                + f" co={censo} cs={censc}")

    for xv_length in xv_lengths:
        LOGGER.info(f"xv = {xv_length}", ntab=1, nret=1)
        plt.close("all")
        nm = len(metric_names)
        ncols = 1
        mosaic = np.array_split(metric_names, nm // ncols + (nm % ncols != 0))
        mosaic = [l.tolist() + ["."] * (ncols - len(l)) for l in mosaic]
        nrows = len(mosaic)

        fz = (ncols * awidth, nrows * aheight)
        # constrained layout crops text, so abandonned
        fig = plt.figure(figsize=fz)
        kw = dict(left=0.02, right=0.95, bottom=0.06, top=0.94,
                  hspace=0.3)
        axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

        has_some_data = False
        iplot = 0

        for aname, ax in axs.items():
            LOGGER.info(f"Metric {aname}", ntab=2, nret=1)
            metric_name = aname

            ii = me.xv_length == xv_length

            if ii.sum() == 0:
                ax.axis("off")
                LOGGER.info("No data, skip.", ntab=2)
                continue

            agg = dict(index=["stationid", "xvperid"],
                       columns=["covariate",
                                "cens_prob_thresh_cov"],
                       values=metric_name)
            df = pd.pivot_table(me.loc[ii], **agg)

            if not re.search("logpv|crps|nse|skew",
                             metric_name):
                df = df.abs()

            # Ensure all columns have same number of missing
            iok = df.notnull().all(axis=1)
            nval = len(df)
            df = df.loc[iok]
            if len(df) == 0:
                LOGGER.info("No data. Skip.", ntab=2)
                continue

            # Rename columns for ease
            coln = lambda cn: f"cov={cn[0]}"\
                    + (f"\ncsim={cn[1]}" if cn[0]!="none" else "")
            df.columns = [coln(cn) for cn in df.columns]

            try:
                cref = next(cn for cn in df.columns if re.search("none", cn))
                cc = [cref] + [cn for cn in df.columns if cn != cref]
                df = df.loc[:, cc]
            except:
                pass

            # Check data is there
            cc = [cn for cn in df.columns if not re.search("none", cn)]
            if len(cc) == 0:
                LOGGER.info("Missing either nocov or cov. Skip.", ntab=2)
                continue

            # Equivalent length
            equiv_metric = None
            if equiv is not None:
                ieqq = equiv.metric_name == metric_name
                ccd = [cn for cn in df.columns
                       if not re.search("none", cn)]
                cce = [f"CS{re.split('=', cn)[-1]}_XV{xv_length:02d}"
                       for cn in ccd]
                equiv_data = equiv.loc[ieqq, cce] - xv_length
                ct = [-100] + np.arange(-15, 20, 10).tolist() + [100]
                tmp = equiv_data.apply(lambda x: pd.cut(x,
                                                        ct, labels=False).value_counts())
                labs = [f"{c:+0.0f} to\n{ct[i+1]:+0.0f}"
                        for i, c in enumerate(ct[:-1])]
                labs[0] = f"<{ct[1]:+0.0f}"
                labs[-1] = f">{ct[-2]:+0.0f}"
                equiv_metric = pd.DataFrame(0,
                                            index=np.arange(len(labs)),
                                            columns=tmp.columns)
                equiv_metric.loc[tmp.index] = tmp.values
                equiv_metric.loc[:, "lab"] = labs
                equiv_metric.set_index("lab", inplace=True)
                equiv_metric *= 100 / equiv_metric.sum()

            # Worst perf
            cs = censc
            try:
                cns = next(cn for cn in df.columns if re.search(f"sim={cs}", cn))
            except StopIteration:
                LOGGER.info("Not enough perf data. Skip.", ntab=2)
                continue

            dfm = df.groupby("stationid").mean()

            key = (marginal, copula, cov, censo, censc,
                   xv_length, aname)
            archive[key] = dfm

            if not cref in dfm.columns:
                LOGGER.info("No ref perf data. Skip.", ntab=2)
                continue
            has_some_data = True

            LOGGER.info("Best and worst fit:", ntab=2)
            bestworst = dfm.apply(lambda x: x.idxmax()).tolist()\
                + dfm.apply(lambda x: x.idxmin()).tolist()
            for sid in bestworst:
                w = dfm.loc[sid]
                LOGGER.info(f"{sid} : nocov={w[cref]:0.2f} cov={w[cns]:0.2f}", ntab=3)

            # Create side axes
            divider = make_axes_locatable(ax)
            pad = 0.02
            ax_better = divider.append_axes("right", size="70%",
                                            pad=pad)
            ax_ari = divider.append_axes("left", size="10%",
                                            pad=pad)
            ax_ari.axis("off")
            if equiv_metric is not None:
                ax_equiv = divider.append_axes("right", size="70%",
                                           pad=pad)

            # Show violin
            df_plot = df.loc[:, [cref, cns]]
            df_plot.columns = [
                f"Univariate $D_{{G,{xv_length}}}^{{(k)}}$",
                f"CoFFI $D_{{F,{xv_length}}}^{{(k)}}$"
                ]
            vl = violinplot.Violin(df_plot)
            vl.draw(ax=ax)

            if re.search("scorecrps", metric_name):
                ylim = (0, 1)
            elif re.search("nse", metric_name):
                ylim = (0, 1)
            elif re.search("logpv", metric_name):
                ylim = (-3, -1)
            elif re.search("skew$", metric_name):
                ylim = (-0.5, 5.)
            elif re.search("skewlog$", metric_name):
                ylim = (-0.5, 1.5)
            else:
                ylim = (0., 1.2)

            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            yticks = ax.get_yticks()
            ax.set_yticks(yticks[1:-1])
            ax.tick_params(axis="y",direction="in",
                                  pad=-22)

            putils.line(ax, 1, 0, 0, 0, "k-", lw=0.8)

            ari = re.search("(?<=ari)[\d]+", metric_name).group()
            title = f"({letters[iplot]}) Discrepancy metric"
            ax.set_title(title)

            if show_count:
                txt = f"{len(df_plot):,d} xv results"
                ax.text(0.98, 0.98, txt, va="top",
                        ha="right", transform=ax.transAxes)

            ylabel = "Discrepancy [-]"
            ax.text(0.02, 0.98, ylabel,
                    va="top", ha="left",
                    transform=ax.transAxes)

            ax_ari.text(0.1, 0.5, f"AEP 1:{ari}",
                        rotation=90, fontweight="bold",
                        fontsize="x-large", va="center",
                        ha="left", transform=ax_ari.transAxes)

            iplot += 1

            if df.shape[1] > 1:
                cs = censc
                ddf = np.abs(df.filter(regex=f"none|csim={cs}", axis=1))
                try:
                    c1 = next(cn for cn in ddf.columns if re.search("none", cn))
                    c2 = next(cn for cn in ddf.columns if re.search(cov, cn))
                except:
                    continue

                if re.search("obscond", metric_name):
                    c1, c2 = c2, c1

                diff = np.abs(ddf.loc[:, c2]) - np.abs(ddf.loc[:, c1])
                mdiff = diff.mean()

                cuts = [diff.min()-1e-10, -0.05, 0.05, diff.max()+1e-10]
                diff = pd.cut(diff, cuts)
                diff = diff.value_counts()
                diff *= 100 / diff.sum()
                idx = []
                for inv, v in diff.items():
                    if inv.left == -0.05:
                        idx.append("-0.05 to\n+0.05")
                    elif inv.left == 0.05:
                        idx.append(">+0.05")
                    else:
                        idx.append("<-0.05")

                diff.index = idx
                diff = diff.loc[["<-0.05", "-0.05 to\n+0.05", ">+0.05"]]

                diff.plot(ax=ax_better, kind="bar",
                          color=["tab:blue", "grey", "tab:red"],
                          rot=0)
                for k, (_, v) in enumerate(diff.items()):
                    t = f"{v:0.0f}% "
                    if v > 0:
                        ax_better.text(k, v, t, rotation=90, color="w",
                                       fontweight="bold", ha="center", va="top")

                title = f"({letters[iplot]}) Difference in discrepancies\n"\
                        + "(Univariate - COFFI)"
                iplot += 1
                ax_better.set(title=title, ylim=(0, 68),
                              yticks=[20, 40, 60], xlabel="")
                ax_better.tick_params(axis="y",direction="in",
                                      pad=-22)
                txt = f"Mean = {mdiff:+0.2f}"
                ax_better.text(0.95, 0.95, txt, va="top", ha="right",
                               fontsize="large",
                               transform=ax_better.transAxes)
                ax_better.text(0.02, 0.98, "% site",
                              transform=ax_better.transAxes,
                              va="top", ha="left")

                if equiv_metric is not None:
                    # Inset axes to count show the % of improved/worsen
                    cn = f"CS{censc}_XV{xv_length:02d}"
                    sec = equiv_metric.loc[:, cn]
                    cols = putils.cmap2colors(len(sec), "PiYG")
                    cols[len(cols)//2] = "grey"
                    sec.plot(ax=ax_equiv, kind="bar", color=cols,
                             rot=0)
                    for k, (_, v) in enumerate(sec.items()):
                        t = f"{v:0.0f}% "
                        if v < 0.5:
                            continue
                        ax_equiv.text(k, v, t, rotation=90, color="w",
                                      fontweight="bold", ha="center", va="top")

                    txt = f"Mean = {equiv_data.loc[:, cn].mean():+0.1f} years"
                    ax_equiv.text(0.95, 0.95, txt, va="top", ha="right",
                                  fontsize="large",
                                  transform=ax_equiv.transAxes)

                    title = f"({letters[iplot]}) Equivalent "\
                            + r"record gain $\delta_{10}[α]$"
                    iplot += 1
                    ax_equiv.set(title=title, ylim=(0, 68),
                                 yticks=[20, 40, 60], xlabel="")
                    ax_equiv.text(0.02, 0.98, "% site",
                                  transform=ax_equiv.transAxes,
                                  va="top", ha="left")
                    ax_equiv.tick_params(axis="y",direction="in",
                                         pad=-22)

        if not has_some_data:
            continue

        fname = f"{fimg.parts[-1]}_{marginal}_{copula}_{cov}_"\
                + f"CO{censo}_CS{censc}_XV{xv_length}.png"
        fp = fimg / f"{fname}_v{version}.png"
        fig.savefig(fp, dpi=fdpi)


LOGGER.completed()
