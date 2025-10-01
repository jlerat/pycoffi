#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2025-06-26 Thu 11:37 AM
## Comment : Figure showing posterior predictive checks
##
## ------------------------------
import re
import sys
from pathlib import Path
from string import ascii_letters as letters
from itertools import product as prod
import argparse
import math

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe

from netCDF4 import Dataset

from hydrodiy.io import csv, iutils
from hydrodiy.io import hyruns
from hydrodiy.plot import putils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
data_version = "5.0"

version = "7.3"

cobs = 30
ccov = 30
stationid = "138001"
copula = "Gumbel"
marginal_name = "GEV"

metric_names = [f"obs_ari{ari}_postpred_logbias"
                for ari in [10, 100]]

awidth = 5
aheight = 5
fdpi = 300

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

fm = fout / f"covarfit_metrics_v{version}.csv"
metrics, _ = csv.read_csv(fm, dtype={"stationid": str})

idx = metrics.marginal_name == marginal_name
idx &= metrics.cens_prob_thresh_obs == cobs
idx &= metrics.cens_prob_thresh_cov == ccov
idx &= metrics.copula_name == copula
idx &= metrics.prior == "default"
idx &= ~metrics.clip_to_obs
idx &= metrics.timeperiod == "ALL"
idx &= metrics.stationid == stationid

cc = ["stationid", "xvperid", "covariate", "xv_length"]\
     + metric_names

metrics = metrics.loc[idx, cc]
metrics.loc[:, metric_names] = metrics.loc[:, metric_names].abs()

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info(f"Dealing with {stationid}")
idx0 = metrics.stationid == stationid
idx = idx0 & (metrics.covariate == "NONE")
refm = metrics.loc[idx, ["xv_length"] +
                   metric_names].groupby("xv_length").mean()
refs = metrics.loc[idx, ["xv_length"] +
                   metric_names].groupby("xv_length").std()

idx = idx0 & (metrics.covariate == "QTOT_AWRAL_MEAN")
altm = metrics.loc[idx, ["xv_length"] +
                   metric_names].groupby("xv_length").mean()
alts = metrics.loc[idx, ["xv_length"] +
                   metric_names].groupby("xv_length").std()

plt.close("all")
mosaic = [["ARI10", "ARI100"]]
nrows, ncols = len(mosaic), len(mosaic[0])
fig = plt.figure(figsize=(awidth * ncols, aheight * nrows),
                 layout="constrained")
kw = dict(hspace=0.08)
axs = fig.subplot_mosaic(mosaic, sharex=True,
                         gridspec_kw=kw)

y1 = -np.inf
for iax, (aname, ax) in enumerate(axs.items()):
    ari = re.sub("ARI", "", aname)
    cn = next(cn for cn in refm.columns if re.search(f"ari{ari}_", cn))
    yr = refm.loc[:, cn]
    ya = altm.loc[:, cn]
    k = np.abs(yr - ya[10]).idxmin()
    yar = ya[10]

    label = r"$\bar{D}_{G,t}$ Univariate"
    ax.plot(yr.index, yr, "o-", label=label,
            color="tab:blue", ms=7)

    label = r"$\bar{D}_{F,t}$ CoFFI"
    ax.plot(ya.index, ya, "o-",
            color="tab:red", alpha=0.2, ms=5)

    ax.plot(10, ya[10], "o", label=label,
            color="tab:red", ms=7)

    ax.plot(k, yr[k], "o", ms=12, mfc="none", mec="tab:red",
            label="Univariate discrepancy\nequivalent to CoFFI",
            mew=2)

    props = dict(arrowstyle="-|>", linestyle="--",
                 lw=1.5, color="0.4")
    ax.annotate("", xytext=(10.5, yar), xy=(k-0.5, yar),
                arrowprops=props)

    ax.annotate("", xytext=(k, yr[k]-0.02), xy=(k, 0.005),
                arrowprops=props)

    ax.text(k, 0.01, "$\Delta_{10}[α] = $" + f"{k} years  ", color="0.3",
            fontsize="large", va="bottom", ha="right",
            fontstyle="italic")

    title = f"({letters[iax]}) " + r"Flood AEP level $\alpha$ = 1:" + ari
    xlab = r"Restricted observed record duration $t$ [years]"
    ax.set(title=title,
           xlabel=xlab,
           ylabel=r"Average discrepancy $\bar{D}$ [-]")

    ax.legend(loc=1, framealpha=0, fontsize="large")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    _, y = ax.get_ylim()
    y1 = max(y, y1)

for _, ax in axs.items():
    ax.set_ylim((0, y1))

fp = fimg / f"{fimg.parts[-1]}_v{version}_{stationid}.png"
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()

