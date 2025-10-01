#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2025-10-01 Wed 02:18 PM
## Comment : Figure 2 - timeseries data
##
## ------------------------------
import sys
import math
import json
import re
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from scipy.stats import kendalltau

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
awidth = 11
aheight = 3
fdpi = 300
nb_fmt = "{x:0,.0f}"

data_version = "5.0"
stationid = "138001"
xvperid = "XVP_10_019"

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
basename = source_file.stem

froot = source_file.parent.parent

fdata = froot / "data"

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

fd = fdata / "covariate" / \
    f"{stationid}_covariate_ams_data_v{data_version}.csv"
df, _ = csv.read_csv(fd, index_col="YEAR")

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info(f"Plotting {stationid}", nret=1)
qobs = df.loc[:, "OBS_STREAMFLOW_AMS[m3_s-1]"]
qsim = df.loc[:, "AWRAL_STREAMFLOW_AMS[mm_day-1]"]

area = stations.loc[stationid, "CATCHMENTAREA[km2]"]
qsim *= area / 86.4

fx = fdata / f"xvperiods" / f"{stationid}_xvperiods_v{data_version}.csv"
xvperiods, _ = csv.read_csv(fx)
idx = xvperiods.xvperid == xvperid
xv = xvperiods.loc[idx].squeeze()

xvse = qobs.copy()
xvy = xv.index[xv == 0].astype(int)
xvse.loc[xvy] = np.nan

plt.close("all")
fig = plt.figure(layout="constrained",
                 figsize=(awidth, aheight * 2))
mosaic = [["qobs", "sc"], ["qsim", "."]]
kw = dict(width_ratios=[3, 1])
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

for aname, ax in axs.items():
    if aname == "qobs":
        se = qobs
        ylabel = "Streamflow [m3.s-1]"
        title = f"(a) Annual maximums of observed streamflow"
    elif aname == "qsim":
        se = qsim
        ylabel = "Streamflow [m3.s-1]"
        title = f"(b) Annual maximums of AWRAL simulated streamflow"

    if aname == "sc":
        ax.plot(qsim, qobs, "o", mfc="none", mec="k",
                ms=6, alpha=0.7)

        ixv = xvse.index[xvse.notnull()]
        ax.plot(qsim[ixv], qobs[ixv], "o",
                ms=8, mfc="none", mec="tab:orange",
                markeredgewidth=2)

        putils.line(ax, 1, 1, 0, 0, "k:", lw=0.8)

        iok = qobs.notnull() & qsim.notnull()
        tau = kendalltau(qobs[iok], qsim[iok]).statistic
        ax.text(0.05, 0.95, r"$\tau =$" + f"{tau:0.2f}",
                transform=ax.transAxes,
                va="top", ha="left")

        title = f"(c) Comparison Obs / AWRAL"
        ylabel = "Streamflow [m3.s-1]"
        ax.set(xlabel="AWRAL [m3.s-1]",
               ylabel=ylabel, yticks=[],
               title=title)

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(4))
            axis.set_major_formatter(ticker.StrMethodFormatter(nb_fmt))

        continue

    se.plot(ax=ax, color="0.5", marker="o",
            mec="k", mfc="w", label="")

    # XV periods
    if aname == "qobs":
        xvse.plot(ax=ax, linestyle="none",
                  marker="o", markersize=11,
                  markeredgewidth=2,
                  mfc="none", mec="tab:orange",
                  label=f"Observation retained\nin restricted subset")

        ax.legend(loc=2, framealpha=0.)

    ax.set(ylabel=ylabel, title=title)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(nb_fmt))


fp = fimg / f"{basename}_{stationid}.png"
fig.savefig(fp)

LOGGER.completed()
