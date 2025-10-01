#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2025-10-01 Wed 03:03 PM
## Comment : Frequency plot
##
## ------------------------------
import sys
import math
import re
import json
from string import ascii_letters as letters
from pathlib import Path
from itertools import product as prod
import argparse

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patheffects as pe

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import violinplot, putils

from floodstan import freqplots

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
version = "7.3"

timeperiod = "ALL"
prior = "default"

data_version = "5.0"

xvperid = "XVP_10_019"
stationid = "138001"

cobs = 30
ccov = 30
xv_length = 10
marginal_name = "GEV"

awidth = 6
aheight = 4
fdpi = 300
nb_fmt = "{x:0,.0f}"

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
basename = source_file.stem
froot = source_file.parent.parent

fdata = froot / "data"
fout = froot / "outputs" / f"results_{stationid}_v{version}"
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

data = {}
for dtype in ["nocov-ref", "nocov",
              "cov-ref", "cov"]:
    fs = f"{stationid}_{marginal_name}_CO{cobs}" \
         + f"_CV{ccov}_XV{xv_length}_v{version}__{dtype}.csv"
    fs = fout / fs
    df, _ = csv.read_csv(fs)
    data[dtype] = df

# ----------------------------------------------------------------------
# @Utils
# ----------------------------------------------------------------------
def plot_ffa(ax, quantiles, q0_col, q1_col,
             col, ptype, lab, lw, linestyle="-"):
    freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                      center_column="center",
                                      q0_column=q0_col,
                                      q1_column=q1_col,
                                      label=lab,
                                      alpha=0.3,
                                      color=col,
                                      lw=2,
                                      linestyle=linestyle,
                                      facecolor=col,
                                      edgecolor="none")

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info(f"{stationid} : Plotting {marginal_name}, "
            + f"co={cobs} cs={ccov} xv={xv_length}",
            nret=1)

# Build dataframe
data["nocov"] = data["nocov"].loc[data["nocov"].xvperid == xvperid, :]
data["cov"] = data["cov"].loc[data["cov"].xvperid == xvperid, :]
data = pd.DataFrame({n: data[n].squeeze() for n in data})

# Get obs data
obs = data.filter(regex="^obs_\d", axis=0)
obs = obs.loc[obs.notnull().any(axis=1)].astype(float)
obs.loc[:, "subsample"] = obs.loc[:, "cov"].notnull()
obs = obs.loc[:, ["nocov-ref", "subsample"]].sort_values("nocov-ref")
y = obs.index.str.replace("obs_|_value", "", regex=True).astype(int)
obs = obs.set_index(y).sort_values("nocov-ref")

# Get ams fit
ams = {}
for dtype, res in data.items():
    x1 = res.filter(regex=f"^obs_ari.*_postpred$")
    x0 = res.filter(regex="^obs_ari.*_q5")
    x2 = res.filter(regex="^obs_ari.*_q95")

    aris = x0.index.str.replace("_ari1_", "_ari1.")\
            .str.replace("obs_ari|_[^_]+$", "", regex=True).astype(float)
    ams[dtype] = pd.DataFrame({"center": x1.values, "q0": x0.values,
                               "q1": x2.values}, index=aris).astype(float)

# Plot configs
colors = {"nocov": "tab:blue",
          "nocov-ref": "0.3",
          "cov-ref": putils.darken_or_lighten("tab:red", 0.2),
          "cov": "tab:red",
          }

# Plots
ptype = "gumbel"
plt.close("all")

mosaic = [[f"ax1_{c}", f"ax2_{c}"] for c in ["uni", "biv"]]
nrows, ncols = len(mosaic), len(mosaic[0])
figsize = (awidth * ncols,
           aheight * nrows)
fig = plt.figure(figsize=figsize,
                 layout="constrained")

kw = dict(wspace=0.07, hspace=0.06)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw,
                         sharex=True)

# plot data
for mod in ["uni", "biv"]:
    ax1 = axs[f"ax1_{mod}"]
    ax2 = axs[f"ax2_{mod}"]

    nval = len(obs)
    xobs, _ = freqplots.plot_data(ax1, obs.loc[:, "nocov-ref"],
                                  ptype,
                                  markersize=5,
                                  label=f"All obs ({nval})")
    x = xobs[obs.subsample]
    y = obs.loc[obs.subsample, "nocov-ref"]
    nval = len(x)
    lab =f"Restricted obs ({nval})"
    ax1.plot(x, y, "o", ms=9,
             mec="tab:orange", mfc="none",
             markeredgewidth=2.0,
             label=lab)

    freqplots.plot_data(ax2, y, ptype,
                        markersize=5)

    x = xobs[obs.subsample]
    kw = dict(ms=9, markeredgewidth=2.0,
              markeredgecolor="tab:orange",
              markerfacecolor="none")
    freqplots.plot_data(ax2, y, ptype,
                        label=lab, **kw)

labels =  {
        "nocov-ref": r"$G^*_X$ Univariate reference",
        "nocov": r"$G_{X,10}^{(k)}$ Univariate",
        "cov": r"$F_{X,10}^{(k)}$ CoFFI",
        }
# Plot quantiles
for iax, (aname, ax) in enumerate(axs.items()):
    if re.search("uni", aname):
        keys = ["nocov-ref", "nocov"]
    else:
        keys = ["nocov-ref", "cov"]

    LOGGER.info(f"{aname} -> {keys}", ntab=2)
    acfg = re.sub("_.*", "", aname)

    for qname in keys:
        quantiles = ams[qname]

        show_ci = False
        lw = 2
        ls = "--" if qname == "nocov-ref" else "-"
        col = colors[qname]

        if show_ci:
            q0_col = "q0"
            q1_col = "q1"
        else:
            q0_col = "none"
            q1_col = "none"

        if re.search("ref", qname):
            nval = len(obs)
        else:
            nval = obs.subsample.sum()

        lab = f"{labels[qname]} ({nval} obs)"
        plot_ffa(ax, quantiles, q0_col, q1_col,
                 col, ptype, lab, lw, ls)

# Decorate
for iax, (aname, ax) in enumerate(axs.items()):
    freqplots.set_xlim(ax, ptype, 1.02, 500)
    freqplots.set_xlabel(ax, ptype)

    y0 = 0
    ymax = obs.loc[:, "nocov-ref"].max() * 2
    ax.set_ylim((y0, ymax))

    retp = [10, 100]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype,
                                            return_periods=retp)

    ylab = "Annual Maximum Streamflow [m3.s-1]"
    if aname.startswith("ax1"):
        title = f"({letters[iax]}) Frequency plot using all data"
    elif aname.startswith("ax2"):
        title = f"({letters[iax]}) Frequency plot using restricted dataset"

    if re.search("biv", aname):
        title += " / CoFFI"
    else:
        title += " / univariate"

    ax.set(ylabel=ylab, title=title)
    ax.legend(loc=2)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(nb_fmt))

    if iax < 2:
        ax.set_xlabel("")

fname = f"{fimg.parts[-1]}_{stationid}_{marginal_name}_CO{cobs}" \
        + f"_CV{ccov}_XV{xv_length}_v{version}.png"
fp = fimg / fname
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()
