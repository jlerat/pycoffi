#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2025-10-01 Wed 04:10 PM
## Comment : Predictive checks maps
##
## ------------------------------
import sys
import json
import re
from string import ascii_letters as letters
from pathlib import Path
from itertools import product as prod
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils
from hydrodiy.gis import oz

import path_utils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
cobs = 30
ccov = 30
xv_length = 10
marginal_name = "GEV"
copula = "Gumbel"

version = "7.3"

metric_names = {
        "postpredcheck_obs_pvalue_lskewness2":
           r"$\mathscr{S}_{skew}^X$ - LH Skewness coefficient - observed flow",
        "postpredcheck_cov_pvalue_lskewness2":
           r"$\mathscr{S}_{skew}^Y$ - LH Skewness coefficient - AWRA-L flow",
        "postpredcheck_dep_pvalue_kendalltau":
           r"$\mathscr{S}_{\tau}$ - Kendall-$\tau$ all Obs/AWRA-L",
        "postpredcheck_dep_pvalue_kendalltauhigh":
           r"$\mathscr{S}_{\tau}^+$ - Kendall-$\tau$ all high/AWRA-L"
    }

xlim = [112., 154.]
ylim = [-44., -10.]

awidth = 7.
aheight = 0.8 * awidth * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
fdpi = 300

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
basename = source_file.stem
froot = source_file.parent.parent

fdata = froot / "data"
fout = froot / "outputs" / f"posterior_predictive_checks_v{version}"
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
lon = stations.loc[:, "LONGITUDE[arc_degree]"]
lat = stations.loc[:, "LATITUDE[arc_degree]"]

fd = f"covarfit_{marginal_name}_CO{cobs}" \
     + f"_CV{ccov}_XV{xv_length}_v{version}__predictive_checks.csv"
fd = fout / fd
results, _ = csv.read_csv(fd,
                          dtype={"stationid": str})

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info(f"Plotting {copula}")

plt.close("all")
ncols, nrows = 2, 2
m = list(metric_names.keys())
mosaic = [[m[ir * nrows + ic] for ic in range(ncols)]
          for ir in range(nrows)]
fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                 layout="constrained")

axs = fig.subplot_mosaic(mosaic)

cuts = [0, 0.05, 0.25, 0.75, 0.95, 1.]
bins = np.arange(21) * 0.05
icopula = results.copula_name == copula

for iax, (aname, ax) in enumerate(axs.items()):
    metric_name = aname
    se = results.loc[icopula, ["stationid",
                         metric_name]].set_index("stationid").squeeze()
    x = lon.loc[se.index].values
    y = lat.loc[se.index].values
    z = se.values

    obj = putils.scattercat(ax, x, y, z, cuts=cuts,
                      cmap="PiYG_r", ec="k", ms=60,
                      alphas=0.8, sel=False, scl=True,
                      fmt="0.2f")

    oz.ozlayer(ax, "ozcoast50m", color="k", lw=0.9)

    ax.legend(loc=2, framealpha=1)

    txt = metric_names[metric_name]
    title  = f"({letters[iax]}) {txt}"
    txt = metric_names[metric_name]
    ax.set(title=title, xticks=[], yticks=[])

    axi = ax.inset_axes([0.35, 0.35, 0.42, 0.42])
    se.plot(ax=axi, kind="hist", bins=bins,
            color="0.8", edgecolor="0.2")

    m = re.sub(" -.*", "", txt)
    axi.set(xlabel=r"Posterior predictive check metric $\Pi($" + m + "$)$",
            ylabel="Number of sites")

    axi.text(0.02, 0.98, f"Mean = {se.mean():0.2f}",
             va="top", ha="left",
             fontweight="bold",
             transform=axi.transAxes,
             path_effects=[pe.withStroke(linewidth=3,
                                         foreground="w")])

    b = axi.get_tightbbox().transformed(ax.transData.inverted())
    r = Rectangle((b.x0, b.y0), b.width, b.height,
                  edgecolor="none", facecolor="w",
                  zorder=axi.get_zorder() - 0.1)
    ax.add_patch(r)

fp = fimg / f"{fimg.parts[-1]}_{copula}_v{version}.png"
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()

