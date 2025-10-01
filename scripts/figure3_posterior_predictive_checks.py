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
from matplotlib import ticker
from matplotlib import patches
import matplotlib.patheffects as pe

from shapely import Polygon

from hydrodiy.io import csv, iutils
from hydrodiy.io import hyruns
from hydrodiy.plot import putils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
version = "7.3"

stationid = "138001"

data_version = "5.0"

awidth = 5
aheight = 3
fdpi = 300
nb_fmt = "{x:0,.0f}"

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent
basename = source_file.stem

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
fs = fdata / f"stations.csv"
stations, _ = csv.read_csv(fs, index_col="STATIONID",
                           dtype={"STATIONID": str})

data = {}
for f in list(fout.glob("*.csv")) + list(fout.glob("*.zip")):
    name = re.sub(".*check_", "", f.stem)
    df, _ = csv.read_csv(f, index_col=0)
    data[name] = df

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

# plot
plt.close("all")
mosaic = [["obs/lskewness2", "cov/lskewness2"],
          ["dep/kendalltau", "dep/kendalltauhigh"]]
ncols, nrows = len(mosaic[0]), len(mosaic)
fig = plt.figure(layout="constrained",
                 figsize=(ncols * awidth, nrows * aheight))
kw = dict(hspace=0.08)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

for iax, (aname, ax) in enumerate(axs.items()):
    varn, metric = re.split("/", aname)

    if varn == "obs":
        k = "univy"
        varntxt = "LH Skewness coefficient - observed flow"
        pc = data["y"]
    elif varn == "cov":
        k = "univz"
        varntxt = "LH Skewness coefficient - AWRA-L flow"
        pc = data["z"]
    else:
        k = varn
        if re.search("high", metric):
            varntxt = r"Kendall-$\tau$ high Obs/AWRA-L"
        else:
            varntxt = r"Kendall-$\tau$ all Obs/AWRA-L"
        pc = data["dep"]

    if re.search("skewness", aname):
        stxt = r"$\mathscr{S}_{skew}^X$" if re.search("obs", aname) \
                else r"$\mathscr{S}_{skew}^Y$"
    else:
        stxt = r"$\mathscr{S}_{\tau}^+$" if re.search("high", aname) \
                else r"$\mathscr{S}_{τ}$"

    pvalue = pc.loc[metric, "pvalue"]
    pvaluediscr = pc.loc[metric, "pvaluediscr"]

    obs = data[f"{k}_obs"].loc[metric]
    sim = data[f"{k}_sim"].loc[:, metric]

    if re.search("tailcoeff", metric):
        bins = np.arange(sim.min(), sim.max() + 1)
        xticks = bins[:-1] + 0.5
        xticklabels = [f"{b:0.0f}" for b in bins[:-1]]
        delta = 0.5
        rwidth = 0.8
    else:
        bins = None
        xticks = None
        xticklabels = None
        delta = 0.
        rwidth = 1.

    h = ax.hist(sim, bins=bins, rwidth=rwidth,
            facecolor="lightblue",
            edgecolor="darkgrey",
            label="Computed from\nCoFFI samples")

    putils.line(ax, 0, 1, obs + delta, 0, "k--",
                label="Computed from\ndata")

    # Polygon region representing Pi(S)
    p = Polygon()
    nh = len(h[0])
    for ib in range(nh):
        y = max(1e-5, h[0][ib])
        x0, x1 = h[1][[ib, ib+1]]
        cc = [(x0, 0), (x1, 0),
              (x1, y), (x0, y),
              (x0, 0)]
        p = p.union(Polygon(cc))

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cc = [(obs, y0), (obs, y1),
          (1+x1, y1), (1+x1, y0), (obs, y0)]
    p = p.intersection(Polygon(cc))
    if hasattr(p, "geoms"):
        p = next(b for b in p.geoms if isinstance(b, Polygon))

    xtxt = 0.7 * obs + 0.3 * p.centroid.x
    ytxt = p.centroid.y

    bnd = p.exterior.coords
    bnd = np.array([c for c in bnd])
    p = patches.Polygon(bnd, hatch="/", fc="none")
    ax.add_patch(p)

    if iax == 0:
        ax.legend(loc=1, framealpha=0.)

    p = r"$\Pi$" + "(" + stxt + ")=" + f"{pvalue:0.2f}"
    ax.text(xtxt, ytxt, p,
            va="center", ha="left",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=6,
                                        foreground="w")])

    title = f"({letters[iax]}) {stxt} - {varntxt}"
    ax.set(xlabel=f"{stxt} (-)", ylabel="Number of samples",
           title=title)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(nb_fmt))

    if not xticks is None:
        ax.set(xticks=xticks,
               xticklabels=xticklabels)

fp = fimg / f"FIGB_{stationid}_posterior_checks_v{version}.png"
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()

