#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2025-10-01 Wed 04:09 PM
## Comment : Impact of record length
##
## ------------------------------
import sys
import json
import re
from string import ascii_letters as letters
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

marginal_name = "GEV"
cobs = 30
csim = 30
xv_length = 10
copula = "Gumbel"
prior = "default"
clip2obs = False
timeperiod = "ALL"

output_name = "covarfit"

aris = [10, 100]
metric_names = [f"obs_ari{ari}_postpred_logbias" for ari in aris]

awidth = 7
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


def reformat(cov):
    cov = cov.str.replace("_MEA.*", "", regex=True)
    return cov.str.replace(".*_", "", regex=True).str.lower()

fr = fout / f"{output_name}_metrics_equiv_v{version}.csv"
metrics_equiv, _ = csv.read_csv(fr, dtype={"stationid": str})
metrics_equiv.loc[:, "covariate"] = reformat(metrics_equiv.covariate)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
ieq = metrics_equiv.cens_prob_thresh_obs == cobs
ieq &= metrics_equiv.prior == prior
ieq &= metrics_equiv.covariate == "awral"
ieq &= metrics_equiv.timeperiod == timeperiod
ieq &= metrics_equiv.marginal_name == marginal_name
ieq &= metrics_equiv.clip_to_obs == clip2obs
ieq &= metrics_equiv.stationid.isin(stations.index)
equiv = metrics_equiv.loc[ieq]

cc = [cn for cn in equiv.columns
      if re.search(f"stationid|metric|CS{csim}", cn)]
equiv = equiv.loc[:, cc]
equiv.columns = [re.sub("CS[\d]+_", "", cn) for cn in equiv.columns]

plt.close("all")
nm = len(metric_names)
mosaic = [[m] for m in metric_names]
ncols, nrows = len(mosaic[0]), len(mosaic)

fz = (ncols * awidth, nrows * aheight)
fig = plt.figure(layout="constrained",
                 figsize=fz)
kw = dict(hspace=0.1)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw, sharey=True)

for iax, (metric_name, ax) in enumerate(axs.items()):
    equivm = equiv.loc[equiv.metric_name == metric_name]\
                .filter(regex="XV", axis=1)
    equivm.columns = [int(re.sub("XV", "", cn)) for cn in equivm.columns]

    T = equivm.columns.values[None, :]
    equivm = equivm - T

    vl = violinplot.Violin(equivm,
                           number_format="0.1f")
    vl.draw(ax=ax)

    putils.line(ax, 1, 0, 0, 0, "k-", lw=0.8)

    ari = re.search("(?<=ari)[\d]+", metric_name).group()
    title = f"({letters[iax]}) Equivalent record "\
            + f"gain for AEP $\\alpha$ = 1:{ari}"
    xlabel = "Restricted observed record length $T$ [years]"
    ylabel = r"Equivalent record gain $\delta_T[\alpha]$ [years]"
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", lw=0.5, ls="--", color="grey")

fname = f"FIGC_{marginal_name}_{copula}"\
        + f"_CO{cobs}_PR{prior}_XV{xv_length}_T{timeperiod}"\
        + f"_C2O{clip2obs}_v{version}"
fp = fimg / f"{fname}.png"
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()
