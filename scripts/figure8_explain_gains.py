#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2025-10-01 Wed 04:04 PM
## Comment : Explain CoFFI performance gains
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
from scipy.stats import kendalltau

import matplotlib.pyplot as plt
from matplotlib import ticker


from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
version = "7.3"

marginal_ref = "GEV"
copula_ref = "Gumbel"
cens_prob_thresh_obs_ref = 30
cens_prob_thresh_cov_ref = 30
prior_ref = "default"
timeperiod_ref = "ALL"
clip_to_obs_ref = False
xv_length_ref = 10
obs_type = "obs"

aris = [10]
metric_names = [f"obs_ari{ari}_postpred_logbias" for ari in aris]

output_name = "covarfit"

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

fr = fout / f"{output_name}_metrics_v{version}.csv"
metrics, _ = csv.read_csv(fr, dtype={"stationid": str})

def reformat(cov):
    cov = cov.str.replace("_MEA.*", "", regex=True)
    return cov.str.replace(".*_", "", regex=True).str.lower()
metrics.loc[:, "covariate"] = reformat(metrics.covariate)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
index = metrics.cens_prob_thresh_obs == cens_prob_thresh_obs_ref
index &= metrics.prior == prior_ref
index &= metrics.timeperiod == timeperiod_ref
index &= metrics.obs_type == obs_type
index &= metrics.marginal_name == marginal_ref
index &= metrics.clip_to_obs == clip_to_obs_ref
index &= metrics.cens_prob_thresh_cov == cens_prob_thresh_cov_ref
index &= metrics.copula_name == copula_ref
index &= metrics.covariate.isin(["none", "awral"])
index &= metrics.xv_length == xv_length_ref
index &= metrics.stationid.isin(stations.index)
me = metrics.loc[index]

plt.close("all")

mosaic = [[m] for m in metric_names]
fig = plt.figure(figsize=(awidth, len(metric_names) * aheight),
                 layout="constrained")
axs = fig.subplot_mosaic(mosaic)

for metric_name, ax in axs.items():
    df = pd.pivot_table(me, index=["stationid", "xvperid"],
                        columns="covariate",
                        values=metric_name).reset_index()

    df = df.drop("xvperid", axis=1)\
        .groupby("stationid")\
        .apply(lambda x: x.abs().mean())

    cn = "KENDALLTAUHIGH_ALL[-]"
    df = pd.concat([df, stations.loc[:, cn]], axis=1)

    x = df.loc[:, cn]
    y = df.loc[:, "none"] - df.loc[:, "awral"]
    tau = kendalltau(x, y).statistic
    LOGGER.info(f"{metric_name}: tau = {tau:0.3f}")
    ax.plot(x, y, "o")

    X = np.column_stack([np.ones_like(x), x])
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-6)
    r2 = np.corrcoef(x, y)[0, 1]
    label = f"Regression line y = {theta[0]:+0.2f}{theta[1]:+0.2f} x"\
            + " (r$^2$=" + f"{r2:0.2f})"
    putils.line(ax, 1, theta[1], 0, theta[0], "k--", label=label)

    ari = int(re.search("(?<=ari)[\d]+", metric_name).group())
    alpha = f"{1./ari:0.1f}"
    ylabel = "Difference in discrepancies $\\bar{D}_{G,10}-\\bar{D}_{F,10}$ [-]"
    ax.set(xlabel=r"$\mathscr{S}_\tau^+$ Kendall-tau above median Obs/AWRA-L [-]",
           ylabel=ylabel)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.legend(loc=2)

fp = fimg / f"{fimg.parts[-1]}_v{version}.png"
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()
