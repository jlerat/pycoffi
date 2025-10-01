#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO Environment
## Created : 2025-10-01 Wed 02:31 PM
## Comment : Site map
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

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils
from hydrodiy.gis import oz

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
data_version = "5.0"

xlim = [112., 154.]
ylim = [-44., -10.]

awidth = 7.
aheight = awidth * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
fdpi = 300

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

lon = stations.loc[:, "LONGITUDE[arc_degree]"]
lat = stations.loc[:, "LATITUDE[arc_degree]"]

qm = stations.loc[:, "STREAMFLOW_MEAN[mm/yr]"]
pm = stations.loc[:, "PRECIPITATION_MEAN[mm/yr]"]
dur = stations.loc[:, "DURATION[yr]"]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
# Charac table
cc = [
    "CATCHMENTAREA[km2]",
    "DURATION[yr]",
    "STREAMFLOW_MEAN[mm/yr]",
    "PRECIPITATION_MEAN[mm/yr]",
    "MANNKENDALL[-]"
    ]
df = stations.loc[:, cc]

idx = ["min", "25%", "50%", "75%", "max"]
st = df.describe().loc[idx, :].T.copy()
st.loc[:, "MIVA"] = stations.loc["138001", cc].values

ck = "MANNKENDALL[-]"
cc = [cn for cn in cc if cn != ck]

st.loc[cc, :] = st.loc[cc, :].astype(int)
st.loc[ck, :] = st.loc[ck, :].astype(float).apply(lambda x: f"{x:0.2f}")

df.loc[:, cc] = df.loc[:, cc].astype(int)
df.loc[:, ck] = df.loc[:, ck].astype(float).apply(lambda x: f"{x:0.2f}")
df = pd.concat([stations.NAME, df], axis=1)

fd = fimg / f"table_dv{data_version}.csv"
csv.write_csv(df, fd,
              "Summary table", source_file,
              compress=False, write_index=True,
              lineterminator="\n")


fd = fimg / f"summary_dv{data_version}.csv"
csv.write_csv(st, fd,
              "Summary table", source_file,
              compress=False, write_index=True,
              lineterminator="\n")

# Site map
plt.close("all")
fig, ax = plt.subplots(figsize=(awidth, aheight),
                     layout="constrained")

putils.scattercat(ax, lon, lat, qm / pm,
                  cmap="viridis_r", ec="k", ms=70,
                  alphas=0.8, scl=True,
                  fmt="0.2f",
                  linewidth=2)

oz.ozlayer(ax, "ozcoast50m", color="k", lw=0.9)

ax.legend(loc=2, title="Runoff Coefficient\nmean(Q)/mean(P) [-]")

ax.set(xticks=[], yticks=[],
       xlim=xlim, ylim=ylim)

fp = fimg / f"{basename}_v{data_version}.png"
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()

