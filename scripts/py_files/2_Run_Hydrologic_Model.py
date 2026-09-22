# %% [markdown]
# # Run Hydrologic Model
#
# In this notebook, we take the data that we downloaded in `1_Download_Data.ipynb`
# and we run a hydrologic model. The steps in this notebook are:
#
# 1. Load the input data
# 2. Create the input forcing data object
# 3. Define the model parameters
# 4. Create the hydrologic model
# 5. Run the hydrologic model
# 6. View the model outputs object
# 7. Plot the simulation against the model inputs
#
# This script is written as a Jupytext-style "percent" script: code cells are
# delimited by `# %%` and text (markdown) cells are delimited by
# `# %% [markdown]`. To convert it to a Jupyter notebook, run, for example:
#
#     jupytext --to notebook -o 2_Run_Hydrologic_Model.ipynb 2_Run_Hydrologic_Model.py

# %%
# Import the packages we need. `potions` (imported as `pt`) is the reactive
# transport / hydrologic modeling package that the rest of this tutorial uses.
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, Series

import potions as pt

# The notebooks in this repo are expected to be run from the `scripts`
# directory, so the data lives one level up, in `../data`.
data_dir: str = "../data"
forcing_path: str = os.path.join(data_dir, "model_inputs", "forcing.csv")
measured_path: str = os.path.join(data_dir, "model_inputs", "measured_data.csv")

# %% [markdown]
# ## 1. Load the input data
#
# Two CSV files were written out by the first notebook:
#
# - `model_inputs/forcing.csv` holds the daily driver series: precipitation
#   (`ppt`, mm/day), air temperature (`temp`, deg C), and potential
#   evapotranspiration (`pet`, mm/day).
# - `model_inputs/measured_data.csv` holds the observed streamflow (`q_mmd`,
#   mm/day) and the observed dissolved organic carbon concentration
#   (`doc_mol_l`, mol/L). Only the streamflow column is needed for a hydrologic
#   run; the DOC column is used later by the reactive transport notebooks.
#
# We read both into DataFrames with a parsed datetime index.

# %%
forcing_df: DataFrame = pd.read_csv(forcing_path, index_col=0, parse_dates=True)
measured_df: DataFrame = pd.read_csv(measured_path, index_col=0, parse_dates=True)

print(
    f"Forcing spans {forcing_df.index[0].date()} to {forcing_df.index[-1].date()} ({len(forcing_df)} days)"
)
print(
    f"Measured streamflow spans {measured_df.index[0].date()} to {measured_df.index[-1].date()} ({len(measured_df)} days)"
)

forcing_df.head()
# %% [markdown]
# ## 2. Create the input forcing data object
#
# A model is driven by a `pt.ForcingData` object, which is simply a container
# bundling the three daily meteorological time series used by the model:
# precipitation, temperature, and potential evapotranspiration.
#
# **Governing idea.** The model advances one time step at a time. At each step it
# consumes one value of each of these series, all aligned to the same daily date
# index. The first row of each series corresponds to the first day the model
# simulates.

# %%
# Bundle the three CSV columns into a single forcing object. These series must
# line up on the same date index so that step *i* of the simulation uses the
# `i`-th value of each series.
forcing: pt.ForcingData = pt.ForcingData(
    precip=forcing_df["ppt"],
    temp=forcing_df["temp"],
    pet=forcing_df["pet"],
)

# The observed streamflow (mm/day) used to score the simulation.
meas_streamflow: Series = measured_df["q_mmd"]

print(f"Number of forcing time steps: {len(forcing.precip)}")
print(f"Number of measured streamflow values: {len(meas_streamflow)}")

# %% [markdown]
# ## 3. Define the model parameters
#
# A hydrologic model is a stack of conceptual storage "zones" through which water
# moves. The built-in `HbvModel` structure (an HBV-style model) has four zones,
# from the top of the column to the bottom:
#
# - **Snow**: stores snow and melts it. Parameters `tt` (threshold temperature,
#   deg C) and `fmax` (max daily melt, mm/deg C). When the air temperature is at
#   or below `tt`, precipitation is stored as snow; otherwise the snow pack melts
#   at a rate proportional to how far the temperature is above `tt`.
# - **Surface**: the soil / fast-response store. Parameters `fc` (field capacity,
#   mm), `lp` (lip limit, 0-1), `beta` (infiltration exponent), `k0` (quick
#   runoff coefficient), and `thr` (runoff threshold, mm). Evaporation is limited
#   by `pet` and the stored water; once storage exceeds `thr` the surplus runs off
#   quickly, while the remainder infiltrates at a rate set by `(s/fc)**beta`.
# - **Shallow groundwater**: a linear reservoir with a recession constant `k`
#   (1/day), an exponent `alpha`, and a percolation limit `perc` (mm). Baseflow
#   out is `k * s**alpha`.
# - **Deep groundwater**: a second, slower linear reservoir with recession
#   constant `k` and exponent `alpha`. It has no deeper vertical outflow, so it
#   only drains to the river.
#
# The sum of the **lateral outflows** (`q_lat`) from the bottom of each column
# (here the shallow and deep groundwater zones) is the simulated streamflow.
#
# Every zone ships with **default** parameters, so a model can be run instantly.
# Here we override a few zones to tailor the response to our catchment by passing
# a dictionary of `zones` keyed by zone **name**. Any zone we do not mention keeps
# its defaults.

# %%
# Build the custom zone objects. We are free to change whichever parameters we
# like; the ones not passed here take the class defaults. See the zone classes
# (e.g. `pt.SurfaceZone`) for the full list of parameters.
custom_zones: dict[str, pt.HydrologicZone] = {
    # Raise the melting threshold a little above freezing so the snow pack
    # holds onto water a bit longer and melts more strongly as it warms.
    "snow": pt.SnowZone(tt=0.5, fmax=2.0),
    # A larger field capacity lets the surface store hold more water before
    # runoff kicks in; a higher delay threshold does the same for quick runoff.
    "surface": pt.SurfaceZone(fc=250.0, thr=25.0),
    # A faster shallow recession and a tighter percolation limit send water to
    # the deep reservoir a little more readily.
    "shallow": pt.GroundZone(k=2.0e-2, perc=0.5),
    # The deep reservoir is left at its defaults.
}

# %% [markdown]
# ## 4. Create the hydrologic model
#
# We instantiate the model, passing in our custom zones. `HbvModel` wires up the
# four-zone structure for us; providing `zones` replaces the default parameters
# for whichever zones we named. The model validates its own structure (that the
# zones stack and connect correctly) when it is run.

# %%
model: pt.HbvModel = pt.HbvModel(zones=custom_zones)

print("Zones in the model (top to bottom):")
for zone_name in model.get_zone_names():
    print(f"  - {zone_name}: {model[zone_name].__class__.__name__}")

# The complete parameterisation, keyed as '<zone>.<parameter>'.
model.to_dict()

# %% [markdown]
# ## 5. Run the hydrologic model
#
# Running the model steps the whole column forward in time, one day at a time,
# using the forcing data. We pass in the observed streamflow so the run
# automatically computes a suite of performance metrics (KGE, NSE, bias, and
# correlation) and returns a `pt.HydroModelResults` object.
#
# **Notes.** The model starts from its default initial storage in every zone (a
# small amount of water in the snow pack and the reservoirs), approximating a
# "warmed-up" seasonal start. Because the forcing begins on 2008-10-01 but the
# first measured value is on 2009-10-01, the first year of the simulation acts
# as a spin-up that is not directly compared to observations.

# %%
hydro_res: pt.HydroModelResults = model.run_hydro_model(
    forc=forcing,
    meas_streamflow=meas_streamflow,
)

print("Simulation complete.")

# %% [markdown]
# ## 6. View the model outputs object
#
# `run_hydro_model` returns a `pt.HydroModelResults` object. It holds three
# things:
#
# - `simulation`: a `DataFrame` with one column group per zone. For each zone
#   you get its storage (`s_<zone>`) and its fluxes: incoming forcing
#   (`q_forc_<zone>`), evapotranspiration (`q_vap_<zone>`), lateral outflow /
#   runoff (`q_lat_<zone>`), vertical outflow / percolation
#   (`q_vert_<zone>`), and the total incoming flux (`q_in_<zone>`). Two more
#   columns summarise the column as a whole: `streamflow_sim` (the modeled
#   streamflow, mm/day) and `meas_streamflow` (the observed streamflow,
#   mm/day).
# - `objective_functions`: a `Series` of the performance metrics computed
#   against the observed streamflow.
# - `forcing`: the forcing data used for the run.
#
# Let's inspect all of it.

# %%
# The performance metrics comparing simulated to observed streamflow.
print("Performance metrics:")
print(hydro_res.objective_functions)
print()

# The full simulation table. It is long, so we only show the first few rows and
# the shape here.
simulation_df: DataFrame = hydro_res.simulation
print(f"Simulation output shape: {simulation_df.shape}")
print()
simulation_df.head()

# %% [markdown]
# ## 7. Plot the simulation against the model inputs
#
# We now compare the modeled streamflow to the observations, and look at the
# internal water storages over time. A few small style settings keep the figures
# clean. The top panel shows the daily drivers (precipitation as bars,
# temperature as a line) so we can relate the streamflow response to what
# drove it; the bottom panel shows the storage in each of the four zones.

# %%
plt.rcParams["figure.dpi"] = 200
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.right"] = False

# A shared x-axis window: the observed period only.
obs_start: pd.Timestamp = meas_streamflow.index[0]
obs_end: pd.Timestamp = meas_streamflow.index[-1]

# ---- Panel 1: streamflow vs. precipitation and temperature ---- #
fig: Figure = plt.figure(figsize=(10, 4))
ax: Axes = fig.gca()

# Drivers on secondary axes for context.
ax_ppt: Axes = ax.twinx()
ax_ppt.bar(
    forcing_df.index,
    forcing_df["ppt"],
    width=1.0,
    alpha=0.2,
    color="skyblue",
    label="Precipitation (mm/d)",
)
ax_ppt.set_ylabel("Precipitation (mm/d)", color="steelblue")
ax_ppt.tick_params(axis="y", labelcolor="steelblue")
ax_ppt.set_ylim(0, forcing_df["ppt"].max() * 8)

ax_temp: Axes = ax.twinx()
ax_temp.plot(
    forcing_df.index,
    forcing_df["temp"],
    color="lightcoral",
    linewidth=0.6,
    label="Temperature (deg C)",
)
ax_temp.set_ylabel("Temperature (deg C)", color="lightcoral")
ax_temp.tick_params(axis="y", labelcolor="lightcoral")

# Observed and simulated streamflow on the primary axis.
ax.plot(
    simulation_df.index,
    simulation_df["streamflow_sim"],
    color="royalblue",
    label="Simulated",
)
ax.scatter(
    meas_streamflow.index,
    meas_streamflow,
    s=2,
    color="black",
    label="Measured",
)
ax.set_xlim(obs_start, obs_end)
ax.set_ylim(0, simulation_df["streamflow_sim"].quantile(0.99) * 10)
ax.set_ylabel("Streamflow (mm/d)")
ax.set_title("Simulated vs. observed streamflow with daily drivers")
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4)
fig.tight_layout()
plt.show()

# ---- Panel 2: zone storages over time ---- #
fig2: Figure = plt.figure(figsize=(10, 3))
ax2: Axes = fig2.gca()

ax2.plot(simulation_df.index, simulation_df["s_snow"], color="teal", label="Snow")
ax2.plot(
    simulation_df.index, simulation_df["s_surface"], color="indianred", label="Surface"
)
ax2.plot(
    simulation_df.index,
    simulation_df["s_shallow"],
    color="darkgoldenrod",
    label="Shallow",
)
ax2.plot(
    simulation_df.index, simulation_df["s_deep"], color="forestgreen", label="Deep"
)
ax2.set_xlim(obs_start, obs_end)
ax2.set_ylabel("Water storage (mm)")
ax2.set_title("Zone water storages")
ax2.legend()
fig2.tight_layout()
plt.show()
