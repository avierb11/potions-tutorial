# %% [markdown]
# # Define your own model structure
#
# In this notebook we look at how to define your _own_ hydrologic model structure.
# If none of the included structures fit the system you are studying, you can
# describe your own as a list of "layers", where each layer is a list of zones.
# We build a five-zone model with two lateral columns (a hilly, upland **hillslope**
# and a wetter, valley-bottom **riparian** zone) and a single shared groundwater
# bucket at the bottom:
#
# - Layer 1: `snow_hs`, `snow_rp` (two snow zones)
# - Layer 2: `surface_hs`, `surface_rp` (two surface zones)
# - Layer 3: `ground` (one groundwater zone)
#
# The steps are:
#
# 1. Load the downloaded input hydrology data
# 2. Construct the forcing data object(s)
# 3. Overview the built-in model structures
# 4. Define our new model structure
# 5. Run the simulation with our new model
# 6. Observe our new model results (focus on the hillslope and riparian zones)
#
# As with the previous script, this is a Jupytext-style percent script: code
# cells are delimited by `# %%` and text cells by `# %% [markdown]`.

# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, Series

import potions as pt

data_dir: str = "../data"
forcing_path: str = os.path.join(data_dir, "model_inputs", "forcing.csv")
measured_path: str = os.path.join(data_dir, "model_inputs", "measured_data.csv")

# %% [markdown]
# ## 1. Load the input hydrology data
#
# This is the same data used in `2_Run_Hydrologic_Model.py`: daily forcing
# (`ppt`, `temp`, `pet`) and the observed streamflow (`q_mmd`). We read both into
# DataFrames with a parsed datetime index.

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
# ## 2. Construct the input forcing data object
#
# The new model has **two surface zones** (the hillslope and the riparian zone),
# so it expects **two** `pt.ForcingData` objects - one per column. For this
# tutorial we use the same daily drivers for both columns (a single weather
# record); a real application could apply lapse-rate scaling to give each column
# its own series. The observed streamflow is still a single series.

# %%
hs_forcing: pt.ForcingData = pt.ForcingData(
    precip=forcing_df["ppt"], temp=forcing_df["temp"], pet=forcing_df["pet"]
)
rp_forcing: pt.ForcingData = pt.ForcingData(
    precip=forcing_df["ppt"], temp=forcing_df["temp"], pet=forcing_df["pet"]
)

# A list of ForcingData objects, one per surface column, ordered [hillslope, riparian].
forcing_list: list[pt.ForcingData] = [hs_forcing, rp_forcing]

meas_streamflow: Series = measured_df["q_mmd"]
print(f"Number of surface forcing series: {len(forcing_list)}")
print(f"Number of measured streamflow values: {len(meas_streamflow)}")

# %% [markdown]
# ## 3. Overview of the built-in model structures
#
# `potions` bundles several ready-made `Model` subclasses. Each is just a list of
# layers, where a layer is a list of zones ordered from the top of the column to
# the bottom. A couple of them are shown below; note that they are read-only
# *templates* - to use your own layout you subclass `pt.Model` yourself.
#
# - `HbvModel` - a single column: snow, surface, shallow, deep.
# - `HbvLateralModel` - two parallel columns (e.g. hillslope + riparian), each
#   with snow, surface, shallow, deep (8 zones total).


# %%
def layer_summary(structure: list[list[pt.HydrologicZone]]) -> list[str]:
    """Return a short, human-readable description of each layer in a structure."""
    return [", ".join(zone.__class__.__name__ for zone in layer) for layer in structure]


print("Built-in single-column structure (HbvModel):")
for line in layer_summary(pt.HbvModel.structure):
    print(f"  - {line}")
print()
print("Built-in two-column structure (HbvLateralModel):")
for line in layer_summary(pt.HbvLateralModel.structure):
    print(f"  - {line}")

# %% [markdown]
# ## 4. Define our new model structure
#
# We subclass `pt.Model` and assign a `structure` class attribute: a list of
# layers, each a list of zones. This example is a *tapered* column:
#
# - **Snow layer** - two zones, `snow_hs` (hillslope) and `snow_rp` (riparian)
# - **Surface layer** - two zones, `surface_hs` and `surface_rp`
# - **Groundwater layer** - a single aggregated `ground` zone
#
# The connection rules are: zones flow vertically to the zone directly below
# them when both layers have the same number of zones, and to a lone
# aggregated zone below when the lower layer has a single one; within a layer,
# zones flow laterally toward the last zone of that layer, and the last zone of
# each layer discharges to the river. Here both surface columns therefore feed
# down into the one `ground` zone, and the shared groundwater storage is what
# forms the outlet streamflow.
#
# We also tell the model the fraction of the catchment each column represents
# with `scales` (they must sum to 1). The hillslope is the larger, drier part.


# %%
class TwinColumnModel(pt.Model):
    """A five-zone model: two lateral columns (hillslope, riparian) over one
    shared groundwater reservoir."""

    structure: list[list[pt.HydrologicZone]] = [
        [pt.SnowZone(name="snow_hs"), pt.SnowZone(name="snow_rp")],
        [pt.SurfaceZone(name="surface_hs"), pt.SurfaceZone(name="surface_rp")],
        [pt.GroundZoneB(name="ground")],
    ]


# Fractional catchment area of each surface column (must sum to 1).
hillslope_scale: float = 0.7
riparian_scale: float = 0.3
scales: list[float] = [hillslope_scale, riparian_scale]

# Optional: tune a couple of the zones. The riparian column is wetter, so it
# stores more water in its surface store; the shared groundwater reservoir is
# kept at its defaults. Any zone we omit keeps its class defaults.
custom_zones: dict[str, pt.HydrologicZone] = {
    "surface_hs": pt.SurfaceZone(fc=200.0, thr=20.0),
    "surface_rp": pt.SurfaceZone(fc=300.0, thr=30.0),
}

# Create the model, injecting our custom zones and column scales.
model: TwinColumnModel = TwinColumnModel(zones=custom_zones, scales=scales)

print("Zones in the model (top to bottom):")
for zone_name in model.get_zone_names():
    print(f"  - {zone_name}: {model[zone_name].__class__.__name__}")
print(f"Surface columns: {model.num_surface_zones}, scales: {model.scales}")

# Complete parameterisation, keyed as '<zone>.<parameter>' (plus column scales).
model.to_dict()

# %% [markdown]
# ## 5. Run the model
#
# We step the model forward in time, passing the list of two forcing series
# (one per column) and the observed streamflow so the run also reports the
# performance metrics.

# %%
hydro_res: pt.HydroModelResults = model.run_hydro_model(
    forc=forcing_list,
    meas_streamflow=meas_streamflow,
)

print("Simulation complete.")

# %% [markdown]
# ## 6. View the model outputs
#
# As before, `run_hydro_model` returns a `pt.HydroModelResults`. Its
# `simulation` DataFrame has one block of eight columns per zone (storage plus
# the inflow, evaporation, lateral, vertical, and external fluxes), plus
# `streamflow_sim` and `meas_streamflow`. Because there are five zones,
# there are 5 * 8 + 2 = 42 columns.

# %%
print("Performance metrics:")
print(hydro_res.objective_functions)
print()

simulation_df: DataFrame = hydro_res.simulation
print(f"Simulation output shape: {simulation_df.shape}")
print()
simulation_df.head()

# %% [markdown]
# ## 7. Plot the results (hillslope vs. riparian)
#
# We focus the visualization on the two lateral columns - the **hillslope**
# (`_hs`) and the **riparian** (`_rp`) zones - since distinguishing the
# upland and valley-bottom behavior is the whole point of adding a second
# column. We compare their storages and outlet fluxes, and show the shared
# streamflow response.

# %%
plt.rcParams["figure.dpi"] = 200
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.right"] = False

obs_start: pd.Timestamp = meas_streamflow.index[0]
obs_end: pd.Timestamp = meas_streamflow.index[-1]

# ---- Panel A: snow storage, hillslope vs. riparian ---- #
fig_a: Figure = plt.figure(figsize=(10, 3))
ax_a: Axes = fig_a.gca()
ax_a.plot(
    simulation_df.index,
    simulation_df["s_snow_hs"],
    color="teal",
    label="Snow, hillslope",
)
ax_a.plot(
    simulation_df.index,
    simulation_df["s_snow_rp"],
    color="skyblue",
    label="Snow, riparian",
)
ax_a.set_xlim(obs_start, obs_end)
ax_a.set_ylabel("Snow storage (mm)")
ax_a.set_title("Snow storage: hillslope vs. riparian")
ax_a.legend()
fig_a.tight_layout()
plt.show()

# ---- Panel B: surface storage, hillslope vs. riparian ---- #
fig_b: Figure = plt.figure(figsize=(10, 3))
ax_b: Axes = fig_b.gca()
ax_b.plot(
    simulation_df.index,
    simulation_df["s_surface_hs"],
    color="indianred",
    label="Surface, hillslope",
)
ax_b.plot(
    simulation_df.index,
    simulation_df["s_surface_rp"],
    color="darkorange",
    label="Surface, riparian",
)
ax_b.set_xlim(obs_start, obs_end)
ax_b.set_ylabel("Surface storage (mm)")
ax_b.set_title("Surface storage: hillslope vs. riparian")
ax_b.legend()
fig_b.tight_layout()
plt.show()

# ---- Panel C: lateral (outlet) flux from the riparian zone + shared streamflow ---- #
fig_c: Figure = plt.figure(figsize=(10, 3))
ax_c: Axes = fig_c.gca()
ax_c.plot(
    simulation_df.index,
    simulation_df["q_lat_surface_rp"],
    color="darkorange",
    label="Riparian surface runoff",
)
ax_c.plot(
    simulation_df.index,
    simulation_df["streamflow_sim"],
    color="royalblue",
    label="Simulated streamflow",
)
ax_c.scatter(
    meas_streamflow.index,
    meas_streamflow,
    s=2,
    color="black",
    label="Measured streamflow",
)
ax_c.set_xlim(obs_start, obs_end)
ax_c.set_ylabel("Flux (mm/d)")
ax_c.set_title("Riparian runoff and streamflow")
ax_c.legend()
fig_c.tight_layout()
plt.show()
