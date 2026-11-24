# Forecast-Augmented Lyapunov-based Dispatch of Hybrid Power Plants in Island Power Systems

This repository contains the online dispatch code for the study Short-term Forecast-Augmented Surrogate Lyapunov-based Real-Time Dispatch of Hybrid Power Plants in Island Power Systems.

## Requirements and run order

The code requires Python, Gurobi 12.0.1, and a valid Gurobi license. The dependencies in `requirements.txt` include:

- `gurobipy==12.0.1`
- `numpy==1.18.5`
- `pandas==1.0.4`
- `openpyxl==3.0.10`

Activate the Python environment you want to use. Then, from the project folder, install the dependencies and run the script:

```bash
python -m pip install -r requirements.txt
python HPP_real_time_dispatch.py
```

Keep the `Inputs/` folder beside `HPP_real_time_dispatch.py`. Set up the files and parameters below before running the script.

## Input files

The `Inputs/` folder contains the following Excel workbooks. Each workbook uses a sheet with the same name as the file, without the extension. 


### Parameter descriptions

| Parameter | Description |
| --- | --- |
| `future_lookahead` | Number of future hours included after the current hour |
| `beta` | Lyapunov drift coefficient |
| `theta_soc` | Virtual queue bias parameter |
| `sigma_loss` | Starting storage-loss penalty for the linear BESS model |
| `max_sigma_iters` | Maximum number of solves in each storage-loss penalty adjustment loop |
| `sigma_scale` | Multiplier applied when the storage-loss penalty is increased |
| `scd_tol` | Tolerance used to detect simultaneous charging and discharging |
| `reserve_reaction_time` | Reserve reaction time in minutes |
| `lol_penalty` | Penalty reserve_reaction_time for loss of load |
| `curtail_penalty` | Penalty coefficient for PV curtailment |
| `reserve_short_penalty` | Penalty coefficient for reserve shortfall |
| `lineflow_penalty` | Penalty coefficient for line-limit slack |
| `netcharge_penalty` | System operator’s HPP operating preference parameter |
| `loss_factor` | Network loss rate |
| `reserve_pv_factor` | Factor applied to the stand alone PV-based reserve requirement |
| `sf_epsilon` | GSF coefficients with an absolute value at or below this threshold are ignored |
| `progress_every` | Number of intervals between progress messages |
| `solver_output` | Gurobi console output: 0 for off and 1 for on |

The utility-specific values for loss_factor, reserve_pv_factor, and the utility's system and time-series data are confidential and are not included. Their data Contents and structures are described below.

| File | Contents |
| --- | --- |
| `Generator.xlsx` | Generator names, buses, output limits, costs, ramp limits, and emergency priorities |
| `BESS.xlsx` | Battery names, buses, energy and power limits, efficiencies, SOC limits |
| `PV.xlsx` | PV names, buses, forecast and realized-data paths, and reserve weights |
| `HPP.xlsx` | HPP names, PV and BESS members, and HPP capacity limits |
| `OtherGeneration.xlsx` | Names and buses of non-dispatchable generators |
| `Other.xlsx` | Paths to commitment, network, load, other-generation data|



Referenced data files can be Excel or CSV. For Excel files, the script reads the first sheet.

| Data | Structure |
| --- | --- |
| PV forecast and realized power | One numeric power column per file, with an optional Hour column |
| Generator commitment | One binary column per generator, with an optional Hour column |
| Nodal load | hour followed by one column for each bus ID |
| Other generation | hour followed by one column for each resource name |
| GSF matrix | Line names in the first column and bus IDs in the remaining column headers |
| Line parameters | StartBusName, EndBusName, X, and FlowLim |

All dispatch intervals are one hour long.

## What the script does

At each hour, HPP_real_time_dispatch.py solves a forecast-augmented Real-Time HPP Dispatch dispatch problem. It uses observed PV power for the current hour and forecast PV power for the future hours in the look-ahead window.

The model uses power system operator oriented HPP integrated modeling,  which consider operator requirements such as loss of load avoidance, PV curtailment mitigation, resrever sufficiency, and transmission security. A generic HPP identification is adopted using incidence-matrix-based modeling.  A Lyapunov-aware relaxation of the bidirectional BESS model is used.  If the continuous BESS solution charges and discharges at the same time, the script increases the storage-loss penalty and solves again. An expert-knowledge-informed operational flexibility is included,  if loss of load remains in the current hour, it attempts flexible generator startup based on expert-knowledge.

Only the decisions for the current hour are applied. The script then updates states and queues before moving to the next hour. 
