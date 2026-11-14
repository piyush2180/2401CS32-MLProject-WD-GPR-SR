# WD-GPR-SR: Battery State of Health Estimation on Unseen Li ion Cells

PBL Machine Learning project, Problem Statement 2.
Piyush Sinha, 2401CS32, Department of Computer Science and Engineering, IIT Patna.

This project estimates the state of health (SOH) of lithium ion cells that the model has never seen. The model is a sparse Gaussian process whose kernel adds a linear part to a small neural network part, with an optional symbolic correction equation and a calibrated 95 % interval.

## What is in this repository

| Path | What it is |
|---|---|
| `2401CS32_MLProject.ipynb` | The full notebook, with the outputs of the final Colab run |
| `requirements.txt` | Library versions used for the final run |

The report and poster are submitted separately.

## How to run it

1. Open `2401CS32_MLProject.ipynb` in Google Colab (a CPU runtime is enough).
2. Run all cells. The notebook installs the few missing packages, downloads the public data by itself and writes tables to `results/` and plots to `results/figures/`.
3. The full run takes about 2.7 hours on a Colab CPU. Setting `PBL_QUICK=1` runs a short version that only checks the code works.

## Data

Both datasets are public and are downloaded inside the notebook.

* NASA Ames PCoE Li ion Battery Aging data, cells B0005, B0006, B0007 and B0018: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
* CALCE battery data, University of Maryland, cells CS2_35 to CS2_38 and CX2_34, 36, 37 and 38: https://web.calce.umd.edu/batteries/data/

## Main results (unseen cells, leave one cell out, 3 seeds)

| Dataset | Our RMSE (SOH %) | Our R² | Best baseline |
|---|---|---|---|
| CALCE CX2 | 1.262 | 0.993 | Ridge 1.506 |
| CALCE CS2 | 1.072 | 0.994 | Ridge 0.827 (no significant difference) |
| NASA | 2.235 | 0.948 | Deep kernel GPR 2.086 |

The model is compared with 15 baselines, with 7 ablations and 7 XAI methods. The exported NumPy model predicts in 0.06 ms on a CPU. No NVIDIA Jetson was available, so edge numbers are CPU measurements.

## References

* B. Saha and K. Goebel, Battery Data Set, NASA Ames Prognostics Data Repository, 2007.
* W. He et al., Journal of Power Sources 196(23), 10314 to 10321, 2011 (CALCE data).
* The full reference list is in the report.
