# AneRBC audit: label validity and haemoglobin regression

Code and derived data for the paper **"Label Validity and Haemoglobin Regression in the AneRBC Blood Smear Benchmark: An Audit Against Haematology Analyser Measurements"** (N. Shivane, R. Paswan, A. Sontakke, P. Salve, D. Pardeshi; SCTR's Pune Institute of Computer Technology).

## Main findings

- The AneRBC anemic/healthy folder label agrees with the pathologist's morphology report for 88.1% of patients, but predicts measured haemoglobin with an AUC of only 0.635.
- 50.1% of patients in the "Healthy" folder are anemic by WHO criteria.
- AneRBC-II is exact 4 x 3 crops of AneRBC-I, so random image-level splits leak.
- 35.9% of CBC records are duplicated, and 82 (8.2%) are unusable, leaving 917 analysable patients.
- A frozen ResNet18 with 14 handcrafted features predicts haemoglobin with MAE 2.145 g/dL under patient-wise cross-validation.

## Repository layout

| Path | Contents |
| --- | --- |
| `validate_anerbc.py` | The audit: tile structure, pixel reassembly, duplicate CBCs, Wintrobe consistency checks, label vs haemoglobin vs morphology |
| `notebooks/anemia_hb_v5_final_H2.ipynb` | Full pipeline: geometry analysis, haemoglobin regression ablation (5 folds x 3 seeds), uncertainty, external test |
| `data/anerbc_patients.csv.gz` | One row per AneRBC-I patient: parsed CBC values, morphology flag, consistency flags (`unusable`) |
| `data/anerbc_tiles.csv.gz` | One row per AneRBC-II tile with its parent patient, for patient-grouped splitting |
| `results/` | Tables behind the paper, the 60-run log and out-of-fold predictions |
| `figures/` | Figures used in the paper |
| `history/` | Earlier versions of the project (see below) |

## Reproducing the audit

1. Download AneRBC from its authors: Shahzad et al., *Database* 2024, baae120, https://doi.org/10.1093/database/baae120
2. Run:

```
pip install numpy pandas pillow
python validate_anerbc.py /path/to/AneRBC_dataset
```

The script prints every audit number in the paper and writes `anerbc_patients.csv` and `anerbc_tiles.csv`. Analysable patients are the rows with `unusable == False` and a haemoglobin value.

The notebook was run on Kaggle with one NVIDIA T4 GPU.

## External laboratory data

The 33 smear images and blood count reports from the external laboratory are **not included** to protect patient privacy. Outputs in the notebook that show per-slide external data have been removed. De-identified derived data are available from the corresponding author on reasonable request.

## Project history

`history/` keeps the earlier classification notebooks, from before the audit. They train anemic/healthy classifiers with **random image-level splits on AneRBC-II**, so the 90-96% accuracies they report come from tile leakage and are **not valid results**. They are kept to show how the project developed: finding that leak is what led to the audit.

## Citation

If you use this code, please cite the paper (citation will be added after publication) and the AneRBC dataset.

## Licence

MIT. See `LICENSE`. AneRBC itself is distributed by its authors under their own terms.
