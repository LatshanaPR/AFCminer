# AFCminer

Quick project flow:
- `preprocess_fb100.py`: load and prepare Facebook100 data
- `experiment_utils.py`: shared graph builder for all experiments
- `experiment1.py`: first experiment
- `experiment2.py`: second experiment placeholder
- `experiment3.py`: third experiment placeholder
- `bk.py`: Bron-Kerbosch baseline
- `optimal_afc.py`: optimized AFCMiner
- `run_fb.py`: quick single-run validation

## Run

Place the dataset in `facebook100/American75.mat` and run:

```powershell
python experimentx.py
```

Quick validation:

```powershell
python run_fb.py
```

```powershell
python bk.py
```