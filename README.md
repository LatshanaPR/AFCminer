# AFCminer

## Project Files

- `run_fb.py`: runs the optimized Facebook100 experiment.
- `optimal_afc.py`: optimized AFCMiner implementation used by `run_fb.py`.
- `backup_afc.py`: reference AFCMiner implementation and shared fairness helpers.
- `bk.py`: Bron-Kerbosch baseline used for comparison.
- `preprocess_fb100.py`: loads and preprocesses Facebook100 `.mat` datasets.
- `FCA.py`: standalone FCA concept-builder example.
- `timeline.py`: generates the project timeline outputs.

## Run

Place the dataset at `facebook100/American75.mat`, then run:

```powershell
python run_fb.py
```

Run the Bron-Kerbosch comparison with:

```powershell
python bk.py
```