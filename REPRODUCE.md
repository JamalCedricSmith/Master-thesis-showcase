# Reproducing the Benchmark Results

> 🔒 **Commercial R&D Notice**:
> Raw model checkpoints (`*.pt`), configuration files (`config.yaml`), and training run directories (`code/models/runs/`) have been **withheld** from the public repository to protect proprietary architecture IP and commercial wearable hardware development.
>
> When run artifacts are restored or generated locally, the analysis notebooks execute offline on CPU without requiring GPU compute or active API billing.

---

## 1. Environment Setup (3 min)

```bash
conda create -n lmu_emg python=3.13 
conda activate lmu_emg
pip install -r code/requirements.txt
python -m ipykernel install --user --name lmu_emg --display-name "Python (lmu_emg)"
cd code && jupyter lab
```

**Rule**: Select the `Python (lmu_emg)` kernel in every notebook.

---

## 2. Immediate Reproduction (No Corpus Download Needed)

> 🔒 **Notice**: Experiment evaluation notebooks and raw training runs have been withheld for commercial IP and R&D protection.

---

## 3. Corpus-Dependent Notebooks (Requires DVSS Corpus)

The following notebooks analyze raw signals or perform corpus characterisation and require the 6.4 GB DVSS dataset (see [TRAIN.md](TRAIN.md)):

| Notebook | Focus |
|---|---|
| `data/analysis/corpus_summary_tables.ipynb` | Corpus summary statistics, duration distributions |
| `data/analysis/dataset_examples_per_token.ipynb` | Token distribution across utterances |
| `data/analysis/oov_split_justification.ipynb` | Unseen-word (banpool) partition verification |
| `data/analysis/tokenization_analysis.ipynb` | Tokenizer vocabulary coverage |
