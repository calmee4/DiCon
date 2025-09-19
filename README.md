# DiCon: Dual Drug Interaction and Contextual Pre-training for Effective Medication Recommendation 

## Introduction 🚀

<img alt="Introduction" src="https://github.com/user-attachments/assets/d32211c7-de3a-46a1-ab31-778a6b2dcd6e" />

## How to Extract Positive Drug-Drug Interactions (DDIs) 💊🔍

<img width="350" height="500" alt="Positive_DDIs_Pipeline" src="https://github.com/user-attachments/assets/fe342bee-423f-4c1e-be76-d6d900417223" />

For easy reproduction, we provide **Drug Interaction information** in `./data/descriptions.txt`. If you need more detailed drug information, you can download `database.xml` from DrugBank. 🏥

✨ **Note**: This research contributes to safer medication recommendation systems! 🎯

# How to Use Our Model 🛠️

## 1. Create a new environment 🌟:

```bash
conda create -n DiCon python=3.12 -y
```

## 2. Activate the environment ⚡:

```bash
conda activate DiCon
```

## 3. Install dependencies 📦:

```bash
pip install -r requirements.txt
```

If anything fails during installation, follow `requirements_step_by_step.txt` to install packages incrementally.

## 4. Run evaluation 🧪:

```bash
cd src/Baselines
./test.sh
```

or run directly:

```python
python main_OurModel.py --cuda 0 --dataset "mimic-iii" --dim 1536 --Test true --resume_path "./saved/mimic-iii/OurModel/Epoch_6_JA_0.5537_DDI_0.07087.model"
python main_OurModel.py --cuda 0 --dataset "mimic-iv"  --dim 1536 --Test true --resume_path "./saved/mimic-iv/OurModel/Epoch_7_JA_0.4692_DDI_0.07334.model"
```

## 5. Train the model 🏋️‍♀️:

```bash
cd src/Baselines
./run.sh
```

or run directly:

```python
python main_OurModel.py --cuda 0  --dim 1536 --dataset "mimic-iii"
python main_OurModel.py --cuda 0  --dim 1536 --dataset "mimic-iv"
```

## 6. Prepare the raw MIMIC tables as follows 📊:

You can refer to mimic-iii / mimic-iv documentation for the following files:

**mimic-iii**: `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, `PROCEDURES_ICD.csv`

**mimic-iv**: `diagnoses_icd.csv`, `prescriptions.csv`, `procedures_icd.csv`

Place them under: `./data/input/mimic-iii` and `./data/input/mimic-iv`

Then run:

```python
python process.py
```

## 7. Model Parameter Details ⚙️

```bash
usage: main_OurModel.py 
[-h] [-n NOTE] [--debug DEBUG] [--Test TEST] [--model_name MODEL_NAME] [--dataset DATASET]
[--resume_path RESUME_PATH] [--cuda CUDA] [--dim DIM] [--lr LR] [--dp DP] [--regular REGULAR]
[--target_ddi TARGET_DDI] [--target_ddi_positive TARGET_DDI_POSITIVE] [--coef COEF]
[--epochs EPOCHS] [--weight_decay WEIGHT_DECAY] [--nhead NHEAD] [--early_stop EARLY_STOP]
[--threshold THRESHOLD] [--sample SAMPLE]
```

**Optional arguments:**

- `-h, --help` Show help message and exit.
- `-n NOTE, --note NOTE` Free-form note for the current run.
- `--debug DEBUG` Debug mode (uses a very small subset and very few epochs; CPU-friendly).
- `--Test TEST` Evaluation-only mode (no training).
- `--model_name MODEL_NAME` Model identifier used in logging/checkpoints.
- `--dataset DATASET` Dataset name (`mimic-iii` or `mimic-iv`).
- `--resume_path RESUME_PATH` Path to a trained checkpoint; used only during evaluation.
- `--cuda CUDA` GPU index to use (e.g., `0`).
- `--dim DIM` Model hidden dimension / embedding size.
- `--lr LR` Learning rate.
- `--dp DP` Dropout ratio.
- `--regular REGULAR` Regularization coefficient (e.g., L2 weight).
- `--target_ddi TARGET_DDI` Target DDI rate for training (overall constraint).
- `--target_ddi_positive TARGET_DDI_POSITIVE` Target positive-DDI rate for training.
- `--coef COEF` Coefficient for DDI-loss weight annealing.
- `--epochs EPOCHS` Number of training epochs.
- `--weight_decay WEIGHT_DECAY` Weight decay factor.
- `--nhead NHEAD` Number of attention heads.
- `--early_stop EARLY_STOP` Stop training if no improvement after N epochs.
- `--threshold THRESHOLD` Decision threshold for converting scores to 0/1 labels.
- `--sample SAMPLE` Positive DDI sampling ratio used during training.
