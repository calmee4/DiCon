# DiCon: Dual Drug Interaction and Contextual Pre-training for Effective Medication Recommendation 

## Introduction 🚀

<img width="1700" height="600" alt="Introduction" src="https://github.com/user-attachments/assets/d32211c7-de3a-46a1-ab31-778a6b2dcd6e" />

## How to Extract Positive Drug-Drug Interactions (DDIs) 💊🔍

<img width="350" height="500" alt="Positive_DDIs_Pipeline" src="https://github.com/user-attachments/assets/fe342bee-423f-4c1e-be76-d6d900417223" />

For easy reproduction, we provide the **Drug Interaction information**. You can download [`descriptions.txt`](https://drive.google.com/drive/folders/1tPZgq_7Z9_ej-oYQnXFO-DiDNfgTBGoX?usp=sharing) and place it in the `./data/` directory.
 If you need more comprehensive drug information, you may also download `database.xml` from DrugBank. 🏥

✨ **Note**: This research contributes to safer medication recommendation systems! 🎯

# How to Use Our Model 🛠️

## 1. Install and create a new environment 🌟:
```bash
git clone https://github.com/calmee4/DiCon.git
```
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

Due to storage limitations, you can download our best models from [`best_models`](https://drive.google.com/drive/folders/1tPZgq_7Z9_ej-oYQnXFO-DiDNfgTBGoX?usp=sharing).
 Place **`Epoch_6_JA_0.5537_DDI_0.07087.model`** in `./src/Baselines/saved/mimic-iii/OurModel`, and **`Epoch_7_JA_0.4692_DDI_0.07334.model`** in `./src/Baselines/saved/mimic-iv/OurModel`.

then

```bash
cd src/Baselines
./test.sh
```

or run directly

```python
python main_OurModel.py --cuda 0 --dataset "mimic-iii" --dim 1536 --Test true --resume_path "./saved/mimic-iii/OurModel/Epoch_6_JA_0.5537_DDI_0.07087.model"
python main_OurModel.py --cuda 0 --dataset "mimic-iv"  --dim 1536 --Test true --resume_path "./saved/mimic-iv/OurModel/Epoch_7_JA_0.4692_DDI_0.07334.model"
```

## 5. Train the model 🏋️‍♀️:

```bash
cd src/Baselines
./run.sh
```

or run directly

```bash
cd src/Baselines
python main_OurModel.py --cuda 0  --dim 1536 --dataset "mimic-iii"
python main_OurModel.py --cuda 0  --dim 1536 --dataset "mimic-iv"
```

## 6. Prepare the raw MIMIC tables as follows 📊:

Before running the project, please make sure you have completed the following preparations:

1. **Access approval**
    Apply for and obtain permission to use the [MIMIC-III](https://physionet.org/content/mimiciii/) and [MIMIC-IV](https://physionet.org/content/mimiciv/) databases.
2. **Prepare the datasets**
   - Download and extract the MIMIC-III and MIMIC-IV datasets.
   - Place them under the `./data/input/` directory.
   - From **MIMIC-III**, you will need:
      `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, `PROCEDURES_ICD.csv`
   - From **MIMIC-IV**, you will need:
      `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, `PROCEDURES_ICD.csv`
3. **Supplementary drug data**
   - Download [`drugbank_drugs_info.csv`](https://drive.google.com/drive/folders/1tPZgq_7Z9_ej-oYQnXFO-DiDNfgTBGoX?usp=sharing), [`descriptions.txt`](https://drive.google.com/drive/folders/1tPZgq_7Z9_ej-oYQnXFO-DiDNfgTBGoX?usp=sharing), and [`drug-DDI.csv`](https://drive.google.com/drive/folders/1tPZgq_7Z9_ej-oYQnXFO-DiDNfgTBGoX?usp=sharing).
   - Place all of them in the `./data/input/` directory.

Once all required files are in place, run:

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
