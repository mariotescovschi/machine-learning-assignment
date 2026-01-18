# ML Practical Assignment - Restaurant Sauce Prediction

## Instalare

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rulare

```bash
# Preprocesare date
python process_dataset.py
python prepare_lr1.py
python prepare_lr2.py

# LR #1: Crazy Sauce | Crazy Schnitzel
python train_lr1.py

# LR #2: Multi-sauce + recomandari
python train_lr2.py

# Ranking: Naive Bayes upsell
python train_ranking.py
```

## Structura

```
├── data/
│   ├── ap_dataset.csv      # Date brute
│   ├── receipts.json       # Bonuri grupate
│   ├── dataset_lr1.csv     # Date LR #1
│   └── dataset_lr2.csv     # Date LR #2
├── logistic_regression.py  # LR de la zero (Gradient Descent)
├── naive_bayes.py          # Naive Bayes de la zero
├── process_dataset.py      # CSV → JSON
├── prepare_lr1.py          # Pregătire LR #1
├── prepare_lr2.py          # Pregătire LR #2
├── train_lr1.py            # Antrenare LR #1
├── train_lr2.py            # Antrenare LR #2
└── train_ranking.py        # Antrenare ranking
```
