# iBreast SmartCare: code, models and data

Code accompanying the manuscript *[TITLE]* ([JOURNAL], [YEAR]).

**Research use only. Not a medical device.** The models were trained on milk
from ten donors with self-reported labels. They are not clinically validated
and their outputs are uncalibrated classifier scores, not disease risk.

- Live app: [STREAMLIT APP URL]
- Archived release (DOI): [ZENODO DOI]

## Contents

| Folder | Contents |
|---|---|
| `app/` | `streamlit_app.py`, the browser app. Four sensor readings are typed in manually; the app computes the electrolyte ratios and applies the trained models. |
| `models/` | Trained scikit-learn pipelines (`.pkl`), `models.json` listing each model's file and feature order, and `train_and_save_models.py`, which regenerates the `.pkl` files. |
| `analysis/` | `MULTIVARIATE_LAB_FINAL.py` (single-loop search over 627 feature subsets x 6 algorithms) and `nested_cv_permutation.py` (nested leave-one-donor-out cross-validation, confidence intervals and permutation tests). |
| `data/` | De-identified sensor measurements and binary labels for the ten donors. |

## Run the app locally

```bash
git clone [REPOSITORY URL]
cd iBreast-SmartCare
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501.

## Reproduce the analysis

```bash
# Table 1 of the manuscript (nested cross-validation, CIs, permutation test)
python analysis/nested_cv_permutation.py \
    --lab data/[SENSOR FILE] --q data/[LABEL FILE] \
    --out results --jobs 8 --perm-search 200

# Regenerate the deployed models
python models/train_and_save_models.py
```

`--perm-search 200` repeats the full model search on 200 label permutations and
takes tens of core-hours; use `--perm-search 0` to skip it.

## Software versions

scikit-learn 1.7.2. Pickled models are only guaranteed
to load with the scikit-learn version that created them; see `requirements.txt`.

## Data

`data/` contains only what is needed to reproduce the analysis: recoded donor
IDs, the four sensor readings and the binary labels. No other questionnaire
information is included. [STATE IRB / CONSENT BASIS FOR SHARING.]

## Citation

[CITATION]

## Licence

Code: [e.g. MIT]. Data: [e.g. CC BY 4.0].
