# ML Activities Dashboard

Interactive Streamlit dashboard for the ML Activities project.

---

## Step-by-step setup

### Step 1 — Install Python dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

---

### Step 2 — Add your CSV data files

Copy all your CSV files into the `data/` folder:

```
data/
  dim_unit.csv
  dim_temps.csv
  dim_country.csv
  dim_adherent.csv
  fact_activities.csv
  dim_budget.csv
  dim_activity.csv
  fact_finance.csv
```

---

### Step 3 — Train the models

This will fit all models and save them to the `models/` folder.
It only needs to be run **once** (or again if your data changes).

```bash
python train.py
```

You should see output like:
```
Loading data...
Building master table...
Training classification models...
  Logistic Regression — F1: 0.8123
  Random Forest — F1: 0.8456
Training regression models...
...
✅ All artifacts saved. You can now run: streamlit run app.py
```

---

### Step 4 — Launch the dashboard

```bash
streamlit run app.py
```

Your browser will open automatically at http://localhost:8501

---

## Deploy to the internet (free) — Streamlit Community Cloud

1. Create a free account at https://github.com and push this project:

```bash
git init
git add app.py requirements.txt .streamlit/
git commit -m "initial commit"
```

Then create a new repo on GitHub and push to it.

2. **Important:** The `models/` and `data/` folders are in `.gitignore`
   because they can be large. For deployment you have two options:

   **Option A (simplest):** Remove `models/` and `data/` from `.gitignore`,
   commit everything, and push. Only works if your files are small (< 100MB total).

   **Option B (recommended for large data):** Keep data out of git.
   Add a `@st.cache_resource` section in `app.py` that loads CSVs from
   Google Drive or an S3 bucket instead.

3. Go to https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub repo
   - Set Main file path to: `app.py`
   - Click Deploy

Your app will be live at a public URL in ~2 minutes.

---

## Project structure

```
ml_dashboard/
├── app.py              ← Streamlit dashboard (this is what runs)
├── train.py            ← Run once to fit models and save .pkl files
├── requirements.txt    ← Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml     ← Dark theme settings
├── data/               ← Put your CSV files here (not committed to git)
│   └── *.csv
└── models/             ← Auto-created by train.py (not committed to git)
    └── *.pkl
```
