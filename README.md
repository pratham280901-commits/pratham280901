# 🍬 Nassau Candy Distributor – Factory Optimization Project

## Project Overview
An end-to-end ML system that predicts shipping lead times, simulates factory–product
reassignment scenarios, and recommends optimal configurations to reduce lead times
and protect profit margins.

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `Nassau_Candy_Colab.ipynb` | Google Colab notebook (EDA → ML → Simulation → Recommendations) |
| `nassau_candy_app.py` | Streamlit dashboard (4-module interactive web app) |
| `requirements.txt` | Python dependencies |

---

## 🚀 How to Run (Google Colab)

### Step 1 – Open the Notebook
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `Nassau_Candy_Colab.ipynb`

### Step 2 – Run All Cells in Order
- **Cell 1:** Installs all required libraries
- **Cell 2:** Uploads your `Nassau_Candy_Distributor.csv`
- **Cells 3–8:** EDA, Feature Engineering, Model Training, Clustering, Simulation, Recommendations
- **Cell 9:** Exports results and downloads output files
- **Cell 10:** Uploads `nassau_candy_app.py`
- **Cell 11:** Launches the Streamlit dashboard with a live public URL

### Step 3 – Use the Dashboard
1. Click the ngrok URL that appears in Cell 11
2. Upload your CSV in the sidebar
3. Explore all 4 tabs

---

## 📊 Dashboard Modules

| Tab | What It Shows |
|-----|--------------|
| **Overview & EDA** | Dataset KPIs, sales charts, lead time analysis, factory map |
| **ML Models** | Model comparison (RMSE/MAE/R²), feature importance, correlation heatmap |
| **Factory Simulator** | Select product + region + ship mode → see predicted lead times for all factories |
| **Recommendations & Risk** | Top reassignment recommendations, risk alerts, congestion scores |

---

## 🤖 ML Models Used

| Model | Role |
|-------|------|
| Linear Regression | Baseline model |
| Random Forest | Ensemble model (typically best) |
| Gradient Boosting | Boosted ensemble model |

**Target:** Predict Lead Time (days)  
**Features:** Distance (km), Ship Speed, Region, Factory, Division, Units, Cost

---

## 🏭 Factory Reference

| Factory | Location | Products |
|---------|---------|---------|
| Lot's O' Nuts | Arizona | Wonka Bars (3 varieties) |
| Wicked Choccy's | Georgia | Wonka Bars (2 varieties) |
| Sugar Shack | Minnesota | Laffy Taffy, SweeTARTS, Nerds, Fun Dip, Fizzy Lifting Drinks |
| Secret Factory | Illinois | Everlasting Gobstopper, Lickable Wallpaper, Wonka Gum |
| The Other Factory | Tennessee | Hair Toffee, Kazookles |

---

## 📈 Key KPIs Tracked

- Lead Time Reduction (%)
- Profit Impact Stability  
- Scenario Confidence Score
- Recommendation Coverage
- Factory Congestion Score
