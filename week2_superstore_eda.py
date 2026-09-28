"""
Week 2 Internship Task: Exploratory Data Analysis and Visualization
Dataset: Tableau Sample - Superstore (9,994 rows, 21 columns)

This script downloads/reads the public Sample Superstore CSV, performs
basic cleaning, descriptive analysis, aggregation, visualization, and
exports summary tables and charts.

If you already downloaded the CSV, place it beside this script as:
    Sample - Superstore.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"
LOCAL_FILE = "superstore.csv"
OUT_DIR = "eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load data
if os.path.exists(LOCAL_FILE):
    df = pd.read_csv(LOCAL_FILE, encoding="latin-1")
else:
    df = pd.read_csv(DATA_URL, encoding="latin-1")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# 2. Basic cleaning
df.columns = df.columns.str.strip()
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove exact duplicate rows if present
df = df.drop_duplicates().copy()

# 3. Feature engineering
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Month Name"] = df["Order Date"].dt.month_name()
df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
df["Profit Margin %"] = np.where(
    df["Sales"] != 0, df["Profit"] / df["Sales"] * 100, np.nan
)
df["Profit Status"] = np.where(df["Profit"] >= 0, "Profitable", "Loss")

# 4. Descriptive statistics
print("\nDescriptive statistics:")
print(df[numeric_cols].describe().round(2))

# 5. KPI summary
kpis = {
    "Rows": len(df),
    "Total Sales": df["Sales"].sum(),
    "Total Profit": df["Profit"].sum(),
    "Total Quantity": df["Quantity"].sum(),
    "Overall Profit Margin %": df["Profit"].sum() / df["Sales"].sum() * 100,
}
print("\nKPI Summary:")
for k, v in kpis.items():
    print(f"{k}: {v:,.2f}" if isinstance(v, float) else f"{k}: {v:,}")

# 6. Category analysis
category = (
    df.groupby("Category")[["Sales", "Profit"]]
      .sum()
      .sort_values("Sales", ascending=False)
)
category["Profit Margin %"] = category["Profit"] / category["Sales"] * 100
print("\nCategory summary:")
print(category.round(2))
category.to_csv(os.path.join(OUT_DIR, "category_summary.csv"))

# 7. Region analysis
region = (
    df.groupby("Region")[["Sales", "Profit"]]
      .sum()
      .sort_values("Sales", ascending=False)
)
region["Profit Margin %"] = region["Profit"] / region["Sales"] * 100
print("\nRegion summary:")
print(region.round(2))
region.to_csv(os.path.join(OUT_DIR, "region_summary.csv"))

# 8. Segment analysis
segment = (
    df.groupby("Segment")[["Sales", "Profit", "Quantity"]]
      .sum()
      .sort_values("Sales", ascending=False)
)
print("\nSegment summary:")
print(segment.round(2))
segment.to_csv(os.path.join(OUT_DIR, "segment_summary.csv"))

# 9. Sub-category analysis
subcategory = (
    df.groupby("Sub-Category")[["Sales", "Profit"]]
      .sum()
      .sort_values("Profit")
)
subcategory["Profit Margin %"] = subcategory["Profit"] / subcategory["Sales"] * 100
print("\nLoss-making sub-categories:")
print(subcategory[subcategory["Profit"] < 0].round(2))
subcategory.to_csv(os.path.join(OUT_DIR, "subcategory_summary.csv"))

# 10. Monthly trend
monthly = (
    df.groupby("Year-Month")[["Sales", "Profit"]]
      .sum()
      .reset_index()
)
monthly["Year-Month"] = pd.to_datetime(monthly["Year-Month"])
monthly = monthly.sort_values("Year-Month")

plt.figure(figsize=(12, 5))
plt.plot(monthly["Year-Month"], monthly["Sales"], label="Sales")
plt.plot(monthly["Year-Month"], monthly["Profit"], label="Profit")
plt.title("Monthly Sales and Profit Trend")
plt.xlabel("Month")
plt.ylabel("Amount (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "monthly_sales_profit.png"), dpi=150)
plt.close()

# 11. Sales and profit by category
ax = category[["Sales", "Profit"]].plot(kind="bar", figsize=(9, 5))
ax.set_title("Sales and Profit by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Amount (USD)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "category_sales_profit.png"), dpi=150)
plt.close()

# 12. Profit by sub-category
plt.figure(figsize=(10, 7))
sns.barplot(
    data=subcategory.reset_index(),
    y="Sub-Category",
    x="Profit",
    orient="h"
)
plt.axvline(0, linewidth=0.8)
plt.title("Profit by Sub-Category")
plt.xlabel("Total Profit (USD)")
plt.ylabel("Sub-Category")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "subcategory_profit.png"), dpi=150)
plt.close()

# 13. Discount vs profit
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df.sample(min(3000, len(df)), random_state=42),
    x="Discount",
    y="Profit",
    alpha=0.35
)
plt.axhline(0, linewidth=0.8)
plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit (USD)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "discount_vs_profit.png"), dpi=150)
plt.close()

# 14. Correlation matrix
corr = df[["Sales", "Quantity", "Discount", "Profit"]].corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_matrix.png"), dpi=150)
plt.close()

# 15. Top 10 customers by sales
top_customers = (
    df.groupby("Customer Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)
top_customers.to_csv(os.path.join(OUT_DIR, "top10_customers.csv"))

print("\nEDA complete. Outputs saved in:", OUT_DIR)
