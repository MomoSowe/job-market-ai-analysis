# Analysis of the REAL dataset: data/real_automation_risk_by_occupation.csv
# See data/SOURCES.md for provenance (Frey & Osborne 2013 computerisation
# probabilities + real May-2016 U.S. BLS wage/employment data by occupation).
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

OUTPUT_DIR = "output_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def zoom_ylim(values, pad_frac=0.2):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = (vmax - vmin) * pad_frac if vmax > vmin else max(abs(vmax) * 0.05, 0.5)
    plt.ylim(vmin - pad, vmax + pad)


df = pd.read_csv("data/real_automation_risk_by_occupation.csv", encoding="utf-8")
df["automation_risk_pct"] = df["automation_probability"] * 100

# Question 1: Does required education relate to automation risk?
EDU_ORDER = ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"]
by_education = (
    df.groupby("required_education")["automation_risk_pct"]
      .mean()
      .reindex(EDU_ORDER)
)

print(by_education)

plt.figure(figsize=(8, 5))
by_education.plot(kind="bar", color="#2a78d6")
plt.title("Average Automation Risk by Required Education (Real Data)")
plt.xlabel("Required Education")
plt.ylabel("Automation Risk (%)")
plt.xticks(rotation=20)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_automation_risk_by_education.png"), dpi=150)
plt.show()

# Question 2: Which real occupations are most at risk of automation?
top_risk = df.nlargest(15, "automation_risk_pct").set_index("occupation")["automation_risk_pct"]

plt.figure(figsize=(12, 6))
top_risk.plot(kind="barh", color="#e34948")
plt.title("15 U.S. Occupations Most Exposed to Automation (Real Data)")
plt.xlabel("Automation Risk (%)")
plt.gca().invert_yaxis()
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_highest_risk_occupations.png"), dpi=150)
plt.show()

# Question 3: Which real occupations are least at risk of automation?
low_risk = df.nsmallest(15, "automation_risk_pct").set_index("occupation")["automation_risk_pct"]

plt.figure(figsize=(12, 6))
low_risk.plot(kind="barh", color="#1baf7a")
plt.title("15 U.S. Occupations Least Exposed to Automation (Real Data)")
plt.xlabel("Automation Risk (%)")
plt.gca().invert_yaxis()
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_lowest_risk_occupations.png"), dpi=150)
plt.show()

# Question 4: Does wage relate to automation risk? (real finding: r ~ -0.53)
corr = df["automation_risk_pct"].corr(df["median_annual_wage_usd"])

bin_edges = np.arange(0, 101, 10)
bin_labels = [f"{lo}-{hi}" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
risk_bin = pd.cut(df["automation_risk_pct"], bins=bin_edges, include_lowest=True, labels=bin_labels)
wage_by_risk = df.groupby(risk_bin, observed=True)["median_annual_wage_usd"].mean()

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.bar(wage_by_risk.index.astype(str), wage_by_risk.values, color="#2a78d6", width=0.65)
fig.suptitle("Median Wage by Automation Risk Bracket (Real Data)", fontsize=13, fontweight="bold", y=0.98)
ax.set_title(f"Higher-risk occupations pay less on average (r = {corr:.2f})",
             fontsize=10, color="#52514e", pad=12)
ax.set_xlabel("Automation Risk (%)")
ax.set_ylabel("Median Annual Wage (USD)")
plt.xticks(rotation=0)
plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig(os.path.join(OUTPUT_DIR, "04_wage_by_automation_risk.png"), dpi=150)
plt.show()

# Question 5: How exposed is the actual U.S. workforce, weighted by real employment?
weighted_mean = np.average(df["automation_risk_pct"], weights=df["employed_may_2016"])
print(f"Employment-weighted average automation risk: {weighted_mean:.1f}%")

plt.figure(figsize=(9, 6))
plt.hist(
    df["automation_risk_pct"], bins=20, weights=df["employed_may_2016"],
    color="#2a78d6", edgecolor="white",
)
plt.axvline(weighted_mean, color="#898781", linewidth=1.2)
plt.text(weighted_mean + 1.5, plt.ylim()[1] * 0.95,
          f"Employment-weighted avg: {weighted_mean:.1f}%", color="#52514e", fontsize=9)
plt.title("Automation Risk Across the Real U.S. Workforce\n(weighted by number of people employed in each occupation)")
plt.xlabel("Automation Risk (%)")
plt.ylabel("People Employed (May 2016)")
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_workforce_exposure_weighted.png"), dpi=150)
plt.show()