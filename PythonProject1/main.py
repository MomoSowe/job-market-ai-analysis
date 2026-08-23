# Import Files
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Show full-width output instead of truncating columns/rows
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Folder where all charts get saved
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def zoom_ylim(values, pad_frac=0.2):
    """Zoom the y-axis into the actual data range instead of starting at 0,
    so bars with close-but-different values don't all look identical."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = (vmax - vmin) * pad_frac if vmax > vmin else max(abs(vmax) * 0.05, 0.5)
    plt.ylim(vmin - pad, vmax + pad)

# Load the data
# ai_job_trends_dataset.csv is a SYNTHETIC dataset (see data/build_datasets.py
# for the generation code and data/SOURCES.md for provenance) built from real
# occupation titles, with intentional, documented relationships between
# fields - unlike the original file it replaced, whose columns were close to
# uncorrelated random noise. For a real (non-synthetic) dataset covering
# automation risk, wages, and education by U.S. occupation, see
# data/real_automation_risk_by_occupation.csv, analyzed in main_real.py.
df = pd.read_csv('data/ai_job_trends_dataset.csv', encoding='utf-8')

df.head()

df.info()

df.describe()

df.isnull().sum()

# Clean the Data
df = df.drop_duplicates()

df.columns = (
    df.columns.str.strip()
      .str.lower()
      .str.replace(r"[^0-9a-z]+", "_", regex=True)
      .str.strip("_")
)

df.isnull().sum()

# Automation risk comes in TWO flavors here (see data/build_datasets.py):
# "Physical" = classical robotic-automation risk (Frey & Osborne 2013):
#   can a machine physically do this task? Manual/routine/on-site work is
#   highest-risk; remote work is protective.
# "Generative AI" = LLM-exposure risk (Eloundou et al. 2023): can an LLM do
#   this work? Computer-based/remote-friendly knowledge work is highest-risk
#   instead - the opposite pattern. The two are shown side by side wherever
#   they might disagree, rather than picking one as "the" automation risk.
RISK_COLS = ["automation_risk_physical", "automation_risk_generative_ai"]
RISK_LABELS = ["Physical (robotic)", "Generative AI (LLM)"]
RISK_COLORS = ["#2a78d6", "#eb6834"]  # categorical slots 1 & 2

# Question 1: Which industries have the highest automation risk - and does
# the answer depend on which kind of automation you mean?
industry_risk = df.groupby("industry")[RISK_COLS].mean()
industry_risk = industry_risk.loc[industry_risk.mean(axis=1).sort_values(ascending=False).index]

print(industry_risk)

ax = industry_risk.rename(columns=dict(zip(RISK_COLS, RISK_LABELS))).plot(
    kind="bar", figsize=(11, 6), color=RISK_COLORS, width=0.75,
)
plt.title("Average Automation Risk by Industry: Physical vs Generative AI")
plt.xlabel("Industry")
plt.ylabel("Automation Risk (%)")
plt.xticks(rotation=45)
plt.ylim(0, 100)
ax.legend(title=None, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_automation_risk_by_industry.png"), dpi=150)
plt.show()

# Question 2: Which job titles have the highest automation risk, under each
# framing? (Two charts - the top 15 lists are usually different job titles.)
for col, label, color, suffix in zip(
    RISK_COLS, RISK_LABELS, RISK_COLORS, ["physical_automation_risk", "generative_ai_risk"]
):
    job_risk = df.groupby("job_title")[col].mean().sort_values(ascending=False).head(15)
    print(job_risk)

    plt.figure(figsize=(12, 6))
    job_risk.plot(kind="bar", color=color)
    plt.title(f"Top 15 Job Titles by {label} Automation Risk")
    plt.xlabel("Job Title")
    plt.ylabel("Automation Risk (%)")
    plt.xticks(rotation=75)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"02_top_job_roles_{suffix}.png"), dpi=150)
    plt.show()

# Question 3: Which locations have the highest automation risk?
# (Location only affects simulated salary in this dataset, not risk directly,
# so this reflects which countries happen to have more at-risk occupations
# sampled into them - shown for the physical-automation framing.)
location_risk = (
    df.groupby("location")["automation_risk_physical"]
      .mean()
      .sort_values(ascending=False)
)

print(location_risk)

plt.figure(figsize=(10,6))
location_risk.plot(kind="bar", color=RISK_COLORS[0])
plt.title("Average Physical Automation Risk by Location")
plt.xlabel("Location")
plt.ylabel("Automation Risk (%)")
plt.xticks(rotation=45)
zoom_ylim(location_risk)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_ai_disruption_by_country.png"), dpi=150)
plt.show()

# Question 4: How does median salary vary by industry?

salary = (
    df.groupby("industry")["median_salary_usd"]
      .mean()
      .sort_values(ascending=False)
)

print(salary)

plt.figure(figsize=(10,6))
salary.plot(kind="bar")
plt.title("Average Median Salary by Industry")
plt.xlabel("Industry")
plt.ylabel("Median Salary (USD)")
plt.xticks(rotation=45)
zoom_ylim(salary)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_salary_before_after.png"), dpi=150)
plt.show()

# Question 5: Does education reduce automation risk? Physical risk: yes -
# routine work needs less credentialing. Generative-AI risk: the opposite -
# Eloundou et al. found LLM exposure does NOT fall with seniority/education
# the way physical-automation risk does (more credentialed knowledge work
# is often more exposed, not less).
EDU_ORDER = ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"]
education = df.groupby("required_education")[RISK_COLS].mean()
education = education.reindex([e for e in EDU_ORDER if e in education.index])

print(education)

ax = education.rename(columns=dict(zip(RISK_COLS, RISK_LABELS))).plot(
    kind="bar", figsize=(9, 5.5), color=RISK_COLORS, width=0.75,
)
plt.title("Automation Risk by Required Education: Physical vs Generative AI")
plt.xlabel("Required Education")
plt.ylabel("Automation Risk (%)")
plt.xticks(rotation=20)
plt.ylim(0, 100)
ax.legend(title=None, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_automation_risk_by_education.png"), dpi=150)
plt.show()

# Question 6: Which industries have the biggest projected job growth (2024 -> 2030)?

job_growth = (
    df.assign(job_growth=df["projected_openings_2030"] - df["job_openings_2024"])
      .groupby("industry")["job_growth"]
      .sum()
      .sort_values(ascending=False)
)

print(job_growth)

plt.figure(figsize=(10,6))
job_growth.plot(kind="bar")
plt.title("Projected Job Growth by Industry (2024 to 2030)")
plt.xlabel("Industry")
plt.ylabel("Net New Openings")
plt.xticks(rotation=45)
zoom_ylim(job_growth)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_skill_gap_by_industry.png"), dpi=150)
plt.show()

# Question 7: Which industries have the highest share of shrinking (decreasing) job status?

reskill = (
    df.assign(is_decreasing=(df["job_status"] == "Decreasing").astype(int))
      .groupby("industry")["is_decreasing"]
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)

print(reskill)

plt.figure(figsize=(10,6))
reskill.plot(kind="bar")
plt.title("Share of Decreasing-Status Jobs by Industry")
plt.xlabel("Industry")
plt.ylabel("Decreasing Jobs (%)")
plt.xticks(rotation=45)
zoom_ylim(reskill)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_reskilling_urgency_by_industry.png"), dpi=150)
plt.show()

# Question 8: Is remote work associated with automation risk?
# The honest answer is "it depends which automation you mean" - physical and
# generative-AI risk move in OPPOSITE directions with remote work ratio, so
# a single scatter/bar answering this with one number would be misleading.
# Two panels, same y-axis, side by side, each binned (raw ~30k-point scatter
# would just be an overplotted blob given how noisy any one job posting is).

GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_SECONDARY = "#52514e"

bin_edges = np.arange(0, 101, 10)
bin_labels = [f"{lo}-{hi}" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
remote_bin = pd.cut(df["remote_work_ratio"], bins=bin_edges, include_lowest=True, labels=bin_labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, col, label, color in zip(axes, RISK_COLS, RISK_LABELS, RISK_COLORS):
    corr = df["remote_work_ratio"].corr(df[col])
    binned = df.groupby(remote_bin, observed=True)[col].agg(["mean", "std", "count"])
    binned["ci95"] = 1.96 * binned["std"] / np.sqrt(binned["count"])

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)

    ax.bar(
        binned.index.astype(str), binned["mean"], yerr=binned["ci95"],
        color=color, edgecolor="none", width=0.65, capsize=3,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capthick": 1},
    )
    ax.set_title(f"{label}  (r = {corr:.2f})", fontsize=11, color=INK_SECONDARY, pad=10)
    ax.set_xlabel("Remote Work Ratio (%)")
    ax.tick_params(axis="x", rotation=45)

axes[0].set_ylabel("Average Automation Risk (%)")
axes[0].set_ylim(0, 100)

fig.suptitle("Remote Work Ratio vs Automation Risk: Two Different Stories", fontsize=13, fontweight="bold", y=1.0)
fig.text(0.5, 0.94,
          "More remote work LOWERS physical-automation risk but RAISES generative-AI risk",
          ha="center", color=INK_SECONDARY, fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.90))
plt.savefig(os.path.join(OUTPUT_DIR, "08_remote_feasibility_vs_automation_risk.png"), dpi=150)
plt.show()

# Question 9: Which industries have the highest AI impact level?
# ("AI Impact Level" is bucketed from Generative AI risk, not physical risk -
# see data/build_datasets.py - since it's meant to capture LLM/AI exposure.)

ai_impact_map = {"Low": 1, "Moderate": 2, "High": 3}
ai_impact_score = df["ai_impact_level"].map(ai_impact_map)

ai_adoption = (
    df.assign(ai_impact_score=ai_impact_score)
      .groupby("industry")["ai_impact_score"]
      .mean()
      .sort_values(ascending=False)
)

print(ai_adoption)

plt.figure(figsize=(10,6))
ai_adoption.plot(kind="bar", color=RISK_COLORS[1])
plt.title("Average AI Impact Level by Industry (Generative AI exposure)")
plt.xlabel("Industry")
plt.ylabel("AI Impact Level (1=Low, 3=High)")
plt.xticks(rotation=45)
zoom_ylim(ai_adoption)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_ai_adoption_by_industry.png"), dpi=150)
plt.show()

# Question 10: What factors are most strongly correlated?

numeric = df.select_dtypes(include="number")

correlation = numeric.corr()

print(correlation)

plt.figure(figsize=(12,10))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Heatmap of Numeric Features")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_correlation_heatmap.png"), dpi=150)
plt.show()