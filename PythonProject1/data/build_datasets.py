"""
Builds the two datasets that replaced the original ai_job_trends_dataset.csv.

1. real_automation_risk_by_occupation.csv
   Cleaned from Plotly's public datasets repo (MIT-licensed):
   https://github.com/plotly/datasets/blob/master/job-automation-probability.csv
   That file joins two real sources:
     - Frey & Osborne (2013), "The Future of Employment: How Susceptible Are
       Jobs to Computerisation?" - per-occupation computerisation probability.
     - U.S. BLS Occupational Employment Statistics, May 2016 - wages and
       employment counts by SOC code.
   No values here are invented; only column names/units were tidied.

2. ai_job_trends_dataset.csv
   A synthetic dataset with the SAME SCHEMA as the original file it replaces,
   so main.py keeps working unchanged. Unlike the original (whose columns
   were essentially uncorrelated random noise - see the near-zero
   correlations main.py prints), every relationship here is intentional and
   documented in the comments below, so charts built from it show real
   (if synthetic) patterns instead of noise dressed up as data.

   Occupation titles are sampled from the real dataset above (weighted by
   real employment size), so titles are genuine job titles even though the
   industry/location/salary/etc. built around them are simulated.

   "Automation risk" is modeled as TWO separate columns, because "can a
   robot do this?" and "can an LLM do this?" point in different directions:
     - Automation Risk (Physical) %: classical robotic-automation framing
       (Frey & Osborne 2013, grounded in the real per-occupation
       probability) - manual/routine/on-site work scores highest, remote
       work is protective.
     - Automation Risk (Generative AI) %: generative-AI exposure framing
       (Eloundou et al. 2023 "GPTs are GPTs") - computer-based,
       writing/analysis/coding-heavy, remote-friendly work scores highest;
       remote work RAISES this one instead of lowering it.
   The two are independently modeled and allowed to disagree (e.g. IT skews
   low physical risk / high generative-AI risk).

This script reads its real-world source from the already-cleaned
real_automation_risk_by_occupation.csv committed in this folder (so it's
reproducible offline). To rebuild that file itself from scratch, re-download
https://raw.githubusercontent.com/plotly/datasets/master/job-automation-probability.csv
and re-run the one-time cleaning step (see git history of this file for it).

Run: python data/build_datasets.py   (writes ai_job_trends_dataset.csv into data/)
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Real dataset: already cleaned and committed - just load it.
# ---------------------------------------------------------------------------
real = pd.read_csv(os.path.join(HERE, "real_automation_risk_by_occupation.csv"))
print(f"real_automation_risk_by_occupation.csv: {len(real)} occupations")

# ---------------------------------------------------------------------------
# 2. Synthetic dataset: same schema as the original ai_job_trends_dataset.csv,
#    built with documented, intentional relationships.
# ---------------------------------------------------------------------------
N = 20000

INDUSTRY_KEYWORDS = [
    ("Healthcare", ["nurs", "medical", "health", "therap", "physician", "dental",
                     "clinical", "diagnos", "surg", "pharmac"]),
    ("IT", ["software", "computer", "data", "web", "network", "information security",
            "systems analy", "programmer", "database"]),
    ("Education", ["teacher", "instructor", "professor", "librarian", "education",
                    "tutor", "school"]),
    ("Finance", ["financ", "account", "actuar", "bank", "audit", "insurance",
                  "budget", "credit", "tax"]),
    ("Retail", ["sales", "retail", "cashier", "merchandis", "store"]),
    ("Entertainment", ["actor", "artist", "musician", "media", "broadcast",
                         "entertain", "writer", "editor", "design"]),
]
DEFAULT_INDUSTRY = "Manufacturing"  # the real source file skews industrial/manual

def classify_industry(title: str) -> str:
    t = title.lower()
    for industry, keywords in INDUSTRY_KEYWORDS:
        if any(k in t for k in keywords):
            return industry
    return DEFAULT_INDUSTRY

real["industry"] = real["occupation"].apply(classify_industry)

EDU_LEVEL = {"High School": 0, "Associate Degree": 1, "Bachelor’s Degree": 2,
             "Master’s Degree": 3, "PhD": 4}

LOCATIONS = ["USA", "Canada", "UK", "Australia", "Germany", "Brazil", "India", "China"]
# Rough relative wage-level multipliers (USD terms) used only to scale the
# synthetic salary field - not a claim about any other variable.
LOCATION_WAGE_MULT = {"USA": 1.00, "Australia": 0.95, "Canada": 0.85, "UK": 0.85,
                        "Germany": 0.85, "China": 0.45, "Brazil": 0.40, "India": 0.28}

# Higher remote-work base rate for desk/office-heavy industries, lower for
# hands-on/physical ones.
INDUSTRY_REMOTE_BASE = {"IT": 65, "Finance": 55, "Education": 45, "Entertainment": 35,
                          "Healthcare": 20, "Retail": 10, "Manufacturing": 8}

# Sample rows: draw occupations weighted by their real employment size, so
# common real jobs show up more often than rare ones (still every draw is an
# independent synthetic "job posting", not a real individual).
weights = np.sqrt(real["employed_may_2016"].to_numpy())
weights = weights / weights.sum()
idx = rng.choice(len(real), size=N, p=weights, replace=True)
rows = real.iloc[idx].reset_index(drop=True)

edu_level = rows["required_education"].map(EDU_LEVEL).to_numpy()

# --- Automation Risk (Physical) %: the real Frey & Osborne (2013)
# computerisation probability for that occupation, jittered with synthetic
# noise. This asks "can a machine/robot physically do this task?" - the
# classical robotic-automation framing, and the one field here most directly
# grounded in real data. Routine/manual/on-site work scores highest; it is
# NOT the same question as generative-AI exposure below, and the two are
# deliberately allowed to disagree (see Automation Risk (Generative AI) %).
automation_risk_physical = np.clip(
    rows["automation_probability"].to_numpy() * 100 + rng.normal(0, 6, N), 0, 100
)
automation_risk = automation_risk_physical  # legacy alias used below

# --- Required Education: inherited directly from the real occupation.
required_education = rows["required_education"].to_numpy()

# --- Experience Required (Years): rises with education level, falls
# slightly with automation risk (higher-risk roles skew entry-level), plus
# noise. Clipped to a plausible 0-20 range.
experience_required_years = np.clip(
    np.round(edu_level * 2.2 - automation_risk * 0.03 + rng.normal(3, 2.5, N)), 0, 20
).astype(int)

# --- Median Salary (USD): the real median wage for that occupation, scaled
# by a location wage multiplier and jittered +/-15%, minus a small penalty
# for automation risk (routine/at-risk work tends to pay less).
location = rng.choice(LOCATIONS, size=N)
loc_mult = np.array([LOCATION_WAGE_MULT[l] for l in location])
median_salary = (
    rows["median_annual_wage_usd"].to_numpy() * loc_mult
    * rng.normal(1.0, 0.15, N)
    - automation_risk * 80
).round(2)
median_salary = np.clip(median_salary, 15000, None)

# --- Remote Work Ratio (%): industry base rate, reduced for higher physical
# automation risk (manual/routine work needs a physical presence), plus noise.
remote_base = rows["industry"].map(INDUSTRY_REMOTE_BASE).to_numpy()
remote_work_ratio = np.clip(
    remote_base - automation_risk_physical * 0.25 + rng.normal(0, 12, N), 0, 100
).round(2)

# --- Automation Risk (Generative AI) %: a SEPARATE, deliberately different
# question - "can an LLM do this work?" (Eloundou et al. 2023, "GPTs are
# GPTs"). That research found the opposite pattern from classical robotic
# automation: computer-based, writing/analysis/coding-heavy, remote-friendly
# work is the MOST exposed, and hands-on physical work is the LEAST exposed;
# exposure also doesn't fall with seniority/education the way physical risk
# does. So this is modeled from industry base rate + remote work ratio
# (positive) + education level (mildly positive) - it does not reuse the
# physical-risk value at all, and the two columns are free to disagree (e.g.
# IT: low physical risk, high generative-AI risk).
GENAI_INDUSTRY_BASE = {"IT": 62, "Finance": 55, "Education": 48, "Entertainment": 45,
                         "Retail": 32, "Healthcare": 26, "Manufacturing": 15}
genai_base = rows["industry"].map(GENAI_INDUSTRY_BASE).to_numpy()
automation_risk_genai = np.clip(
    genai_base + remote_work_ratio * 0.30 + edu_level * 2.5 + rng.normal(0, 10, N), 0, 100
)

# --- Projected growth: occupations with lower automation risk get a growth
# multiplier > 1, higher-risk occupations get < 1, plus noise. Job Openings
# (2024) is scaled from the occupation's real employment size (capped) so
# common real jobs post more openings than rare ones.
job_openings_2024 = np.clip(
    (rows["employed_may_2016"].to_numpy() / 400) * rng.uniform(0.5, 1.5, N), 20, 10000
).round().astype(int)
growth_factor = 1.15 - (automation_risk / 100) * 0.5 + rng.normal(0, 0.08, N)
projected_openings_2030 = np.clip(
    (job_openings_2024 * growth_factor).round().astype(int), 0, None
)
job_status = np.where(
    (projected_openings_2030 >= job_openings_2024) ^ (rng.random(N) < 0.08),
    "Increasing", "Decreasing",
)

# --- AI Impact Level: bucketed from the GENERATIVE-AI risk (this column is
# about AI impact, not robotic/physical automation), with a small amount of
# label noise so it isn't a perfectly deterministic re-statement of the %.
bucket = np.select(
    [automation_risk_genai < 33, automation_risk_genai < 66],
    ["Low", "Moderate"], default="High",
)
noisy_slot = rng.random(N) < 0.12
ai_impact_level = np.where(
    noisy_slot, rng.choice(["Low", "Moderate", "High"], size=N), bucket
)

# --- Gender Diversity (%): deliberately left as unstructured noise, not
# tied to industry or any other field - we have no real, non-stereotyped
# basis to correlate it, so it's an honest placeholder rather than a
# fabricated relationship.
gender_diversity = np.clip(rng.normal(50, 15, N), 5, 95).round(2)

synthetic = pd.DataFrame({
    "Job Title": rows["occupation"].to_numpy(),
    "Industry": rows["industry"].to_numpy(),
    "Job Status": job_status,
    "AI Impact Level": ai_impact_level,
    "Median Salary (USD)": median_salary,
    "Required Education": required_education,
    "Experience Required (Years)": experience_required_years,
    "Job Openings (2024)": job_openings_2024,
    "Projected Openings (2030)": projected_openings_2030,
    "Remote Work Ratio (%)": remote_work_ratio,
    "Automation Risk (Physical) %": automation_risk_physical.round(2),
    "Automation Risk (Generative AI) %": automation_risk_genai.round(2),
    "Location": location,
    "Gender Diversity (%)": gender_diversity,
})

out_path = os.path.join(HERE, "ai_job_trends_dataset.csv")
synthetic.to_csv(out_path, index=False, encoding="utf-8")
print(f"ai_job_trends_dataset.csv: {len(synthetic)} rows")
num = synthetic.select_dtypes(include="number")
print("\ncorrelation with Remote Work Ratio (%):")
print(num.corr()["Remote Work Ratio (%)"][["Automation Risk (Physical) %", "Automation Risk (Generative AI) %"]])
print("\ncorrelation between the two risk columns:")
print(num["Automation Risk (Physical) %"].corr(num["Automation Risk (Generative AI) %"]))