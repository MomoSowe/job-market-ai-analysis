# Job Market & AI Disruption Analysis

A data analysis project exploring the relationship between AI automation risk, job displacement, and workforce outcomes using real labor market data from the Anthropic Economic Index and O*NET.

## Overview

This project analyzes a 15,000-record labor market dataset across 19 features to understand how AI automation risk varies across industries, job roles, countries, and education levels. The goal was to move beyond surface-level observations and answer specific, structured business questions using real aggregation and statistical analysis techniques.

## Tools & Technologies

- **Python**
- **Pandas** — data cleaning, transformation, and groupby aggregations
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization and styling

## What This Project Does

- Analyzes a 15,000-record labor market dataset across 19 features to explore relationships between AI automation risk, job displacement, and workforce outcomes
- Answers 10 structured business questions using pandas groupby aggregations, identifying automation risk patterns by industry, job role, country, and education level
- Includes a custom visualization function that dynamically scales chart axes, improving readability of close-value comparisons across 9 bar chart visualizations
- Builds a full correlation matrix and heatmap across all numeric features to identify redundant metrics and validate relationships prior to deeper analysis
- Automates an image export pipeline, programmatically saving all generated visualizations for reporting and presentation use

## Data Sources

- **Anthropic Economic Index** — real-world data on AI usage and economic impact
- **O\*NET** — U.S. Department of Labor occupational database, providing detailed job role and skill data

## Project Structure

```
job-market-ai-analysis/
├── data/                  # Raw and/or cleaned datasets
├── notebooks/ or scripts/ # Data cleaning, analysis, and visualization code
├── outputs/                # Generated charts and visualizations
├── requirements.txt        # Python dependencies
└── README.md
```

## Key Skills Demonstrated

- Data cleaning and preprocessing on a large, multi-feature dataset
- Exploratory data analysis (EDA) and structured business-question answering
- Correlation analysis and heatmap visualization
- Custom, reusable visualization functions
- Automated reporting/export pipelines

## Author

Muhammad Sowe — Information Systems student at UMBC, seeking Data Analyst / SQL Developer opportunities.
[GitHub](https://github.com/MomoSowe) | sowem2765@gmail.com
