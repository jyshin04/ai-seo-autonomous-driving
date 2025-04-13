import pandas as pd
import ast
from collections import Counter

# Load combined ranked CSV
df = pd.read_csv("../data/raw/ranked_combined.csv")

# Weights for ChatGPT and Claude
chatgpt_weight = 0.8
claude_weight = 0.2

# Function to parse string representation of dictionaries
def parse_ranking(ranking_str):
    try:
        return ast.literal_eval(ranking_str)
    except Exception:
        return {}

# Function to assign weighted scores by rank position
def compute_weighted_scores(rankings, weight_scheme=None):
    if weight_scheme is None:
        weight_scheme = {i: 6 - i for i in range(1, 6)}  # 1st=5 pts, 2nd=4, ..., 5th=1
    score = Counter()
    for pos, company in rankings.items():
        try:
            pos = int(pos)
            score[company] += weight_scheme.get(pos, 0)
        except Exception:
            continue
    return score

# Initialize storage for aggregated results
service_term_data = {}
category_data = {}

# Process each row in the DataFrame
for _, row in df.iterrows():
    service_term = row['Service Term']
    category = row['Category']
    
    chatgpt_ranks = parse_ranking(row['ChatGPT'])
    claude_ranks = parse_ranking(row['Claude'])

    chatgpt_scores = compute_weighted_scores(chatgpt_ranks)
    claude_scores = compute_weighted_scores(claude_ranks)

    # Aggregate by service term
    if service_term not in service_term_data:
        service_term_data[service_term] = {"ChatGPT": Counter(), "Claude": Counter()}
    service_term_data[service_term]["ChatGPT"].update(chatgpt_scores)
    service_term_data[service_term]["Claude"].update(claude_scores)

    # Aggregate by category
    if category not in category_data:
        category_data[category] = {"ChatGPT": Counter(), "Claude": Counter()}
    category_data[category]["ChatGPT"].update(chatgpt_scores)
    category_data[category]["Claude"].update(claude_scores)

# Helper function to convert nested counters to DataFrame
def convert_aggregation_to_df(agg_data, label_name):
    rows = []
    for label, model_data in agg_data.items():
        all_companies = set(model_data["ChatGPT"].keys()).union(model_data["Claude"].keys())
        for company in all_companies:
            chatgpt_score = model_data["ChatGPT"].get(company, 0)
            claude_score = model_data["Claude"].get(company, 0)
            combined_score = chatgpt_score * chatgpt_weight + claude_score * claude_weight
            rows.append({
                label_name: label,
                "Company": company,
                "ChatGPT_WeightedScore": chatgpt_score,
                "Claude_WeightedScore": claude_score,
                "Combined_WeightedScore": combined_score
            })
    return pd.DataFrame(rows)

# Create final DataFrames
service_term_df = convert_aggregation_to_df(service_term_data, "Service Term")
category_df = convert_aggregation_to_df(category_data, "Category")

# Save as CSVs
service_term_df.to_csv("../data/intermediate/analysis_by_service_term.csv", index=False)
category_df.to_csv("../data/intermediate/analysis_by_category.csv", index=False)
