import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
category_df = pd.read_csv("../data/intermediate/analysis_by_category.csv")
service_term_df = pd.read_csv("../data/intermediate/analysis_by_service_term.csv")

# === Global Style Settings ===
TITLE_FONT = {"fontsize": 15, "color": "#2c3e50"}
LABEL_FONT = {"fontsize": 10, "color": "#2c3e50"}
FILL_COLOR = "#a8dadc"
LINE_COLOR = "#2c3e50"
BG_COLOR = "#f8f9fa"
COLOR_PALETTE = plt.cm.Pastel1  # Used for other plots (not the pie)

# 1. Radar Chart
def plot_company_radar(company_name, df, save_path):
    filtered_df = df[(df["Company"].str.lower() == company_name.lower()) & (df["Category"] != "General")]
    categories = filtered_df["Category"].tolist()
    values = filtered_df["Combined_WeightedScore"].tolist()

    if len(categories) < 3:
        print(f"Not enough categories for {company_name} to create a radar chart.")
        return

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]
    categories += categories[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.spines['polar'].set_visible(False)
    ax.grid(color="lightgray", linestyle="--", linewidth=0.7)
    ax.tick_params(colors="#2c3e50")

    ax.plot(angles, values, linewidth=2, color=LINE_COLOR)
    ax.fill(angles, values, color=FILL_COLOR, alpha=0.6)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1], **LABEL_FONT)

    title_text = rf"$\bf{{{company_name}}}$ Performance by Keywords"
    ax.set_title(title_text, pad=20, **TITLE_FONT)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# 2. Pie Chart (Green-Blue Tones)
def plot_dominance_pie(df, save_path, label_col, label_value, score_col="Combined_WeightedScore", top_n=6):
    filtered = df[df[label_col] == label_value]
    sorted_df = filtered.sort_values(by=score_col, ascending=False)

    top = sorted_df.head(top_n)
    others = sorted_df.iloc[top_n:]
    others_score = others[score_col].sum()
    if others_score > 0:
        top = pd.concat([top, pd.DataFrame({
            label_col: [label_value],
            "Company": ["Others"],
            score_col: [others_score]
        })])

    # Green-blue color palette
    green_blue_palette = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(top)))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
    wedges, texts, autotexts = ax.pie(
        top[score_col],
        labels=top["Company"],
        autopct="%1.1f%%",
        startangle=140,
        colors=green_blue_palette,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10, "color": "#2c3e50"}
    )

    label_type = "Terminology" if label_col == "Service Term" else "Keyword"
    title_text = rf'Dominance in {label_type}: $\bf{{{label_value}}}$'
    ax.set_title(title_text, **TITLE_FONT, pad=20)

    if "Others" in top["Company"].values:
        others_list = others["Company"].tolist()
        small_text = ", ".join(others_list)
        fig.text(0.5, 0.01, f"Others include: {small_text}", wrap=True,
                 ha='center', fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# === Generate Radar Charts ===
companies = ["Waymo", "Cruise", "Tesla", "Zoox"]
for company in companies:
    file_name = f"{company.lower()}_radar_chart.png"
    save_path = os.path.join("../data/final/radar/", file_name)
    plot_company_radar(company, category_df, save_path)

# === Generate Category Pie Charts ===
categories = category_df["Category"].unique()
for cat in categories:
    file_name = f"{cat.lower().replace(' ', '_')}_dominance_pie.png"
    save_path = os.path.join("../data/final/category/", file_name)
    plot_dominance_pie(category_df, save_path, label_col="Category", label_value=cat)

# === Generate Service Term Pie Charts ===
service_terms = service_term_df["Service Term"].unique()
for term in service_terms:
    file_name = f"{term.lower().replace(' ', '_')}_dominance_pie.png"
    save_path = os.path.join("../data/final/service_term/", file_name)
    plot_dominance_pie(service_term_df, save_path, label_col="Service Term", label_value=term)
