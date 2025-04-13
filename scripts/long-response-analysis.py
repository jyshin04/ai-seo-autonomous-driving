import pandas as pd
import re
import random
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# === Step 1: Load Data ===
file_path = "../data/raw/long_response_combined.csv"
df = pd.read_csv(file_path)

# === Step 2: Text Cleaning for Each LLM Column ===
service_terms = {"robotaxi", "autonomous", "vehicle", "vehicles", "self-driving", "driverless", "car", "safety", "technology"}
company_names = {"waymo", "cruise", "tesla", "zoox"}

custom_stopwords = {
    'the', 'and', 'for', 'are', 'with', 'that', 'has', 'was', 'have', 'but', 'not', 'you',
    'your', 'this', 'can', 'they', 'from', 'been', 'their', 'will', 'also', 'had', 'about',
    'who', 'which', 'would', 'there', 'more', 'some', 'such', 'its', 'into', 'all', 'what',
    'than', 'when', 'how', 'out', 'any', 'our', 'his', 'her', 'she', 'him', 'them', 'because',
    'these', 'those', 'being', 'where', 'after', 'before', 'very', 'just', 'over', 'while'
}.union(service_terms).union(company_names)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in custom_stopwords and len(w) > 2]
    return " ".join(words)

df["Claude_Cleaned"] = df["Claude"].apply(clean_text)
df["ChatGPT_1_Cleaned"] = df["ChatGPT_1"].apply(clean_text)
df["ChatGPT_2_Cleaned"] = df["ChatGPT_2"].apply(clean_text)

# === Step 3: Sentiment Calculation ===
df["Claude_Sentiment"] = df["Claude_Cleaned"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["ChatGPT_1_Sentiment"] = df["ChatGPT_1_Cleaned"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["ChatGPT_2_Sentiment"] = df["ChatGPT_2_Cleaned"].apply(lambda x: TextBlob(x).sentiment.polarity)

# === Step 4: Save Per-Answer Sentiment ===
sentiment_columns = [
    "Company", "Claude_Sentiment", "ChatGPT_1_Sentiment", "ChatGPT_2_Sentiment"
]
df[sentiment_columns].to_csv("../data/final/sentiment/sentiment_by_answer.csv", index=False)

# === Step 5: Final Company-Level Sentiment (All Models Averaged) ===
df["Avg_Sentiment"] = df[["Claude_Sentiment", "ChatGPT_1_Sentiment", "ChatGPT_2_Sentiment"]].mean(axis=1)
company_sentiment = df.groupby("Company")["Avg_Sentiment"].mean().reset_index()
company_sentiment.columns = ["Company", "Sentiment_Score"]

custom_order = ["Tesla", "Waymo", "Zoox", "Cruise"]
company_sentiment["Company"] = pd.Categorical(company_sentiment["Company"], categories=custom_order, ordered=True)
company_sentiment = company_sentiment.sort_values("Company")
company_sentiment.to_csv("../data/final/sentiment/sentiment_by_company.csv", index=False)

# === Step 6: Color and Style Setup ===
bar_colors = {
    "Tesla": "#76D7C4", "Waymo": "#48C9B0",
    "Zoox": "#7FB3D5", "Cruise": "#5DADE2"
}
wordcloud_colormaps = {
    "Tesla": "BuGn", "Waymo": "GnBu",
    "Zoox": "PuBu", "Cruise": "Blues"
}

font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
plt.rcParams["font.family"] = "DejaVu Sans"

# === Step 7: Bar Chart — Sentiment by Company ===
plt.figure(figsize=(10, 6))
bars = plt.bar(
    company_sentiment["Company"],
    company_sentiment["Sentiment_Score"],
    color=[bar_colors[c] for c in company_sentiment["Company"]],
    width=0.5
)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.2f}", ha="center", va="bottom", fontsize=12)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.xticks(fontweight="bold")
plt.title("Average Sentiment Score by Company", fontsize=18, weight='bold')
plt.ylabel("Sentiment Score", fontsize=14)
plt.ylim(0, 0.25)
plt.tight_layout()
plt.savefig("../data/final/sentiment/sentiment_bar_chart.png", dpi=300)
plt.close()

# === Step 8: Claude vs ChatGPT per Company (Grouped Bar Chart) ===
company_model_sentiment = df.groupby("Company")[["Claude_Sentiment", "ChatGPT_1_Sentiment", "ChatGPT_2_Sentiment"]].mean()
company_model_sentiment["ChatGPT_Sentiment"] = company_model_sentiment[["ChatGPT_1_Sentiment", "ChatGPT_2_Sentiment"]].mean(axis=1)
company_model_sentiment = company_model_sentiment[["Claude_Sentiment", "ChatGPT_Sentiment"]].reset_index()

company_model_sentiment["Company"] = pd.Categorical(company_model_sentiment["Company"], categories=custom_order, ordered=True)
company_model_sentiment = company_model_sentiment.sort_values("Company")

plt.figure(figsize=(10, 6))
bar_width = 0.35
x = range(len(company_model_sentiment))

bars1 = plt.bar(
    [i - bar_width/2 for i in x],
    company_model_sentiment["Claude_Sentiment"],
    width=bar_width,
    color="#5DADE2",
    label="Claude"
)

bars2 = plt.bar(
    [i + bar_width/2 for i in x],
    company_model_sentiment["ChatGPT_Sentiment"],
    width=bar_width,
    color="#48C9B0",
    label="ChatGPT"
)

# Add text labels
for i, val in enumerate(company_model_sentiment["Claude_Sentiment"]):
    plt.text(i - bar_width/2, val + 0.005, f"{val:.2f}", ha='center', va='bottom', fontsize=12)

for i, val in enumerate(company_model_sentiment["ChatGPT_Sentiment"]):
    plt.text(i + bar_width/2, val + 0.005, f"{val:.2f}", ha='center', va='bottom', fontsize=12)

plt.xticks(ticks=range(len(company_model_sentiment)), labels=company_model_sentiment["Company"], fontsize=13, fontweight="bold")
plt.yticks(fontsize=11)
plt.ylabel("Sentiment Score", fontsize=14)
plt.title("Sentiment Score by Company and Model", fontsize=18, weight="bold")
plt.ylim(0, 0.25)

# Match styling of sentiment_bar_chart
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.legend(title="Model", fontsize=11)
plt.tight_layout()
plt.savefig("../data/final/sentiment/sentiment_company_by_model_chart.png", dpi=300)
plt.close()

# === Step 9: Word Clouds ===
vectorizer = CountVectorizer(stop_words="english", max_features=100)
X = vectorizer.fit_transform(df["Claude_Cleaned"] + " " + df["ChatGPT_1_Cleaned"] + " " + df["ChatGPT_2_Cleaned"])
word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_freq["Company"] = df["Company"]
word_freq_grouped = word_freq.groupby("Company").sum()

def color_func_factory(colormap_name):
    cmap = plt.get_cmap(colormap_name)
    def color_func(*args, **kwargs):
        r, g, b, _ = cmap(random.uniform(0.3, 0.7))
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    return color_func

for company in custom_order:
    freqs = word_freq_grouped.loc[company].to_dict()
    filtered_freqs = {
        k: v for k, v in freqs.items()
        if company.lower() not in k and k not in ["vehicle", "vehicles"]
    }

    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        font_path=font_path,
        max_words=100,
        colormap=wordcloud_colormaps[company],
        color_func=color_func_factory(wordcloud_colormaps[company])
    ).generate_from_frequencies(filtered_freqs)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Top Words Associated with {company}", fontsize=18, weight='bold', pad=30)
    plt.tight_layout(pad=3)
    plt.savefig(f"../data/final/wordcloud/wordcloud_{company.lower()}.png", dpi=300)
    plt.close()
