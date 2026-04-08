
import requests
import time
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants (from existing cells) ---
TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"
headers = {"User-Agent": "TrendPulse/1.0"}

CATEGORIES = {
    "technology": ["ai", "software", "tech", "code", "computer", "data", "cloud", "api", "gpu", "llm"],
    "worldnews": ["war", "government", "country", "president", "election", "climate", "attack", "global"],
    "sports": ["nfl", "nba", "fifa", "sport", "game", "team", "player", "league", "championship"],
    "science": ["research", "study", "space", "physics", "biology", "discovery", "nasa", "genome"],
    "entertainment": ["movie", "film", "music", "netflix", "game", "book", "show", "award", "streaming"]
}

# --- Helper Function (from existing cells) ---
def get_category(title):
    title = title.lower()
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in title:
                return category
    return "others"

# --- Pipeline Functions ---

def collect_data_step(data_dir):
    """
    Collects top Hacker News stories and saves them to a JSON file.
    Returns the path to the saved JSON file.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = os.path.join(data_dir, f"trends_{datetime.now().strftime('%Y%m%d')}.json")

    try:
        response = requests.get(TOP_STORIES_URL, headers=headers)
        story_ids = response.json()[:500]
    except Exception as e:
        print("Failed to fetch top stories:", e)
        story_ids = []
        return None

    stories = []
    category_count = {cat: 0 for cat in CATEGORIES}

    for story_id in story_ids:
        try:
            res = requests.get(ITEM_URL.format(story_id), headers=headers)
            data = res.json()

            if not data or "title" not in data:
                continue

            category = get_category(data["title"])

            if category in category_count and category_count[category] >= 25:
                continue

            story = {
                "post_id": data.get("id"),
                "title": data.get("title"),
                "category": category,
                "score": data.get("score", 0),
                "num_comments": data.get("descendants", 0),
                "author": data.get("by", ""),
                "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            stories.append(story)

            if category in category_count:
                category_count[category] += 1

            if sum(category_count.values()) >= 125:
                break

        except Exception as e:
            print(f"Error fetching story {story_id}: {e}")
            continue

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(stories, f, indent=4)

    print(f"Collected {len(stories)} stories. Saved to {filename}")
    return filename

def clean_data_step(json_file_path):
    """
    Loads data from a JSON file, cleans it, and returns a Pandas DataFrame.
    """
    try:
        df = pd.read_json(json_file_path)
        print(f"Successfully loaded JSON file: {json_file_path}")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"Number of rows loaded: {len(df)}")

    # 1. Remove duplicate rows based on 'post_id'
    initial_shape = df.shape[0]
    df.drop_duplicates(subset=['post_id'], inplace=True)
    print(f"Removed {initial_shape - df.shape[0]} duplicate rows based on 'post_id'. Current shape: {df.shape}")

    # 2. Drop rows with missing values in 'post_id', 'title', or 'score'
    initial_shape = df.shape[0]
    df.dropna(subset=['post_id', 'title', 'score'], inplace=True)
    print(f"Removed {initial_shape - df.shape[0]} rows with missing essential values. Current shape: {df.shape}")

    # 3. Convert 'score' and 'num_comments' to integer type
    df['score'] = df['score'].astype(int)
    df['num_comments'] = df['num_comments'].fillna(0).astype(int)
    print("Converted 'score' and 'num_comments' to integer type.")

    # 4. Remove stories where 'score' is less than 5
    initial_shape = df.shape[0]
    df = df[df['score'] >= 5]
    print(f"Removed {initial_shape - df.shape[0]} stories with score less than 5. Current shape: {df.shape}")

    # 5. Strip extra spaces from the 'title' column
    df['title'] = df['title'].str.strip()
    print("Whitespace stripped from 'title' column.")

    print("\nFirst 5 rows of the cleaned DataFrame:")
    print(df.head().to_markdown(index=False))

    return df

def analyze_data_step(df):
    """
    Performs basic analysis and feature engineering on the DataFrame.
    """
    if df.empty:
        print("DataFrame is empty, skipping analysis.")
        return pd.DataFrame()

    print("\n--- Basic Analysis ---")
    average_score = df['score'].mean()
    average_comments = df['num_comments'].mean()
    print(f"Average Score across all stories: {average_score:.2f}")
    print(f"Average Number of Comments across all stories: {average_comments:.2f}")

    print("\n--- NumPy Analysis ---")
    mean_score = np.mean(df['score'])
    median_score = np.median(df['score'])
    std_dev_score = np.std(df['score'])
    print(f"Mean Score: {mean_score:.2f}")
    print(f"Median Score: {median_score:.2f}")
    print(f"Standard Deviation of Score: {std_dev_score:.2f}")

    highest_score = np.max(df['score'])
    lowest_score = np.min(df['score'])
    print(f"Highest Score: {highest_score}")
    print(f"Lowest Score: {lowest_score}")

    most_stories_category = df['category'].value_counts().idxmax()
    num_most_stories = df['category'].value_counts().max()
    print(f"Category with the most stories: {most_stories_category} ({num_most_stories} stories)")

    most_comments_story = df.loc[df['num_comments'].idxmax()]
    print("\nStory with the most comments:")
    print(f"  Title: {most_comments_story['title']}")
    print(f"  Comment Count: {most_comments_story['num_comments']}")

    print("\n--- Feature Engineering ---")
    df['engagement'] = df['num_comments'] / (df['score'] + 1)
    df['is_popular'] = df['score'] > average_score
    print("Added 'engagement' and 'is_popular' columns.")
    print("\nFirst 5 rows of the DataFrame with new columns:")
    print(df.head().to_markdown(index=False))

    return df

def visualize_data_step(df, output_dir):
    """
    Generates and saves visualizations based on the analyzed DataFrame.
    """
    if df.empty:
        print("DataFrame is empty, skipping visualizations.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")

    # Prepare data for Chart 1: Top 10 Stories by Score
    top_10_stories_by_score = df.sort_values(by='score', ascending=False).head(10)
    top_10_stories_by_score['short_title'] = top_10_stories_by_score['title'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

    # Prepare data for Chart 2: Number of Stories Per Category
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    # Create the figure and a set of subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # 1 row, 3 columns

    # Plot Chart 1: Top 10 Hacker News Stories by Score
    sns.barplot(x='score', y='short_title', data=top_10_stories_by_score, palette='viridis', ax=axes[0])
    axes[0].set_title('Top 10 Hacker News Stories by Score')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Story Title')

    # Plot Chart 2: Number of Stories Per Category
    sns.barplot(x='category', y='count', data=category_counts, palette='Paired', hue='category', legend=False, ax=axes[1])
    axes[1].set_title('Number of Stories Per Category')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Number of Stories')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot Chart 3: Hacker News Stories: Score vs. Number of Comments (Popularity Highlighted)
    sns.scatterplot(x='score', y='num_comments', hue='is_popular', data=df, palette='coolwarm', s=100, alpha=0.7, ax=axes[2])
    axes[2].set_title('Score vs. Comments (Popularity)')
    axes[2].set_xlabel('Score')
    axes[2].set_ylabel('Number of Comments')
    axes[2].legend(title='Is Popular')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    # Add an overall title to the figure
    fig.suptitle('TrendPulse Dashboard', fontsize=20, y=1.02)

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle

    # Save the dashboard
dashboard_filename = os.path.join(output_dir, 'dashboard.png')
    plt.savefig(dashboard_filename)
    print(f"Dashboard saved to {dashboard_filename}")

    plt.close(fig) # Close the figure to prevent it from displaying twice in some environments


def main():
    data_dir = "data"
    output_dir = "outputs"

    # Step 1: Collect Data
    print("--- Starting Data Collection ---")
    raw_json_path = collect_data_step(data_dir)
    if not raw_json_path:
        print("Data collection failed. Exiting pipeline.")
        return

    # Step 2: Clean Data
    print("\n--- Starting Data Cleaning ---")
    cleaned_df = clean_data_step(raw_json_path)
    if cleaned_df.empty:
        print("No data available after cleaning. Exiting pipeline.")
        return
    cleaned_csv_path = os.path.join(data_dir, "trends_clean.csv")
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned data saved to {cleaned_csv_path}")

    # Step 3: Analyze Data
    print("\n--- Starting Data Analysis ---")
    analysed_df = analyze_data_step(cleaned_df.copy())
    analysed_csv_path = os.path.join(data_dir, "trends_analysed.csv")
    analysed_df.to_csv(analysed_csv_path, index=False)
    print(f"Analyzed data saved to {analysed_csv_path}")

    # Step 4: Visualize Data
    print("\n--- Starting Data Visualization ---")
    visualize_data_step(analysed_df.copy(), output_dir)
    print(f"All visualizations generated and saved to {output_dir}")

    print("\n--- Data Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
