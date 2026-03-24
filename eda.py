import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data.loader import load_transactions
from features.engineering import add_features

# creates plots/ folder if it doesn't exist, exist_ok means no error if it already does
Path("plots").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def plot_spend_distribution(df):
    # two subplots: full range shows outliers, under $500 shows the bulk of transactions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["amt"], bins=100, color="steelblue", edgecolor="none")
    axes[0].set_title("Spend Distribution (full range)")
    axes[0].set_xlabel("Amount ($)")

    # filter to under $500 to see the real distribution without extreme outliers squashing it
    axes[1].hist(df[df["amt"] < 500]["amt"], bins=100, color="steelblue", edgecolor="none")
    axes[1].set_title("Spend Distribution (under $500)")
    axes[1].set_xlabel("Amount ($)")

    plt.tight_layout()
    plt.savefig("plots/spend_distribution.png", dpi=150)
    plt.close()


def plot_fraud_rate(df):
    fraud_counts = df["is_fraud"].value_counts()

    plt.figure(figsize=(6, 5))
    plt.bar(["Legitimate", "Fraud"], fraud_counts.values, color=["steelblue", "tomato"], edgecolor="none")
    # show the actual fraud percentage in the title so it's immediately readable
    plt.title(f"Fraud vs Legitimate ({fraud_counts[1] / len(df) * 100:.2f}% fraud)")
    plt.ylabel("Transaction count")
    plt.tight_layout()
    plt.savefig("plots/fraud_rate.png", dpi=150)
    plt.close()


def plot_spend_by_category(df):
    # agg lets you compute multiple statistics in one groupby call
    category_stats = (
        df.groupby("category")["amt"]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # barh = horizontal bar chart, easier to read category names than vertical
    axes[0].barh(category_stats.index, category_stats["mean"], color="steelblue", edgecolor="none")
    axes[0].set_title("Mean Spend by Category")
    axes[0].set_xlabel("Mean Amount ($)")

    axes[1].barh(category_stats.index, category_stats["count"], color="mediumseagreen", edgecolor="none")
    axes[1].set_title("Transaction Count by Category")
    axes[1].set_xlabel("Number of Transactions")

    plt.tight_layout()
    plt.savefig("plots/spend_by_category.png", dpi=150)
    plt.close()


def plot_fraud_by_category(df):
    # .mean() on a 0/1 column gives the proportion, multiply by 100 for percentage
    fraud_rate = (
        df.groupby("category")["is_fraud"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(fraud_rate.index, fraud_rate.values * 100, color="tomato", edgecolor="none")
    plt.title("Fraud Rate by Category (%)")
    plt.xlabel("Fraud Rate (%)")
    plt.tight_layout()
    plt.savefig("plots/fraud_by_category.png", dpi=150)
    plt.close()


def plot_spend_by_hour(df):
    hourly = df.groupby("hour")["amt"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly.index, hourly.values, color="steelblue", linewidth=2)
    plt.title("Average Spend by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Mean Amount ($)")
    plt.xticks(range(0, 24))  # force all 24 hours to show on x axis
    plt.tight_layout()
    plt.savefig("plots/spend_by_hour.png", dpi=150)
    plt.close()


def plot_fraud_vs_legit_spend(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, color in [(0, "steelblue"), (1, "tomato")]:
        subset = df[df["is_fraud"] == label]["amt"]
        subset = subset[subset < 500]
        # alpha=0.6 makes bars semi-transparent so both distributions are visible where they overlap
        ax.hist(subset, bins=80, alpha=0.6, color=color,
                label="Legitimate" if label == 0 else "Fraud", edgecolor="none")

    ax.set_title("Spend Distribution: Fraud vs Legitimate (under $500)")
    ax.set_xlabel("Amount ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/fraud_vs_legit_spend.png", dpi=150)
    plt.close()

def plot_fraud_spend_only(df):
    fraud = df[df["is_fraud"] == 1]["amt"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(fraud, bins=100, color="tomato", edgecolor="none")
    axes[0].set_title("Fraud Spend Distribution (full range)")
    axes[0].set_xlabel("Amount ($)")
    
    # cap at 500 to see the bulk of fraud transactions
    axes[1].hist(fraud[fraud < 500], bins=100, color="tomato", edgecolor="none")
    axes[1].set_title("Fraud Spend Distribution (under $500)")
    axes[1].set_xlabel("Amount ($)")
    
    plt.tight_layout()
    plt.savefig("plots/fraud_spend_only.png", dpi=150)
    plt.close()
    
    print(f"\nFraud spend stats:")
    print(fraud.describe().round(2))

def print_summary(df):
    print(f"Total transactions: {len(df):,}")
    print(f"Date range: {df['trans_date_trans_time'].min()} to {df['trans_date_trans_time'].max()}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.3f}%")
    print(f"Unique categories: {df['category'].nunique()}")
    print(f"\nSpend stats:")
    print(df["amt"].describe().round(2))
    print(f"\nAge range: {df['age'].min()} to {df['age'].max()}")
    print(f"Distance range: {df['distance_km'].min():.1f} to {df['distance_km'].max():.1f} km")


if __name__ == "__main__":
    print("Loading data...")
    df = load_transactions()
    df = add_features(df)

    print("Running EDA...")
    print_summary(df)
    plot_spend_distribution(df)
    plot_fraud_rate(df)
    plot_spend_by_category(df)
    plot_fraud_by_category(df)
    plot_spend_by_hour(df)
    plot_fraud_vs_legit_spend(df)
    plot_fraud_spend_only(df)


    print("Done. Plots saved to plots/")