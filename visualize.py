import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


METHOD_COLORS = {
    "tfidf": "#E45C1D",    # orange-red
    "sbert": "#E5F016",    # bright yellow
    "scibert": "#24DDCA"   # teal
}

OUTPUT_DIR = "results/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results(csv_path="results/evaluation.csv"):
    df = pd.read_csv(csv_path)
    return df

# 1: Bar chart — average entropy per method
def average_entropy_plot(df):
    """ Plot average entropy for each method
    to Shows which method produces the most balanced section coverage on average """

    # method is the index, to change it to a regular column we should reset_index
    avg = df.groupby("method")["entropy"].mean().reset_index()  #  average entropy for each group
    std = df.groupby("method")["entropy"].std().reset_index()  # standard deviation — how much entropy varies across papers

    # Create the figure and axis
    fig, ax = plt.subplots()

    ax.set_title("Average Section Entropy by Method", pad=20)
    ax.set_xlabel("Method", labelpad=10)
    ax.set_ylabel("Average Section Entropy", labelpad=10)
    ax.set_xticklabels([m.upper() for m in avg["method"]])

    bars = ax.bar(
        x=avg["method"],
        height=avg["entropy"],
        color= [METHOD_COLORS.get(m, "gray") for m in avg["method"]],
        yerr = std["entropy"], # error bars, standard deviation.
        edgecolor="black",
        capsize=3.5,
        # width= 0.6
    )

    # value labels on top of each bar
    for bar, y_value in zip(bars, avg["entropy"]):
        ax.text(
            x = bar.get_x() + bar.get_width() / 2,
            y = y_value,
            s = f"{y_value:.2f}",
            va="bottom",
            # ha="center"
        )

    path = os.path.join(OUTPUT_DIR, "avg_entropy_per_method.png")
    plt.savefig(path)
    plt.close()

# 2: Box plot — distribution of entropy values for each method
def entropy_distribution_plot(df):
    """ Box plot of entropy distribution for each method """

    methods = sorted(df["method"].unique())
    entropy_per_method = [df[df["method"] == m]["entropy"].values for m in methods]

    fig, ax = plt.subplots()

    ax.set_title("Distribution of Section Coverage Entropy Across Papers",
                 pad=15)
    ax.set_xlabel("Summarization Method",labelpad=10)
    ax.set_ylabel("Entropy",labelpad=10)
    ax.set_xticklabels([m.upper() for m in methods])

    boxplot = ax.boxplot(
        entropy_per_method,
        patch_artist=True,  # to color the boxes
        notch=False,  # straight edges (True = pinched middle)
        widths=0.5,
    )

    for patch, method in zip(boxplot["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "gray"))
        patch.set_alpha(0.7)

    path = os.path.join(OUTPUT_DIR, "entropy_distribution_boxplot.png")
    plt.savefig(path)
    plt.close()



if __name__ == "__main__":
    df = load_results("results/evaluation.csv")

    print("Generating visualizations...")
    average_entropy_plot(df)
    entropy_distribution_plot(df)