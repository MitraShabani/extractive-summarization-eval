import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


METHOD_COLORS = {
    "tfidf": "#C94B35",    # orange-red
    "sbert": "#C4C41A",    # yellow
    "scibert": "#3BB8B0"   # teal
}

OUTPUT_DIR = "results/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results(csv_path="results/evaluation.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
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

# 3: Bar chart — average kl_divergence
def plot_average_kl(df):
    """
    Shows which method best mirrors the paper's original structure.
    Lower KL divergence = better structural faithfulness.
    """
    avg = df.groupby("method")["kl_divergence"].mean().reset_index()
    std = df.groupby("method")["kl_divergence"].std().reset_index()

    # Create the figure and axis
    fig, ax = plt.subplots()

    ax.set_title("Average KL Divergence by Summarization Method", pad=15)
    ax.set_xlabel("Summarization Method",labelpad=10)
    ax.set_ylabel("KL Divergence (lower = more faithful to paper structure)",labelpad=10)
    ax.set_xticks(range(len(avg["method"])))
    ax.set_xticklabels([m.upper() for m in avg["method"]])

    bars = ax.bar(
        avg["method"],
        avg["kl_divergence"],
        yerr=std["kl_divergence"],
        color=[METHOD_COLORS.get(m, "gray") for m in avg["method"]],
        capsize=3.5,
        edgecolor="Black"
    )

    for bar, y_value in zip(bars, avg["kl_divergence"]):
        ax.text(
            x = bar.get_x() + bar.get_width() / 2,
            y = y_value,
            s = f"{y_value:.2f}",
            va="bottom",
        )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "avg_kl_per_method.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

# 4: Box plot — distribution of kl_divergence for each method
def plot_kl_boxplot(df):
    """
    Shows consistency of structural faithfulness across papers.
    """
    methods = sorted(df["method"].unique())
    data_per_method = [df[df["method"] == m]["kl_divergence"].values
                       for m in methods]

    fig, ax = plt.subplots()

    ax.set_title("Distribution of KL Divergence Across Papers", pad=15)
    ax.set_xlabel("Summarization Method",labelpad=10)
    ax.set_ylabel("KL Divergence (lower = more faithful to paper structure)",labelpad=10)
    # ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([m.upper() for m in methods])

    boxplot = ax.boxplot(
        data_per_method,
        patch_artist=True,
        notch=False,
        widths=0.4
    )

    for patch, method in zip(boxplot["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "gray"))
        patch.set_alpha(0.7)


    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "kl_distribution_boxplot.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

def combined_metrics_plot(df, results_df):

    """ Shows entropy and KL divergence metrics side by side per method.
        and also we wanna answer that 'Are the differences reader sees in the plot statistically real?',
        by presenting the statistical test datain a table.
    """

    # method is the index, to change it to a regular column we should reset_index
    avg_entropy  = df.groupby("method")["entropy"].mean().reset_index()
    avg_kl  = df.groupby("method")["kl_divergence"].mean().reset_index()

    methods = [m.upper() for m in avg_entropy["method"]]
    entropy_values = avg_entropy["entropy"].values
    kl_values = avg_kl["kl_divergence"].values

    # Create the figure and axis
    fig, (ax_plot, ax_table) = plt.subplots(2,1,figsize=(8, 10))  # 2 row and 1 columns


    " First panel: plot"

    x = np.arange(len(methods))  # We divide a given interval equally between the number of methods. [0,1,2]
    width = 0.40  # how wide each bar is

    # first y-axis for entropy (left side)
    ax_plot.set_ylabel("Entropy (↑ higher is better)", color="orange", labelpad=10)
    ax_plot.tick_params(axis='y', labelcolor="orange")

    # create second y-axis for KL divergence (right side)
    ax2 = ax_plot.twinx()
    ax2.set_ylabel("KL Divergence (↓ lower is better)", color="cornflowerblue", labelpad=10)
    ax2.tick_params(axis='y', labelcolor="cornflowerblue")

    # we need both bars for each method
    bar1= ax_plot.bar( x = x - width/2, height = entropy_values, width = width, color = "orange" ,label="Entropy"), # entropy bars — shifted LEFT of center
    bar2= ax2.bar( x = x + width/2, height = kl_values, width = width, color = "cornflowerblue" ,label="KL Divergence") # kl bars — shifted RIGHT of center

    ax_plot.set_xticks(x)  # where to put tick marks
    ax_plot.set_xticklabels(methods)  # replace the numeric value with actual names
    ax_plot.set_title("Entropy and KL Divergence by Summarization Methods", y=-0.25, pad=20)

    lines = [bar1, bar2]
    labels = ["Entropy (↑)", "KL Divergence (↓)"]
    ax_plot.legend(lines, labels, loc="upper center")  # legend box


    """ Second panel: table"""

    # function to convert p-value to stars automatically
    def get_stars(p_value):
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "n.s."

    data_table = []

    for index,row in results_df.iterrows():

        extracted_row = []
        methods = f"{row['method_a']} vs {row['method_b']}"

        extracted_row.append(methods)
        extracted_row.append(row["metric"])
        extracted_row.append(f"{row['p_value']:.4f}")
        extracted_row.append(get_stars(row["p_value"]))

        data_table.append(extracted_row)

    table = ax_table.table(
    colLabels = ["Comparison", "Metric", "p-value", "Significance"],
    cellText = data_table,
    cellLoc="center",
    loc ="center"
    )


    """ Customize table appearance"""

    table.scale(1, 1.5)
    for col in range(4):
        table[0, col].set_facecolor("#2C3E50")
        table[0, col].set_text_props(color="white", fontweight="bold")

    # color significance column based on value
    for row in range(1, len(data_table) + 1):
        stars = data_table[row-1][3]
        if stars == "n.s.":
            table[row, 3].set_facecolor("#ffcccc")
        else:
            table[row, 3].set_facecolor("#ccffcc")

    ax_table.axis('off')
    ax_table.annotate("** p < 0.01, * p < 0.05, n.s. = not significant",
            xy=(0.5, 0.15),
            xycoords='axes fraction',
            ha='center',
            fontsize=9,
            style='italic')


    path = os.path.join(OUTPUT_DIR, "combined_metrics.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")



if __name__ == "__main__":
    df = load_results("results/evaluation.csv")
    results_df = pd.read_csv("results/statistical_tests.csv")

    print("Generating visualizations...")
    average_entropy_plot(df)
    entropy_distribution_plot(df)
    plot_average_kl(df)
    plot_kl_boxplot(df)
    combined_metrics_plot(df, results_df)