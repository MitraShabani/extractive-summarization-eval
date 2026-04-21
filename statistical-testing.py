import pandas as pd
from scipy.stats import wilcoxon
import os

def run_statistical_tests(csv_path = "results/evaluation.csv"):

    """ We should check whether differences between summarization methods
    are statistically significant or not.
    I used WILCOXON because -data is paired( same paper, different methods)
                        and -our data doesn't follow normal distribution.
    """

    df = pd.read_csv(csv_path)
    results = []

    # Defining the pairs we want to compare and the metrics
    comparisons = [
        ("scibert", "tfidf"),
        ("scibert", "sbert"),
        ("sbert",   "tfidf")
    ]

    metrics = ["entropy", "kl_divergence"]

    """ We need every combination of metric and comparison.
    2 metrics and 3 comparisons """

    for metric in metrics:
        # in 'comparisons -> 'method-a: left column method-b: right column
        for method_a, method_b in comparisons:

            # one score per paper per method per metric
            method_a_scores = df[df["method"] == method_a].sort_values("paper")[metric].values
            method_b_scores  = df[df["method"] == method_b].sort_values("paper")[metric].values

            """ All the math happens here,
            if for instance for method_a = "scibert" & metric = "kl_divergence" we have:

            method_a_scores = [0.101, 0.115, 0.033, ...]  # scibert KL per paper
            method_b_scores = [0.435, 0.198, 0.456, ...]  # tfidf KL per paper

            wilcoxon calculates differences between method_a_scores[i] - method_b_scores[i] for each paper
            Then asks: "are these differences consistently pointing in the same direction across all 50 papers, or are they random?"
            """
            # run wilcoxon test
            stat, p_value = wilcoxon(method_a_scores, method_b_scores)

            """ p tells us "how likely would I be to see a difference this large just by chance?"
            """
            # determine significance
            if p_value < 0.01:
                significance = "strongly significant"  # Less than 1% chance this is random — very strong evidence
            elif p_value < 0.05:
                significance = "significant"  # Less than 5% chance — standard academic threshold
            elif p_value < 0.10:
                significance = "marginally significant"  # Less than 10% chance — weak, borderline evidence
            else:
                significance = "not significant"  # Too likely to be random — can't claim significance

            """ Determining the better method based on comparing the average entropy and KL values
            between each pair """
            mean_a = method_a_scores.mean()
            mean_b = method_b_scores.mean()

            if metric == "entropy":
                # higher entropy is better
                winner = method_a if mean_a > mean_b else method_b
            else:
                # lower KL divergence is better
                winner = method_a if mean_a < mean_b else method_b

            results.append({
                "metric":        metric,
                "method_a":      method_a,
                "method_b":      method_b,
                "mean_a":        round(mean_a, 4),
                "mean_b":        round(mean_b, 4),
                "statistic":     round(stat, 4),
                "p_value":       round(p_value, 4),
                "significance":  significance,
                "better_method": winner
            })

    return results

# save as csv file
if __name__ == "__main__":
    results = run_statistical_tests()
    df_results = pd.DataFrame(results)
    output_path = "results/statistical_tests.csv"

    if not os.path.exists("results/evaluation.csv"):
        print("Error: results/evaluation.csv not found.")
        print("Please run evaluate.py first.")
        exit()

    df_results.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")