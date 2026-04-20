import os
import math
import pandas as pd
from pdf_parser.blocks import extract_blocks
from pdf_parser.sentences import split_into_sentences
from summarizers.tfidf import summarize as summarize_tfidf
from summarizers.scibert import summarize as summarize_scibert
from summarizers.sbert import summarize_sbert


def section_coverage(summary):
    coverage = {}
    for item in summary:
        header = item["header"] or "No Header"
        coverage[header] = coverage.get(header, 0) + 1
    return coverage


def section_entropy(summary):
    coverage = {}
    # count sentences per section
    for item in summary:
        header = item["header"] or "No Header"
        coverage[header] = coverage.get(header, 0) + 1

    total = sum(coverage.values())
    num_sections = len(coverage)

    # calculate proportion for each section
    """ Entropy measures how spread out a distribution is """
    entropy = 0
    for count in coverage.values():
        p = count / total
        entropy -= p * math.log(p)  # -p*log(p) is the contribution of this section to the total entropy

    """because we're comparing papers with different structures,
        we should normalize the entropy.
        entropy is maximized when all sections are equally represented,
        we can normalize by the max possible entropy which is log(num_sections)"""
    # normalize by max possible entropy
    if num_sections > 1:
        entropy = entropy / math.log(num_sections)

    return entropy  # now always between 0 and 1

def kl_divergence(summary, data):
    """ Measures how much the summary's section distribution diverges
    from the original paper's section distribution.
    Lower KL divergence = better coverage of original structure
    """
    # step 1: count sentences per section in the ORIGINAL paper
    original_counts = {}
    for item in data:
        header = item["header"] or "No Header"
        original_counts[header] = original_counts.get(header, 0) + 1 # header is the key, count is the value

    # step 2: count sentences per section in the SUMMARY
    summary_counts = {}
    for item in summary:
        header = item["header"] or "No Header"
        summary_counts[header] = summary_counts.get(header, 0) + 1

    # step 3: compute KL divergence
    # KL(actual || expected) = sum of actual * log(actual / expected)
    kl_div = 0
    for header, original_count  in original_counts.items():

            """ Section in paper, ignored by summary -> actual =0 -> log(0) is undefined
            in this case we can act proportional to how important the section was in the paper
            a large section being ignored = larger penalty
            """
            # how much this section contributes to summary
            actual = summary_counts.get(header, 0) / sum(summary_counts.values())
            # how much this section occupies in the paper
            expected = original_count / sum(original_counts.values())

            if actual > 0 :
                kl_div += actual * math.log(actual / expected)
            else:
                kl_div += expected  # += expected for ignored sections

    return kl_div



def evaluate_methods(pdf_path):
    doc, pages = extract_blocks(pdf_path)
    data = split_into_sentences(pages)

    results = []
    for method_name, summarizer in [
        ("tfidf", summarize_tfidf),
        ("scibert", summarize_scibert),
        ("sbert", summarize_sbert)
    ]:
        summary = summarizer(data)
        coverage = section_coverage(summary)
        entropy = section_entropy(summary)
        kl = kl_divergence(summary, data)
        row = {
            "paper": os.path.basename(pdf_path),
            "method": method_name,
            "section_count": len(coverage),
            "entropy": entropy,
            "kl_divergence": kl
        }

        results.append(row)

    return results

if __name__ == "__main__":
    papers_dir = "data/"
    all_results = []

    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            path = os.path.join(papers_dir, filename) # full path
            all_results.extend(evaluate_methods(path))

    df = pd.DataFrame(all_results)
    df.to_csv("results/evaluation.csv", index=False)
    print("Saved to results/evaluation.csv")