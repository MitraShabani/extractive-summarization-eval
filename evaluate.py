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
        row = {
            "paper": os.path.basename(pdf_path),
            "method": method_name,
            "section_count": len(coverage),
            "entropy": entropy,
        }

        results.append(row)
        df=pd.DataFrame(summary)
        df.to_csv(f"results/{method_name}.txt", index=False)
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