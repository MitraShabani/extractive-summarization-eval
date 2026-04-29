# Extractive Summarization Evaluation for Scientific Papers

## Research Question

"Does using domain-specific sentence-level embeddings (SciBERT) produce more balanced section coverage than lexical (TF-IDF) and general semantic (SBERT) methods in extractive summarization of scientific papers?"

## Methods Compared
| Method  | Type     | Domain-specific |
|---------|----------|-----------------|
| TF-IDF  | Lexical  | No              |
| SBERT   | Semantic | No              |
| SciBERT | Semantic | Yes             |

## Evaluation Metrics
  Standard ROUGE metrics were intentionally excluded — they measure
  similarity to the abstract, which introduces bias toward introductory
  content and ignores structural faithfulness.
  Two established metrics were selected instead:

- **Entropy** — measures balance of section coverage
- **KL Divergence** — measures how faithfully the summary mirrors
  the paper's own section proportions. Ignored sections are penalized proportionally to their importance in the original paper.

## Key Design Decisions

**SciBERT over SPECTER** :
SPECTER is trained for document-level embeddings. Since our pipeline
operates at sentence level, SciBERT is more architecturally appropriate.

**SBERT as third method** :
SBERT provides a middle ground semantic but not domain-specific.
This isolates two separate effects:
- Does semantic understanding help over pure frequency? (TF-IDF vs SBERT)
- Does domain-specificity add value beyond general semantics? (SBERT vs   SciBERT)


**arXiv API over manual collection** :
Automated collection ensures reproducibility. Single-field papers
(CS/NLP) were chosen for consistent section naming conventions.

**KL Divergence over Entropy alone** :
Entropy treats all sections equally, a known limitation. KL Divergence
accounts for section importance by comparing against the paper's own
proportional structure.

## How to Run

1. `python download_papers.py`   - collect 50 PDFs from arXiv
2. `python evaluate.py`          - compute entropy and KL divergence
3. `python statistical-testing.py`        - run Wilcoxon significance tests
4. `python visualize.py`        - generate figures

## Results Summary

| Metric | Best Method | Significant? |
|--------|-------------|--------------|
| Entropy | TF-IDF | Yes (p < 0.05) |
| KL Divergence | SciBERT | Yes (p < 0.01) |

**Key finding:** TF-IDF achieves higher entropy but worse KL divergence.
It appears balanced while failing to respect the paper's
actual structure.

SciBERT significantly outperforms both methods on
structural faithfulness.

**Notable observation:** Entropy and KL divergence tell opposite stories
in our results. This contradiction confirms that entropy alone is
insufficient for evaluating structural faithfulness in scientific paper
summarization, a method can appear balanced while systematically
ignoring important sections. KL divergence captures this distinction
where entropy cannot.

## Limitations
- **PDF parsing inconsistencies** : section headers may be incorrectly
  extracted from some papers, affecting entropy and KL measurements
- **Single field bias** : papers collected from CS/NLP only;
  results may not generalize to other scientific domains
- **KL divergence penalization** : standard KL divergence is undefined
  when a section is completely ignored by the summarizer (i.e. when
  actual = 0, since log(0) is undefined). To handle this, ignored sections
  receive an explicit penalty proportional to their weight in the original
  paper:

  Covered sections (actual > 0) :
    KL += actual × log(actual / expected)

  Ignored sections (actual = 0) :
    KL += expected


      For example

      expected = 0.40  (Methods section contains 40% of paper)
      actual   = 0.00  (summarizer selected 0 sentences from Methods)
      Standard KL term: undefined (log(0/0.40) = log(0) = undefined)
      Applied penalty:  KL += 0.40

      Compare this to a minor section being ignored:

      expected = 0.02  (Acknowledgements section contains 2% of paper)
      actual   = 0.00
      Applied penalty: KL += 0.02

      Ignoring Methods contributes 20× more penalty than ignoring
      Acknowledgements — reflecting the structural importance of each section.

- **Generalizability** : 50 papers may not be sufficient for
  strong generalization claims

## Project Structure
```
project/
├── download_papers.py     # arXiv API paper collection
├── evaluate.py            # runs all methods, computes metrics
├── statistical-testing.py # run Wilcoxon significance tests
├── visualize.py           # generates academic figures
├── pdf_parser/
│   ├── blocks.py
│   ├── formula.py
│   ├── headers.py
│   └── sentences.py
├── summarizers/
│   ├── tfidf.py
│   ├── sbert.py
│   └── scibert.py
├── data/                  # downloaded PDFs
└── results/               # CSV results and figures
```
