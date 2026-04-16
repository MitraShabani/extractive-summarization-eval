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
- **Entropy** — measures balance of section coverage

## Key Design Decisions
- Used SciBERT over SPECTER: SPECTER is document-level,
  our pipeline is sentence-level
- Used SBERT as middle ground: isolates domain-specificity
  effect from general semantics
- Used arXiv API for data collection: 50 papers,
  single field for structural consistency

## Project Structure
```
project/
├── download_papers.py   # arXiv API paper collection
├── evaluate.py          # runs all methods, computes metrics
├── visualize.py         # generates academic figures
├── pdf_parser/
│   ├── blocks.py
│   ├── formula.py
│   ├── headers.py
│   └── sentences.py
├── summarizers/
│   ├── tfidf.py
│   ├── sbert.py
│   └── scibert.py
├── data/                # downloaded PDFs
└── results/             # CSV results and figures
```