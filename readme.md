# Search Engine Evaluation Project

This project explores and evaluates traditional and neural-based retrieval models for information retrieval tasks using a community-driven question-answering dataset. The main focus is on comparing the effectiveness of BM25, TF-IDF, and a hybrid approach that combines BM25 with Bi-Encoder embeddings.

## Project Overview

1. **Baseline Retrieval Models**  
   - Models: BM25 and TF-IDF  
   - Evaluation Metrics:  
     - Mean Average Precision (MAP)  
     - Normalized Discounted Cumulative Gain (NDCG)  
     - Precision at 10 (P@10)  

2. **Neural-Based Approach**  
   - Bi-Encoder embeddings generated using the SentenceTransformer model.  
   - Combined with BM25 using a weighted sum for hybrid retrieval.  

## Methodology

- **Data Preprocessing:** Text cleaning and tokenization of corpus and queries.  
- **Indexing:** Performed using PyTerrier's `IterDictIndexer`.  
- **Evaluation:** Conducted on validation and test datasets using the metrics listed above.  

## Results Summary

- **Traditional Models:** TF-IDF outperformed BM25 across all metrics.  
- **Hybrid Approach:** The combination of BM25 and Bi-Encoder embeddings demonstrated better performance compared to standalone BM25 or Bi-Encoder models.  

## How to Run

1. Ensure all dependencies, libraries are installed.  
2. Adjust the file paths in the code to match your local machine setup.  
   - Example: Update paths for datasets and output files in sections like `corpus_df`, `val_queries`, `test_queries`, etc.  
3. Run the scripts (BM25_TF-IDF.py and neural-links.py) to reproduce the experiments and view the results.  

## Note

- All file paths in the project are currently hardcoded and need to be modified for your local setup.
- Place datasets in the appropriate locations.

- Data for the project : 