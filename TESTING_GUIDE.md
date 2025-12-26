# Testing the Recall@K and Precision@K Fix

## How to Test the Fix

### 1. Run the Application

```bash
streamlit run EnhancedRAG10.py
```

### 2. Prepare Test Data

Create a CSV file with test questions and gold answers:

**Format**: `id,question,answer`

Example (`test_eval.csv`):
```csv
id,question,answer
1,What is the capital of France?,Paris is the capital of France
2,Who wrote Romeo and Juliet?,William Shakespeare wrote Romeo and Juliet
3,What is the speed of light?,The speed of light is approximately 299,792,458 meters per second
```

### 3. Upload Documents

1. Upload PDF files through the application UI
2. Click "Index / Re-index uploaded PDFs"
3. Wait for indexing to complete

### 4. Run Evaluation

1. In the sidebar, upload your test CSV file
2. Configure evaluation settings:
   - Set "Evaluate up to K" (e.g., 5)
   - Enable "Use BERTScore for semantic matching" (recommended)
   - Optionally enable "Run LLM to compute EM/F1" (if you have API key)
3. Click "Run Evaluation"

### 5. Check Results

The application will display:
- **Aggregated Retrieval Metrics**:
  - Recall@1, Recall@3, Recall@5
  - Precision@1, Precision@3, Precision@5
  - Best semantic scores

Expected improvements:
- ✅ Recall@3 should now be **> 0.5** (was 0.033)
- ✅ Precision@3 should now be **> 0.3** (was 0.022)
- ✅ Scores should reflect semantic understanding of documents

### 6. Download Results

Click "Download evaluation CSV" to get detailed per-question results including:
- Individual recall and precision scores
- Semantic similarity scores
- Retrieved documents for each question
- LLM-generated answers (if enabled)

## Understanding the Metrics

### Recall@K
**Definition**: Did we retrieve at least one relevant document in the top-K results?
- 1 = Yes, found at least one relevant doc
- 0 = No relevant docs in top-K

**Relevance Criteria** (NEW):
- Document has semantic similarity >= 0.5 with gold answer, OR
- Document contains the exact gold answer as substring

### Precision@K
**Definition**: What fraction of the top-K results are relevant?
- Formula: (# relevant docs in top-K) / K
- Range: 0.0 to 1.0

**Example**:
- Top-3 results: [relevant, not relevant, relevant]
- Precision@3 = 2/3 = 0.667

## Comparing Old vs New

### Old Behavior (Exact Substring Only)
```python
# Only matched if exact gold answer phrase appeared in retrieved text
is_relevant = "Paris is the capital of France" in retrieved_text
```
- Very strict
- Missed paraphrases
- Poor scores (~0.03)

### New Behavior (Semantic Similarity with Adaptive Thresholds)
```python
# Compute semantic scores
sem_scores, sem_method, _ = semantic_scores_for_retrieved(...)

# Adaptive threshold based on method
if sem_method == "bert_score_rescaled":
    RELEVANCE_THRESHOLD = 0.0  # Rescaled BERTScore centered around 0
else:
    RELEVANCE_THRESHOLD = 0.5  # Cosine similarity in [0,1]

# Matches if semantically similar OR contains exact phrase
is_relevant = (semantic_score >= RELEVANCE_THRESHOLD) or (gold_answer in retrieved_text)
```
- Captures semantic meaning with better discrimination
- Uses rescaled BERTScore for wider score range
- Handles paraphrases
- Expected scores vary by method (rescaled: -1.0 to 2.0, cosine: 0.0 to 1.0)

## Tuning the Threshold

The system uses adaptive thresholds by default, but you can manually adjust:

```python
# In compute_retrieval_stats function
# For rescaled BERTScore (default: 0.0)
if sem_method == "bert_score_rescaled":
    RELEVANCE_THRESHOLD = 0.0  # Adjust between -0.5 and 0.5
    
# For cosine similarity (default: 0.5)
else:
    RELEVANCE_THRESHOLD = 0.5  # Adjust between 0.4 and 0.7

# Lower threshold: More lenient, higher recall, lower precision
# Higher threshold: More strict, lower recall, higher precision
```

Recommended ranges:
- **Rescaled BERTScore**: -0.5 to 0.5 (default: 0.0)
- **Cosine similarity**: 0.4 to 0.7 (default: 0.5)

## Troubleshooting

### Issue: Scores still low
- **Check**: Are your test questions answerable from the uploaded documents?
- **Check**: Is the semantic similarity computation working? (Look for "bert_score_rescaled" or "embedding_cosine" in method column)
- **Try**: Lowering the threshold (to -0.2 for BERTScore, or 0.4 for cosine)

### Issue: Scores too high (suspicious)
- **Check**: Might indicate test questions are too easy or documents contain explicit answers
- **Try**: Raising the threshold (to 0.2 for BERTScore, or 0.6-0.7 for cosine)

### Issue: BERT score variation too small
- **Fixed**: Now using `rescale_with_baseline=True` which provides better discrimination
- Scores now range from negative to positive values instead of narrow 0.85-0.95 range
- If you still see issues, ensure bert-score package is properly installed

### Issue: BERTScore not available
- **Check**: Install bert-score: `pip install bert-score`
- **Fallback**: System will automatically use embedding cosine similarity

## Additional Notes

1. **BERTScore is preferred** over embedding cosine for better semantic understanding
2. **Larger K values** (e.g., K=10) give more opportunities for recall but lower precision
3. **Cross-encoder reranking** (if enabled) improves ranking but doesn't affect relevance determination
4. **Exact substring matching** is still checked as a strong positive signal

## Support

See `RECALL_PRECISION_FIX.md` for technical details and rationale behind the fix.
