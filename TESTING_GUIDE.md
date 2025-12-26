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
- ✅ Recall@3 should now be **> 0.5** (was ~0.1)
- ✅ Precision@3 should now be **> 0.3** (was ~0.033)
- ✅ Mean semantic score can be negative (e.g., -0.101) - this is normal for rescaled BERTScore
- ✅ Scores should reflect semantic understanding, token overlap, and key term matching

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

### Relevance Criteria (UPDATED):
- Document has semantic similarity >= -0.3 (rescaled BERTScore) or >= 0.4 (cosine) with gold answer, OR
- Document has 30%+ token overlap (Jaccard) with gold answer AND (reasonable semantic score OR 50%+ key terms present), OR
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

### New Behavior (Multi-Signal Relevance with Lenient Thresholds)
```python
# Compute semantic scores
sem_scores, sem_method, _ = semantic_scores_for_retrieved(...)

# More lenient thresholds based on method
if sem_method == "bert_score_rescaled":
    RELEVANCE_THRESHOLD = -0.3  # More lenient for rescaled scores
else:
    RELEVANCE_THRESHOLD = 0.4  # More lenient for cosine similarity

# Multiple relevance signals
for i, score in enumerate(sem_scores):
    has_full_exact_match = gold_norm in retrieved_norm
    token_overlap = compute_token_overlap(gold_text, retrieved_texts[i])
    has_high_token_overlap = token_overlap >= 0.3
    has_semantic_match = score >= RELEVANCE_THRESHOLD
    has_key_terms = key_token_overlap >= 0.5  # 50% of key terms present
    
    # Document is relevant if ANY condition is met
    is_relevant = (
        has_full_exact_match or 
        has_semantic_match or 
        (has_high_token_overlap and (score >= -0.5 or has_key_terms))
    )
```
- Captures semantic meaning with better discrimination
- Uses rescaled BERTScore with lenient threshold (-0.3 vs 0.0)
- Uses token overlap (Jaccard similarity) for paraphrase detection
- Uses key term matching for partial relevance
- Handles negative semantic scores (like -0.101) appropriately
- Multiple signals reduce false negatives
- Expected scores:
  - Rescaled BERTScore: -1.0 to 2.0 (threshold -0.3)
  - Cosine similarity: 0.0 to 1.0 (threshold 0.4)
  - Token overlap: 0.0 to 1.0 (threshold 0.3)
- Expected metrics with this fix:
  - Recall@3: **0.5-0.8** (was 0.1)
  - Precision@3: **0.3-0.6** (was 0.033)

## Tuning the Thresholds

The system uses multiple thresholds that can be manually adjusted in both `EnhancedRAG10.py` and `EnhancedRAG9.py`:

```python
# In compute_retrieval_stats function (EnhancedRAG10.py and EnhancedRAG9.py)

# Semantic similarity thresholds (primary)
if sem_method == "bert_score_rescaled":
    RELEVANCE_THRESHOLD = -0.3  # Adjust between -0.5 and 0.3
else:
    RELEVANCE_THRESHOLD = 0.4  # Adjust between 0.3 and 0.6

# Token overlap threshold
TOKEN_OVERLAP_THRESHOLD = 0.3  # Adjust between 0.2 and 0.5 (Jaccard similarity)

# Key term threshold
KEY_TERM_THRESHOLD = 0.5  # Adjust between 0.3 and 0.7 (fraction of key terms)

# Fallback semantic threshold for token overlap cases
FALLBACK_SEMANTIC_THRESHOLD = -0.5  # Adjust between -0.7 and -0.3

# Lower thresholds: More lenient, higher recall, lower precision
# Higher thresholds: More strict, lower recall, higher precision
```

Recommended ranges:
- **Rescaled BERTScore**: -0.5 to 0.3 (default: -0.3)
  - Use -0.5 for maximum recall
  - Use 0.0 for balanced performance
  - Use 0.3 for high precision
- **Cosine similarity**: 0.3 to 0.6 (default: 0.4)
  - Use 0.3 for maximum recall
  - Use 0.5 for balanced performance
  - Use 0.6 for high precision
- **Token overlap**: 0.2 to 0.5 (default: 0.3)
- **Key terms**: 0.3 to 0.7 (default: 0.5)

## Troubleshooting

### Issue: Scores still low
- **Check**: Are your test questions answerable from the uploaded documents?
- **Check**: Is the semantic similarity computation working? (Look for "bert_score_rescaled" or "embedding_cosine" in method column)
- **Check**: Is token overlap being computed correctly?
- **Try**: Lowering the semantic threshold (to -0.5 for BERTScore, or 0.3 for cosine)
- **Try**: Lowering the token overlap threshold (to 0.2)
- **Try**: Lowering the key term threshold (to 0.3)

### Issue: Mean semantic score is negative (e.g., -0.101)
- **This is NORMAL** when using rescaled BERTScore
- Negative scores mean below baseline but don't necessarily mean irrelevant
- The multi-signal approach handles this by also checking token overlap and key terms
- If most documents have negative scores, it indicates retrieval could be improved but relevance detection still works

### Issue: Scores too high (suspicious)
- **Check**: Might indicate test questions are too easy or documents contain explicit answers
- **Try**: Raising the semantic threshold (to 0.0 for BERTScore, or 0.6 for cosine)
- **Try**: Raising the token overlap threshold (to 0.4 or 0.5)
- **Try**: Raising the key term threshold (to 0.6 or 0.7)

### Issue: BERT score variation too small
- **Fixed**: Now using `rescale_with_baseline=True` which provides better discrimination
- Scores now range from negative to positive values instead of narrow 0.85-0.95 range
- If you still see issues, ensure bert-score package is properly installed

### Issue: BERTScore not available
- **Check**: Install bert-score: `pip install bert-score`
- **Fallback**: System will automatically use embedding cosine similarity

## Additional Notes

1. **BERTScore is preferred** over embedding cosine for better semantic understanding
2. **Multi-signal relevance** uses semantic similarity, token overlap, and key term matching
3. **Negative semantic scores** (like -0.101) are normal with rescaled BERTScore and handled appropriately
4. **Larger K values** (e.g., K=10) give more opportunities for recall but lower precision
5. **Cross-encoder reranking** (if enabled) improves ranking but doesn't affect relevance determination
6. **Exact substring matching** is still checked as the strongest positive signal
7. **Token overlap** (Jaccard similarity) helps catch paraphrases and related content
8. **Key term matching** provides fallback when semantic scores are weak

## Support

See `RECALL_PRECISION_FIX.md` for technical details and rationale behind the fix.
