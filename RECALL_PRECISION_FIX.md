# Recall@K and Precision@K Metrics Fix

## Problem Statement

The RAG system was producing very poor recall and precision scores:
- Recall@3: 0.033
- Recall@5: 0.033  
- Precision@1: 0.033
- Precision@3: 0.022
- Precision@5: 0.013

## Root Cause

The evaluation metrics were using **exact substring matching** to determine if a retrieved document was relevant:

```python
# OLD CODE (problematic)
is_relevant = gold_norm in normalize_text(retrieved_text)
```

This approach was too strict because:
1. Gold answers may be phrased differently than document text
2. No semantic understanding was applied
3. Documents semantically similar to the answer were marked as irrelevant

## Solution

We updated the relevance determination to use **semantic similarity scores with adaptive thresholds**:

```python
# NEW CODE (improved)
sem_scores, sem_method, _ = semantic_scores_for_retrieved(
    gold_text, retrieved_texts, embedding_manager, use_bertscore
)

# Adaptive threshold based on scoring method
if sem_method == "bert_score_rescaled":
    RELEVANCE_THRESHOLD = 0.0  # Rescaled scores centered around 0
else:
    RELEVANCE_THRESHOLD = 0.5  # Cosine similarity in [0,1]

for i, score in enumerate(sem_scores):
    has_exact_match = gold_norm in normalize_text(retrieved_texts[i])
    is_relevant = (score >= RELEVANCE_THRESHOLD) or has_exact_match
```

### Key Changes:

1. **Semantic Similarity**: Uses BERTScore with baseline rescaling or embedding cosine similarity
2. **BERTScore Improvement**: Now uses `rescale_with_baseline=True` for better discrimination between relevant/irrelevant documents
3. **Adaptive Threshold**: Threshold adjusts based on scoring method (0.0 for rescaled BERTScore, 0.5 for cosine)
4. **Fallback to Exact Match**: Maintains exact substring detection as a strong positive signal (OR condition)
5. **Safety Check**: Handles edge cases where semantic scores may be unavailable

## Expected Impact

With the new semantic-based relevance determination using rescaled BERTScore:
- Documents that semantically contain the answer (but phrased differently) will now be counted as relevant
- **Better discrimination**: Rescaled BERTScore provides wider score range, making it easier to distinguish relevant from irrelevant
- **More stable metrics**: Adaptive thresholds ensure consistent behavior across different scoring methods
- Recall should significantly improve as more relevant documents will be identified
- Precision improves due to better discrimination between relevant and irrelevant documents
- The system now properly leverages its semantic understanding capabilities

## Example

**Gold Answer**: "Paris is the capital of France"

**Retrieved Documents** (with rescaled BERTScore):
1. "The French capital city is known as Paris" (rescaled_score=0.85)
2. "London is in England" (rescaled_score=-0.45)  
3. "Paris has many famous landmarks" (rescaled_score=0.25)

**OLD Approach** (exact substring):
- All documents: NOT relevant (exact phrase not found)
- Recall@3 = 0, Precision@3 = 0

**NEW Approach** (rescaled BERTScore with threshold=0.0):
- Doc 1: RELEVANT (rescaled_score 0.85 >= 0.0)
- Doc 2: NOT relevant (rescaled_score -0.45 < 0.0)
- Doc 3: RELEVANT (rescaled_score 0.25 >= 0.0)
- Recall@3 = 1.0, Precision@3 = 0.667

## Modified Files

- `EnhancedRAG10.py` - Primary application file
- `EnhancedRAG9.py` - Legacy version
- `EnhancedRAG10.2_test` - Test version

All three files had the `semantic_scores_for_retrieved` and `compute_retrieval_stats` functions updated consistently.

## Recent Improvements (BERT Score Variation Fix)

**Issue**: BERT scores with `rescale_with_baseline=False` had very small variation (typically 0.85-0.95), making it hard to distinguish relevant from irrelevant documents.

**Solution**: Changed to `rescale_with_baseline=True` which:
- Provides better discrimination with wider score range (can be negative or >1.0)
- Centers scores around baseline (0.0), making threshold of 0.0 natural
- Removes artificial clamping to [0,1] that reduced discrimination

## Testing

A validation test script (`/tmp/test_recall_precision.py`) confirms:
- The new logic correctly identifies relevant documents based on semantic similarity
- Rescaled BERTScore provides better discrimination than raw scores
- Significant improvement over the old exact-match approach
- All edge cases are handled properly

## Configuration

The relevance threshold is now adaptive based on the scoring method:
```python
# Adaptive threshold
if sem_method == "bert_score_rescaled":
    RELEVANCE_THRESHOLD = 0.0  # Rescaled scores centered around 0
else:
    RELEVANCE_THRESHOLD = 0.5  # Cosine similarity in [0,1]
```

For manual tuning:
- **Rescaled BERTScore**: Typical range -1.0 to 2.0, default threshold 0.0
  - Lower threshold (e.g., -0.2): Higher recall, lower precision
  - Higher threshold (e.g., 0.2): Lower recall, higher precision
- **Cosine similarity**: Range 0.0 to 1.0, default threshold 0.5
  - Lower threshold (e.g., 0.4): Higher recall, lower precision
  - Higher threshold (e.g., 0.6): Lower recall, higher precision
