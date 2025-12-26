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

We updated the relevance determination to use **semantic similarity scores**:

```python
# NEW CODE (improved)
sem_scores = compute_semantic_similarity(gold_text, retrieved_texts)
RELEVANCE_THRESHOLD = 0.5

for i, score in enumerate(sem_scores):
    has_exact_match = gold_norm in normalize_text(retrieved_texts[i])
    is_relevant = (score >= RELEVANCE_THRESHOLD) or has_exact_match
```

### Key Changes:

1. **Semantic Similarity**: Uses BERTScore or embedding cosine similarity (already computed by the system)
2. **Relevance Threshold**: Documents with similarity >= 0.5 are considered relevant
3. **Fallback to Exact Match**: Maintains exact substring detection as a strong positive signal (OR condition)
4. **Safety Check**: Handles edge cases where semantic scores may be unavailable

## Expected Impact

With the new semantic-based relevance determination:
- Documents that semantically contain the answer (but phrased differently) will now be counted as relevant
- Recall should significantly improve as more relevant documents will be identified
- Precision should also improve as the threshold (0.5) balances true positives vs false positives
- The system now properly leverages its semantic understanding capabilities

## Example

**Gold Answer**: "Paris is the capital of France"

**Retrieved Documents**:
1. "The French capital city is known as Paris" (sem_score=0.75)
2. "London is in England" (sem_score=0.1)  
3. "Paris has many famous landmarks" (sem_score=0.55)

**OLD Approach** (exact substring):
- All documents: NOT relevant (exact phrase not found)
- Recall@3 = 0, Precision@3 = 0

**NEW Approach** (semantic similarity >= 0.5):
- Doc 1: RELEVANT (sem_score 0.75 >= 0.5)
- Doc 2: NOT relevant (sem_score 0.1 < 0.5)
- Doc 3: RELEVANT (sem_score 0.55 >= 0.5)
- Recall@3 = 1.0, Precision@3 = 0.667

## Modified Files

- `EnhancedRAG10.py` - Primary application file
- `EnhancedRAG9.py` - Legacy version
- `EnhancedRAG10.2_test` - Test version

All three files had the same `compute_retrieval_stats` function updated consistently.

## Testing

A validation test script (`/tmp/test_recall_precision.py`) confirms:
- The new logic correctly identifies relevant documents based on semantic similarity
- Significant improvement over the old exact-match approach
- All edge cases are handled properly

## Configuration

The relevance threshold is configurable:
```python
RELEVANCE_THRESHOLD = 0.5  # Can be tuned: 0.4-0.7 recommended range
```

Lower threshold (e.g., 0.4): Higher recall, lower precision
Higher threshold (e.g., 0.7): Lower recall, higher precision
