# Confidence Score Method - Documentation

## Overview

The confidence score method in `Summarizationcode.py` provides a quantitative assessment of the quality and reliability of summarization results. It helps users understand how trustworthy the generated summaries are based on multiple factors.

## Confidence Score Components

The overall confidence score is a weighted average of four key metrics:

### 1. Snippet Coverage (30% weight)
**What it measures**: How many relevant snippets were found during the map phase.

- **Calculation**: `min(1.0, num_snippets / 5.0)`
- **Target**: 5 or more snippets indicates good coverage
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 100%: 5+ snippets found (excellent coverage)
  - 60%: 3 snippets found (moderate coverage)
  - 20%: 1 snippet found (limited coverage)
  - 0%: No snippets found

### 2. Result Coherence (30% weight)
**What it measures**: Internal consistency and structural completeness of the result.

- **Checks**: Presence of expected fields (summary, bullets, citations, matrix, risks, dashboard, queries)
- **Calculation**: `min(1.0, field_count / 2.0)`
- **Target**: 2+ fields present
- **Range**: 0.0 to 1.0 (or 0.1 if error present)
- **Interpretation**:
  - 100%: 2+ key fields present (well-structured)
  - 50%: 1 key field present (basic structure)
  - 10%: Error in result (failed generation)
  - 0%: No valid structure

### 3. Information Density (20% weight)
**What it measures**: Amount of information extracted relative to the query length.

- **Calculation**: Ratio of result tokens to query tokens
- **Optimal Range**: 10-20x longer than query
- **Range**: 0.0 to 1.0
- **Scoring**:
  - `< 2x`: Linear scaling (result too brief)
  - `2-50x`: Linear scaling toward optimal
  - `> 50x`: Penalty applied (result too verbose)
- **Interpretation**:
  - 100%: ~20x query length (comprehensive answer)
  - 50%: ~10x query length (adequate answer)
  - 25%: ~5x query length (brief answer)
  - 10%: Similar length to query (minimal extraction)

### 4. Citation Quality (20% weight)
**What it measures**: Completeness and quality of page citations.

- **Calculation**: `min(1.0, unique_citations / expected_citations)`
- **Expected Citations**: At least 1 citation per 3 snippets
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 100%: Citations meet or exceed expectations
  - 67%: 2 citations for 3 snippets
  - 33%: 1 citation for 3 snippets
  - 0%: No citations provided

## Overall Confidence Score

**Formula**: 
```
Overall = 0.30×Coverage + 0.30×Coherence + 0.20×Density + 0.20×Citations
```

**Interpretation Thresholds**:
- **≥70%**: **High Confidence** 🎯 (reliable result, green indicator)
- **40-69%**: **Moderate Confidence** ✓ (acceptable result, blue indicator)
- **<40%**: **Low Confidence** ⚠️ (limited information, yellow/orange indicator)

## Usage in Summarizationcode.py

### During Query Processing

The confidence score is computed after the map/reduce phases:

```python
# Compute confidence scores
confidence_scores = compute_confidence_score(mapped, reduced, query)

# Display confidence indicator
overall_conf = confidence_scores["overall_confidence"]
if overall_conf >= 0.7:
    st.success(f"🎯 High Confidence: {overall_conf:.2%} (reliable result)")
elif overall_conf >= 0.4:
    st.info(f"✓ Moderate Confidence: {overall_conf:.2%} (acceptable result)")
else:
    st.warning(f"⚠️ Low Confidence: {overall_conf:.2%} (limited information found)")
```

### In Exports

Confidence scores are included in:
1. **Text Export** (.txt): Confidence breakdown at the start of each query result
2. **JSON Export** (.json): `confidence_scores` field in each result object
3. **Excel Export** (.xlsx): Separate columns for each metric in the General_Summary sheet
4. **Evaluation CSV**: Overall confidence column when running benchmarks

## Example Scenarios

### Scenario 1: High Confidence (89.6%)
```
Input Query: "What are the tender requirements?"
Snippets Found: 5
Result: Comprehensive summary with bullets and citations

Metrics:
- Snippet Coverage: 100% (5 snippets)
- Result Coherence: 100% (summary + bullets + citations)
- Information Density: 48% (24x query length)
- Citation Quality: 100% (5 citations for 5 snippets)
Overall: 89.6% → High Confidence 🎯
```

### Scenario 2: Low Confidence (26%)
```
Input Query: "What are the specific technical requirements?"
Snippets Found: 1
Result: Brief summary with minimal citations

Metrics:
- Snippet Coverage: 20% (1 snippet)
- Result Coherence: 50% (summary only)
- Information Density: 25% (5x query length)
- Citation Quality: 0% (no citations)
Overall: 26% → Low Confidence ⚠️
```

### Scenario 3: No Information (0%)
```
Input Query: "What is the project timeline?"
Snippets Found: 0
Result: "No relevant info found."

All Metrics: 0%
Overall: 0% → Low Confidence ⚠️
```

## Tuning the Confidence Score

You can adjust the weighting or thresholds in the `compute_confidence_score()` function:

```python
# Current weights (line ~535 in Summarizationcode.py)
confidence["overall_confidence"] = (
    0.30 * confidence["snippet_coverage"] +      # Snippet importance
    0.30 * confidence["result_coherence"] +      # Structure importance
    0.20 * confidence["information_density"] +   # Content importance
    0.20 * confidence["citation_confidence"]     # Citation importance
)
```

**Recommendations**:
- Increase snippet coverage weight if retrieval quality is most important
- Increase coherence weight if structured output is critical
- Increase density weight if comprehensive answers are needed
- Increase citation weight if traceability is essential

## Benefits

1. **Transparency**: Users can see why a result has high or low confidence
2. **Quality Control**: Helps identify queries that need refinement
3. **Filtering**: Can filter/sort results by confidence in benchmark mode
4. **Decision Support**: Informs whether to trust the result or investigate further
5. **Diagnostics**: Individual metrics help debug issues (e.g., poor chunking, weak retrieval)

## Limitations

1. **Heuristic-based**: Not a guarantee of semantic accuracy
2. **Query-dependent**: Short queries naturally have higher density scores
3. **Mode-specific**: Some analysis modes don't produce all fields (affects coherence)
4. **No ground truth**: Cannot measure accuracy without reference answers

## Future Enhancements

Potential improvements to the confidence score:
- Add semantic similarity between query and result
- Factor in cross-encoder reranking scores if available
- Consider LLM token probabilities/perplexity
- Add domain-specific weights per analysis mode
- Machine learning calibration based on user feedback
