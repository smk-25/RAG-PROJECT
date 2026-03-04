"""Unit tests for _unwrap_batch_result and related compliance-matrix helpers.

These tests are intentionally self-contained: they copy the minimal logic
from CombinedTenderAnalysis_V3.py so that the heavy Streamlit/ML dependencies
are not required when the CI runs pytest.
"""

# ---------------------------------------------------------------------------
# Inline copy of the functions under test (kept in sync with source)
# ---------------------------------------------------------------------------

_MODE_LIST_KEYS = {
    "Compliance Matrix": ["matrix", "requirements", "compliance_requirements", "items", "findings"],
    "Risk Assessment": ["risks", "risk_items", "risk_findings", "findings", "items"],
    "Ambiguity Scrutiny": ["ambiguities", "ambiguity_findings", "findings", "items"],
    "Entity Dashboard": ["entities", "findings", "items"],
    "Overall Summary & Voice": ["key_findings", "findings", "items"],
    "General Summary": ["key_findings", "findings", "items"],
}


def _unwrap_batch_result(b, mode: str) -> list:
    if b is None:
        return []
    if isinstance(b, list):
        return [item for item in b if isinstance(item, dict) and not item.get("error")]
    if isinstance(b, dict) and not b.get("error"):
        # Try mode-specific keys first, skipping empty / all-error lists
        for key in _MODE_LIST_KEYS.get(mode, []):
            if isinstance(b.get(key), list) and b[key]:
                items = [item for item in b[key] if isinstance(item, dict) and not item.get("error")]
                if items:
                    return items
        # Fall back to the first non-empty list-valued key found in the dict
        for k in b:
            if isinstance(b[k], list) and b[k]:
                items = [item for item in b[k] if isinstance(item, dict) and not item.get("error")]
                if items:
                    return items
        # Last resort: treat the whole dict as a single finding,
        # but ONLY when it has no list-valued keys.
        # A dict that still has list-valued keys (e.g. {"matrix": [], "requirements": []})
        # is a container response with empty lists — NOT a direct finding.  Returning
        # it would populate `mapped` with useless meta-dicts and prevent recovery.
        if b and not any(isinstance(v, list) for v in b.values()):
            return [b]
    return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

REQ = {"item": "Security Standards", "detail": "ISO 27001 required", "page": 3}
REQ2 = {"item": "Submission Deadline", "detail": "Submit by 30 April", "page": 5}


class TestUnwrapBatchResultComplianceMatrix:
    """Tests for Compliance Matrix mode."""

    def test_bare_list_returned_directly(self):
        result = _unwrap_batch_result([REQ], "Compliance Matrix")
        assert result == [REQ]

    def test_dict_with_matrix_key(self):
        b = {"matrix": [REQ]}
        assert _unwrap_batch_result(b, "Compliance Matrix") == [REQ]

    def test_dict_with_requirements_key(self):
        b = {"requirements": [REQ]}
        assert _unwrap_batch_result(b, "Compliance Matrix") == [REQ]

    def test_dict_with_compliance_requirements_key(self):
        b = {"compliance_requirements": [REQ]}
        assert _unwrap_batch_result(b, "Compliance Matrix") == [REQ]

    # --- Bug fix: empty earlier key should not block non-empty later key ---

    def test_empty_matrix_falls_through_to_requirements(self):
        """If 'matrix' is empty but 'requirements' has items, return requirements."""
        b = {"matrix": [], "requirements": [REQ]}
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert result == [REQ], (
            "_unwrap_batch_result must not short-circuit on an empty 'matrix' list"
        )

    def test_all_error_matrix_falls_through_to_items(self):
        """If 'matrix' contains only error dicts, fall through to 'items'."""
        b = {"matrix": [{"error": "API failure"}], "items": [REQ]}
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert result == [REQ]

    def test_empty_matrix_and_requirements_falls_through_to_items(self):
        """Both 'matrix' and 'requirements' empty → use 'items'."""
        b = {"matrix": [], "requirements": [], "items": [REQ]}
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert result == [REQ]

    def test_error_dict_returns_empty(self):
        b = {"error": "API failed"}
        assert _unwrap_batch_result(b, "Compliance Matrix") == []

    def test_none_returns_empty(self):
        assert _unwrap_batch_result(None, "Compliance Matrix") == []

    def test_empty_list_returns_empty(self):
        assert _unwrap_batch_result([], "Compliance Matrix") == []

    def test_items_filtered_for_error_dicts(self):
        b = [REQ, {"error": "bad"}, REQ2]
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert result == [REQ, REQ2]

    def test_last_resort_whole_dict_as_finding(self):
        """A valid dict with no list-valued keys is returned as a single-item list."""
        b = {"item": "Penalty Clause", "detail": "Late penalty 1%/day", "page": 7}
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert result == [b]

    def test_container_dict_with_all_empty_lists_returns_empty(self):
        """A container dict whose only values are empty lists must return [].

        Before the fix, {"matrix": [], "requirements": []} reached the last-resort
        and was returned as [b], which populated `mapped` with a useless meta-dict
        and prevented recovery extraction from triggering.
        """
        b = {"matrix": [], "requirements": []}
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert result == [], (
            "Container dicts with all-empty list values must not be treated as findings"
        )

    def test_container_dict_with_mixed_empty_list_and_scalar_returns_empty(self):
        """Container dict with empty list + scalar meta-fields returns []."""
        b = {"matrix": [], "total_requirements": 0, "summary": "No requirements found"}
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert result == []

    def test_finding_dict_with_no_list_values_is_returned(self):
        """A direct-finding dict that has no list values is returned via last resort."""
        b = {"item": "Bond Requirement", "detail": "Bid bond of 2% required", "page": 4}
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert result == [b]

    def test_multiple_items_in_matrix(self):
        b = {"matrix": [REQ, REQ2]}
        result = _unwrap_batch_result(b, "Compliance Matrix")
        assert len(result) == 2
        assert REQ in result and REQ2 in result


class TestUnwrapBatchResultOtherModes:
    """Smoke tests to ensure other modes are unaffected by the fix."""

    def test_risk_assessment_risks_key(self):
        risk = {"clause": "Late Delivery", "risk_level": "High", "page": 2}
        b = {"risks": [risk]}
        assert _unwrap_batch_result(b, "Risk Assessment") == [risk]

    def test_ambiguity_scrutiny_ambiguities_key(self):
        amb = {"ambiguous_text": "reasonable time", "severity": "Medium", "page": 4}
        b = {"ambiguities": [amb]}
        assert _unwrap_batch_result(b, "Ambiguity Scrutiny") == [amb]

    def test_risk_empty_risks_falls_through_to_items(self):
        risk = {"clause": "Penalty", "risk_level": "High", "page": 1}
        b = {"risks": [], "items": [risk]}
        assert _unwrap_batch_result(b, "Risk Assessment") == [risk]
