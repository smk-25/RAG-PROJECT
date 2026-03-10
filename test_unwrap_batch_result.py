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

_MAX_FALLBACK_ITEM_LENGTH = 80


def _resolve_mandatory_val(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "yes", "mandatory", "must", "shall", "1", "required")
    if isinstance(val, (int, float)):
        return bool(val)
    return False


def _normalize_compliance_item(entry: dict):
    """Inline copy kept in sync with CombinedTenderAnalysis_V3.py."""
    if not isinstance(entry, dict) or "error" in entry:
        return None
    item_text = str(
        entry.get("item") or entry.get("requirement") or entry.get("compliance_item")
        or entry.get("name") or entry.get("clause") or entry.get("topic")
        or entry.get("requirement_name") or ""
    ).strip()
    detail_text = str(
        entry.get("detail") or entry.get("description") or entry.get("details")
        or entry.get("specifics") or entry.get("content") or ""
    ).strip()
    if not item_text and not detail_text:
        return None
    mandatory_raw = (
        entry.get("mandatory") if entry.get("mandatory") is not None
        else entry.get("type") or entry.get("requirement_type") or False
    )
    return {
        "item": item_text or detail_text[:_MAX_FALLBACK_ITEM_LENGTH],
        "detail": detail_text,
        "evidence": str(entry.get("evidence") or ""),
        "category": str(entry.get("category") or entry.get("type") or "General"),
        "mandatory": _resolve_mandatory_val(mandatory_raw),
        "pages": (
            entry.get("pages") if isinstance(entry.get("pages"), list)
            else ([entry["page"]] if entry.get("page") else [])
        ),
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


# ---------------------------------------------------------------------------
# Tests for _normalize_compliance_item (crash-fix validation)
# ---------------------------------------------------------------------------

class TestNormalizeComplianceItem:
    """Tests for _normalize_compliance_item, covering the non-string field crash."""

    def test_normal_string_fields(self):
        entry = {"item": "Bid Bond", "detail": "2% of contract value", "page": 3,
                 "category": "Financial", "mandatory": True, "evidence": "Bidder shall..."}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["item"] == "Bid Bond"
        assert result["detail"] == "2% of contract value"
        assert result["evidence"] == "Bidder shall..."
        assert result["category"] == "Financial"
        assert result["mandatory"] is True
        assert result["pages"] == [3]

    def test_integer_item_field_does_not_crash(self):
        """LLM returns 'item': 1 (integer) — must not raise AttributeError."""
        entry = {"item": 1, "detail": "Submit documents", "page": 4,
                 "mandatory": True, "evidence": "shall submit"}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["item"] == "1"

    def test_integer_detail_field_does_not_crash(self):
        """LLM returns 'detail': 42 — must not raise AttributeError."""
        entry = {"item": "Performance Bond", "detail": 42, "page": 2,
                 "mandatory": True, "evidence": "evidence text"}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["detail"] == "42"

    def test_integer_evidence_field_does_not_crash(self):
        """LLM returns 'evidence': 0 (falsy int) — must give empty string."""
        entry = {"item": "Technical Spec", "detail": "ISO 9001", "page": 5,
                 "mandatory": False, "evidence": 0}
        result = _normalize_compliance_item(entry)
        assert result is not None
        # str(0 or "") = str("") = "" because 0 is falsy
        assert result["evidence"] == ""

    def test_truthy_integer_evidence_field_does_not_crash(self):
        """LLM returns 'evidence': -1 (truthy int) — converted to string, no crash."""
        entry = {"item": "Technical Spec", "detail": "ISO 9001", "page": 5,
                 "mandatory": False, "evidence": -1}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["evidence"] == "-1"

    def test_boolean_false_evidence_gives_empty_string(self):
        """'evidence': False falls through the `or ""` chain → empty string."""
        entry = {"item": "Spec", "detail": "Some spec", "page": 1, "evidence": False}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["evidence"] == ""  # False is falsy, so str(False or "") = ""

    def test_integer_category_field_does_not_crash(self):
        """LLM returns 'category': 3 — must not raise AttributeError."""
        entry = {"item": "Financial Report", "detail": "Annual report required",
                 "page": 1, "mandatory": True, "evidence": "text", "category": 3}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["category"] == "3"

    def test_boolean_false_category_falls_through_to_general(self):
        """'category': False falls through the `or` chain → 'General'."""
        entry = {"item": "Spec", "detail": "Some spec", "page": 1,
                 "category": False, "mandatory": True}
        result = _normalize_compliance_item(entry)
        assert result is not None
        # False is falsy → falls through to "General"
        assert result["category"] == "General"

    def test_none_item_falls_back_to_detail(self):
        entry = {"item": None, "detail": "Provide security deposit", "page": 7,
                 "mandatory": True}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["item"] == "Provide security deposit"[:_MAX_FALLBACK_ITEM_LENGTH]

    def test_empty_item_and_detail_returns_none(self):
        entry = {"item": "", "detail": "", "page": 2, "mandatory": True}
        result = _normalize_compliance_item(entry)
        assert result is None

    def test_error_entry_returns_none(self):
        entry = {"error": "API failed", "item": "Something"}
        result = _normalize_compliance_item(entry)
        assert result is None

    def test_non_dict_returns_none(self):
        assert _normalize_compliance_item("not a dict") is None  # type: ignore
        assert _normalize_compliance_item(42) is None  # type: ignore
        assert _normalize_compliance_item(None) is None  # type: ignore

    def test_pages_from_singular_page(self):
        entry = {"item": "Deadline", "detail": "30 days", "page": 12}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["pages"] == [12]

    def test_pages_preserved_when_list(self):
        entry = {"item": "Deadline", "detail": "30 days", "pages": [3, 7, 12]}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["pages"] == [3, 7, 12]

    def test_alternative_key_requirement(self):
        entry = {"requirement": "ISO Certification", "description": "Must have ISO 9001",
                 "page": 1, "mandatory": True}
        result = _normalize_compliance_item(entry)
        assert result is not None
        assert result["item"] == "ISO Certification"
        assert result["detail"] == "Must have ISO 9001"


# ---------------------------------------------------------------------------
# Tests for evidence-recovery helper behaviour (safe str() coercion)
# ---------------------------------------------------------------------------

class TestEvidenceRecoveryCoercion:
    """Validate that building evidence lookup dicts tolerates non-string LLM values."""

    def _build_evidence_maps(self, mapped):
        """Reproduce the evidence-map-building logic from run_sum_v2."""
        mapped_evidence_by_item = {}
        mapped_evidence_by_detail = {}
        mapped_fallback_by_item = {}
        for m in mapped:
            if isinstance(m, dict):
                ev = m.get("evidence") or ""
                if ev:
                    if m.get("item"):
                        mapped_evidence_by_item[str(m["item"]).lower().strip()] = ev
                    if m.get("detail"):
                        dk = str(m["detail"]).lower().strip()[:80]
                        if dk:
                            mapped_evidence_by_detail[dk] = ev
                elif m.get("detail") and m.get("item"):
                    mapped_fallback_by_item[str(m["item"]).lower().strip()] = str(m["detail"])
        return mapped_evidence_by_item, mapped_evidence_by_detail, mapped_fallback_by_item

    def test_integer_item_does_not_crash(self):
        """m['item'] is an int: str() must be applied before .lower()."""
        mapped = [{"item": 1, "detail": "Submit by Friday", "evidence": "shall submit by Friday", "page": 2}]
        ev_by_item, _, _ = self._build_evidence_maps(mapped)
        assert "1" in ev_by_item

    def test_integer_detail_does_not_crash(self):
        """m['detail'] is an int: str() must be applied before .lower()."""
        mapped = [{"item": "Penalty", "detail": 500, "evidence": "penalty clause", "page": 3}]
        _, ev_by_detail, _ = self._build_evidence_maps(mapped)
        assert "500" in ev_by_detail

    def test_string_fields_work_normally(self):
        mapped = [{"item": "Bid Bond", "detail": "2% required", "evidence": "bidder shall provide", "page": 1}]
        ev_by_item, ev_by_detail, _ = self._build_evidence_maps(mapped)
        assert "bid bond" in ev_by_item
        assert "2% required" in ev_by_detail

    def test_entry_item_key_coercion(self):
        """entry.get('item') is an int in the matrix entry: str() must be used."""
        entry = {"item": 7, "detail": "Some requirement", "evidence": ""}
        # Reproduce the matrix-entry item_key derivation
        item_key = str(entry.get("item") or "").lower().strip()
        assert item_key == "7"

    def test_entry_detail_key_coercion(self):
        """entry.get('detail') is a float in the matrix entry: str() must be used."""
        entry = {"item": "Spec", "detail": 3.14, "evidence": ""}
        detail_key = str(entry.get("detail") or "").lower().strip()[:80]
        assert detail_key == "3.14"

