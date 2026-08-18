from __future__ import annotations

from ecommerce_rag.context_compaction import context_compaction_enabled


def test_canonical_compaction_setting_has_priority(monkeypatch):
    monkeypatch.setenv("ARAG_CONTEXT_COMPACTION", "1")
    monkeypatch.setenv("ERAG_CONTEXT_COMPACTION", "0")
    assert context_compaction_enabled(default=True) is False


def test_legacy_compaction_setting_is_backward_compatible(monkeypatch):
    monkeypatch.delenv("ERAG_CONTEXT_COMPACTION", raising=False)
    monkeypatch.setenv("ARAG_CONTEXT_COMPACTION", "on")
    assert context_compaction_enabled(default=False) is True


def test_compaction_setting_uses_caller_default_when_unset(monkeypatch):
    monkeypatch.delenv("ERAG_CONTEXT_COMPACTION", raising=False)
    monkeypatch.delenv("ARAG_CONTEXT_COMPACTION", raising=False)
    assert context_compaction_enabled(default=False) is False
    assert context_compaction_enabled(default=True) is True
