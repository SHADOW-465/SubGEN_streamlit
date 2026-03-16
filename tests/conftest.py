# tests/conftest.py
"""
Shared pytest fixtures.

Stubs Streamlit's runtime-only calls (set_page_config, markdown) so that
`subgen_ai.app` can be imported in unit tests without a live Streamlit server.
"""
import pytest
import unittest.mock as mock


@pytest.fixture(autouse=True)
def stub_streamlit_runtime(monkeypatch):
    """
    Prevent st.set_page_config / st.markdown from raising StreamlitAPIException
    when app.py is imported outside a running Streamlit server.
    """
    import streamlit as st
    monkeypatch.setattr(st, "set_page_config", lambda **kw: None)
    monkeypatch.setattr(st, "markdown",        lambda *a, **kw: None)
