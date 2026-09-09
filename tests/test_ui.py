"""Playwright validation of the sovren-ai-gpu-server dashboard.

Requires a live server (`./run.sh`, or uvicorn directly) on the port below.
Skips gracefully if nothing is listening there rather than failing the
whole suite -- this is an integration test against a running process, not
a unit test.

The previous version of this file asserted against a tab-based UI
(`.tab-btn`, `#panel-models`, etc.) that no longer exists after the
2026-09 rebuild to a single-page, status-first layout -- replaced rather
than patched, since every selector it checked was gone.
"""
from __future__ import annotations

import socket

import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8082"


def _server_is_up(url: str) -> bool:
    host = url.split("://")[1].split(":")[0]
    port = int(url.rsplit(":", 1)[1])
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


requires_live_server = pytest.mark.skipif(
    not _server_is_up(BASE_URL),
    reason=f"no server listening on {BASE_URL} (start with ./run.sh first)",
)


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1400, "height": 1600})
        yield pg
        browser.close()


@requires_live_server
def test_api_endpoints_respond(page):
    for endpoint in ["/health", "/api/gpus", "/api/metrics/summary", "/api/patterns",
                      "/api/ollama/services", "/api/health_status", "/api/load_cycles",
                      "/api/gpu_hardware", "/api/connections", "/api/task_samples"]:
        resp = page.request.get(BASE_URL + endpoint)
        assert resp.ok, f"{endpoint} returned {resp.status}"
        resp.json()  # must be valid JSON, not an error page


@requires_live_server
def test_dashboard_loads_without_errors(page):
    console_errors = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: console_errors.append(str(exc)))

    page.goto(BASE_URL, wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(2500)

    assert not console_errors, f"console errors: {console_errors}"
    assert page.title() == "Sovren — GPU Ollama Monitor"


@requires_live_server
def test_hero_status_card_renders(page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(2500)
    hero = page.locator("#hero")
    assert hero.locator(".hero-svc").count() >= 1
    text = hero.inner_text()
    assert any(s in text for s in ["STABLE", "IDLE", "RELOADING", "DEGRADED", "disabled"])


@requires_live_server
def test_load_cycle_timeline_renders(page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(2500)
    assert page.locator("#timeline").inner_text().strip() != ""


@requires_live_server
def test_gpu_hardware_table_renders(page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(2500)
    assert page.locator("#hwTable tbody tr").count() >= 1


@requires_live_server
def test_connections_chart_attached(page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(2500)
    assert page.locator("#connChart").count() == 1
    assert page.evaluate("() => window.Chart !== undefined")


@requires_live_server
def test_no_broken_data_binding_leaks_into_ui(page):
    """Regression guard for the class of bug this whole rebuild exists to
    fix: a field that's silently null/undefined rendering as literal text
    instead of a clean placeholder."""
    page.goto(BASE_URL, wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(2500)
    body_text = page.locator("body").inner_text()
    assert "undefined" not in body_text
    assert "[object Object]" not in body_text
    assert "NaN" not in body_text
