"""Capture real screenshots of the running Streamlit app with headless Chromium.

Usage (the app must already be running on http://localhost:8501):
    pip install playwright
    playwright install chromium
    python tools/capture_screenshots.py ../screenshots

Optional: set CHROME_PATH to use an already-installed Chrome instead of the Playwright download.

Every image is a genuine screenshot of the live app - nothing is mocked.
If the Sentence-BERT model is available on your computer, the U3 screenshots show real
retrieval results; if not, they show the app's "model unavailable" message.
"""
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots")
URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8501"
OUT.mkdir(parents=True, exist_ok=True)


def wait_idle(page, pause=2.5, timeout=240):
    """Wait until Streamlit has finished running the script."""
    page.wait_for_timeout(int(pause * 1000))
    end = time.time() + timeout
    while time.time() < end:
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        page.wait_for_timeout(400)
    page.wait_for_timeout(1200)


def shot(page, name, height):
    page.set_viewport_size({"width": 1500, "height": height})
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / f"{name}.png"))
    print("saved", name)


def go(page, label):
    page.locator('[data-testid="stSidebar"]').get_by_text(label, exact=True).first.click()
    wait_idle(page)


def choose_sample(page, label):
    page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first.click()
    page.wait_for_timeout(700)
    page.locator('[role="option"]', has_text=label).first.click()
    page.wait_for_timeout(700)
    page.locator('[data-testid="stSidebar"]').get_by_role("button", name="Load sample").click()
    wait_idle(page, 4)


with sync_playwright() as p:
    kwargs = {"headless": True, "args": ["--no-sandbox"]}
    if os.environ.get("CHROME_PATH"):
        kwargs["executable_path"] = os.environ["CHROME_PATH"]
    browser = p.chromium.launch(**kwargs)
    page = browser.new_page(viewport={"width": 1500, "height": 1000})
    page.goto(URL)
    wait_idle(page, 5)

    shot(page, "01_dashboard_empty", 1000)
    go(page, "Project Input")
    shot(page, "02_project_input", 1300)

    for tag, label in [("healthcare", "Project A"), ("recruitment", "Project B")]:
        choose_sample(page, label)
        go(page, "Dashboard");                 shot(page, f"{tag}_01_dashboard", 1750)
        go(page, "Lifecycle Mapping (U1)");    shot(page, f"{tag}_02_lifecycle_U1", 1700)
        go(page, "Risk Register (U2)");        shot(page, f"{tag}_03_register_U2", 1300)
        go(page, "NIST Guidance (U3)");        wait_idle(page, 8); shot(page, f"{tag}_04_nist_guidance_U3", 1500)
        go(page, "Heatmap & Assessment (U4)"); shot(page, f"{tag}_05_heatmap_U4", 1900)
        go(page, "RMF Classification (U5)");   shot(page, f"{tag}_06_rmf_U5", 1900)

    browser.close()
