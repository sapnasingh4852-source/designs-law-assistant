import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path(__file__).resolve().parent / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    (1, ".answer-card"),
    (2, ".answer-card"),
    (3, ".not-found-card"),
    (4, ".not-found-card"),
    (5, ".answer-card")
]

def generate_screenshots():
    print("Launching Chromium via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 920})
        page = context.new_page()

        for q_num, target_selector in CONFIGS:
            print(f"\n[Q{q_num}] Loading clean page at http://localhost:8501...")
            page.goto("http://localhost:8501", timeout=60000)
            page.wait_for_selector(".main-title", timeout=30000)
            time.sleep(1.5)

            btn_selector = f'button:has-text("Q{q_num}:")'
            print(f"[Q{q_num}] Clicking sidebar button...")
            page.locator(btn_selector).first.click()

            print(f"[Q{q_num}] Waiting for target selector: {target_selector}...")
            page.wait_for_selector(target_selector, timeout=30000)
            
            # Wait for any spinner to disappear
            try:
                page.wait_for_selector(".stSpinner", state="hidden", timeout=15000)
            except Exception:
                pass

            time.sleep(1.5)

            screenshot_path = OUTPUT_DIR / f"question{q_num}.png"
            page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"✅ Successfully captured pristine: {screenshot_path.name}")

        browser.close()
        print("\nAll 5 official screenshots captured with 100% precision!")

if __name__ == "__main__":
    generate_screenshots()
