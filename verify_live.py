from playwright.sync_api import sync_playwright
import time

def verify_live_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Navigate to the app
        print("Navigating to app...")
        page.goto("http://localhost:8501")

        # Wait for sidebar to load
        page.wait_for_selector('section[data-testid="stSidebar"]', timeout=10000)

        # Click "Live Translation" in the radio button
        # Note: Streamlit radio buttons are often labels.
        print("Switching to Live Translation...")
        page.get_by_text("Live Translation").click()

        # Wait for the specific header of the Live Translation page
        page.wait_for_selector("text=Live Real-Time Translation", timeout=10000)

        # Wait for the WebRTC component to be visible (it might be an iframe or a div)
        # We can look for the "Start" button text usually associated with webrtc streamer
        # or the specific help text we added.
        page.wait_for_selector("text=Microphone Input", timeout=5000)

        # Take screenshot
        print("Taking screenshot...")
        page.screenshot(path="/home/jules/verification/live_translation_page.png")

        browser.close()

if __name__ == "__main__":
    verify_live_page()
