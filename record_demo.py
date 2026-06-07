import time
import os
import shutil
from playwright.sync_api import sync_playwright

def record_demo():
    output_dir = "demo_videos"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    with sync_playwright() as p:
        # We need a large viewport to see the dashboard clearly
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            record_video_dir=output_dir,
            record_video_size={'width': 1280, 'height': 800}
        )
        
        page = context.new_page()
        print("Navigating to localhost...")
        page.goto("http://localhost:8501")
        
        # Wait for Streamlit to load completely
        page.wait_for_selector("text=SISTEM DETEKSI DINI BANJIR", timeout=30000)
        time.sleep(3)
        
        print("Demoing Tab 1...")
        # Scroll smoothly
        for _ in range(4):
            page.mouse.wheel(0, 600)
            time.sleep(1.5)
            
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.5)
        
        # Click on Tab 2: Performa ML
        print("Demoing Tab 2...")
        tabs = page.locator("button[role='tab']")
        if tabs.count() > 1:
            tabs.nth(1).click()
            time.sleep(3)
            # Scroll down to show all charts
            for _ in range(6):
                page.mouse.wheel(0, 500)
                time.sleep(1.5)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1.5)
        
        # Click on Tab 3: Prediksi Baru
        print("Demoing Tab 3...")
        if tabs.count() > 2:
            tabs.nth(2).click()
            time.sleep(2)
            
            # Form elements - wait and interact
            # The button to predict might have text "Lakukan Prediksi"
            predict_button = page.locator("button:has-text('Lakukan Prediksi')")
            if predict_button.count() > 0:
                predict_button.first.click()
                time.sleep(5)  # Wait for prediction to render
            
            page.mouse.wheel(0, 400)
            time.sleep(2)
            
        print("Closing browser to save video...")
        context.close()
        browser.close()
        
        # Rename the saved video
        videos = os.listdir(output_dir)
        for vid in videos:
            if vid.endswith(".webm"):
                os.rename(os.path.join(output_dir, vid), "demo_dashboard_floodwatch.webm")
                print("Video saved as demo_dashboard_floodwatch.webm")

if __name__ == "__main__":
    record_demo()
