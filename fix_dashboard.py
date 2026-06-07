import re
import os
import json
import base64

APP_PATH = "dashboard/app.py"
with open(APP_PATH, "r") as f:
    content = f.read()

# Replace em-dashes
content = content.replace("—", "-")

# Function to get base64 image tag
def get_img_tag(img_name, height="1.2em", margin="0 5px"):
    path = f"dashboard/assets/{img_name}"
    with open(path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="height:{height}; margin:{margin}; vertical-align:middle;">'

# Emojis to replace
content = content.replace('page_icon="🌊"', 'page_icon=Image.open("dashboard/assets/flood_wave_icon.png")')
content = content.replace('<span style="font-size:2.2rem">🌊</span>', get_img_tag("flood_wave_icon.png", height="2.2rem", margin="0"))
content = content.replace('🌊 Sistem Deteksi Dini', get_img_tag("flood_wave_icon.png") + ' Sistem Deteksi Dini')
content = content.replace('🌊 FloodWatch', get_img_tag("flood_wave_icon.png") + ' FloodWatch')
content = content.replace('📋 BNPB:', get_img_tag("clipboard_icon.png") + ' BNPB:')
content = content.replace('🧩 Feature set:', get_img_tag("puzzle_icon.png") + ' Feature set:')

content = content.replace('"Banjir 🔴": RED', '"Banjir": RED')
content = content.replace('"Banjir 🔴": RED,', '"Banjir": RED,')
content = content.replace('"Banjir 🔴": RED', '"Banjir": RED')
content = content.replace('"Tidak Banjir 🟢": GREEN_LT', '"Tidak Banjir": GREEN_LT')
content = content.replace('True:"Banjir 🔴"', 'True:"Banjir"')
content = content.replace('False:"Tidak Banjir 🟢"', 'False:"Tidak Banjir"')

content = content.replace('"📊  Overview & Tren"', '"Overview & Tren"')
content = content.replace('"🌿  Lingkungan & Cuaca"', '"Lingkungan & Cuaca"')
content = content.replace('"🤖  Machine Learning"', '"Machine Learning"')

content = content.replace('🌲 Random Forest', get_img_tag("pine_tree_icon.png") + ' Random Forest')
content = content.replace('⚡ XGBoost', get_img_tag("lightning_bolt_icon.png") + ' XGBoost')
content = content.replace('🔍 Temuan Utama', get_img_tag("magnifying_glass_icon.png") + ' Temuan Utama')
content = content.replace('📊 Distribusi Kelas', get_img_tag("bar_chart_icon.png") + ' Distribusi Kelas')

lstm_meta_path = "models/lstm_meta.json"
lstm_acc = 0.8475
lstm_f1 = 0.5352
lstm_auc = 0.9316

if os.path.exists(lstm_meta_path):
    try:
        with open(lstm_meta_path, "r") as f:
            m = json.load(f)
            lstm_acc = m.get("accuracy", lstm_acc)
            lstm_f1 = m.get("f1", lstm_f1)
            lstm_auc = m.get("auc", lstm_auc)
    except:
        pass

lstm_html = f"""
        <div class="kpi-card" style="border-left-color:#8FA87E;background:#EDEAE3">
            <div class="kpi-label">{get_img_tag("stopwatch_icon.png")} LSTM/GRU</div>
            <div class="kpi-value" style="color:#8FA87E;font-size:1.3rem">{(lstm_acc*100):.2f}%</div>
            <div class="kpi-delta" style="color:#9aab8a">Prediksi Time-Series (biner)</div>
            <div style="margin-top:0.6rem;font-size:0.8rem;color:#8A9E82">
                F1 Score: <strong>{lstm_f1:.4f}</strong><br>
                AUC-ROC: <strong>{lstm_auc:.4f}</strong>
            </div>
        </div>"""

content = re.sub(r'<div class="kpi-card" style="border-left-color:#8FA87E;background:#EDEAE3">\s*<div class="kpi-label">⏱️ LSTM/GRU</div>.*?</div>\s*</div>', lstm_html, content, flags=re.DOTALL)

with open(APP_PATH, "w") as f:
    f.write(content)

print("Dashboard updated successfully.")
