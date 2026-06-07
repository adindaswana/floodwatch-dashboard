import re

APP_PATH = "dashboard/app.py"
with open(APP_PATH, "r") as f:
    content = f.read()

# Replace double quotes in img tags that are breaking f-strings
# Specifically the base64 src and the style
content = re.sub(r'<img src="data:image/png;base64,([^"]+)"', r"<img src='data:image/png;base64,\1'", content)
content = re.sub(r'style="([^"]+)">', r"style='\1'>", content)

with open(APP_PATH, "w") as f:
    f.write(content)

print("Fixed quotes.")
