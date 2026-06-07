import re
with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

text_no_base64 = re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]*', '', text)
emojis = re.findall(r'[^\x00-\x7F]', text_no_base64)
unique_emojis = set(emojis)
print(f'Found non-ASCII symbols: {unique_emojis}')
