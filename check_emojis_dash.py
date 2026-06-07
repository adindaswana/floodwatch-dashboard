import re
with open('dashboard/dashboard.py', 'r', encoding='utf-8') as f:
    text = f.read()

emojis = re.findall(r'[^\x00-\x7F]', text)
unique_emojis = set(emojis)
print(f'Found non-ASCII symbols: {unique_emojis}')
