import re

with open("src/dashboard/dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

emojis = [
    "🛰️", "⚙️", "🌍", "📤", "🌿", "⚠️", "📈", "🗺️", "🌤️", "🔄",
    "🌡️", "💧", "🌧️", "📍", "🔴", "🟡", "🟢", "🤖", "ℹ️", "📄",
    "✅", "📥", "📋", "⚪"
]

new_lines = []
for line in lines:
    # If it's the page_icon line, just skip it to remove it
    if "page_icon=" in line:
        continue
    
    # Strip emojis
    new_line = line
    for emoji in emojis:
        new_line = new_line.replace(emoji + " ", "")
        new_line = new_line.replace(emoji, "")
    
    new_lines.append(new_line)

with open("src/dashboard/dashboard.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Emojis stripped successfully.")
