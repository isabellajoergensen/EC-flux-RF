"""Quick fix to replace remaining color assignments in Plots.py"""
import re

# Read the file
with open("Plots.py", "r", encoding="utf-8") as f:
    content = f.read()

# Pattern to find and replace
old_pattern = r"colors = plt\.cm\.tab20\(np\.linspace\(0, 1, (?:len\(all_cols\)|max\(1, len\(all_cols\)\))\)\)\n\tcolor_map = \{col: colors\[i\] for i, col in enumerate\(all_cols\)\}"
new_text = "color_map = _get_feature_color_map(all_cols)"

# Replace all occurrences
content_new = re.sub(old_pattern, new_text, content)

# Write back
with open("Plots.py", "w", encoding="utf-8") as f:
    f.write(content_new)

print("Color assignments fixed!")
print(f"Replaced {len(re.findall(old_pattern, content))} occurrences")
