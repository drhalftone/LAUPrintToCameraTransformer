"""Generate a clean side-by-side comparison figure for the paper."""

from PIL import Image
import os

# Load the results image
results_path = "../outputs_pix2pix_reverse/results_final.png"
img = Image.open(results_path)

width, height = img.size
print(f"Image size: {width}x{height}")

# The grid is 3 columns x 4 rows of 512x512 images
# Each image is 512x512, so total grid is 1536 x 2048
# Below that is the loss curve chart
cell_size = 512
col_width = cell_size
row_height = cell_size

print(f"Cell size: {col_width}x{row_height}")

# Pick the third row (index 2) - the dark cat image
row_idx = 2  # 0-indexed

# Extract the three images from that row
top = row_idx * row_height
bottom = top + row_height

print(f"Extracting row {row_idx}: y={top} to y={bottom}")

# Column positions
input_scanned = img.crop((0, top, col_width, bottom))
prediction_restored = img.crop((col_width, top, 2*col_width, bottom))
ground_truth_original = img.crop((2*col_width, top, 3*col_width, bottom))

# Create new image with white gaps
gap = 10  # white gap in pixels
new_width = 3 * col_width + 2 * gap
new_height = row_height

# Create white background
result = Image.new('RGB', (new_width, new_height), (255, 255, 255))

# Paste in order: Original | Scanned | Simulated (restored)
result.paste(ground_truth_original, (0, 0))
result.paste(input_scanned, (col_width + gap, 0))
result.paste(prediction_restored, (2 * col_width + 2 * gap, 0))

# Save
output_path = "comparison_figure.png"
result.save(output_path, quality=95)
print(f"Saved to {output_path}")

# Also create a version with labels
from PIL import ImageDraw, ImageFont

# Create larger canvas for labels
label_height = 30
labeled_result = Image.new('RGB', (new_width, new_height + label_height), (255, 255, 255))
labeled_result.paste(result, (0, label_height))

draw = ImageDraw.Draw(labeled_result)

# Try to use a reasonable font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

labels = ["(a) Original", "(b) Scanned", "(c) Restored"]
positions = [col_width // 2, col_width + gap + col_width // 2, 2 * col_width + 2 * gap + col_width // 2]

for label, x_pos in zip(labels, positions):
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text((x_pos - text_width // 2, 5), label, fill=(0, 0, 0), font=font)

labeled_result.save("comparison_figure_labeled.png", quality=95)
print("Saved labeled version to comparison_figure_labeled.png")
