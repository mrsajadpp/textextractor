import os
import random
from PIL import Image, ImageDraw, ImageFont
import zipfile
import nltk
nltk.download('words')
from nltk.corpus import words

# Configuration for data generation
output_dir = "newdata"
os.makedirs(output_dir, exist_ok=True)

# Excluded fonts (e.g., "ml", "fml")
excluded_fonts = ["ml", "fml"]

# Collect all fonts from system font directories
font_dirs = [
    "C:/Windows/Fonts",
    os.path.expanduser("~/.fonts")
]
all_fonts = []
for font_dir in font_dirs:
    for root, _, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".otf"):
                font_path = os.path.join(root, file)
                # Exclude problematic fonts
                if not any(excluded in file.lower() for excluded in excluded_fonts):
                    all_fonts.append(font_path)

# Ensure fonts are available
if not all_fonts:
    raise FileNotFoundError("No fonts found in the specified directories, or all were excluded.")

# Select a random subset of computer fonts
computer_fonts = random.sample(all_fonts, min(len(all_fonts), 20))

# Print the number of fonts selected for debugging
print(f"Using {len(computer_fonts)} computer fonts.")

# Sample text data
sample_texts = ["Dear","User,","Handwritten","uses","robotic","handwriting","machines","that","use","an","actual","pen","to","write","your","message.","The","results","are","virtually","indistinguishable","from","actual","handwriting.","Try","it","today!","The","Robot"]

# Function to generate images
def generate_images(texts, fonts, output_folder, prefix):
    for i, text in enumerate(texts):
        font_path = random.choice(fonts)
        try:
            font = ImageFont.truetype(font_path, 24)
            img = Image.new('L', (256, 64), color=255)  # Larger canvas for text
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), text, font=font, fill=0)  # Text in black
            img = img.resize((128, 32))  # Resize to standard dimensions
            img.save(os.path.join(output_folder, f"{prefix}_{i}.png"))
            with open(os.path.join(output_folder, f"{prefix}_{i}.txt"), "w") as f:
                f.write(text)
        except Exception as e:
            print(f"Skipping font {font_path} due to an error: {e}")

# Generate computer text data only
generate_images(sample_texts * 10, computer_fonts, output_dir, "computer")

# Create ZIP file
zip_path = "./text_images_large.zip"
with zipfile.ZipFile(zip_path, 'w') as zf:
    for file_name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, file_name)
        zf.write(full_path, arcname=os.path.relpath(full_path, output_dir))

print(f"Data generation complete. ZIP file created at: {zip_path}")
