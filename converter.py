import os
from PIL import Image
import pillow_heif

input_folder = "/Users/allikhankoshamet/Desktop/dl_project/final_test"
CREATE_JPG = True

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(input_folder, filename)

        heif_file = pillow_heif.open_heif(heic_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
        )

        base_name = os.path.splitext(filename)[0]

        if CREATE_JPG:
            jpg_path = os.path.join(input_folder, base_name + ".jpg")
            image.save(jpg_path, "JPEG", quality=90)
            print(f"Saved JPG: {jpg_path}")

print("DONE ✅")