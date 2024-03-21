from PIL import Image
from rembg import remove

import os

# removes background of image
def remove_background(image_path):
    try:
        # Load the image using PIL
        img = Image.open(image_path)

        # Remove background using rembg
        img_without_bg = remove(img)

        return img_without_bg

    except Exception as e:
        print(f"Error removing background: {e}")
        return None  # Return None in case of errors
    
input_path = "../test_images"
output_path = "../no_background_images"

# Loads in files and does whole process of removing background and saving image.
def remove_and_save(filename):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        print("Processing", filename)
        # Construct the full path to the image file
        image_path = os.path.join(input_path, filename)

        # Process the image
        img_without_bg = remove_background(image_path)
        
        # Check if the background removal was successful
        if img_without_bg is not None:
            # Save the image with the transparent background
            save_path = os.path.join(output_path, f"{filename}")
            img_without_bg.save(save_path, format="PNG")
        else:
            print(f"Error removing background for image: {filename}")

# Iterate over all files in the folder
for filename in os.listdir(input_path):
    remove_and_save(filename)
    