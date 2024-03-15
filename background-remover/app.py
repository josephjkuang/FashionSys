from rembg import remove
from PIL import Image

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

# Usage

image_path = "test_shorts.png"  # Replace with your image path
img_without_bg = remove_background(image_path)

if img_without_bg is not None:
    # Save the image with the transparent background
    img_without_bg.save("image_without_background.png", format="PNG")
    img_without_bg.show()
else:
    print("Error removing background. Check the image path and try again.")
