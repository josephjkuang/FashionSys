
import numpy as np
import requests
import webcolors

from io import BytesIO
from PIL import Image
from Pylette.src.color import Color
from Pylette.src.palette import Palette
from Pylette.src.utils import ColorBox
from sklearn.cluster import KMeans

def k_means_extraction(arr, height, width, palette_size):
    arr = np.reshape(arr, (width * height, -1))
    model = KMeans(n_clusters=palette_size, n_init="auto", init="k-means++")
    labels = model.fit_predict(arr)
    palette = np.array(model.cluster_centers_, dtype=int)
    color_count = np.bincount(labels)
    color_frequency = color_count / float(np.sum(color_count))
    colors = []
    for color, freq in zip(palette, color_frequency):
        # Check if the alpha channel is not small
        if color[3] > 20 and not is_white(color[0], color[1], color[2]):
            colors.append(Color(color, freq))
    return colors

def median_cut_extraction(arr, height, width, palette_size):
    arr = arr.reshape((width * height, -1))
    c = [ColorBox(arr)]
    full_box_size = c[0].size

    while len(c) < palette_size:
        largest_c_idx = np.argmax(c)
        c = c[:largest_c_idx] + c[largest_c_idx].split() + c[largest_c_idx + 1 :]

    colors = [Color(map(int, box.average), box.size / full_box_size) for box in c if np.any(box.average[3] != 0)]

    return colors

def is_white(r, g, b):
    white = (254, 254, 254)
    threshold = 10
    if (abs(r - white[0]) <= threshold and
        abs(g - white[1]) <= threshold and
        abs(b - white[2]) <= threshold):
        return True
    else:
        return False

def extract_colors(
    image=None,
    image_url: str = None,
    palette_size=5,
    resize=True,
    mode="KM",
    sort_mode=None,
):
    if image is None and image_url is None:
        raise ValueError("No image provided")

    if image is None and image_url is not None:
        response = requests.get(image_url)
        if response.status_code == 200 and "image" in response.headers.get(
            "Content-Type", ""
        ):
            img = Image.open(BytesIO(response.content)).convert("RGBA")
        else:
            raise ValueError("The URL did not point to a valid image.")
    else:
        img = Image.open(image).convert("RGBA")

    if resize:
        img = img.resize((256, 256))
    width, height = img.size
    arr = np.asarray(img)

    if mode == "KM":
        colors = k_means_extraction(arr, height, width, palette_size)
    elif mode == "MC":
        colors = median_cut_extraction(arr, height, width, palette_size)
    else:
        raise NotImplementedError("Extraction mode not implemented")

    if sort_mode == "luminance":
        colors.sort(key=lambda c: c.luminance, reverse=False)
    else:
        colors.sort(reverse=True)

    return Palette(colors)

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_color(image_path):
    palette = extract_colors(image=image_path, palette_size=3, resize=True)
    return palette[0].rgb[:3]