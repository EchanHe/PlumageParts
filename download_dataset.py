import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from tqdm import tqdm
from PIL import Image
import io

# Parameters
CSV_PATH = "./PlumageParts_Macaulay_catalogue.csv"
OUTPUT_DIR = "./PlumageParts_Macaulay_images"
LOG_CSV_PATH = os.path.join(OUTPUT_DIR, "download_log.csv")
MAX_WORKERS = 10
RETRY_LIMIT = 3
REQUEST_TIMEOUT = 30
# PlumageParts masks were prepared for the 1200-pixel-long-edge image set.
IMAGE_SIZES = ["1200"]
PRINT_FALLBACK_ERRORS = True

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV with appropriate encoding
df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

# Fill missing sex with "Unknown" and drop duplicate photo entries
df["sex_bird_photo"] = df["sex_bird_photo"].fillna("Unknown")
image_data = df[
    [
        "sci_name",
        "sex_bird_photo",
        "macaulay_photo_catalog_id",
        "photo_width_px",
        "photo_height_px",
    ]
].drop_duplicates()

# Sanitize strings for file system
def sanitize(s):
    return s.replace(" ", "_").replace("/", "_")

def parse_pixel_value(value):
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

# Function to validate image and read dimensions
def get_image_size(img_data):
    try:
        img = Image.open(io.BytesIO(img_data))
        width, height = img.size
        img.verify()
        if width < 10 or height < 10:  # Check if image is too small
            return None
        return width, height
    except Exception:
        return None

def get_existing_image_size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None

def format_fallback_errors(errors):
    return " | ".join(f"{size}: {error}" for size, error in errors)

# Function to download a single image
def download_image(row):
    raw_sci_name = row["sci_name"]
    raw_sex = row["sex_bird_photo"]
    sci_name = sanitize(raw_sci_name)
    sex = sanitize(raw_sex)
    photo_id = str(row["macaulay_photo_catalog_id"])
    original_width = parse_pixel_value(row["photo_width_px"])
    original_height = parse_pixel_value(row["photo_height_px"])
    
    filename = f"{sci_name}_{sex}_{photo_id}.jpg"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        existing_size = get_existing_image_size(output_path)
        final_width, final_height = existing_size if existing_size else (None, None)
        return {
            "sci_name": raw_sci_name,
            "sex_bird_photo": raw_sex,
            "macaulay_photo_catalog_id": photo_id,
            "filename": filename,
            "status": "skipped_exists",
            "downloaded_size": "existing",
            "original_width": original_width,
            "original_height": original_height,
            "final_width": final_width,
            "final_height": final_height,
            "fallback_errors": None,
            "error": None,
        }

    last_error = None
    fallback_errors = []
    for image_size in IMAGE_SIZES:
        url = f"https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{photo_id}/{image_size}"
        image_size_error = None
        for attempt in range(RETRY_LIMIT):
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    image_dimensions = get_image_size(response.content)
                    if image_dimensions:
                        final_width, final_height = image_dimensions
                        downloaded_size = image_size
                        if (
                            original_width
                            and original_height
                            and final_width == original_width
                            and final_height == original_height
                        ):
                            downloaded_size = "original"
                        with open(output_path, "wb") as f:
                            f.write(response.content)
                        return {
                            "sci_name": raw_sci_name,
                            "sex_bird_photo": raw_sex,
                            "macaulay_photo_catalog_id": photo_id,
                            "filename": filename,
                            "status": "downloaded",
                            "downloaded_size": downloaded_size,
                            "original_width": original_width,
                            "original_height": original_height,
                            "final_width": final_width,
                            "final_height": final_height,
                            "fallback_errors": format_fallback_errors(fallback_errors),
                            "error": None,
                        }
                    raise Exception("invalid image data received")
                raise Exception(f"HTTP {response.status_code}")
            except Exception as e:
                last_error = e
                image_size_error = f"attempt {attempt + 1}/{RETRY_LIMIT}: {e}"
                if str(e).startswith(("HTTP 404", "HTTP 422")):
                    break
                if attempt < RETRY_LIMIT - 1:
                    sleep(1)

        fallback_errors.append((image_size, image_size_error or str(last_error)))
        if PRINT_FALLBACK_ERRORS:
            tqdm.write(f"Fallback for {filename}: size {image_size} failed ({fallback_errors[-1][1]})")

    return {
        "sci_name": raw_sci_name,
        "sex_bird_photo": raw_sex,
        "macaulay_photo_catalog_id": photo_id,
        "filename": filename,
        "status": "failed",
        "downloaded_size": None,
        "original_width": original_width,
        "original_height": original_height,
        "final_width": None,
        "final_height": None,
        "fallback_errors": format_fallback_errors(fallback_errors),
        "error": str(last_error),
    }

# Launch multi-threaded download with progress bar
results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_image, row) for _, row in image_data.iterrows()]
    
    # Create progress bar
    with tqdm(total=len(futures), desc="Downloading images") as progress_bar:
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress_bar.update(1)

# Print summary at the end
downloads = sum(1 for r in results if r["status"] == "downloaded")
skipped = sum(1 for r in results if r["status"] == "skipped_exists")
failed = sum(1 for r in results if r["status"] == "failed")

pd.DataFrame(results).to_csv(LOG_CSV_PATH, index=False)

print(f"\nSummary: {downloads} downloaded, {skipped} skipped, {failed} failed")
print(f"Download log saved to: {LOG_CSV_PATH}")
