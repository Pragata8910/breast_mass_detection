# Please note that this is the final code for the project with capabilities of handling multiple masks for a single image and extracting the information required from proper csv files.
# The code is written in a way that it can be run in a single go and will process all the images and masks in the given csv files.
# The code will also log the processing of each image and mask in a log file and will also save the results in a csv file.
# The code will also print the total number of images processed and the number of images that were successfully processed.
# The code will also print the number of images that failed to process.
# The code will also print the location of the log file and the output directory where the processed images are saved.
# The code will also print the percentage of images processed after every 10 images processed.
# The code will also print the time of processing for each image and mask.
# The code will also print the error if any error occurs while processing the image and mask.

import cv2
import csv
import os
import pandas as pd
from pathlib import Path
from datetime import datetime


# Please note that this function requires full path for both image and ts masks
def full_image_processing(full_image_path, output_dir, UID, list_of_masks):
    try:
        print(f"Processing image {UID} with {len(list_of_masks)} masks")
        original_img = cv2.imread(full_image_path)
        if original_img is None:
            raise FileNotFoundError(f"Full image not found: {full_image_path}")

        # Check if we actually have masks
        if len(list_of_masks) == 0:
            print(f"Warning: No masks found for image {UID}")

        mask_applied = False  # Flag to track if any mask was successfully applied

        for i, each_mask in enumerate(list_of_masks):
            print(
                f"  Processing mask {i + 1}/{len(list_of_masks)}: {os.path.basename(each_mask)}"
            )
            mask_img = cv2.imread(each_mask)
            if mask_img is None:
                print(f"  Warning: Mask image not found: {each_mask}")
                continue

            # Log mask dimensions
            print(f"  Mask dimensions: {mask_img.shape}")

            mask_img = cv2.resize(
                mask_img,
                (original_img.shape[1], original_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

            # Check if mask has any non-zero pixels
            non_zero = cv2.countNonZero(gray_mask)
            print(f"  Mask has {non_zero} non-zero pixels")
            if non_zero == 0:
                print("  Warning: Mask is completely black")
                continue

            _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

            # Check binary mask
            binary_non_zero = cv2.countNonZero(binary_mask)
            print(f"  Binary mask has {binary_non_zero} non-zero pixels")

            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            print(f"  Found {len(contours)} contours")

            if len(contours) == 0:
                print(
                    f"  Warning: No contours found in mask {os.path.basename(each_mask)}"
                )
                continue

            contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(contour)
            print(f"  Largest contour area: {contour_area}")

            # Skip very small contours
            if contour_area < 100:  # Adjust this threshold as needed
                print(f"  Warning: Contour area too small: {contour_area}")
                continue

            x, y, w, h = cv2.boundingRect(contour)
            print(f"  Drawing rectangle at ({x}, {y}) with width {w} and height {h}")
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 15)
            mask_applied = True

        if not mask_applied:
            print(f"Warning: No masks were successfully applied to image {UID}")

        output_path = os.path.join(output_dir, f"{UID}.png")
        cv2.imwrite(output_path, original_img)
        print(f"Saved result to {output_path}")

        return True, None

    except Exception as e:
        import traceback

        print(f"Error processing {UID}: {str(e)}")
        print(traceback.format_exc())
        return False, str(e)


# Defining the main function, for processing all the parameter criteria and output paths of image processing function
def main():
    # Defining the output directory first
    output_dir = Path("/Users/pragata/Datas(all)/cbis ddsm/results/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionaries for the mapping
    # Processing the dicom file first
    full_ids = []
    mask_ids = {}

    # Then modify the CSV parsing section

    with open(
        "/Users/pragata/Datas(all)/cbis ddsm/csv/dicom_info.csv",
        mode="r",
        encoding="utf8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            patient_id = row["PatientID"]
            if "Mass" not in patient_id:
                continue
            if row["SeriesDescription"] == "full mammogram images":
                full_ids.append(patient_id)
            elif row["SeriesDescription"] == "ROI mask images":
                parts = patient_id.split("_")
                base_id = "_".join(parts[:-1])
                if base_id not in mask_ids:
                    mask_ids[base_id] = []
                mask_ids[base_id].append(patient_id)

    full_to_masks = {}
    for full_id in full_ids:
        if full_id in mask_ids:
            full_to_masks[full_id] = mask_ids[full_id]
        else:
            full_to_masks[full_id] = []
    # Defining the log file and results file
    log_file = output_dir / "processing_log.txt"
    results = []
    # Reading the master combined sheet
    df = pd.read_csv("/Users/pragata/Datas(all)/cbis ddsm/csv/combined_test_sheet.csv")

    for key, value in full_to_masks.items():
        if key in df["image_patient_id"].values:
            full_image_path = df.loc[
                df["image_patient_id"] == key, "image file path"
            ].values[0]
            uid = df.loc[df["image_patient_id"] == key, "UID"].values[0]
        else:
            continue
        list_of_masks = []
        for item in value:
            if item in df["ROI_patient_id"].values:
                mask_path = df.loc[
                    df["ROI_patient_id"] == item, "ROI mask file path"
                ].values[0]
                list_of_masks.append(mask_path)
            else:
                continue

        success, error = full_image_processing(
            full_image_path, output_dir, uid, list_of_masks
        )

        with open(log_file, "a") as f:
            current_time = datetime.now().time()
            if success:
                f.write(f"Succesfully processed image with UID:{uid}\n")
                f.write(f"Exact time of processing:{current_time}\n")
            else:
                f.write(f"Error occurred while processing image with UID:{uid}\n")
                f.write(f"Error: {error}\n")
                f.write(f"Time of error: {current_time}\n")

        results.append(
            {
                "success": success,
                "error": error if not success else None,
                "UID": uid if uid is not None else "N/A",
            }
        )

        if len(results) % 10 == 0:
            progress = ((len(results)) / 602) * 100
            print(f"Processed {progress:.2f}% images. Thank you for your patience\n")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "Processing results.csv", index=False)

    total = len(results)
    successful = sum(1 for r in results if r["success"])
    print("\nProcessing complete!")
    print(f"Successfully processed: {successful}/{total} images")
    print(f"Failed: {total - successful}/{total} images")
    print(f"\nCheck {log_file} for detailed processing log")
    print(f"Output images saved in: {output_dir}")


if __name__ == "__main__":
    main()
