from main import detect_license_plate
from difflib import SequenceMatcher
import json

folder_path = "Images"

image_dict = json.load(open('solution.json'))

matching_ratio_average = 0.0
plate_not_found_counter = 0

print(f"\nTesting on average pictures from the {folder_path} folder.\n\n")
for filename, expected_text in image_dict.items():
    plate_text = detect_license_plate(f"{folder_path}/{filename}")
    if plate_text is None or plate_text == "No plate detected":
        print(f"Filename: {filename}\tLicense plate not found or can't read.")
        plate_not_found_counter += 1
        continue
    matching_ratio = SequenceMatcher(a=plate_text, b=expected_text).ratio()
    print(f"Filename: {filename}\tExpected output: {expected_text}\tCalculated output: {plate_text}\tMatching ratio: {round((matching_ratio * 100), 2)}%.")
    matching_ratio_average += matching_ratio

print(f"\nLength of dataset: {len(image_dict)}\tAverage matching ratio: {round((matching_ratio_average / len(image_dict) * 100), 2)}%.\n")
print(f"Plate not found or couldn't read: {plate_not_found_counter}\tAverage matching ratio with only found plates: {round((matching_ratio_average / (len(image_dict) - plate_not_found_counter) * 100), 2)}%.\n")
