#This python script analyzes files containing the results of the chatbot experiments. These results ar in json format.
#the fileds or interest are: "image_file_name" and "expert_reply_json", this last field contains a json with the expert's answers.
#The script counts the number of times each expert answer contains in its "family" field unique values, and group them by the "image_file_name" field.
#Only those "image_file_name" fields that contain the string ".jpg" are considered.
#The json files to be analyzed are in the same folder as this script and have the name "*2025_.json".
#The results are printed in the console and also saved in a csv file.
import os
import json
import csv
import glob

def analyze_logs():
    results = {}  # {image_file_name: {family: count}}

    # find all matching files in current folder
    json_files = glob.glob("*2025_log.json")

    for file_name in json_files:
        with open(file_name, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON file: {file_name}")
                continue

        # each file is a dict with timestamps as keys
        for _, record in data.items():
            image_name = record.get("image_file_name", "")
            if ".jpg" not in image_name:
                continue

            # clean up "uploaded_image: " prefix if present
            image_name = image_name.replace("uploaded_image: ", "").strip()

            expert_json_str = record.get("expert_reply_json")
            if not expert_json_str or not isinstance(expert_json_str, str):
                continue  # skip if missing or invalid

            try:
                expert_data = json.loads(expert_json_str)
            except json.JSONDecodeError:
                continue

            families = expert_data.get("family", [])
            if not isinstance(families, list):
                families = [families]

            if image_name not in results:
                results[image_name] = {}

            for fam in families:
                results[image_name][fam] = results[image_name].get(fam, 0) + 1

    # print results
    print("=== Results ===")
    for image, fam_counts in results.items():
        print(f"\nImage: {image}")
        for fam, count in fam_counts.items():
            print(f"  {fam}: {count}")

    # save to CSV
    csv_file = "chatbot_experiment_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image filename", "Family", "Count"])
        for image, fam_counts in results.items():
            for fam, count in fam_counts.items():
                writer.writerow([image, fam, count])

    print(f"\n✅ Results saved to {csv_file}")

if __name__ == "__main__":
    analyze_logs()
