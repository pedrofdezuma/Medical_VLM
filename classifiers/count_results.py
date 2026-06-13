import json
import os
from collections import defaultdict

def count_inferences():
    base_dir = "D:"

    actual_view_counts = defaultdict(int)
    classifier_view_counts = defaultdict(int)
    caption_view_counts = defaultdict(int)

    actual_sequence_counts = defaultdict(int)
    classifier_sequence_counts = defaultdict(int)
    caption_sequence_counts = defaultdict(int)

    total_images_analyzed = 0

    ignored_values = {"unknown", "non_brain_mri", "others"}

    for i in range(1, 10):
        filename_ext = f"inference_results_{i}.jsonl"
        filename_no_ext = f"inference_results_{i}"

        filepath = os.path.join(base_dir, filename_ext)
        if not os.path.exists(filepath):
            filepath = os.path.join(base_dir, filename_no_ext)
            if not os.path.exists(filepath):
                print(f"File not found: {filename_ext} or {filename_no_ext} in {base_dir}")
                continue

        print(f"Processing {filepath}...")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        total_images_analyzed += 1

                        # Extract all view and sequence types, handling potential null values
                        actual_view = (data.get("actual_view") or "").strip().lower()
                        classifier_view = (data.get("classifier_view") or "").strip().lower()
                        caption_view = (data.get("caption_view") or "").strip().lower()

                        actual_sequence = (data.get("actual_sequence") or "").strip().lower()
                        classifier_sequence = (data.get("classifier_sequence") or "").strip().lower()
                        caption_sequence = (data.get("caption_sequence") or "").strip().lower()

                        # Count occurrences if not in the ignored values list
                        if actual_view and actual_view not in ignored_values:
                            actual_view_counts[actual_view] += 1
                        if classifier_view and classifier_view not in ignored_values:
                            classifier_view_counts[classifier_view] += 1
                        if caption_view and caption_view not in ignored_values:
                            caption_view_counts[caption_view] += 1

                        if actual_sequence and actual_sequence not in ignored_values:
                            actual_sequence_counts[actual_sequence] += 1
                        if classifier_sequence and classifier_sequence not in ignored_values:
                            classifier_sequence_counts[classifier_sequence] += 1
                        if caption_sequence and caption_sequence not in ignored_values:
                            caption_sequence_counts[caption_sequence] += 1

                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in {filepath}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    print("-" * 30)
    print("RESULTS SUMMARY")
    print("-" * 30)
    print(f"Total images analyzed: {total_images_analyzed}")

    def print_counts(title, counts, expected_keys):
        print(title)
        for key in expected_keys:
            if key not in counts:
                counts[key] = 0
        for item, count in sorted(counts.items()):
            print(f"  - {item}: {count}")
        print()

    expected_views = ["axial", "sagittal", "coronal"]
    expected_sequences = ["t1-w", "t1-ce", "t2-w", "flair"]

    print_counts("Actual View Occurrences:", actual_view_counts, expected_views)
    print_counts("Classifier View Occurrences:", classifier_view_counts, expected_views)
    print_counts("Caption View Occurrences:", caption_view_counts, expected_views)

    print_counts("Actual Sequence Occurrences:", actual_sequence_counts, expected_sequences)
    print_counts("Classifier Sequence Occurrences:", classifier_sequence_counts, expected_sequences)
    print_counts("Caption Sequence Occurrences:", caption_sequence_counts, expected_sequences)

if __name__ == "__main__":
    count_inferences()
