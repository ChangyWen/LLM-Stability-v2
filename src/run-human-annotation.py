import json
import os


INPUT_PATH = "./data_for_human_annotation.jsonl"
OUTPUT_PATH = "./human_annotation_labels.jsonl"


def load_annotated_keys(output_path):
    """
    Load already-annotated items so the script can resume safely.
    Each key is identified by (dataset, idx, uuid, inner_idx).
    """
    annotated = set()

    if not os.path.exists(output_path):
        return annotated

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                key = (
                    data["dataset"],
                    data["idx"],
                    data["uuid"],
                    data["inner_idx"],
                )
                annotated.add(key)
            except Exception:
                continue

    return annotated


def ask_for_label():
    """
    Prompt the annotator for a label.
    Returns:
        1 or 0 for valid labels
        None if the user wants to quit
    """
    while True:
        user_input = input("\nLabel [1=correct, 0=incorrect, q=quit]: ").strip().lower()

        if user_input == "1":
            return 1
        elif user_input == "0":
            return 0
        elif user_input in {"q", "quit", "exit"}:
            return None
        else:
            print("Invalid input. Please enter 1, 0, or q.")


if __name__ == "__main__":
    annotated_keys = load_annotated_keys(OUTPUT_PATH)
    num_new_annotations = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        all_data = []
        remaining_count = 0

        for line in fin:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            all_data.append(data)

            key = (
                data["dataset"],
                data["idx"],
                data["uuid"],
                data["inner_idx"],
            )
            if key not in annotated_keys:
                remaining_count += 1

    print(f"Remaining labeling tasks: {remaining_count}")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
        for data in all_data:
            dataset = data["dataset"]
            idx = data["idx"]
            uuid = data["uuid"]
            inner_idx = data["inner_idx"]

            question = data["question"]
            response = data["response"]
            extracted_answer = data["extracted_answer"]

            key = (dataset, idx, uuid, inner_idx)
            if key in annotated_keys:
                continue

            print("\n" * 4)
            print("=" * 100)
            print(f"Dataset: {dataset}")
            print("-" * 100)
            print("Question:")
            print(question)
            print("-" * 100)
            print("Response:")
            print(response)
            print("-" * 100)
            print("Extracted answer:")
            print(extracted_answer)
            print("=" * 100)

            label = ask_for_label()
            if label is None:
                print(f"\nStopped by user. New annotations written: {num_new_annotations}")
                break

            annotation = {
                "dataset": dataset,
                "idx": idx,
                "uuid": uuid,
                "inner_idx": inner_idx,
                "label": label,
            }

            fout.write(json.dumps(annotation, ensure_ascii=False) + "\n")
            fout.flush()

            annotated_keys.add(key)
            num_new_annotations += 1

            print(f"Saved label={label}")

    print(f"Done. New annotations written: {num_new_annotations}")