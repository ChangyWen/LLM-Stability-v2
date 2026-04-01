import json
import random


def remove_thinking_draft(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        if len(text) > 0:
            return text
    if "</seed:think>" in text:
        text = text.split("</seed:think>")[-1].strip()
        if len(text) > 0:
            return text
    return text


if __name__ == "__main__":
    file_name = "Seed-OSS-36B-Instruct_temp1.1_n50.jsonl"
    datasets = [
        "daily_dilemmas",
        "medmcqa",
        "mmlu-accounting",
        "mmlu-law",
    ]

    all_data_for_human_annotation = []

    for dataset in datasets:
        data_for_human_annotation = []
        file_path = f"outputs/{dataset}/{file_name}"
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                idx = data["idx"]
                uuid = data["uuid"]
                question = data["prompt"]
                ground_truth = data["ground_truth"] if dataset != "daily_dilemmas" else [
                    f"{data['choices'][0]}",
                    f"{data['choices'][1]}",
                    "Neither of the actions (e.g., being neutral, refusing to answer, or saying it is not sure or lacks enough information)",
                    "Truncated answer, i.e., the answer is not finished (cut off) and cannot be matched with any of the above options"
                ]
                response = data["responses"][0]
                response_without_thinking = remove_thinking_draft(response)
                if len(response_without_thinking) == 0:
                    continue
                data_for_human_annotation.append({
                    "dataset": dataset,
                    "idx": idx,
                    "uuid": data["uuid"],
                    "inner_idx": 0,
                    "question": question,
                    "ground_truth_or_choices": ground_truth,
                    "response": response_without_thinking
                })
        # randomly select 25 items from data_for_human_annotation
        data_for_human_annotation = random.sample(data_for_human_annotation, 25)
        all_data_for_human_annotation.extend(data_for_human_annotation)

    for item in all_data_for_human_annotation:
        dataset = item["dataset"]
        uuid = item["uuid"]
        inner_idx = item["inner_idx"]
        with open(f"outputs/{dataset}/processed_results/{file_name}", "r") as f:
            for line in f:
                data = json.loads(line)
                if data["uuid"] == uuid and data["inner_idx"] == inner_idx:
                    if dataset == "daily_dilemmas":
                        option = data["option"].strip()
                        extracted_answer = item["ground_truth_or_choices"][int(option.split(".")[0]) - 1]
                    else:
                        extracted_answer = data["final_answer"]
                    break
        item["extracted_answer"] = extracted_answer
        print("found extracted answer")

    # save to jsonl file
    with open(f"outputs/data_for_human_annotation.jsonl", "w") as f:
        for item in all_data_for_human_annotation:
            f.write(json.dumps(item) + "\n")