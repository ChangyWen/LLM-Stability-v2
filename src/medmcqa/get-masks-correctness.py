import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents import chat
import re
import random


def extract_from_tags(text, tag):
    if text is None:
        return None
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def complete(prompt, reasoning_effort_or_thinking_budget="high"):
    response = chat(
        max_retry=1,
        prompt=prompt,
        model_name="msra-gpt-5",
        enable_search=False,
        enable_thinking=True,
        reasoning_effort_or_thinking_budget=reasoning_effort_or_thinking_budget,
        temperature=1.0,
        top_p=1.0,
    )
    if response is None:
        return None
    return response["value"]


def get_correctness(question, answer, ground_truth):
    repeat_count = 0
    while True:
        if repeat_count >= 3:
            return None
        prompt = f"""
Below are a question.
**Question:**
{question}

The correct answer to the above question is *{ground_truth}*.

A model generated the following answer to the question: *{answer}*.

**Your task:**
Read the question carefully, understand the correct answer ({ground_truth}), and then evaluate whether the model's answer ({answer}) is correct.
* Guidelines:
1) The model's answer does not need to exactly match the wording of the correct answer — it only needs to be semantically equivalent (i.e., express the same meaning) in the context of the question.
2) Minor differences in phrasing, formatting, or unit representation are acceptable as long as the meaning remains consistent.
3) If the model's answer is correct, put <correctness> True </correctness> at the end of your response.
4) Otherwise, put <correctness> False </correctness> at the end of your response.
        """.strip()

        response = complete(prompt)
        correctness = extract_from_tags(response, "correctness")
        try:
            if correctness is None:
                repeat_count += 1
                continue
            correctness = correctness.strip().lower()
            if correctness == "true":
                return True
            elif correctness == "false":
                return False
            else:
                repeat_count += 1
                continue
        except Exception as e:
            print(f"Error: {e}")
            repeat_count += 1


if __name__ == "__main__":
    total_count = int(sys.argv[1])
    index = int(sys.argv[2])
    file_name = sys.argv[3]

    save_file = f"outputs/medmcqa/processed_results/{file_name}_correctness.jsonl"
    idx_uuid_done = []
    if not os.path.exists(save_file):
        with open(save_file, "w") as f:
            pass
    else:
        with open(save_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                idx = str(item["idx"])
                uuid = str(item["uuid"])
                partial_response_label = str(item["partial_response_label"])
                idx_uuid = idx + "_" + uuid + "_" + partial_response_label
                idx_uuid_done.append(idx_uuid)
    all_idx_uuid = set()
    with open(f"outputs/medmcqa/processed_results/{file_name}_counts.jsonl", "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = str(item["idx"])
            uuid = str(item["uuid"])
            partial_response_label = str(item["partial_response_label"])
            idx_uuid = idx + "_" + uuid + "_" + partial_response_label
            all_idx_uuid.add(idx_uuid)
    remaining_idx_uuid = all_idx_uuid - set(idx_uuid_done)
    remaining_idx_uuid = sorted(list(remaining_idx_uuid))
    remaining_idx_uuid = [remaining_idx_uuid[i::total_count] for i in range(total_count)]
    remaining_idx_uuid = remaining_idx_uuid[index]
    print(f"chunk {index} size: {len(remaining_idx_uuid)}")


    idx_to_ground_truth = {}
    with open(f"datasets/medmcqa/medmcqa_non_mcq.jsonl", "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = str(item["idx"])
            ground_truth = item["ground_truth"]
            idx_to_ground_truth[idx] = ground_truth


    with open(f"outputs/medmcqa/processed_results/{file_name}_counts.jsonl", "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = str(item["idx"])
            uuid = str(item["uuid"])
            question = item["question"]
            answer_counts = item["answer_counts"]
            partial_response_label = item["partial_response_label"]
            if idx + "_" + uuid + "_" + partial_response_label not in remaining_idx_uuid:
                continue
            ground_truth = idx_to_ground_truth[idx]
            correct_count = 0
            total_count = sum(answer_counts.values())
            skipped = False
            answer_to_correctness = {}
            for answer, count in answer_counts.items():
                correctness = get_correctness(question, answer, ground_truth)
                if correctness is None:
                    skipped = True
                    break
                if correctness:
                    correct_count += count
                answer_to_correctness[answer] = correctness
            if skipped:
                continue
            with open(save_file, "a") as f:
                f.write(json.dumps({
                    "idx": idx,
                    "uuid": uuid,
                    "partial_response_label": partial_response_label,
                    "correct_count": correct_count,
                    "total_count": total_count,
                    "answer_to_correctness": answer_to_correctness,
                }) + "\n")