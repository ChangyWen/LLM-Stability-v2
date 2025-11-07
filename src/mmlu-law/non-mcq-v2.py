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


def check_can_rewrite(question, correct_answer):
    prompt = f"""
Below is a multiple-choice question (within <question> </question> tags):
<question>
{question}
</question>

The correct answer to the above question is *{correct_answer}*.

Your task:
Evaluate whether this question meets both of the following two conditions.
1. Single definitive answer:
The question has one and only one correct answer (i.e., *{correct_answer}*) even when the answer is not limited to the options provided — that is, there is no ambiguity or reasonable alternative answer.
2. Option-independent answerability:
The question can be answered correctly without seeing the multiple-choice options, based on the information in the question alone.

If both conditions are true, put <answer> True </answer> at the end of your response.
Otherwise, put <answer> False </answer> at the end of your response.
    """.strip()
    can_rewrite = complete(prompt)
    can_rewrite = extract_from_tags(can_rewrite, "answer")
    return can_rewrite


def rewrite_question(question):
    prompt = f"""
Below is a multiple-choice question (within <question> </question> tags):
<question>
{question}
</question>

Your task:
Rewrite the given question so that it becomes a non-multiple-choice question — that is, ask for the same information or decision without including or referencing any answer options.

Requirements:
- Preserve the original meaning and intent of the question.
- Do not alter, paraphrase, or simplify any legal terminology (e.g., legal cases, legal rules, principles, or other legal concepts).
- Ensure the rewritten question remains grammatically complete and natural as a standalone question.
- Place the final rewritten version inside <rewritten-question> </rewritten-question> tags.
    """.strip()
    rewritten_question = complete(prompt)
    rewritten_question = extract_from_tags(rewritten_question, "rewritten-question")
    return rewritten_question


if __name__ == "__main__":
    total_count = int(sys.argv[1])
    index = int(sys.argv[2])
    save_file = f"datasets/mmlu-law/mmlu-law_non_mcq.jsonl"
    idx_done = []
    if not os.path.exists(save_file):
        with open(save_file, "w") as f:
            pass
    else:
        with open(save_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                idx = str(item["idx"])
                idx_done.append(idx)
    all_idx = set()
    line_count = 0
    with open(f"datasets/mmlu-law/mmlu-law.jsonl", "r") as f:
        for line in f:
            line_count += 1
            if line_count <= -1:
                continue
            item = json.loads(line.strip())
            idx = str(item["idx"])
            all_idx.add(idx)
    remaining_idx = all_idx - set(idx_done)
    remaining_idx = sorted(list(remaining_idx))
    remaining_idx = [remaining_idx[i::total_count] for i in range(total_count)]
    remaining_idx = remaining_idx[index]
    print(f"chunk {index} size: {len(remaining_idx)}")

    with open(f"datasets/mmlu-law/mmlu-law.jsonl", "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = str(item["idx"])
            if idx not in remaining_idx:
                continue
            question_context = item["question"]
            word_count = len(question_context.split())
            if word_count < 15:
                continue
            options = item["choices"]
            all_paraphrases = [
                f"""
{question_context}

Select the correct answer from the following choices: *{options[0]}*, *{options[1]}*, *{options[2]}*, *{options[3]}*.
                """.strip()
            ]
            ground_truth = options[item["answer"]]
            non_mcq_paraphrases = []
            paraphrase_0 = all_paraphrases[0]
            can_rewrite = check_can_rewrite(paraphrase_0, ground_truth)
            if isinstance(can_rewrite, str) and can_rewrite.upper() == "TRUE":
                for paraphrase in all_paraphrases:
                    rewritten_question = rewrite_question(paraphrase)
                    if isinstance(rewritten_question, str) and len(rewritten_question) > 0:
                        non_mcq_paraphrases.append(rewritten_question)
                        break
            else:
                print(f"[{idx}] {paraphrase_0} [cannot rewrite]")
            if len(non_mcq_paraphrases) <= 0:
                continue
            with open(save_file, "a") as f:
                f.write(json.dumps({
                    "idx": idx,
                    "question": item["question"],
                    "options": options,
                    "ground_truth": ground_truth,
                    "paraphrases": non_mcq_paraphrases
                }) + "\n")