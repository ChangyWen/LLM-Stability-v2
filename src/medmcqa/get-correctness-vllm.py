import os
import json
import argparse
import jsonlines
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from vllm import LLM, SamplingParams
from typing import Optional, List
import re
import time
from uuid import uuid4
import re
import sys


PROMPT = """
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


def extract_from_tags(text, tag):
    if text is None:
        return None
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


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


def load_dataset(file, tokenizer):
    prompts = []
    questions = []
    answers = []
    ground_truths = []
    uuids = []
    idxs = []

    idx_to_ground_truth = {}
    with open(f"/mnt/blob_output/v-dachengwen/LLM-Stability-v2/datasets/medmcqa/medmcqa_non_mcq.jsonl", "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = str(item["idx"])
            ground_truth = item["ground_truth"]
            idx_to_ground_truth[idx] = ground_truth

    file = os.path.join("/mnt/blob_output/v-dachengwen/LLM-Stability-v2/backup-from-sandbox-2026-0315-2324/outputs/medmcqa/processed_results", file)
    with open(file, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                idx = str(item["idx"])
                uuid = str(item["uuid"])
                question = item["question"]
                answer_counts = item["answer_counts"]
                ground_truth = idx_to_ground_truth[idx]

                for answer, _ in answer_counts.items():
                    questions.append(question)
                    answers.append(answer)
                    ground_truths.append(ground_truth)
                    uuids.append(uuid)
                    idxs.append(idx)

                    raw_prompt = PROMPT.format(question=question, answer=answer, ground_truth=ground_truth)
                    prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True)
                    prompt_len = len(tokenizer.encode(prompt))
                    if prompt_len > 10240:
                        continue
                    prompts.append(prompt)
            except Exception as e:
                continue

    return prompts, questions, answers, ground_truths, uuids, idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, default="Qwen/Qwen3-235B-A22B-Thinking-2507")
    parser.add_argument("--tensor_parallel_size", type=int, required=False, default=8)
    parser.add_argument("--max_prompt_length", type=int, required=False, default=10240)
    parser.add_argument("--max_completion_length", type=int, required=False, default=10240)
    parser.add_argument("--temperature", type=float, required=False, default=0.6)
    parser.add_argument("--file_name", type=str, required=False)
    args = parser.parse_args()

    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.8,
        max_model_len=args.max_prompt_length + args.max_completion_length,
        max_num_batched_tokens=args.max_prompt_length + args.max_completion_length,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_completion_length,
    )

    for file_name in [
        "Qwen3-4B_temp0.6_n50",
        "Qwen3-4B_temp0.6_n50_dt",
        "Qwen3-32B_temp0.6_n50",
        "Qwen3-32B_temp0.6_n50_dt",
        "Qwen3-30B-A3B_temp0.6_n50",
        "Qwen3-30B-A3B_temp0.6_n50_dt",
        "Seed-OSS-36B-Instruct_temp1.1_n50",
        "Seed-OSS-36B-Instruct_temp1.1_n50_dt",
        "NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50",
        "NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt",
        "NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50",
        "NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt",
    ]:
        file_name = file_name + "_counts.jsonl"
        all_files = os.listdir("/mnt/blob_output/v-dachengwen/LLM-Stability-v2/backup-from-sandbox-2026-0315-2324/outputs/medmcqa/processed_results")
        assert file_name in all_files

        save_dir = "/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/medmcqa/processed_results"
        save_file = os.path.join(save_dir, file_name.replace("_counts.jsonl", "_correctness-vllm-v2.jsonl"))

        if os.path.exists(save_file):
            print(f"Results already exist in [{save_file}], skipping ...")
            continue

        prompts, questions, answers, ground_truths, uuids, idxs = load_dataset(file_name, llm.get_tokenizer())

        print(f"Processing [{args.model_name}] on [{file_name}] ({len(prompts)} prompts) ...")
        print(f"Results will be saved to [{save_file}] ...")

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        res = []
        for i in range(len(outputs)):
            assert len(outputs[i].outputs) == 1, f"Expected {1} outputs, but got {len(outputs[i].outputs)}"
            response = outputs[i].outputs[0].text.strip()
            correctness = extract_from_tags(remove_thinking_draft(response), "correctness")
            res.append({
                "idx": idxs[i],
                "uuid": uuids[i],
                "question": questions[i],
                "answer": answers[i],
                "ground_truth": ground_truths[i],
                "correctness": correctness,
            })

        # save to jsonl file
        with open(save_file, "a") as f:
            for item in res:
                f.write(json.dumps(item) + "\n")
