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
    file = os.path.join("/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/medmcqa", file)

    prompts = []
    questions = []
    original_responses = []
    partial_responses = []
    ground_truths = []
    partial_response_labels = []
    inner_idxs = []
    uuids = []
    idxs = []

    with open(file, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = item["idx"]
            uuid = item["uuid"]
            inner_idx = item["inner_idx"]
            question = item["question"]
            original_response = item["original_response"]
            current_partial_response_labels = ["step_1", "step_2", "step_3", "step_4"]
            ground_truth = item["ground_truth"]

            for partial_response_label in current_partial_response_labels:
                partial_response = item[partial_response_label]
                if partial_response is None:
                    continue

                prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
                prompt += f"<think>\n{partial_response}\n</think>\n\n"

                prompt_len = len(tokenizer.encode(prompt))
                if prompt_len > 10240:
                    continue

                prompts.append(prompt)
                questions.append(question)
                original_responses.append(original_response)
                partial_responses.append(partial_response)
                ground_truths.append(ground_truth)
                partial_response_labels.append(partial_response_label)
                inner_idxs.append(inner_idx)
                uuids.append(uuid)
                idxs.append(idx)

    return prompts, questions, original_responses, partial_responses, ground_truths, partial_response_labels, inner_idxs, uuids, idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, default="Qwen/Qwen3-4B")
    parser.add_argument("--tensor_parallel_size", type=int, required=False, default=8)
    parser.add_argument("--max_prompt_length", type=int, required=False, default=10240)
    parser.add_argument("--max_completion_length", type=int, required=False, default=8192)
    parser.add_argument("--temperature", type=float, required=False, default=0.6)
    parser.add_argument("--file_name", type=str, required=True)
    args = parser.parse_args()

    file_name = args.file_name + ".jsonl"
    all_files = os.listdir("/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/medmcqa")
    assert file_name in all_files

    save_dir = "/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/medmcqa/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, file_name.replace(".jsonl", "_completion.jsonl"))

    if os.path.exists(save_file):
        print(f"Results already exist in [{save_file}], skipping ...")
        exit()

    if "Qwen3-4B" in args.file_name:
        model_name = "Qwen/Qwen3-4B"
    elif "Qwen3-32B" in args.file_name:
        model_name = "Qwen/Qwen3-32B"
    elif "Qwen3-30B-A3B" in args.file_name:
        model_name = "Qwen/Qwen3-30B-A3B"
    elif "Seed-OSS-36B-Instruct" in args.file_name:
        model_name = "ByteDance-Seed/Seed-OSS-36B-Instruct"
    else:
        raise ValueError(f"Invalid file name: {args.file_name}")

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.8,
        max_model_len=args.max_prompt_length + args.max_completion_length,
        max_num_batched_tokens=args.max_prompt_length + args.max_completion_length,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    prompts, questions, original_responses, partial_responses, ground_truths, partial_response_labels, inner_idxs, uuids, idxs = load_dataset(file_name, llm.get_tokenizer())

    print(f"Processing [{model_name}] on [{file_name}] ({len(prompts)} prompts) ...")
    print(f"Results will be saved to [{save_file}] ...")
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_completion_length,
        logprobs=1,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    res = []
    for i in range(len(outputs)):
        assert len(outputs[i].outputs) == 1, f"Expected {1} outputs, but got {len(outputs[i].outputs)}"
        raw_output = outputs[i].outputs[0]
        text_output = raw_output.text.strip()
        token_ids = raw_output.token_ids
        raw_logprobs = raw_output.logprobs
        logprobs = []
        for token_id, logprob_obj in zip(token_ids, raw_logprobs):
            logprobs.append(logprob_obj[token_id].logprob)
        res.append({
            "idx": idxs[i],
            "uuid": uuids[i],
            "inner_idx": inner_idxs[i],
            "partial_response_label": partial_response_labels[i],
            "question": questions[i],
            "original_response": original_responses[i],
            "partial_response": partial_responses[i],
            "ground_truth": ground_truths[i],
            "response": text_output,
            "token_ids": token_ids,
            "logprobs": logprobs,
            "length": len(token_ids),
        })

    # save to jsonl file
    with open(save_file, "a") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")