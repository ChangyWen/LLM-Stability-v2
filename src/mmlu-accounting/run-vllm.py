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
import torch


def get_prompt(raw_prompt, tokenizer, model_name, disable_thinking, thinking_budget):
    if not disable_thinking:
        if thinking_budget is None:
            if model_name.startswith("nvidia/NVIDIA-Nemotron"):
                prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": "/think"},
                    {"role": "user", "content": raw_prompt}
                ], tokenize=False, add_generation_prompt=True)
            else:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        else:
            assert isinstance(thinking_budget, int), f"thinking_budget must be an integer, but got {thinking_budget}"
            assert model_name == "ByteDance-Seed/Seed-OSS-36B-Instruct", f"thinking_budget is only supported for model: {model_name}"
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True, thinking_budget=thinking_budget)
    else:
        if model_name.startswith("Qwen/Qwen3") and ("2507" not in model_name) and ("Next" not in model_name):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        elif model_name == "ByteDance-Seed/Seed-OSS-36B-Instruct":
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True, thinking_budget=0)
        elif model_name.startswith("LGAI-EXAONE/EXAONE-4.0"):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        elif model_name.startswith("nvidia/NVIDIA-Nemotron"):
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": raw_prompt}
            ], tokenize=False, add_generation_prompt=True)
        else:
            assert False, f"Disable thinking is not supported for model: {model_name}"
    return prompt


def load_dataset(dataset_path, tokenizer, model_name, disable_thinking, thinking_budget):
    prompts = []
    raw_prompts = []
    ground_truths = []
    idxs = []
    total_count = 0

    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = str(item["idx"])
            original_question = item["paraphrases"][0]
            ground_truth = item["ground_truth"]
            prompt = get_prompt(original_question, tokenizer, model_name, disable_thinking, thinking_budget)
            prompts.append(prompt)
            raw_prompts.append(original_question)
            ground_truths.append(ground_truth)
            idxs.append(idx)

            total_count += 1
            if total_count >= 500:
                break

    return prompts, raw_prompts, ground_truths, idxs


def main(
    llm,
    n: int,
    temperature: float,
    model_name: str,
    max_completion_length: int,
    disable_thinking: bool,
    thinking_budget: int,
):
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_completion_length,
        logprobs=1,
    )
    tokenizer = llm.get_tokenizer()

    test_data_file = "/mnt/blob_output/v-dachengwen/LLM-Stability-v2/datasets/mmlu-accounting/mmlu-accounting_non_mcq.jsonl"
    os.makedirs("/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/mmlu-accounting/", exist_ok=True)
    _model_name = model_name.split("/")[-1]

    if not disable_thinking:
        save_file = f"/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/mmlu-accounting/{_model_name}_temp{str(temperature)}_n{str(n)}.jsonl"
    else:
        save_file = f"/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/mmlu-accounting/{_model_name}_temp{str(temperature)}_n{str(n)}_dt.jsonl"

    if isinstance(thinking_budget, int):
        assert not disable_thinking, f"thinking_budget is only supported when disable_thinking is False"
        assert thinking_budget > 0, f"thinking_budget must be greater than 0, but got {thinking_budget}"
        save_file = save_file.replace(".jsonl", f"_tb{str(thinking_budget)}.jsonl")

    prompts, raw_prompts, ground_truths, idxs = load_dataset(test_data_file, tokenizer, model_name, disable_thinking, thinking_budget)

    print(f"Processing [{model_name}] on [{test_data_file}] ({len(prompts)} prompts) ...")
    print(f"Results will be saved to [{save_file}] ...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    res = []
    for i in range(len(outputs)):
        all_responses = []
        all_logprobs = []
        all_token_ids = []

        assert len(outputs[i].outputs) == n, f"Expected {n} outputs, but got {len(outputs[i].outputs)}"
        for j in range(n):
            raw_output = outputs[i].outputs[j]
            response = raw_output.text.strip()
            token_ids = raw_output.token_ids
            raw_logprobs = raw_output.logprobs
            logprobs = []
            for token_id, logprob_obj in zip(token_ids, raw_logprobs):
                logprobs.append(logprob_obj[token_id].logprob)

            all_responses.append(response)
            all_logprobs.append(logprobs)
            all_token_ids.append(token_ids)

        res.append({
            "idx": idxs[i],
            "prompt": raw_prompts[i],
            "ground_truth": ground_truths[i],
            "responses": all_responses,
            "logprobs": all_logprobs,
            "token_ids": all_token_ids,
            "uuid": str(uuid4())
        })

    # save to jsonl file
    with open(save_file, "a") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, required=False, default=8)
    parser.add_argument("--max_prompt_length", type=int, required=False, default=512)
    parser.add_argument("--max_completion_length", type=int, required=False, default=8192)
    parser.add_argument("--temperature", type=float, required=False, default=0.6)
    parser.add_argument("--disable_thinking", action=argparse.BooleanOptionalAction, required=False, default=False)
    parser.add_argument("--thinking_budget", type=int, required=False, default=None)
    parser.add_argument("--repeated_times", type=int, required=False, default=50)
    args = parser.parse_args()

    # check how many GPUs are available
    print(f"Available GPUs: {torch.cuda.device_count()}")
    # set tensor parallel size to the number of available GPUs
    args.tensor_parallel_size = torch.cuda.device_count()
    print(f"Setting tensor parallel size to: {args.tensor_parallel_size}")

    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_prompt_length + args.max_completion_length,
        max_num_batched_tokens=args.max_prompt_length + args.max_completion_length,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    n = 1 if args.temperature == 0.0 else args.repeated_times
    main(
        llm,
        n,
        args.temperature,
        args.model_name,
        args.max_completion_length,
        args.disable_thinking,
        args.thinking_budget,
    )