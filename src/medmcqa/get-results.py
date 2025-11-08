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
Below are a question (within <question> </question> tags) and a response to the question (within <response> </response> tags).
<question>
{question}
</question>

<response>
{response}
</response>

**Your Task:**
Read the response carefully and extract the final answer — the specific conclusion that directly answers the question. Follow these guidelines:
1. Ignore reasoning and explanations. Do not include rationales, justifications, or discussion. Extract only the final, conclusive answer.
2. Exclude irrelevant words. Remove hedging, filler, or meta-phrases (e.g., "I think", "it seems", "the answer should be", "in my opinion", "the answer is", etc.).
3. Be concise and literal. Output only the essential final answer (e.g., a term, number, or phrase).
*Output format:*
- If the response contains a valid final answer, put the extracted final answer within <final-answer> </final-answer> tags at the end of your response.
- If the response does not contain a valid final answer (e.g., expresses uncertainty, refusal, or lack of information), output "<final-answer> None </final-answer>".
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
    save_dir = "/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/medmcqa/processed_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = os.path.join("/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/medmcqa", file)

    prompts = []
    raw_prompts = []
    questions = []
    ground_truths = []
    inner_idxs = []
    uuids = []
    idxs = []
    with open(file, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = item["idx"]
            question = item["prompt"]
            uuid = item["uuid"]
            ground_truth = item["ground_truth"]
            responses = item["responses"]
            for i, response in enumerate(responses):
                try:
                    answer_after_think = remove_thinking_draft(response)
                    raw_prompt = PROMPT.format(question=question, response=answer_after_think)
                    prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True)
                    prompt_len = len(tokenizer.encode(prompt))
                    if prompt_len > 10240:
                        continue
                    prompts.append(prompt)
                    raw_prompts.append(raw_prompt)
                    questions.append(question)
                    ground_truths.append(ground_truth)
                    inner_idxs.append(i)
                    uuids.append(uuid)
                    idxs.append(idx)
                except Exception as e:
                    continue
    return prompts, raw_prompts, questions, ground_truths, inner_idxs, uuids, idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--tensor_parallel_size", type=int, required=False, default=8)
    parser.add_argument("--max_prompt_length", type=int, required=False, default=10240)
    parser.add_argument("--max_completion_length", type=int, required=False, default=5120)
    parser.add_argument("--temperature", type=float, required=False, default=0.6)
    parser.add_argument("--file_name", type=str, required=True)
    args = parser.parse_args()

    file_name = args.file_name + ".jsonl"
    all_files = os.listdir("/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/medmcqa")
    assert file_name in all_files

    save_dir = "/mnt/blob_output/v-dachengwen/LLM-Stability-v2/outputs/medmcqa/processed_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, file_name)
    print(f"Results will be saved to [{save_file}] ...")

    if os.path.exists(save_file):
        print(f"Results already exist in [{save_file}], skipping ...")
        exit()

    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_prompt_length + args.max_completion_length,
        max_num_batched_tokens=args.max_prompt_length + args.max_completion_length,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    prompts, raw_prompts, questions, ground_truths, inner_idxs, uuids, idxs = load_dataset(file_name, llm.get_tokenizer())

    print(f"Processing [{args.model_name}] on [{file_name}] ({len(prompts)} prompts) ...")
    print(f"Results will be saved to [{save_file}] ...")
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_completion_length,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    res = []
    for i in range(len(outputs)):
        assert len(outputs[i].outputs) == 1, f"Expected {1} outputs, but got {len(outputs[i].outputs)}"
        response = outputs[i].outputs[0].text.strip()
        final_answer = extract_from_tags(remove_thinking_draft(response), "final-answer")
        res.append({
            "idx": idxs[i],
            "uuid": uuids[i],
            "inner_idx": inner_idxs[i],
            "final_answer": final_answer,
            "question": questions[i],
            "ground_truth": ground_truths[i],
        })

    # save to jsonl file
    with open(save_file, "a") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")