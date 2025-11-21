from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What type of reaction is the conversion of glucose-6-P to fructose-6-P?"
messages = [
    {"role": "user", "content": prompt}
]

non_reasoning_question_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
non_reasoning_question_model_inputs = tokenizer([non_reasoning_question_text], return_tensors="pt")

question_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
print("question_text: ", question_text)
question_model_inputs = tokenizer([question_text], return_tensors="pt")

step_1_text = question_text + "<think>\nOkay, so I need to figure out what type of reaction the conversion of glucose-6-P to fructose-6-P is. Let me start by recalling what I know about glucose and fructose. Both are hexoses, six-carbon sugars. Glucose-6-phosphate (G6P) is a phosphorylated form of glucose, and fructose-6-phosphate (F6P) is the phosphorylated form of fructose.\n</think>\n\n"
step_1_model_inputs = tokenizer([step_1_text], return_tensors="pt")

step_2_text = question_text + "<think>\nFirst, I remember that in glycolysis, there are several steps where phosphates are added or removed. For example, the first step is the phosphorylation of glucose to form G6P by hexokinase. Then, in the next step, G6P is converted to F6P. Wait, is that right? Oh yeah, in glycolysis, the reaction is catalyzed by phosphoglucose isomerase. So that's the enzyme involved here.\nNow, the question is about the type of reaction. So, what kind of reaction is it when a molecule is converted from one isomer to another? Since glucose and fructose are isomers (structural isomers), this must be an isomerization reaction. Isomerization reactions involve the rearrangement of atoms within a molecule to form a different isomer. So, in this case, the phosphate group is on the same carbon (carbon 6) in both G6P and F6P, but the rest of the molecule is rearranged.\n</think>\n\n"
step_2_model_inputs = tokenizer([step_2_text], return_tensors="pt")

step_3_text = question_text + "<think>\nWait, but phosphoglucose isomerase is an enzyme that catalyzes this conversion. So the reaction is an isomerization. Is there a specific term for this? Maybe it's a phosphoglucoisomerase reaction. But the main point is that it's an isomerization. Let me think if there are other possibilities. For example, could it be a dehydration or hydration reaction? No, because the phosphate is still on the same carbon. The key is that the structure of the sugar changes, but the phosphate remains in the same position. So, the reaction is an isomerization.\n</think>\n\n"
step_3_model_inputs = tokenizer([step_3_text], return_tensors="pt")

step_4_text = question_text + "<think>\nI think that's it. The conversion of glucose-6-P to fructose-6-P is an isomerization reaction, specifically a phosphoglucoisomerization. So the answer should be that it's an isomerization reaction.\n</think>\n\n"
step_4_model_inputs = tokenizer([step_4_text], return_tensors="pt")

all_model_inputs = {
    "non_reasoning_question_model_inputs": non_reasoning_question_model_inputs,
    "question_model_inputs": question_model_inputs,
    "step_1_model_inputs": step_1_model_inputs,
    "step_2_model_inputs": step_2_model_inputs,
    "step_3_model_inputs": step_3_model_inputs,
    "step_4_model_inputs": step_4_model_inputs,
}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


for key, model_inputs in all_model_inputs.items():
    model_inputs = model_inputs.to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    print("\n\n\n\n\n")
    print("-" * 100)
    print(f"******** {key} ********\n", output_text)
