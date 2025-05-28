import os
import re
import json
import torch
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ----------------------------------------------
# 1) Load environment variables from a .env file
# ----------------------------------------------
load_dotenv()

# Retrieve the Hugging Face token from the environment
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set!")

# -------------------------------------------------
# 2) Set up a Transformers cache directory (optional)
# -------------------------------------------------
cache_dir = "/tmp/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir

# -------------------------------------------------
# 3) Model & tokenizer load
# -------------------------------------------------

adapter_model_path = (
    "Sanjayan201/mistral-7b-v0.3-finetuned-for-mindmap-lora"
)
base_model_name = "mistralai/Mistral-7B-v0.3"
max_seq_length = 2048

print("Loading base model on CPU (float32)...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    use_auth_token=hf_token
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    use_auth_token=hf_token
)

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    adapter_model_path,
    use_auth_token=hf_token
)

print("Merging LoRA adapter into the model ...")
model = model.merge_and_unload()
model.eval()


print("Model loaded on CPU.")

# Alpaca-style prompt template
alpaca_prompt = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n"
    "### Instruction:\n{}\n"
    "### Input:\n{}\n"
    "### Response:\n{}"
)

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
async def root():
    return {
        "status": "ok",
        "service": "mindmap",
        "version": "1.0.0",
    }


class MindMapRequest(BaseModel):
    content: str = ""


def remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def preprocess_text(text: str) -> str:
    text = remove_emoji(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_prompt(instruction: str, content: str):
    prompt = alpaca_prompt.format(instruction, content, "")
    inputs = tokenizer([prompt], return_tensors="pt", truncation=False)
    if inputs.input_ids.shape[1] > max_seq_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input text too long, got "
                f"{inputs.input_ids.shape[1]} tokens."
            ),
        )
    return inputs


def get_response(instruction: str, content: str) -> str:
    with torch.no_grad():
        inputs = tokenize_prompt(instruction, content)
        # Ensure on CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
        full_response = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]
    if "### Response:" in full_response:
        return full_response.split("### Response:")[1].strip()
    else:
        return full_response.strip()


def fix_incomplete_json(json_str: str) -> str:
    stack = []
    in_string = False
    escape = False

    for char in json_str:
        if char == '"' and not escape:
            in_string = not in_string
        if in_string:
            if char == '\\' and not escape:
                escape = True
            else:
                escape = False
            continue

        if char in '{[':
            stack.append(char)
        elif char in '}]':
            if stack:
                last = stack[-1]
                if (
                    (last == "{" and char == "}")
                    or (last == "[" and char == "]")
                ):
                    stack.pop()
    closing_map = {'{': '}', '[': ']'}
    while stack:
        json_str += closing_map[stack.pop()]
    return json_str


def get_mindmap(text: str) -> dict:
    mindmap_response = get_response(
        "Convert the following text into a structured JSON mind map "
        "with parent node and logical nested subnodes:"
        text
    )
    try:
        return json.loads(mindmap_response)
    except json.JSONDecodeError:
        fixed_response = fix_incomplete_json(mindmap_response)
        try:
            return json.loads(fixed_response)
        except json.JSONDecodeError as e2:
            raise HTTPException(
                status_code=500,
                detail="Error parsing JSON from model output after fixing: "
                       + str(e2) + ". Raw response: " + mindmap_response
            )


@app.post("/generate")
async def generate_mindmap(request: MindMapRequest):
    try:
        cleaned_text = preprocess_text(request.content)
        mindmap_obj = get_mindmap(cleaned_text)
        return mindmap_obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simplify")
async def simplify_text(request: MindMapRequest):
    try:
        cleaned_text = preprocess_text(request.content)
        simplified_text = get_response(
            "Summarize and shorten the following text :",
            cleaned_text
        )
        mindmap_obj = get_mindmap(simplified_text)
        return mindmap_obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extend")
async def extend_text(request: MindMapRequest):
    try:
        cleaned_text = preprocess_text(request.content)
        extended_text = get_response(
            "Extend, expand, and enrich the following text by providing deeper "
            "detail, relevant examples, and additional context, while preserving "
            "its original meaning and style:"
            cleaned_text
        )
        mindmap_obj = get_mindmap(extended_text)
        return mindmap_obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# You can either run uvicorn directly from code:
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)
