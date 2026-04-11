import json
import os
import sys
from typing import Optional
from openai import OpenAI
import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
BASE_URL = "http://localhost:7860"

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an Eco-Resilient Logistics Agent. Your goal is to fulfill orders while minimizing CO2.
Respond ONLY with a valid JSON completion.

Available Actions:
{
  "action_type": "order_parts | produce | offset | skip",
  "part_type": "chips | sensors | batteries | casing",
  "quantity": number,
  "mode": "sea | air | rail | road",
  "product": "EcoPhone | GreenTab",
  "offset_amount": number
}
"""

def get_action(obs):
    prompt = f"Current State: {json.dumps(obs, indent=2)}\nChoose next action:"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_task(task_name: str):
    print(f"[START] task={task_name} env=atlas-greenpath model={MODEL_NAME}")
    
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as app:
        obs = app.post("/reset", json={"task": task_name}).json()
        
        done = False
        step = 0
        while not done and step < 50:
            step += 1
            action_json = get_action(obs)
            resp = app.post("/step", json=action_json).json()
            
            obs = resp["observation"]
            reward = resp["reward"]
            done = resp["done"]
            
            print(f"[STEP]  step={step} action={action_json['action_type']} reward={reward:.2f} done={done} error=null")
            
            if done:
                score = resp["info"].get("final_score", 0.0)
                print(f"[END]   success=true steps={step} score={score:.4f} rewards=...")

if __name__ == "__main__":
    # In a real environment, we'd start the server first.
    # For baseline reproducibility, we assume the server is running on 7860.
    run_task("easy")
