import copy
import os
import time
import json

import argparse
from tqdm import tqdm

import requests
from openai import OpenAI

from prompt import SCORER_TEMPLATE
from utils import try_parse_llm_score


def get_chat_response(client, model_version, content, max_tokens, retries=5):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the correctness of the answer.",
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": model_version,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            return content
        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt + 1}: {e}")
            time.sleep(5.0)
            if attempt == retries - 1:
                print(f"Failed to get response after {retries} attempts")
                return ""
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            return ""


if __name__ == '__main__':
    # parse argments (input file and output file)
    parser = argparse.ArgumentParser(description='Evaluate model performance.')
    parser.add_argument('--base_url', type=str, required=False, default='')
    parser.add_argument('--api_key', type=str, required=False, default='')
    parser.add_argument('--model_version', type=str, required=False, default='o3')
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # read instruct.json
    instruct_file = './data/veriGUI.json'
    _veriGUI = json.load(open(instruct_file, 'r', encoding='utf-8'))
    veriGUI = {}
    for item in _veriGUI:
        if item['type'] != "global":
            continue
        name = item['name']
        instruction = item['instruction']
        answer = item['answer']
        veriGUI[name] = {
            "name": name,
            "instruction": instruction,
            "answer": answer,
            "type": item['type'],
        }

    # iterate all json files in the 'predictions' directory
    for filename in os.listdir('./predictions'):
        if not (filename.endswith(".json") or filename.endswith(".jsonl")):
            continue
        input_file = os.path.join('./predictions', filename)
        output_file = os.path.join('./evaluated', filename.replace("jsonl", "json"))

        if os.path.exists(output_file):
            continue

        print(f"Evaluating {input_file}, output will be saved to {output_file}")
        if input_file.endswith(".json"):
            with open(input_file, 'r') as file:
                data = json.load(file)
        elif input_file.endswith(".jsonl"):
            data = []
            with open(input_file, 'r') as file:
                for line in file:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Unsupported file format: {input_file}")

        formatted_data = {}
        for datum in data:
            if "name" in datum:
                name = datum["name"]
            elif "folder" in datum:
                name = datum["folder"]
            else:
                raise ValueError(f"name not found")

            if "prediction" in datum:
                prediction = datum["prediction"]
            elif "result" in datum:
                prediction = datum["result"]
            elif "answer" in datum:
                prediction = datum["answer"]
            elif "model_output" in datum:
                prediction = datum["model_output"]
            else:
                print("No prediction found for", name)
                prediction = ""

            if "nsteps" in datum:
                n_steps = datum["nsteps"]
            elif "executor_trace" in datum:
                n_steps = len(datum.get("executor_trace", []))
            elif "tool_call_count" in datum:
                n_steps = int(datum.get("tool_call_count", 0))
            else:
                n_steps = 0

            if prediction is None or prediction == "[executor reach max turns]":
                prediction = ""
                n_steps = 0

            formatted_data[name] = {
                "prediction": prediction,
                "nsteps": n_steps,
            }

        output = []
        for vname, vdata in tqdm(veriGUI.items()):
            _vdata = copy.deepcopy(vdata)
            if vname not in formatted_data:
                _vdata["score"] = 0
                _vdata["nsteps"] = 0
            else:
                question = _vdata["instruction"]
                answer = _vdata["answer"]
                prediction = formatted_data[vname]["prediction"]
                nsteps = formatted_data[vname]["nsteps"]

                judge_prompt = SCORER_TEMPLATE.format(question=question, answer=answer, pred=prediction)
                score = get_chat_response(client, args.model_version, judge_prompt, max_tokens=1024 * 32)
                score = try_parse_llm_score(score)

                _vdata["prediction"] = prediction
                _vdata["score"] = score
                _vdata["nsteps"] = nsteps
            output.append(_vdata)
        with open(output_file, 'w') as file:
            json.dump(output, file, indent=4, ensure_ascii=False)
