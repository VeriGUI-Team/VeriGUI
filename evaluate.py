import time
import json

import argparse


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
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file containing evaluation data')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file for results')
    parser.add_argument('--base_url', type=str, required=False, default='http://35.220.164.252:3888/v1')
    parser.add_argument('--api_key', type=str, required=False, default='sk-xxxx')
    parser.add_argument('--model_version', type=str, required=False, default='gpt-4.1')
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    with open(args.input_file, 'r') as file:
        data = json.load(file)

    output = []
    for datum in data:
        question = datum["instruction"]
        answer = datum["answer"]

        if "prediction" not in datum:
            datum["score"] = 0.0
            output.append(datum)
            continue

        prediction = datum["prediction"]

        judge_prompt = SCORER_TEMPLATE.format(question=question, answer=answer, pred=prediction)
        score = get_chat_response(client, args.model_version, judge_prompt, max_tokens=20)
        score = try_parse_llm_score(score)

        datum["score"] = score
        output.append(datum)

    with open(args.output_file, 'w') as file:
        json.dump(output, file, indent=4, ensure_ascii=False)





