import os
import json
import random
import time
from functools import partial

import requests
import numpy as np

from openai import OpenAI

from prompt import SCORER_TEMPLATE, CAUSAL_AGENT_TEMPLATE
from utils import get_n_items, try_parse_llm_score

from tqdm import tqdm


class Submission(object):
    def __init__(self, answer, actions):
        self.answer = answer
        self.actions = actions


class AgentWorkflowBenchmark:
    NUM_SECONDS_TO_SLEEP = 5
    MAX_TOKENS = 20

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.data = {}

        self.submit_required = True
        self.scores = {}
        self.responses = []

        self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_dir, "version-0716.json"), 'r') as f:
            try:
                records = json.load(f)
                for record in records:
                    self.data[record["folder"]] = record
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return
        print(f"A total of {len(self.data)} tasks loaded from {self.data_dir}")

    def __iter__(self):
        for task_id, (name, content) in enumerate(self.data.items()):
            # global accuracy: only validate the final result
            metadata = {
                "id": name,
                "instruction": content["instruct"],
                "type": "global",
                "task_id": task_id,
            }
            ground_truth = content["result"]
            yield metadata, ground_truth

            # casual accuracy: validate each step in the workflow
            actions = [(action["instruct"], action["result"]) for action in content["sub_tasks"]]
            previous_actions = []
            for i, (sub_instruct, sub_ground_truth) in enumerate(actions):
                history_actions = "\n\n".join([f"({i}). Goal: {_a} \n Answer: {_gt}" for i, (_a, _gt) in enumerate(previous_actions)])
                prompt = CAUSAL_AGENT_TEMPLATE.format(ginstruction=content["instruct"], history=history_actions)
                metadata = {
                    "id": name,
                    "instruction": prompt,
                    "type": "causal",
                    "task_id": task_id,
                }
                if i >= len(actions) - 3:
                    yield metadata, ground_truth
                previous_actions.append((sub_instruct, sub_ground_truth))


if __name__ == '__main__':
    benchmark = AgentWorkflowBenchmark(data_dir='./data')
    dataset = []
    for i, (metadata, ground) in tqdm(enumerate(benchmark)):
        dataset.append({
            "id": metadata["task_id"],
            "name": metadata["id"],
            "type": metadata["type"],
            "instruction": metadata["instruction"],
            "answer": json.dumps(ground),
        })

    # save dataset to json file
    with open("data/veriGUI.json", "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
