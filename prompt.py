SCORER_TEMPLATE = """
You are a strict evaluator assessing answer correctness. You must score the model's prediction on a scale from 0 to 10, where 0 represents an entirely incorrect answer and 10 indicates a highly correct answer.

# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- Assign a high score if the prediction matches the answer semantically, considering variations in format.
- Deduct points for partially correct answers or those with incorrect additional information.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Scoring Guide
Provide a single integer from 0 to 10 to reflect your judgment of the answer's correctness.

# Strict Output format example
4"""


CAUSAL_AGENT_TEMPLATE = """You are an AI agent designed to complete the global instruction. You will be given the global instruction and your previous actions. Your task is to complete the global instruction based on your previous actions.

# Input
Global Instruction:
```
{ginstruction}
```

Previous Actions:
```
{history}
```
"""