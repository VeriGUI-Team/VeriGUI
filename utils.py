import re


def get_n_items(obj):
    total_length = 0
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, list) or isinstance(item, dict):
                total_length += get_n_items(item)
            else:
                total_length += 1
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, list) or isinstance(value, dict):
                total_length += get_n_items(value)
            else:
                total_length += 1

    return total_length


def try_parse_llm_score(score):
    try:
        pattern = r"(\d+)"
        match = re.search(pattern, score)

        if match:
            score = float(match.group(1))
        else:
            score = 0
    except:
        score = 0

    return score
