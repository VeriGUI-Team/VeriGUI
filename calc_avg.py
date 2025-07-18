import json

import argparse


if __name__ == '__main__':
    # parse argments (input file)
    parser = argparse.ArgumentParser(description='Evaluate model performance.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file containing evaluation data')
    args = parser.parse_args()

    with open(args.input_file, 'r') as file:
        data = json.load(file)

    with open('category.json', 'r') as f:
        cat = json.load(f)

    n_full_score = 0
    scores, nsteps = {}, {}
    cat_full_score, cat_num = {}, {}
    cat_scores, cat_nsteps = {}, {}
    for c in ["finance", "scientific", "arts", "technology", "social"]:
        cat_scores[c] = {}
        cat_nsteps[c] = {}
        cat_full_score[c] = 0
        cat_num[c] = 0

    for datum in data:
        name = datum.get("name", None).lower()
        name = f"{name.split('_')[0]}_{name.split('_')[1].zfill(2)}"
        q_type = cat[name]
        data_type = datum.get("type", None)
        score = datum.get("score", None)
        nstep = datum.get("nsteps", None)

        if data_type != "global":
            continue

        scores[name] = score
        cat_scores[q_type][name] = score

        if nstep > 0:
            nsteps[name] = nstep
            cat_nsteps[q_type][name] = nstep

        if score == 10:
            n_full_score += 1
            cat_full_score[q_type] += 1
        cat_num[q_type] += 1

    avg_score = sum(scores.values()) / len(scores) if scores else 0
    avg_nsteps = sum(nsteps.values()) / len(nsteps) if nsteps else 0

    avg_cat_scores = {k: sum(v.values()) / len(v) if v else 0 for k, v in cat_scores.items()}
    avg_cat_nsteps = {k: sum(v.values()) / len(v) if v else 0 for k, v in cat_nsteps.items()}
    avt_cat_full_score = {k: v / cat_num[k] if cat_num[k] > 0 else 0 for k, v in cat_full_score.items()}

    print(f"Full Score: {n_full_score * 100 / len(data)}")
    print(f"Average Full Score by Category:")
    for category, full_score in avt_cat_full_score.items():
        print(f"\t{category}: {full_score * 100:.1f}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Number of Steps: {avg_nsteps:.2f}")
    print("Average Scores by Category:")
    for category, score in avg_cat_scores.items():
        print(f"\t{category}: {score:.2f}")
    print("Average Number of Steps by Category:")
    for category, nstep in avg_cat_nsteps.items():
        print(f"\t{category}: {nstep:.2f}")
