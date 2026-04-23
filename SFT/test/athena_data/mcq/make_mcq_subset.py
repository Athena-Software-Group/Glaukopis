import argparse
import json
import random
from typing import List


def read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: List[dict]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create a 3k MCQ subset from updated dataset, skipping rows with updated_answer == 'X'.")
    parser.add_argument('--input', default='benchmark_data/athena_bench/athena-cti-mcq-updated.jsonl', help='Path to updated MCQ JSONL')
    parser.add_argument('--output', default='benchmark_data/athena_bench/athena-cti-mcq-3k.jsonl', help='Path to write subset JSONL')
    parser.add_argument('--size', type=int, default=3000, help='Number of questions to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    data = read_jsonl(args.input)
    indexed = list(enumerate(data))  # keep original 0-based index as id
    filtered = [(i, r) for i, r in indexed if str(r.get('updated_answer', '')).strip().upper() != 'X']

    if len(filtered) < args.size:
        raise ValueError(f"Not enough questions after filtering X. Needed {args.size}, found {len(filtered)}.")

    rng = random.Random(args.seed)
    sampled = rng.sample(filtered, args.size)  # list of (id, rec)
    sampled.sort(key=lambda t: t[0])  # store sequentially by original id

    # Ensure 'answer' matches 'updated_answer' and attach original id
    output_records = []
    for orig_id, rec in sampled:
        rec['id'] = orig_id
        rec['answer'] = rec.get('updated_answer', '')
        output_records.append(rec)

    write_jsonl(args.output, output_records)

    print("MCQ 3k subset summary:")
    print(f"- Input records: {len(data)}")
    print(f"- Eligible (updated_answer != 'X'): {len(filtered)}")
    print(f"- Sampled: {len(sampled)}")
    print(f"- Wrote: {args.output}")


if __name__ == '__main__':
    main()
