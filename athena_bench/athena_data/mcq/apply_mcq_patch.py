import argparse
import csv
import json
from typing import Dict, List, Set


VALID_LETTERS = {"A", "B", "C", "D", "E"}


def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_patch_tsv(path: str) -> Dict[int, dict]:
    patch: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required_cols = {"id", "question", "answer"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Patch TSV missing required columns: {sorted(missing)}")
        for row in reader:
            try:
                idx = int(row["id"])  # 0-based index into the JSONL
            except Exception as e:
                raise ValueError(f"Invalid id in patch TSV row: {row}") from e
            patch[idx] = row
    return patch


def compute_counts(
    original: List[dict], patch: Dict[int, dict]
):
    total = len(patch)

    ids_all: Set[int] = set(patch.keys())
    # 1) How many have answer 'X' in patch
    ids_x = {i for i, r in patch.items() if (r.get("answer", "").strip().upper() == "X")}
    count_x = len(ids_x)

    # Remaining after removing 'X'
    remaining = ids_all - ids_x

    # 2) Among remaining, how many had original correct_answer not in [A..E] and have answer in patch
    ids_bad_original: Set[int] = set()
    for i in sorted(remaining):
        orig = original[i]
        orig_correct = str(orig.get("correct_answer", "")).strip().upper()
        patch_ans = str(patch[i].get("answer", "")).strip().upper()
        if orig_correct not in VALID_LETTERS and patch_ans in VALID_LETTERS:
            ids_bad_original.add(i)
    count_bad_original = len(ids_bad_original)

    remaining2 = remaining - ids_bad_original

    # 3) Among remaining, original correct_answer in [A..E] but replaced by patch (different letter)
    ids_replaced: Set[int] = set()
    for i in sorted(remaining2):
        orig = original[i]
        orig_correct = str(orig.get("correct_answer", "")).strip().upper()
        patch_ans = str(patch[i].get("answer", "")).strip().upper()
        if orig_correct in VALID_LETTERS and patch_ans in VALID_LETTERS and patch_ans != orig_correct:
            ids_replaced.add(i)
    count_replaced = len(ids_replaced)

    remaining3 = remaining2 - ids_replaced

    # 4) Remaining: same correct_answer and patch answer
    ids_same: Set[int] = set()
    for i in sorted(remaining3):
        orig = original[i]
        orig_correct = str(orig.get("correct_answer", "")).strip().upper()
        patch_ans = str(patch[i].get("answer", "")).strip().upper()
        if orig_correct in VALID_LETTERS and patch_ans == orig_correct:
            ids_same.add(i)
        else:
            # If we end up here, it means some case not covered above.
            # Treat it conservatively as 'same' only if strings match exactly.
            if patch_ans == orig_correct:
                ids_same.add(i)
    count_same = len(ids_same)

    # Sanity: sums should match total
    if (count_x + count_bad_original + count_replaced + count_same) != total:
        raise AssertionError(
            "Counts do not sum to total patch rows. "
            f"total={total} x={count_x} bad_original={count_bad_original} "
            f"replaced={count_replaced} same={count_same}"
        )

    return {
        "total": total,
        "count_x": count_x,
        "count_bad_original": count_bad_original,
        "count_replaced": count_replaced,
        "count_same": count_same,
    }


def assert_questions_match(original: List[dict], patch: Dict[int, dict]):
    for i, prow in patch.items():
        oq = str(original[i].get("question", "")).strip()
        pq = str(prow.get("question", "")).strip()
        if oq != pq:
            raise AssertionError(
                f"Question mismatch at row {i}:\nOriginal: {oq}\nPatch:    {pq}"
            )


def apply_patch(
    original: List[dict], patch: Dict[int, dict]
) -> List[dict]:
    out: List[dict] = []
    for i, rec in enumerate(original):
        new_rec = dict(rec)
        if i in patch:
            patch_ans = str(patch[i].get("answer", "")).strip().upper()
            new_rec["updated_answer"] = patch_ans
        else:
            # For rows without patch, set updated_answer == original correct_answer
            new_rec["updated_answer"] = str(rec.get("correct_answer", "")).strip()
        out.append(new_rec)
    return out


def main():
    parser = argparse.ArgumentParser(description="Apply MCQ patch TSV to JSONL dataset and produce updated answers with counts.")
    parser.add_argument("--input", default="benchmark_data/athena_bench/athena-cti-mcq.jsonl", help="Path to original MCQ JSONL")
    parser.add_argument("--patch", default="benchmark_data/athena_bench/mcq-patch.tsv", help="Path to TSV patch file (0-based ids)")
    parser.add_argument("--output", default="benchmark_data/athena_bench/athena-cti-mcq-updated.jsonl", help="Path to write updated JSONL")
    args = parser.parse_args()

    original = read_jsonl(args.input)
    patch = read_patch_tsv(args.patch)

    # Ensure patch ids are in-range
    max_id = max(patch.keys()) if patch else -1
    if max_id >= len(original):
        raise IndexError(f"Patch id {max_id} out of range for original dataset size {len(original)}")

    # Assert questions match between original[i] and patch row i
    assert_questions_match(original, patch)

    # Compute and print counts
    counts = compute_counts(original, patch)
    print("Patch application summary:")
    print(f"- Total rows in patch: {counts['total']}")
    print(f"- Answer 'X' in patch: {counts['count_x']}")
    print(
        f"- Original incorrect format (not A-E) but patched with letter: {counts['count_bad_original']}"
    )
    print(
        f"- Original letter (A-E) replaced by different letter: {counts['count_replaced']}"
    )
    print(
        f"- Original letter (A-E) same as patch: {counts['count_same']}"
    )

    # Apply patch and write output
    updated = apply_patch(original, patch)
    write_jsonl(args.output, updated)
    print(f"\nWrote updated dataset with 'updated_answer' to: {args.output}")


if __name__ == "__main__":
    main()

