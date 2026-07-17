"""Shared helpers for JSONL-backed counting datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch.utils.data import Dataset


SplitName: TypeAlias = str
Record: TypeAlias = dict[str, Any]
SPLITS: tuple[str, ...] = ("train", "eval", "test")


class TokenizedCountingDataset(Dataset):
    """JSONL-backed tokenized counting dataset."""

    def __init__(self, root: str | Path, split: SplitName) -> None:
        self.root = Path(root)
        self.split = split
        path = self.root / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing counting split file: {path}")
        self.records: list[Record] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "input_ids": record["input_ids"],
            "labels": record["labels"],
        }


def load_counting_dataset_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "dataset_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing counting dataset config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collate_tokenized_counting(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for row, item in enumerate(batch):
        item_input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        item_labels = torch.tensor(item["labels"], dtype=torch.long)
        length = item_input_ids.numel()
        input_ids[row, :length] = item_input_ids
        labels[row, :length] = item_labels
        attention_mask[row, :length] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def compute_split_counts(total: int, fractions: dict[SplitName, float]) -> dict[SplitName, int]:
    if total < 1:
        raise ValueError("cannot assign splits for an empty dataset")

    splits = tuple(fractions)
    if not splits:
        raise ValueError("at least one split fraction is required")
    positive_splits = [split for split in splits if fractions[split] > 0]
    if len(positive_splits) > total:
        raise ValueError(
            f"{total} examples cannot populate {len(positive_splits)} positive splits"
        )

    exact_counts = {split: total * fractions[split] for split in splits}
    counts = {split: int(exact_counts[split]) for split in splits}
    assigned = sum(counts.values())
    remainders = sorted(
        splits,
        key=lambda split: (exact_counts[split] - counts[split], fractions[split]),
        reverse=True,
    )
    for split in remainders[: total - assigned]:
        counts[split] += 1

    for split in positive_splits:
        if counts[split] > 0:
            continue
        donor = max(splits, key=lambda name: counts[name])
        if counts[donor] <= 1:
            raise ValueError("split allocation could not satisfy positive split fractions")
        counts[donor] -= 1
        counts[split] = 1

    return counts


def write_split_files(output_dir: Path, records: list[Record]) -> None:
    by_split: dict[str, list[Record]] = {}
    for record in records:
        by_split.setdefault(str(record["split"]), []).append(record)

    for split in SPLITS:
        if split not in by_split:
            (output_dir / f"{split}.jsonl").unlink(missing_ok=True)
    standard_splits = [split for split in SPLITS if split in by_split]
    extra_splits = sorted(split for split in by_split if split not in SPLITS)
    for split in (*standard_splits, *extra_splits):
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in by_split[split]:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")


def write_dataset_config(output_dir: Path, dataset_config: dict[str, Any]) -> None:
    with (output_dir / "dataset_config.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def digit_token(digit: int, *, digit_offset: int) -> int:
    if not 0 <= digit <= 9:
        raise ValueError("digit must be in [0, 9]")
    return digit_offset + digit


def number_tokens(value: int, *, digit_offset: int) -> list[int]:
    if value < 0:
        raise ValueError("only non-negative integers can be tokenized")
    return [digit_token(int(char), digit_offset=digit_offset) for char in str(value)]
