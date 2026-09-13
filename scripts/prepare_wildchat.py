#!/usr/bin/env python3
"""Download a deterministic WildChat conversation window and prepare replay data."""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import struct
import sys
import urllib.parse
import urllib.request


DATASET = "allenai/WildChat-4.8M"
CONFIG = "default"
SPLIT = "train"
DEFAULT_OUTPUT = pathlib.Path("data/wildchat-4.8m-english-10k")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def parse_timestamp(value):
    if not value:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.timestamp()
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
        tzinfo=dt.timezone.utc).timestamp()


def prepare_conversation(row):
    if str(row.get("language", "")).casefold() != "english":
        return None
    messages = row.get("conversation") or []
    if any(message.get("redacted") for message in messages):
        return None

    turns = []
    index = 0
    while index + 1 < len(messages):
        user, assistant = messages[index], messages[index + 1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            return None
        if not user.get("content") or not assistant.get("content"):
            return None
        if (user.get("turn_identifier") is not None and
                assistant.get("turn_identifier") is not None and
                user["turn_identifier"] != assistant["turn_identifier"]):
            return None
        created = assistant.get("created")
        if created is None:
            created = parse_timestamp(assistant.get("timestamp"))
        if created is None:
            return None
        output_tokens = assistant.get("token_counter")
        if not isinstance(output_tokens, int) or output_tokens <= 0:
            return None
        turns.append({
            "assistant": assistant["content"],
            "output_tokens": output_tokens,
            "request_timestamp": created,
            "turn_id": assistant.get("turn_identifier"),
            "user": user["content"],
        })
        index += 2
    if index != len(messages) or len(turns) < 3:
        return None
    return {
        "conversation_id": row["conversation_hash"],
        "model": row.get("model"),
        "source_timestamp": str(row.get("timestamp")),
        "turns": turns,
    }


def requests_for_conversation(conversation):
    messages = []
    requests = []
    for turn_index, turn in enumerate(conversation["turns"]):
        messages.append({"role": "user", "content": turn["user"]})
        requests.append({
            "arrival_timestamp": turn["request_timestamp"],
            "conversation_id": conversation["conversation_id"],
            "messages": list(messages),
            "reference_response": turn["assistant"],
            "reference_output_tokens": turn["output_tokens"],
            "source_model": conversation["model"],
            "turn_id": turn["turn_id"],
            "turn_index": turn_index,
        })
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return requests


def render_qwen_prompt(messages):
    parts = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def write_conversation_trace(requests, path, source_path):
    first_ns = round(float(requests[0]["arrival_timestamp"]) * 1_000_000_000)
    last_ns = round(float(requests[-1]["arrival_timestamp"]) * 1_000_000_000)
    digest = bytes.fromhex(sha256(source_path))
    header = struct.pack("<8sIIIIQqq32s16s", b"QSCONV\0\0", 1, 96, 0, 1,
                         len(requests), first_ns, last_ns, digest, bytes(16))
    with path.open("wb") as target:
        target.write(header)
        for request in requests:
            conversation_id = request["conversation_id"].encode("utf-8")
            if len(conversation_id) > 32:
                raise ValueError("conversation_id exceeds 32 UTF-8 bytes")
            prompt = render_qwen_prompt(request["messages"]).encode("utf-8")
            arrival_ns = round(float(request["arrival_timestamp"]) * 1_000_000_000)
            target.write(struct.pack(
                "<QIIII32s", arrival_ns - first_ns,
                request["reference_output_tokens"], request["turn_index"],
                len(prompt), 0, conversation_id.ljust(32, b"\0")))
            target.write(prompt)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(conversations, output_dir, source):
    output_dir.mkdir(parents=True, exist_ok=True)
    conversations = sorted(
        conversations,
        key=lambda item: (item["turns"][0]["request_timestamp"],
                          item["conversation_id"]),
    )
    conversation_path = output_dir / "conversations.jsonl"
    request_path = output_dir / "requests.jsonl"
    with conversation_path.open("wb") as target:
        for conversation in conversations:
            target.write(json_bytes(conversation))
    requests = sorted(
        (request for conversation in conversations
         for request in requests_for_conversation(conversation)),
        key=lambda item: (item["arrival_timestamp"], item["conversation_id"],
                          item["turn_index"]),
    )
    with request_path.open("wb") as target:
        for request in requests:
            target.write(json_bytes(request))
    trace_path = output_dir / "requests.qsc"
    write_conversation_trace(requests, trace_path, request_path)
    manifest = {
        "conversation_count": len(conversations),
        "conversation_sha256": sha256(conversation_path),
        "filters": {"language": "English", "minimum_complete_turns": 3,
                    "redacted": False},
        "first_request_timestamp": requests[0]["arrival_timestamp"],
        "last_request_timestamp": requests[-1]["arrival_timestamp"],
        "request_count": len(requests),
        "request_sha256": sha256(request_path),
        "conversation_trace": trace_path.name,
        "conversation_trace_sha256": sha256(trace_path),
        "selection": "earliest eligible conversations in source shard order",
        "source": source,
        "timing_note": "request_timestamp uses the paired assistant created field",
    }
    (output_dir / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "QuickServe/1"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download(url, destination):
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "QuickServe/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as target:
        while block := response.read(1024 * 1024):
            target.write(block)
    temporary.replace(destination)


def collect(limit, cache_dir):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required; run this script with: uv run --with pyarrow") from error

    dataset_api = "https://huggingface.co/api/datasets/" + DATASET
    dataset_info = get_json(dataset_api)
    parquet_api = ("https://datasets-server.huggingface.co/parquet?" +
                   urllib.parse.urlencode({"dataset": DATASET}))
    files = [item for item in get_json(parquet_api)["parquet_files"]
             if item["config"] == CONFIG and item["split"] == SPLIT]
    conversations = []
    previous_source_timestamp = None
    used_files = []
    columns = ["conversation_hash", "model", "timestamp", "conversation",
               "turn", "language"]
    for index, item in enumerate(files):
        path = cache_dir / f"{index:04d}.parquet"
        print(f"Downloading/scanning shard {index + 1}/{len(files)}", file=sys.stderr)
        download(item["url"], path)
        used_files.append({"url": item["url"], "sha256": sha256(path)})
        table = parquet.read_table(path, columns=columns)
        for row in table.to_pylist():
            timestamp = parse_timestamp(row.get("timestamp"))
            if timestamp is None:
                continue
            if previous_source_timestamp is not None and timestamp < previous_source_timestamp:
                raise RuntimeError("WildChat source shards are not chronologically ordered")
            previous_source_timestamp = timestamp
            prepared = prepare_conversation(row)
            if prepared is not None:
                conversations.append(prepared)
                if len(conversations) == limit:
                    source = {
                        "config": CONFIG,
                        "dataset": DATASET,
                        "dataset_revision": dataset_info["sha"],
                        "parquet_files": used_files,
                        "split": SPLIT,
                    }
                    return conversations, source
    raise RuntimeError(f"only found {len(conversations)} eligible conversations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=10_000)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=pathlib.Path,
                        default=pathlib.Path("data/wildchat-4.8m-cache"))
    args = parser.parse_args()
    if args.conversations <= 0:
        parser.error("--conversations must be positive")
    conversations, source = collect(args.conversations, args.cache_dir)
    manifest = write_dataset(conversations, args.output_dir, source)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
