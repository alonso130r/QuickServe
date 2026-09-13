#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "prepare_wildchat.py"
SPEC = importlib.util.spec_from_file_location("prepare_wildchat", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def turn(role, content, identifier, created=None, redacted=False,
         token_counter=None):
    return {
        "role": role,
        "content": content,
        "turn_identifier": identifier,
        "created": created,
        "redacted": redacted,
        "token_counter": token_counter,
    }


class PrepareWildChatTests(unittest.TestCase):
    def row(self, language="English", turns=3, redacted=False):
        conversation = []
        for index in range(turns):
            identifier = 100 + index
            conversation.extend([
                turn("user", f"question {index}", identifier,
                     redacted=redacted and index == 0),
                turn("assistant", f"answer {index}", identifier,
                     created=1_700_000_000 + index, token_counter=10 + index),
            ])
        return {
            "conversation_hash": "abc",
            "model": "gpt-test",
            "timestamp": "2023-11-14T22:13:22",
            "conversation": conversation,
            "turn": turns,
            "language": language,
        }

    def test_accepts_complete_unredacted_english_conversation(self):
        prepared = MODULE.prepare_conversation(self.row())
        self.assertEqual(prepared["conversation_id"], "abc")
        self.assertEqual(len(prepared["turns"]), 3)

    def test_rejects_wrong_language_short_or_redacted_conversation(self):
        self.assertIsNone(MODULE.prepare_conversation(self.row(language="Spanish")))
        self.assertIsNone(MODULE.prepare_conversation(self.row(turns=2)))
        self.assertIsNone(MODULE.prepare_conversation(self.row(redacted=True)))

    def test_each_request_contains_the_complete_conversation_prefix(self):
        prepared = MODULE.prepare_conversation(self.row())
        requests = MODULE.requests_for_conversation(prepared)
        self.assertEqual([message["content"] for message in requests[0]["messages"]],
                         ["question 0"])
        self.assertEqual([message["content"] for message in requests[2]["messages"]],
                         ["question 0", "answer 0", "question 1", "answer 1",
                          "question 2"])
        self.assertEqual(requests[2]["arrival_timestamp"], 1_700_000_002)
        self.assertEqual(requests[2]["reference_response"], "answer 2")
        self.assertEqual(requests[2]["reference_output_tokens"], 12)

    def test_writer_is_deterministic_and_orders_requests_by_arrival(self):
        later = MODULE.prepare_conversation(self.row())
        earlier_row = self.row()
        earlier_row["conversation_hash"] = "earlier"
        for message in earlier_row["conversation"]:
            if message["role"] == "assistant":
                message["created"] -= 100
        earlier = MODULE.prepare_conversation(earlier_row)
        with tempfile.TemporaryDirectory() as root:
            first = pathlib.Path(root) / "first"
            second = pathlib.Path(root) / "second"
            MODULE.write_dataset([later, earlier], first, {"revision": "fixed"})
            MODULE.write_dataset([later, earlier], second, {"revision": "fixed"})
            self.assertEqual((first / "requests.jsonl").read_bytes(),
                             (second / "requests.jsonl").read_bytes())
            lines = (first / "requests.jsonl").read_text().splitlines()
            self.assertIn('"conversation_id":"earlier"', lines[0])
            self.assertEqual((first / "manifest.json").read_bytes(),
                             (second / "manifest.json").read_bytes())
            self.assertEqual((first / "requests.qsc").read_bytes(),
                             (second / "requests.qsc").read_bytes())
            self.assertEqual((first / "requests.qsc").read_bytes()[:8],
                             b"QSCONV\0\0")


if __name__ == "__main__":
    unittest.main()
