"""Saved-analysis details must describe real work without changing replay."""
from copy import deepcopy
import json
import unittest

from src.analysis_checkpoints import AnalysisCheckpointError
from src.analysis_task_details import summarize_saved_items, update_task_display
from src.resumable_analysis import (
    CallJournal, checkpoint_call, checkpoint_local_value, checkpoint_scope, journal_scope,
)
from src.run_cancellation import AnalysisCancelled


PAYLOAD = {"provider": "OpenAI", "model": "test-model", "prompt": "private prompt"}
SAVED_REQUEST = {
    "operation": "llm.completion", "request": PAYLOAD,
    "fingerprint": "52e50c23b7a78db4ce6317ba9a1326b3176ac39d09e9aa853351c4e0e18cdf3b",
}
SAVED_ITEM = {"index": 0, "payload": SAVED_REQUEST, "status": "complete", "result": "answer", "attempts": 1}


class MemoryStore:
    """Record fenced writes while retaining realistic task/item responses."""
    owner_id = "owner"

    def __init__(self, items=(), *, status="incomplete", fail_phase=None, stop_on_progress=False):
        self.task = {
            "id": "task", "page_key": "ordinary", "status": status,
            "source": {"journal_version": 2}, "stop_requested": False,
            "metadata": {"audit": "keep", "final_session": {"old": 1}, "stop_requested": True},
        }
        self.items = deepcopy(list(items))
        self.events = []
        self.fail_phase = fail_phase
        self.stop_on_progress = stop_on_progress

    def claim(self, task_id, **kwargs):
        self.events.append("claim")
        self.task["lease_token"] = "lease"
        return deepcopy(self.task)

    def load_items(self, task_id, offset, limit):
        return deepcopy(self.items[offset:offset + limit])

    def heartbeat(self, task_id, lease_token, **kwargs):
        assert lease_token == "lease"
        self.events.append("heartbeat")
        return deepcopy(self.task)

    def append_item(self, task_id, index, request, *, lease_token):
        assert lease_token == "lease"
        self.events.append("append")
        self.items.append({"index": index, "payload": deepcopy(request), "status": "pending", "attempts": 0})

    def save_progress(self, task_id, *, metadata, lease_token):
        assert lease_token == "lease"
        phase = metadata["display"]["current_step"]["phase"]
        self.events.append("progress:" + phase)
        if self.fail_phase == phase:
            raise RuntimeError("checkpoint offline")
        self.task["metadata"] = deepcopy(metadata)
        self.task["stop_requested"] |= self.stop_on_progress
        return deepcopy(self.task)

    def save_item(self, task_id, index, **kwargs):
        assert kwargs["lease_token"] == "lease"
        self.events.append("save:" + kwargs["status"])
        self.items[index].update({key: deepcopy(kwargs[key]) for key in ("status", "attempts", "result")})
        self.task["metadata"] = deepcopy(kwargs["metadata"])
        return deepcopy(self.task)

    def release(self, task_id, lease_token, *, status, metadata):
        assert lease_token == "lease"
        self.events.append("release")
        self.task.update(status=status, metadata=deepcopy(metadata))
        return deepcopy(self.task)


class TaskDetailsTests(unittest.TestCase):
    def journal(self, store):
        journal = CallJournal(store, store.task, heartbeat=False)
        self.addCleanup(journal.finish, False)
        return journal

    def test_safe_fields_and_local_sampling(self):
        request = {
            "operation": "llm.completion", "scope": {"path": ["example", 0]},
            "request": {"provider": "OpenAI", "model_name": "test-model", "prompt": "private prompt",
                        "messages": [{"content": "private body"}],
                        "base_url": "https://name:password@api.openai.com/v1?api_key=secret#token"},
        }
        display = update_task_display(None, request, 0, "running")
        self.assertEqual(display["apis"], [{"provider": "OpenAI", "model": "test-model", "endpoint": "https://api.openai.com/v1"}])
        self.assertEqual(display["current_step"]["scope"], ["example", 0])
        for word in ("private", "password", "secret", "messages", "api_key"):
            self.assertNotIn(word, json.dumps(display))
        local = {"operation": "prompt.full", "request": {"input_text": "private prompt"}}
        self.assertEqual(update_task_display(None, local, 1, "complete")["apis"], [])
        self.assertEqual(update_task_display(display, local, 1, "complete")["apis"], display["apis"])

    def test_endpoint_inference_and_agent_model_history(self):
        probe = {"operation": "min_k.completion_probe", "request": {"model": "test", "endpoint": "https://api.openai.com/v1"}}
        self.assertEqual(update_task_display(None, probe, 0, "running")["apis"][0]["provider"], "OpenAI")
        agent = {"operation": "agent.deploy", "request": {"agent_url": "http://localhost:9123/", "model_path": "/models/test", "query": "private"}}
        display = update_task_display(None, agent, 1, "running")
        self.assertEqual(display["apis"], [{"provider": "Deployment agent", "model": "/models/test", "endpoint": "http://localhost:9123"}])
        self.assertEqual(update_task_display(display, probe, 2, "complete")["apis"][0], display["apis"][0])

    def test_legacy_rows_use_latest_updated_work(self):
        display = summarize_saved_items([
            {"input": SAVED_REQUEST, "item_index": 0, "status": "failed", "updated_at": "2026-10-13T02:00:00Z"},
            {"payload": {"operation": "prompt.full", "request": {"input_text": "private"}},
             "index": 1, "status": "complete", "updated_at": "2026-10-13T01:00:00Z"},
        ])
        self.assertEqual(display["current_step"]["index"], 0)
        self.assertEqual(display["current_step"]["phase"], "failed")
        self.assertEqual(len(display["apis"]), 1)
        self.assertNotIn("private", json.dumps(display))

    def test_running_before_invoke_and_complete_metadata(self):
        store = MemoryStore()
        journal = self.journal(store)
        def invoke():
            self.assertEqual(store.events[-1], "progress:running")
            self.assertEqual(store.task["metadata"]["display"]["current_step"]["phase"], "running")
            return "answer"
        self.assertEqual(journal.call("llm.completion", PAYLOAD, invoke), "answer")
        self.assertEqual(store.items[0]["payload"], SAVED_REQUEST)
        journal.finish(True, {"new": 2})
        metadata = store.task["metadata"]
        self.assertEqual(metadata["display"]["current_step"]["phase"], "complete")
        self.assertEqual(metadata["final_session"], {"new": 2})
        self.assertEqual(metadata["audit"], "keep")
        self.assertIs(metadata["stop_requested"], True)

    def test_completed_and_incomplete_cached_replays_skip_writes(self):
        for status in ("complete", "incomplete"):
            with self.subTest(status=status):
                store = MemoryStore([SAVED_ITEM], status=status)
                journal = self.journal(store)
                self.assertEqual(journal.call("llm.completion", PAYLOAD, lambda: self.fail("cached API invoked")), "answer")
                self.assertEqual(store.events, [] if status == "complete" else ["claim"])
                journal.finish(True)
                self.assertEqual(store.task["metadata"]["final_session"], {"old": 1})
                if status == "complete":
                    self.assertEqual(store.events, [])
                else:
                    self.assertEqual(len(store.task["metadata"]["display"]["apis"]), 1)

    def test_resumed_legacy_call_retains_previous_api_history(self):
        store = MemoryStore([SAVED_ITEM])
        journal = self.journal(store)
        journal.call("llm.completion", PAYLOAD, lambda: self.fail("cached API invoked"))
        journal.call("single_choice.evaluate", {"provider": "Anthropic", "model": "judge"}, lambda: "B")
        self.assertEqual([api["provider"] for api in store.task["metadata"]["display"]["apis"]], ["OpenAI", "Anthropic"])

    def test_nested_local_checkpoint_is_suppressed(self):
        store = MemoryStore()
        journal = self.journal(store)
        with journal_scope(journal), checkpoint_scope("row", 0):
            self.assertEqual(checkpoint_local_value("prompt.full", {"input_text": "private"}, lambda: "frozen"), "frozen")
            self.assertEqual(store.task["metadata"]["display"]["apis"], [])
            result = checkpoint_call("llm.completion", PAYLOAD, lambda: checkpoint_local_value("prompt.full", {"input_text": "nested"}, lambda: "answer"))
            self.assertEqual(result, "answer")
        self.assertEqual(len(store.items), 2)
        self.assertEqual(len(store.task["metadata"]["display"]["apis"]), 1)
        self.assertEqual(store.task["metadata"]["display"]["current_step"]["operation"], "llm.completion")

    def test_api_exception_survives_failed_progress_save(self):
        for failed_progress in (False, True):
            with self.subTest(failed_progress=failed_progress):
                store = MemoryStore(fail_phase="failed" if failed_progress else None)
                journal = self.journal(store)
                def invoke():
                    raise ValueError("original API failure")
                with self.assertRaisesRegex(ValueError, "original API failure"):
                    journal.call("llm.completion", PAYLOAD, invoke)
                journal.finish(False)
                self.assertEqual(store.task["metadata"]["display"]["current_step"]["phase"], "failed")
                self.assertEqual(store.task["metadata"]["final_session"], {"old": 1})

    def test_failed_result_records_failed_step(self):
        store = MemoryStore()
        journal = self.journal(store)
        journal.call("llm.completion", PAYLOAD, lambda: "Error: provider rejected")
        journal.finish(True)
        self.assertEqual(store.items[0]["status"], "failed")
        self.assertEqual(store.task["status"], "incomplete")
        self.assertEqual(store.task["metadata"]["display"]["current_step"]["phase"], "failed")

    def test_progress_failure_and_stop_prevent_api_invocation(self):
        for stopped in (False, True):
            with self.subTest(stopped=stopped):
                store = MemoryStore(stop_on_progress=stopped, fail_phase=None if stopped else "running")
                journal = self.journal(store)
                expected = AnalysisCancelled if stopped else AnalysisCheckpointError
                with self.assertRaises(expected):
                    journal.call("llm.completion", PAYLOAD, lambda: self.fail("unfenced or stopped API invoked"))
                self.assertFalse(any(event.startswith("save:") for event in store.events))
                journal.finish(False)
                self.assertEqual(store.task["status"], "incomplete")
                if stopped:
                    self.assertIs(store.task["stop_requested"], True)


if __name__ == "__main__":
    unittest.main()
