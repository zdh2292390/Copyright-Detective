"""Unlearning request, partial-result, and scope regressions without model calls."""

import base64
from contextlib import nullcontext
import io
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PyPDF2 import PdfReader
from streamlit.testing.v1 import AppTest
import requests
import src.pages.unlearning_detection as page
import src.unlearning_detection.remote_execution as remote
import src.unlearning_detection.unlearning as probes
from src.pdf_preview import generate_min_k_prob_pdf_report


def response(code, data):
    item = Mock(status_code=code)
    item.json.return_value = data
    return item


_png_buffer = io.BytesIO()
Image.new('RGB', (1, 1), 'white').save(_png_buffer, format='PNG')
PNG_BYTES = _png_buffer.getvalue()


def payload():
    return {"status": "success", "data": {"visualizations": [{"title": "Plot", "data": base64.b64encode(PNG_BYTES).decode(), "mime_type": "image/png"}], "warnings": []}}


class RemoteAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.code = patch.object(remote, "read_analysis_code_files", return_value={"analysis.py": "pass"})
        self.code.start()
        self.addCleanup(self.code.stop)
        self.clock = SimpleNamespace(value=0)
        self.monotonic = patch.object(remote.time, "monotonic", side_effect=lambda: self.clock.value)
        self.sleep = patch.object(remote.time, "sleep", side_effect=lambda seconds: setattr(self.clock, "value", self.clock.value + seconds))
        self.monotonic.start(); self.sleep.start()
        self.addCleanup(self.monotonic.stop); self.addCleanup(self.sleep.stop)

    def run_analysis(self, **options):
        return remote.execute_analysis_remotely("https://agent.example/", "cka", "base/model", "updated/model", ["query"], **options)

    def test_legacy_sync_artifacts_are_preserved_and_response_closed(self):
        item = response(200, payload())
        with patch.object(remote.requests, "post", return_value=item) as submit:
            result = self.run_analysis(api_key="key")
        self.assertEqual(result.visualizations[0].data, PNG_BYTES)
        self.assertEqual(submit.call_args.args[0], "https://agent.example/run_analysis")
        self.assertEqual(submit.call_args.kwargs["headers"], {"X-API-Key": "key"})
        item.close.assert_called_once()

    def test_temporary_poll_errors_retry_without_duplicate_submission(self):
        status = [response(503, {}), response(202, {"status": "running"}), response(200, {"status": "completed", "result": payload()})]
        with patch.object(remote.requests, "post", return_value=response(202, {"task_id": "task-1"})) as submit, patch.object(remote.requests, "get", side_effect=status) as poll:
            result = self.run_analysis(timeout=3600, max_poll_time=10)
        self.assertEqual(len(result.visualizations), 1)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(poll.call_count, 3)
        self.assertTrue(all(call.kwargs["timeout"] <= 10 for call in poll.call_args_list))
        for item in status:
            item.close.assert_called_once()

    def test_poll_deadline_bounds_requests_and_sleep(self):
        with patch.object(remote.requests, "post", return_value=response(202, {"task_id": "task-1"})), patch.object(remote.requests, "get", side_effect=requests.exceptions.Timeout()) as poll:
            with self.assertRaisesRegex(RuntimeError, "Task task-1 may still be running"):
                self.run_analysis(timeout=3600, max_poll_time=5, poll_interval=2)
        self.assertEqual(self.clock.value, 5)
        self.assertEqual([call.kwargs["timeout"] for call in poll.call_args_list], [5, 3, 1])

    def test_submission_timeout_never_blindly_reposts(self):
        with patch.object(remote.requests, "post", side_effect=requests.exceptions.Timeout()) as submit, patch.object(remote.requests, "get") as poll:
            with self.assertRaisesRegex(RuntimeError, "may have accepted"):
                self.run_analysis()
        submit.assert_called_once()
        poll.assert_not_called()

    def test_invalid_submission_and_status_objects_raise_clear_errors(self):
        for code, body in ((202, {}), (200, []), (200, {"status": "success"}), (401, {})):
            with self.subTest(body=body), patch.object(remote.requests, "post", return_value=response(code, body)), self.assertRaises(RuntimeError):
                self.run_analysis()
        bad = response(202, {})
        bad.json.side_effect = ValueError("not JSON")
        with patch.object(remote.requests, "post", return_value=bad), self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            self.run_analysis()
        for body in ({"status": "unknown"}, {"status": "completed", "result": None}, {"status": "failed", "error": "server error"}):
            with self.subTest(body=body), patch.object(remote.requests, "post", return_value=response(202, {"task_id": "task-1"})), patch.object(remote.requests, "get", return_value=response(200, body)), self.assertRaises(RuntimeError):
                self.run_analysis()

    def test_auth_poll_failure_is_immediate(self):
        with patch.object(remote.requests, "post", return_value=response(202, {"task_id": "task-1"})), patch.object(remote.requests, "get", return_value=response(403, {})) as poll, self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            self.run_analysis()
        poll.assert_called_once()

    def test_invalid_artifact_is_warned_without_losing_valid_sibling(self):
        data = payload()["data"]
        data["visualizations"].append({"data": "not base64!"})
        result = remote._parse_analysis_result(data)
        self.assertEqual(len(result.visualizations), 1)
        self.assertTrue(result.warnings)
        with self.assertRaisesRegex(RuntimeError, "none contained usable"):
            remote._parse_analysis_result({"visualizations": [{"data": "garbage!"}]})
        with self.assertRaises(RuntimeError):
            remote._parse_analysis_result({"warnings": "a warning"})
        self.assertTrue(remote._parse_analysis_result({}).warnings)

    def test_valid_base64_with_invalid_image_bytes_is_not_an_artifact(self):
        bad = base64.b64encode(b"HTML error page disguised as an image").decode()
        with self.assertRaisesRegex(RuntimeError, "none contained usable"):
            remote._parse_analysis_result({"visualizations": [{"data": bad, "mime_type": "image/png"}]})
        pdf = remote._parse_analysis_result({"visualizations": [{"data": base64.b64encode(b"pdf download bytes").decode(), "mime_type": "application/pdf"}]})
        self.assertEqual(pdf.visualizations[0].data, b"pdf download bytes")

    def test_svg_artifacts_remain_supported_with_xml_validation(self):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" /></svg>'
        item = {"data": base64.b64encode(svg).decode(), "mime_type": "image/svg+xml"}
        result = remote._parse_analysis_result({"visualizations": [item]})
        self.assertEqual(result.visualizations[0].data, svg)
        self.assertEqual(result.visualizations[0].mime_type, "image/svg+xml")
        for invalid in (b'<svg', b'<html>error page</html>'):
            item["data"] = base64.b64encode(invalid).decode()
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, "none contained usable"):
                remote._parse_analysis_result({"visualizations": [item]})

    def test_sync_and_async_completed_errors_redact_current_agent_key(self):
        failed = {"status": "error", "msg": "private-agent-key failed"}
        with patch.object(remote.requests, "post", return_value=response(200, failed)):
            with self.assertRaises(RuntimeError) as error:
                self.run_analysis(api_key="private-agent-key")
        self.assertNotIn("private-agent-key", str(error.exception))
        with patch.object(remote.requests, "post", return_value=response(202, {"task_id": "task-1"})), patch.object(remote.requests, "get", return_value=response(200, {"status": "completed", "result": failed})):
            with self.assertRaises(RuntimeError) as error:
                self.run_analysis(api_key="private-agent-key")
        self.assertNotIn("private-agent-key", str(error.exception))

    def test_existing_fim_limits_and_other_parameters_are_unchanged(self):
        with patch.object(remote.requests, "post", return_value=response(200, payload())) as submit:
            remote.execute_analysis_remotely("https://agent", "fim", "ref", "updated", ["q"], batch_size=4, num_batches=10, max_length=256)
        sent = submit.call_args.kwargs["json"]
        self.assertEqual((sent["batch_size"], sent["num_batches"], sent["max_length"]), (1, 3, 256))


class MinKRobustnessTests(unittest.TestCase):
    def fake_ui(self, state=None):
        return SimpleNamespace(session_state={} if state is None else state, spinner=lambda *args: nullcontext(), progress=Mock(return_value=Mock()), empty=Mock(return_value=Mock()), warning=Mock(), error=Mock(), success=Mock())

    def run_batch(self, batch, side_effect):
        ui = self.fake_ui()
        with patch.object(page, "st", ui), patch.object(page, "detection_job", return_value=nullcontext()), patch.object(page, "run_min_k_prob_analysis", side_effect=side_effect) as analyze:
            page._run_batch_evaluation(batch, "gpt2", "key", "model", "OpenAI", 10, 50)
        ui.analysis_calls = analyze.call_args_list
        return ui

    def test_echo_request_timeout_response_validation_and_client_cleanup(self):
        client = Mock()
        client.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(token_logprobs=[None, -1.0, -2.0]))])
        with patch.object(page.openai, "OpenAI", return_value=client) as constructor:
            values, error = page._get_completion_logprobs("Prompt", "key", "model")
        self.assertEqual(values, [-1.0, -2.0]); self.assertIsNone(error)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 120)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        self.assertEqual(client.completions.create.call_args.kwargs["max_tokens"], 0)
        self.assertTrue(client.completions.create.call_args.kwargs["echo"])
        client.close.assert_called_once()
        for invalid in ([None], [math.nan], [math.inf], [1.0]):
            client.completions.create.return_value.choices[0].logprobs.token_logprobs = invalid
            with self.subTest(invalid=invalid), patch.object(page.openai, "OpenAI", return_value=client):
                values, error = page._get_completion_logprobs("Prompt", "key", "model")
            self.assertEqual(values, []); self.assertIsNotNone(error)

    def test_valid_min_k_formula_and_sampling_settings_are_preserved(self):
        with patch.object(page, "_get_completion_logprobs", side_effect=[([-1.0, -2.0, -3.0, -4.0], None), ([-2.0, -4.0], None)]), patch.object(page, "get_llm_completion") as fallback:
            result = page.run_min_k_prob_analysis("Prompt", "key", "model", "OpenAI", k_percentage=50)
        fallback.assert_not_called()
        self.assertEqual(result["min_k_prob"], 3.5)
        self.assertEqual(result["min_k_probs"]["Min_10%_Prob"], 4.0)
        self.assertAlmostEqual(result["perplexity"], math.exp(2.5))
        self.assertAlmostEqual(result["ppl_lowercase"], -1.2)
        self.assertEqual(result["generated_text"], "Prompt")

    def test_missing_and_invalid_chat_logprobs_are_not_fabricated_zero(self):
        for tokens in ([{"token": "word"}], [{"token": "word", "logprob": math.nan}], [{"token": "word", "logprob": 0.5}]):
            with self.subTest(tokens=tokens), patch.object(page, "_get_completion_logprobs", return_value=([], "unsupported")), patch.object(page, "get_llm_completion", return_value=("generated", tokens)), self.assertRaises(ValueError):
                page.run_min_k_prob_analysis("Prompt", "key", "model", "OpenAI")

    def test_large_negative_logprobs_keep_valid_min_k_when_perplexity_overflows(self):
        with patch.object(page, "_get_completion_logprobs", return_value=([-1000.0], None)):
            result = page.run_min_k_prob_analysis("Prompt", "key", "model", "OpenAI")
        self.assertEqual(result["min_k_prob"], 1000.0)
        self.assertTrue(math.isinf(result["perplexity"]))
        self.assertIsNone(result["ppl_lowercase"])

    def test_failed_new_batch_clears_previous_success(self):
        state = {"min_k_predefined_evaluation_results": {"auc": 1.0}, "min_k_predefined_batch_results": [{"old": "result"}]}
        ui = self.fake_ui(state)
        with patch.object(page, "st", ui), patch.object(page, "detection_job", return_value=nullcontext()), patch.object(page, "run_min_k_prob_analysis", side_effect=ValueError("failure")):
            page._run_batch_evaluation([{"text": "new", "label": 0}], "gpt2", "key", "model", "OpenAI", 10, 50)
        self.assertIsNone(state["min_k_predefined_evaluation_results"])
        self.assertEqual(state["min_k_predefined_batch_results"], [])
        self.assertEqual(state["min_k_predefined_batch_progress"]["status"], "incomplete")

    def test_partial_batch_keeps_successes_counts_and_decoding_parameters(self):
        good = {"min_k_prob": 2.0, "perplexity": 3.0, "min_k_probs": {"Min_10%_Prob": 2.0}}
        ui = self.run_batch([{"text": str(i), "label": i % 2} for i in range(3)], [ValueError("failure key"), good, good])
        state = ui.session_state
        scope = state["min_k_predefined_evaluation_results"]["analysis_progress"]
        self.assertEqual((scope["total"], scope["completed"], len(scope["failures"])), (3, 2, 1))
        self.assertEqual(scope["status"], "incomplete")
        self.assertNotIn("key", scope["failures"][0])
        self.assertEqual(len(state["min_k_predefined_batch_results"]), 2)
        for call in ui.analysis_calls:
            self.assertEqual((call.kwargs['temperature'], call.kwargs['top_p'], call.kwargs['max_tokens'], call.kwargs['k_percentage']), (1.0, 1.0, 50, 10))

    def test_optional_incomplete_metric_cannot_misalign_labels_or_drop_primary_scores(self):
        result = page.compute_evaluation_metrics([
            {"label": 0, "pred": {"Min_k%_Prob": 4.0, "optional": 2.0}},
            {"label": 1, "pred": {"Min_k%_Prob": 1.0}},
        ])
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["num_examples"], 2)
        self.assertNotIn("optional", result["all_metrics"])
        self.assertTrue(result["warnings"])

    def test_invalid_single_run_does_not_show_previous_success(self):
        state = {"min_k_user_input_last_result": {"old": True}}
        with patch.object(page, "st", self.fake_ui(state)):
            page._run_single_analysis("Prompt", "invalid name", "key", "model", "OpenAI", 10, 0.7, 0.9, 50)
        self.assertIsNone(state["min_k_user_input_last_result"])


class ProbeAndArtifactTests(unittest.TestCase):
    def test_probe_exception_isolated_and_secret_redacted(self):
        with patch.object(probes, "get_llm_completion", side_effect=[TimeoutError("private-key timeout"), "valid recall"]) as call:
            result = probes.run_unlearning_detection("private-key", "model", "OpenAI", target_description="source", strategy_ids=["direct_question", "indirect_summary"])
        self.assertEqual(len(result.results), 2)
        self.assertTrue(result.results[0].error)
        self.assertNotIn("private-key", result.results[0].error)
        self.assertEqual(result.results[1].response, "valid recall")
        self.assertEqual(call.call_args.kwargs["request_timeout"], 120)

    def test_remote_dependency_failure_does_not_load_local_models(self):
        with patch.dict(sys.modules, {"src.unlearning_detection.remote_execution": None}), patch.object(probes, "_run_feature_analysis") as local, self.assertRaisesRegex(RuntimeError, "Remote representational"):
            probes.run_representational_analysis(feature="cka", model_reference_path="ref", model_path="updated", query="q", agent_url="https://agent")
        local.assert_not_called()

    def test_stale_file_not_reported_as_current_but_overwrite_detected(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "output.pdf"
            target.write_bytes(b"old")
            options = dict(feature="cka", model_reference_path="ref", model_path="updated", query="q", output_path=str(target), device="cpu")
            with patch.object(probes, "_run_feature_analysis", return_value=None):
                result = probes.run_representational_analysis(**options)
            self.assertFalse(result.has_artifacts)
            with patch.object(probes, "_run_feature_analysis", side_effect=lambda **kwargs: target.write_bytes(b"new artifact")):
                result = probes.run_representational_analysis(**options)
            self.assertTrue(result.has_artifacts)

    def test_partial_min_k_pdf_records_scope_and_excludes_full_batch_claim(self):
        evaluation = {"auc": 0.8, "accuracy": 0.7, "tpr_at_5fpr": 0.6, "num_examples": 2, "analysis_progress": {"total": 3, "completed": 2, "failures": {0: "HTTP 429"}, "status": "incomplete"}}
        data = generate_min_k_prob_pdf_report(evaluation, [{}, {}], "model", "OpenAI")
        text = " ".join(" ".join(item.extract_text() or "" for item in PdfReader(io.BytesIO(data)).pages).split())
        self.assertIn("INCOMPLETE ANALYSIS: 2 of 3 requested examples", text)
        self.assertIn("HTTP 429", text)
        self.assertIn("Failed or Skipped Examples", text)
        evaluation["analysis_progress"] = {"total": 2, "completed": 2, "failures": {}, "status": "complete"}
        data = generate_min_k_prob_pdf_report(evaluation, [{}, {}], "model", "OpenAI")
        text = " ".join(item.extract_text() or "" for item in PdfReader(io.BytesIO(data)).pages)
        self.assertNotIn("INCOMPLETE ANALYSIS", text)


REPRESENTATIONAL_APP = """
import streamlit as st
from unittest.mock import Mock, patch
import src.pages.unlearning_detection as page
from src.unlearning_detection.unlearning import RepresentationalAnalysisResult

def analyze(**kwargs):
    st.session_state['test_sent_key'] = kwargs.get('agent_key')
    if st.session_state.get('test_fail'):
        raise RuntimeError('Failure involving secret-agent-key')
    return RepresentationalAnalysisResult('fim', 'Fisher Information Matrix', '', [], [], [])

reply = Mock(status_code=200)
reply.json.return_value = {'status': 'success'}
with patch.object(page, 'run_representational_analysis', side_effect=analyze), patch.object(page.requests, 'post', return_value=reply), patch.object(page, 'is_representational_analysis_available', return_value=True):
    page.render_representational_analysis_page('sidebar-key', 'model', 'OpenAI')
"""


class RepresentationalPageTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_string(REPRESENTATIONAL_APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        app.text_input(key='unlearn_deploy_agent_url_input').set_value('https://agent')
        app.text_input(key='unlearn_deploy_agent_key_input').set_value('secret-agent-key')
        app.text_input(key='representational_reference_model').set_value('gpt2')
        app.text_input(key='representational_updated_model').set_value('gpt2')
        app.text_area(key='representational_query_text').set_value('query').run()
        return app

    def test_request_credentials_not_cached_or_rendered_and_no_artifacts_not_success(self):
        app = self.make_app()
        app.button(key='unlearn_rep_submit_run').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['test_sent_key'], 'secret-agent-key')
        self.assertNotIn('agent_key', app.session_state['unlearn_last_request'])
        self.assertFalse(any('Completed Fisher' in str(item.value) for item in app.success))
        self.assertTrue(any('no usable artifacts' in str(item.value) for item in app.warning))
        for item in app.get('json'):
            self.assertNotIn('secret-agent-key', str(item.value))

    def test_invalid_or_failed_rerun_clears_prior_success(self):
        app = self.make_app()
        app.button(key='unlearn_rep_submit_run').click().run()
        self.assertIsNotNone(app.session_state['unlearn_last_result'])
        app.text_area(key='representational_query_text').set_value('').run()
        app.button(key='unlearn_rep_submit_run').click().run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state['unlearn_last_result'])
        app.text_area(key='representational_query_text').set_value('query').run()
        app.session_state['test_fail'] = True
        app.button(key='unlearn_rep_submit_run').click().run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state['unlearn_last_result'])
        self.assertIsNone(app.session_state['unlearn_last_request'])


if __name__ == "__main__":
    unittest.main()
