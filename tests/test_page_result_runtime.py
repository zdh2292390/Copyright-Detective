"""Regressions for task/report ownership and provider routing, without live APIs."""
import ast
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
import inspect
import re
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
import unittest
from unittest.mock import Mock
import zlib

ROOT = Path(__file__).resolve().parents[1]


def load(path, names, env):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    found = {node.name: node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
    if set(found) != set(names):
        raise AssertionError(set(names) - set(found))
    module = ast.fix_missing_locations(ast.Module(body=[found[name] for name in names], type_ignores=[]))
    exec(compile(module, path, 'exec'), env)


class CheckpointFailure(ValueError):
    pass


class TextFailure(ValueError):
    pass


ADAPTER = {'deepcopy': deepcopy}
load('src/page_jobs.py', ['PageJobResult', '_TaskState', 'TaskUI'], ADAPTER)


class FakePdf:
    def __init__(self):
        self.text = []

    def __getattr__(self, name):
        if name == 'get_y':
            return lambda: 0
        if name == 'output':
            return lambda **kwargs: '\n'.join(self.text)
        def record(*args, **kwargs):
            self.text.append(str(kwargs.get('txt', '')))
        return record


class ForbiddenUI:
    @property
    def session_state(self):
        raise AssertionError('Report rendering read live Streamlit state')


def pdf_environment():
    env = {'Any': Any, 'Dict': Dict, 'Optional': Optional, 'AuditReportPDF': FakePdf,
           'st': ForbiddenUI(), '_sanitize_text_for_pdf': str}
    for name in ('_add_audit_cover_page', '_draw_page_border', '_add_executive_summary_section',
                 '_add_findings_section', '_add_methodology_section', '_add_conclusions_section',
                 '_add_image_to_pdf'):
        env[name] = lambda *args, **kwargs: None
    return env


class ReportOwnership(unittest.TestCase):
    def text_result(self):
        return {'type': 'single', 'text2': 'reference', 'generated_text': 'generated',
                'metrics_map': {'rouge_l': 0.2}, 'rouge_score': 0.2, 'jaccard_index': 0.1,
                'user_inputs': {'input_text': 'submitted source', 'provider': 'OpenAI', 'model': 'old-model'}}

    def test_text_pdf_uses_embedded_confidence_without_live_state(self):
        env = pdf_environment()
        load('src/pdf_preview.py', ['_add_blackbox_analysis_to_pdf', 'generate_text_memorization_pdf_report'], env)
        result = self.text_result()
        result['confidence_analysis_result'] = {'analysis_available': True, 'memorization_score': 0.8}
        report = env['generate_text_memorization_pdf_report'](result, 'Direct Probing', 'old-model').decode('latin-1')
        self.assertIn('Memorization Score: 80.0%', report)

    def test_legacy_text_result_does_not_borrow_another_tasks_confidence(self):
        env = pdf_environment()
        load('src/pdf_preview.py', ['_add_blackbox_analysis_to_pdf', 'generate_text_memorization_pdf_report'], env)
        report = env['generate_text_memorization_pdf_report'](self.text_result(), 'Direct Probing', 'model')
        self.assertNotIn(b'Black-Box Memorization Analysis', report)

    def narrative_environment(self):
        env = {'TextAnalysisError': TextFailure, 'checkpoint_scope': lambda *args: nullcontext(),
               'generate_llm_analysis': Mock(return_value='saved narrative'), 'is_api_error_response': lambda value: False}
        load('src/pages/text_memorization_detection.py', ['_save_text_report_narrative'], env)
        return env

    def test_report_rejects_current_key_from_a_different_provider_before_api_call(self):
        env = self.narrative_environment()
        with self.assertRaisesRegex(TextFailure, 'Select OpenAI'):
            env['_save_text_report_narrative'](self.text_result(), 'kimi-key', 'Direct Probing', 'new-model', 'Kimi', state={})
        env['generate_llm_analysis'].assert_not_called()

    def test_report_keeps_saved_model_when_sidebar_model_changes(self):
        env = self.narrative_environment()
        result = self.text_result()
        env['_save_text_report_narrative'](result, 'openai-key', 'changed-type', 'new-model', 'OpenAI', state={})
        self.assertEqual(env['generate_llm_analysis'].call_args.args[2], 'old-model')
        self.assertEqual(result['report_narrative'], 'saved narrative')

    def test_choice_pdf_and_details_use_evaluated_questions_after_next_task_is_prepared(self):
        env = {'deepcopy': deepcopy}
        load('src/pages/single_choice_detection.py', ['_single_choice_result_context'], env)
        questions = [{'question': 'submitted question', 'correct_option': 'A',
                      'options': [{'label': 'A', 'text': 'submitted answer'}]}]
        results = [[{**deepcopy(questions[0]), 'llm_choice': 'A', 'is_correct': True,
                     'option_probabilities': {'A': 0.75}}]]
        metadata = {'generated_mcqs': deepcopy(questions), 'document_text': 'submitted source'}
        questions[0]['question'] = 'new draft question'
        saved_questions, saved_source = env['_single_choice_result_context'](results, metadata)
        pdf = pdf_environment()
        load('src/pdf_preview.py', ['generate_single_choice_question_pdf_report'], pdf)
        report = pdf['generate_single_choice_question_pdf_report']({
            'results': results, 'metrics': {'overall_accuracy': 1.0, 'total_runs': 1},
            'generated_mcqs': saved_questions, 'document_text': saved_source,
        }, 'old-model', 'OpenAI', 'Input Text').decode('latin-1')
        self.assertIn('submitted question', report)
        self.assertIn('submitted source', report)
        self.assertNotIn('new draft question', report)
        self.assertIn('A: 75.0%', report)

    def test_legacy_choice_report_reconstructs_its_own_evidence(self):
        env = {'deepcopy': deepcopy}
        load('src/pages/single_choice_detection.py', ['_single_choice_result_context'], env)
        results = [[{'question': 'original question', 'options': [], 'correct_option': 'B'}]]
        questions, source = env['_single_choice_result_context'](results, {'model': 'old-model'})
        self.assertEqual(questions[0]['question'], 'original question')
        self.assertEqual(source, '')


class CompletedInferenceFailures(unittest.TestCase):
    def runner(self, *, checkpoint_at=None):
        env = dict(ADAPTER)
        calls = []
        def compare(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == checkpoint_at:
                raise CheckpointFailure('checkpoint could not save')
            return 'generated', {'rouge_l': 0.2}, ['captured logprobs']
        def inferences(count, analyze, save):
            for index in range(count):
                save(index, *analyze(index))
        def fail_report(*args, **kwargs):
            raise CheckpointFailure('report checkpoint failed')
        env.update(frozen_state={}, text1='source', text2='truth', inference_runs=2, temperature=0.2, top_p=0.8,
            continuation_method='Normal Continuation', prompt_mode='Zero-Shot', prompt_type='Direct Probing',
            model_choice='old-model', provider='OpenAI', api_key='key', ground_word_count=1, ground_char_count=5,
            input_word_count=1, input_char_count=6, chunk_size=1, target_word_count=None, prompt_instructions=None,
            custom_template=None, target_char_count=5,
            pd=SimpleNamespace(Timestamp=SimpleNamespace(now=lambda: SimpleNamespace(isoformat=lambda: 'now'))),
            compare_texts=compare, run_persuasion_probe=compare, run_text_inferences=inferences,
            TextAnalysisError=TextFailure, AnalysisCheckpointError=CheckpointFailure,
            TEXT_API_FAILURE_KEY='_text_memorization_api_failure',
            _save_text_report_narrative=fail_report,
            _run_blackbox_analysis_auto=lambda **kwargs: kwargs['state'].update(
                confidence_analysis_result={'analysis_available': True, 'memorization_score': 0.4}))
        load('src/pages/text_memorization_detection.py', ['_text_result_outputs', 'execute_text'], env)
        return env['execute_text'], calls

    def test_optional_report_checkpoint_failure_keeps_all_completed_inferences(self):
        run, calls = self.runner()
        result = run(lambda *args: None)
        self.assertFalse(result.complete)
        self.assertEqual(result['text_analysis_results']['analysis_progress']['completed_runs'], 2)
        self.assertIn('report checkpoint failed', result.error)
        self.assertEqual(len(calls), 2)

    def test_later_checkpoint_failure_keeps_previous_run_and_its_confidence(self):
        run, calls = self.runner(checkpoint_at=2)
        result = run(lambda *args: None)
        self.assertFalse(result.complete)
        self.assertEqual(result['text_analysis_results']['analysis_progress']['completed_runs'], 1)
        self.assertEqual(result['text_analysis_results']['confidence_analysis_result']['memorization_score'], 0.4)
        self.assertIsNone(result['text_pdf_report'])


class MinKProviderRouting(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.env = {'Optional': Optional, 'List': List, 'Dict': Dict, 'Any': Any, 'Tuple': Tuple,
                    'Real': Real, 'math': math, 'zlib': zlib, 'checkpoint_scope': lambda *args: nullcontext()}
        def probe(**kwargs):
            self.calls.append(kwargs)
            return [-1.0, -2.0], None
        self.env['_get_completion_logprobs'] = probe
        load('src/pages/unlearning_detection.py', ['_validate_logprob', '_validate_min_k_parameters', '_min_k_connection', 'run_min_k_prob_analysis'], self.env)

    def analyze(self, provider='OpenAI', **kwargs):
        return self.env['run_min_k_prob_analysis']('submitted source', 'provider-key', 'model', provider, **kwargs)

    def test_prompt_probe_uses_selected_cloud_provider_endpoint(self):
        for provider, endpoint in [('OpenAI', 'https://api.openai.com/v1'),
                                   ('OpenRouter', 'https://openrouter.ai/api/v1'),
                                   ('Kimi', 'https://api.moonshot.cn/v1')]:
            with self.subTest(provider=provider):
                self.calls.clear()
                result = self.analyze(provider)
                self.assertEqual(result['min_k_prob'], 2.0)
                self.assertEqual({call['base_url'] for call in self.calls}, {endpoint})
                self.assertEqual({call['api_key'] for call in self.calls}, {'provider-key'})

    def test_custom_agent_without_key_never_receives_cloud_provider_key(self):
        self.analyze('OpenAI', agent_url='https://agent.invalid/')
        self.assertEqual({call['base_url'] for call in self.calls}, {'https://agent.invalid/v1'})
        self.assertEqual({call['api_key'] for call in self.calls}, {'not-required'})

    def test_custom_agent_uses_its_own_key(self):
        self.analyze('Kimi', agent_url='https://agent.invalid', agent_key='agent-key')
        self.assertEqual({call['api_key'] for call in self.calls}, {'agent-key'})

    def test_local_prompt_probe_uses_frozen_connection(self):
        from src.api_task_context import api_task_scope
        with api_task_scope({'base_url': 'http://submitted-host:9000/v1', 'api_key': 'local-key'}):
            self.analyze('Local vLLM')
        self.assertEqual({call['base_url'] for call in self.calls}, {'http://submitted-host:9000/v1'})

    def test_unsupported_provider_is_rejected_before_transport(self):
        with self.assertRaisesRegex(ValueError, 'OpenAI-compatible'):
            self.analyze('Anthropic')
        self.assertEqual(self.calls, [])

    def test_missing_cloud_key_does_not_fall_back_to_server_environment(self):
        with self.assertRaisesRegex(ValueError, 'Configure an API key'):
            self.env['run_min_k_prob_analysis']('source', '', 'model', 'OpenAI')
        self.assertEqual(self.calls, [])


class GameErrors(unittest.TestCase):
    def test_shared_game_key_is_removed_from_worker_errors(self):
        env = {'contextmanager': contextmanager, 'AnalysisCheckpointError': CheckpointFailure}
        load('src/pages/copyright_game.py', ['_game_api_error_scope'], env)
        with self.assertRaises(RuntimeError) as failure:
            with env['_game_api_error_scope']('organizer-secret'):
                raise ValueError('Bad key organizer-secret')
        self.assertNotIn('organizer-secret', str(failure.exception))
        self.assertIn('[redacted]', str(failure.exception))

    def test_fatal_game_checkpoint_error_keeps_its_identity(self):
        env = {'contextmanager': contextmanager, 'AnalysisCheckpointError': CheckpointFailure}
        load('src/pages/copyright_game.py', ['_game_api_error_scope'], env)
        original = CheckpointFailure('fatal checkpoint')
        with self.assertRaises(CheckpointFailure) as failure:
            with env['_game_api_error_scope']('secret'):
                raise original
        self.assertIs(failure.exception, original)


class ScalingRuntime(unittest.TestCase):
    def test_repeated_harvest_does_not_copy_already_delivered_response_batches(self):
        @dataclass
        class Batch:
            errors: tuple = ()
        payload = Batch()
        reads = []
        saved = {'round_history': [{'id': 'round', 'job_keys': {'OpenAI': 'job'}}]}
        record = {'status': 'completed', 'finished_at': 'finished', 'result': payload}
        def get_job(key, *, include_result=True):
            reads.append(include_result)
            return dict(record) if include_result else {k: v for k, v in record.items() if k != 'result'}
        publish = Mock()
        env = {'st': SimpleNamespace(session_state=saved), 'ScalingBatch': Batch,
               'get_background_job': get_job, '_key': lambda value: value,
               '_round_history': lambda: saved['round_history'], '_format_kst': lambda value: value,
               '_job_key': lambda provider: 'legacy-job', '_record_round_provider': publish}
        load('src/pages/copyright_game2.py', ['_harvest_provider_job'], env)
        env['_harvest_provider_job']('OpenAI', 'job')
        env['_harvest_provider_job']('OpenAI', 'job')
        self.assertEqual(reads.count(True), 1)
        self.assertEqual(publish.call_count, 1)
        self.assertIs(publish.call_args.kwargs['result'], payload)

    def test_scaling_wrapper_preserves_fatal_checkpoint_error(self):
        @dataclass
        class Batch:
            errors: tuple = ()
        pending = []
        saved = {'round_history': [{'id': 'round', 'providers': {'OpenAI': {'stopped': True}}}]}
        original = CheckpointFailure('fatal checkpoint')
        env = {'Optional': Optional, 'ScalingBatch': Batch, 'replace': replace,
               'datetime': datetime, '_KST': timezone.utc, 'st': SimpleNamespace(session_state=saved),
               'AnalysisCheckpointError': CheckpointFailure, '_job_key': lambda *args: 'job',
               '_key': lambda value: value, '_round_history': lambda: saved['round_history'],
               'run_provider_scaling': Mock(side_effect=original),
               'submit_background_job': lambda key, label, runner: pending.append(runner) or True}
        load('src/pages/copyright_game2.py', ['_start_scaling'], env)
        env['_start_scaling']('OpenAI', round_id='round', api_key='key', model='model', runs=1,
                              temperature=0.2, top_p=0.9, prompt_method='Recall',
                              prompt_mode='Direct Probing', custom_template=None)
        self.assertNotIn('OpenAI', saved['round_history'][0]['providers'])
        with self.assertRaises(CheckpointFailure) as failure:
            pending[0](lambda *args: None)
        self.assertIs(failure.exception, original)


class SystemicChoiceFailures(unittest.TestCase):
    def coordinator(self, responses):
        questions = [{'question': f'Question {index}', 'correct_option': 'A',
                      'options': [{'label': 'A', 'text': 'Answer'}]} for index in range(20)]
        evaluate = Mock(side_effect=responses)
        env = {'Any': Any, 'Callable': Callable, 'Dict': Dict, 'List': List, 'Optional': Optional,
               'OPTION_LABELS': ['A', 'B', 'C', 'D'], 'evaluate_single_choice_question': evaluate,
               'report_analysis_progress': lambda *args, **kwargs: None,
               '_raise_checkpoint_error': lambda exc: None}
        load('src/direct_recall/single_choice.py', ['_sc_error_result', 'run_single_choice_evaluation',
                                                  'summarize_single_choice_results'], env)
        return env, evaluate, questions

    def test_auth_and_missing_model_failures_stop_after_preserving_current_error(self):
        for message in ('HTTP 401 provider-key', '403 Forbidden', '404 model not found',
                        'model_not_found', 'Model test does not exist'):
            with self.subTest(message=message):
                env, evaluate, questions = self.coordinator([
                    {'choice': 'A'}, {'choice': '?', 'error': message, 'raw_response': message}])
                results = env['run_single_choice_evaluation'](questions, 'provider-key', 'model', 'OpenAI', num_runs=500)
                self.assertEqual(evaluate.call_count, 2)
                self.assertEqual(len(results[0]), 2)
                self.assertTrue(results[0][0]['is_correct'])
                self.assertIsNone(results[0][1]['is_correct'])
                self.assertNotIn('provider-key', str(results[0][1]))

    def test_ordinary_invalid_choice_does_not_stop_the_remaining_questions(self):
        env, evaluate, questions = self.coordinator(lambda *args, **kwargs: {'choice': 'X'})
        results = env['run_single_choice_evaluation'](questions[:2], 'key', 'model', 'OpenAI', num_runs=2)
        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual(sum(map(len, results)), 4)

    def test_transient_timeout_does_not_stop_the_batch(self):
        env, evaluate, questions = self.coordinator(lambda *args, **kwargs: {
            'choice': '?', 'error': 'Request timed out'})
        env['run_single_choice_evaluation'](questions[:2], 'key', 'model', 'OpenAI', num_runs=2)
        self.assertEqual(evaluate.call_count, 4)

    def test_task_retains_attempted_and_planned_counts_after_early_stop(self):
        env, evaluate, questions = self.coordinator([
            {'choice': 'A'}, {'choice': '?', 'error': '401 Unauthorized'}])
        env.update(ADAPTER)
        env.update(deepcopy=deepcopy, AnalysisCheckpointError=CheckpointFailure, api_key='key',
            model_choice='model', provider='OpenAI', source_mode='Input Text',
            frozen_state={'sc_generated_mcqs': questions, 'sc_document_text': 'original source',
                          'sc_eval_runs': 500, 'sc_eval_temperature': 0.2, 'sc_eval_top_p': 0.8})
        load('src/pages/single_choice_detection.py', ['execute_evaluation'], env)
        outputs = env['execute_evaluation'](lambda *args: None)
        self.assertFalse(outputs.complete)
        self.assertEqual(outputs['sc_evaluation_metadata']['requested_attempts'], 10000)
        self.assertEqual(outputs['sc_evaluation_metadata']['attempted_attempts'], 2)
        self.assertEqual(len(outputs['sc_evaluation_metadata']['generated_mcqs']), 20)
        self.assertEqual(evaluate.call_count, 2)

    def test_open_ended_pdf_distinguishes_attempted_from_unstarted_work(self):
        env = pdf_environment()
        env.update(List=List, pd=SimpleNamespace(Timestamp=SimpleNamespace(
            now=lambda: SimpleNamespace(strftime=lambda pattern: 'now'))))
        load('src/pdf_preview.py', ['generate_open_ended_question_pdf_report'], env)
        report = env['generate_open_ended_question_pdf_report'](
            [[{'error': '401 Unauthorized'}]],
            {'total_attempted': 1, 'total_evaluations': 0, 'failed_evaluations': 1},
            [], 'model', 'Input Text', 20, 500, 0.2, 0.8,
        ).decode('latin-1')
        self.assertIn('1/10000 planned evaluations attempted', report)
        self.assertIn('9999 not attempted', report)


class MinKInputBoundaries(unittest.TestCase):
    setUp = MinKProviderRouting.setUp
    analyze = MinKProviderRouting.analyze

    def test_invalid_numeric_settings_do_not_start_a_paid_probe(self):
        for settings in ({'k_percentage': 0}, {'k_percentage': 101}, {'k_percentage': float('nan')},
                         {'k_percentage': True}, {'temperature': float('inf')}, {'top_p': 1.1},
                         {'max_tokens': 0}, {'max_tokens': True}, {'max_tokens': 4001}):
            with self.subTest(settings=settings):
                self.calls.clear()
                with self.assertRaises(ValueError):
                    self.analyze(**settings)
                self.assertEqual(self.calls, [])

    def test_empty_source_and_model_are_rejected_before_transport(self):
        for prompt, model in (('', 'model'), ('\x00', 'model'), ('source', '   ')):
            with self.subTest(prompt=prompt, model=model):
                with self.assertRaises(ValueError):
                    self.env['run_min_k_prob_analysis'](prompt, 'key', model, 'OpenAI')
                self.assertEqual(self.calls, [])


class MinKBatchFailures(unittest.TestCase):
    def execute(self, failures):
        calls = []
        def analyze(**kwargs):
            calls.append(kwargs)
            error = failures.get(len(calls))
            if error:
                raise RuntimeError(error)
            return {'min_k_prob': 0.5, 'perplexity': 2.0}
        state = {'min_k_user_input_dataset_name': 'submitted dataset'}
        ui = ADAPTER['TaskUI'](state, lambda *args: None)
        env = {'Real': Real, 'math': math, 'AnalysisCheckpointError': CheckpointFailure,
               'checkpoint_scope': lambda *args: nullcontext(), 'run_min_k_prob_analysis': analyze,
               'compute_evaluation_metrics': lambda rows: {'num_examples': len(rows)}, 'os': __import__('os')}
        load('src/pages/unlearning_detection.py', ['_get_mode_key', '_run_batch_evaluation'], env)
        env['_run_batch_evaluation']([{'text': 'source', 'label': 0} for _ in range(10)],
            'org/model', 'key', 'model', 'OpenAI', 10, 100, 'User Input', task_ui=ui)
        return ui.session_state, calls

    def test_auth_failure_preserves_previous_success_and_leaves_remaining_examples_pending(self):
        state, calls = self.execute({2: '401 Unauthorized'})
        progress = state['min_k_user_input_batch_progress']
        self.assertEqual(len(calls), 2)
        self.assertEqual(progress['total'], 10)
        self.assertEqual(progress['completed'], 1)
        self.assertEqual(len(progress['failures']), 1)
        self.assertEqual(progress['status'], 'incomplete')
        self.assertEqual(state['min_k_user_input_evaluation_results']['num_examples'], 1)

    def test_transient_batch_timeout_preserves_later_successes(self):
        state, calls = self.execute({1: 'Request timed out'})
        self.assertEqual(len(calls), 10)
        self.assertEqual(state['min_k_user_input_batch_progress']['completed'], 9)


class SamplingRequests(unittest.TestCase):
    def base_env(self):
        from src.openai_utils import apply_openai_short_answer_compat
        from src.kimi_utils import normalize_kimi_sampling_params, kimi_request_extra_body
        return {'Any': Any, 'Dict': Dict, 'Optional': Optional,
                'apply_openai_short_answer_compat': apply_openai_short_answer_compat,
                'normalize_kimi_sampling_params': normalize_kimi_sampling_params,
                'kimi_request_extra_body': kimi_request_extra_body,
                'limit_api_concurrency': lambda **kwargs: nullcontext(),
                'provider_call': lambda invoke, **kwargs: invoke(**kwargs),
                '_extract_option_from_text': lambda text: text if text in 'ABCD' else None,
                '_parse_openai_top_logprobs': lambda values: None,
                'is_api_error_response': lambda text: str(text).startswith('Error')}

    def test_legacy_openai_probe_sends_selected_sampling_values(self):
        requests = []
        class Client:
            def __init__(self, **kwargs):
                self.completions = SimpleNamespace(create=self.create)
            def create(self, **kwargs):
                requests.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(text='B', logprobs=None)])
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        env = self.base_env()
        env['OpenAI'] = Client
        load('src/direct_recall/single_choice.py', ['_sc_error_result', '_try_openai_style_completion'], env)
        result = env['_try_openai_style_completion']('question', 'key', 'gpt-4o-mini', 'OpenAI', 0.42, 0.66)
        self.assertEqual(requests[0]['temperature'], 0.42)
        self.assertEqual(requests[0]['top_p'], 0.66)
        self.assertEqual(result['generation_parameters'], {'temperature': 0.42, 'top_p': 0.66})
        self.assertTrue(result['generation_parameters_observed'])

    def test_basic_fallback_uses_selected_values_and_labels_normalization_as_unobserved(self):
        env = self.base_env()
        invoke = Mock(return_value='B')
        env['get_llm_completion'] = invoke
        load('src/direct_recall/single_choice.py', ['_sc_error_result', '_evaluate_with_basic_completion'], env)
        result = env['_evaluate_with_basic_completion']('question', 'key', 'gpt-4o-mini', 'OpenAI', 0.42, 0.66)
        self.assertEqual(invoke.call_args.kwargs['temperature'], 0.42)
        self.assertEqual(invoke.call_args.kwargs['top_p'], 0.66)
        self.assertFalse(result['generation_parameters_observed'])

    def test_kimi_fallback_keeps_documented_model_sampling_normalization(self):
        env = self.base_env()
        env['get_llm_completion'] = Mock(return_value='B')
        load('src/direct_recall/single_choice.py', ['_sc_error_result', '_evaluate_with_basic_completion'], env)
        result = env['_evaluate_with_basic_completion']('question', 'key', 'kimi-k2.6', 'Kimi', 0.42, 0.66)
        self.assertEqual(result['generation_parameters'], {'temperature': 0.6, 'top_p': 0.95})

    def test_anthropic_observer_records_final_compatibility_retry_without_wrapping_sdk_method(self):
        sent = []
        class Messages:
            def create(self, **kwargs):
                sent.append(kwargs)
                if 'top_p' in kwargs:
                    raise ValueError('400: cannot combine temperature and top_p')
                return SimpleNamespace(content=[SimpleNamespace(type='text', text='B')])
        class Client:
            def __init__(self, **kwargs):
                self.messages = Messages()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        def provider_call(invoke, **kwargs):
            self.assertIsInstance(invoke.__self__, Messages)
            return invoke(**kwargs)
        env = self.base_env()
        env.update(deepcopy=deepcopy, inspect=inspect, re=re, Mapping=Mapping,
                   check_cancellation=lambda: None, provider_call=provider_call, Anthropic=Client)
        module = ast.parse((ROOT / 'src/anthropic_utils.py').read_text())
        assignments = [node for node in module.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in {
                           '_SAMPLING_RESTRICTED_MODELS', '_ALWAYS_THINKING_MODELS', '_LEGACY_SAMPLING_MODELS'}
                               for target in node.targets)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[])), 'constants', 'exec'), env)
        load('src/anthropic_utils.py', ['_matches_model', 'anthropic_short_answer_error',
             'apply_anthropic_request_compat', 'create_anthropic_message', '_is_sampling_combination_error',
             '_field', 'extract_anthropic_response_text'], env)
        load('src/direct_recall/single_choice.py', ['_sc_error_result', '_try_anthropic_style_completion'], env)
        result = env['_try_anthropic_style_completion']('question', 'key', 'claude-sonnet-4-5', 0.42, 0.66)
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]['top_p'], 0.66)
        self.assertNotIn('top_p', sent[1])
        self.assertEqual(result['generation_parameters'], {'temperature': 0.42, 'top_p': None})
        self.assertTrue(result['generation_parameters_observed'])


if __name__ == '__main__':
    unittest.main()
