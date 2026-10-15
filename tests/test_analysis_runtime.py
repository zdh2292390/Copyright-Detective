"""Runtime regressions with deterministic provider fakes and no UI/network."""
import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace, ModuleType
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Callable
import math
import re
import unittest
import sys
from unittest.mock import patch
from collections.abc import Mapping
from numbers import Real

from src.analysis_checkpoints import AnalysisCheckpointError
from src.analysis_outcomes import task_outcome_errors

ROOT = Path(__file__).resolve().parents[1]


def load_nodes(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names), (filename, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def task_runtime():
    ns = {"__name__": __name__, "deepcopy": deepcopy}
    load_nodes('src/page_jobs.py', {'PageJobResult', '_TaskState', 'TaskUI'}, ns)
    load_nodes('src/ui.py', {'_finish_api_task'}, ns)
    return ns


class ResultClassification(unittest.TestCase):
    def test_guest_qa_partial_is_not_complete(self):
        runtime = task_runtime()
        ui = runtime['TaskUI']({}, lambda *args: None)
        rows = [[{'answer': 'ok'}, {'error': 'provider failure'}]]
        result = runtime['_finish_api_task'](ui, {'qa_evaluation_results': rows})
        self.assertFalse(result.complete)
        self.assertEqual(result['qa_evaluation_results'], rows)
        self.assertIn('1 knowledge', result.error)

    def test_all_qa_errors_are_not_successful_scores(self):
        errors = task_outcome_errors({'qa_evaluation_results': [[{'error': 'unauthorized'}]]})
        self.assertTrue(errors)

    def test_failed_judge_preserves_answer_and_marks_partial(self):
        errors = task_outcome_errors({'qa_evaluation_results': [[{'answer': 'ok', 'llm_judge_error': 'timeout'}]]})
        self.assertEqual(len(errors), 1)
        self.assertIn('judge', errors[0])

    def test_successful_qa_has_no_failure(self):
        self.assertEqual(task_outcome_errors({'qa_evaluation_results': [[{'f1': 0.0, 'llm_judge_score': 0.0}]]}), [])

    def test_generation_resets_do_not_imply_evaluation_failure(self):
        self.assertEqual(task_outcome_errors({'qa_generated_qa_pairs': [{'question': 'Q'}],
                                              'qa_evaluation_results': None, 'qa_sleek_results': None}), [])

    def test_sleek_partial_and_no_success_are_incomplete(self):
        for value in ({'failed_evaluations': 1, 'successful_evaluations': 2},
                      {'failed_evaluations': 0, 'successful_evaluations': 0}):
            with self.subTest(value=value):
                self.assertTrue(task_outcome_errors({'qa_sleek_results': value}))

    def test_persuasion_failed_planned_work_is_incomplete(self):
        outputs = {'persuasion_last_run_summary': {'planned': 4, 'successful': 3, 'failed': 1}}
        self.assertTrue(task_outcome_errors(outputs))

    def test_negative_judgement_is_valid_but_transport_failure_is_not(self):
        output = {'generated_persuasion_mutations': {'prompt': [{'judge_meta': {}, 'judge_passed': False}]}}
        self.assertEqual(task_outcome_errors(output), [])
        output['generated_persuasion_mutations']['prompt'][0]['judge_meta']['secondary_error'] = 'timeout'
        self.assertTrue(task_outcome_errors(output))


class ReportInputs(unittest.TestCase):
    def test_qa_pdf_uses_actual_task_question_count(self):
        ns = task_runtime()
        pairs = [{'question': 'Q1', 'answer': 'A1'}, {'question': 'Q2', 'answer': 'A2'}]
        frozen = {'qa_generated_qa_pairs': pairs, 'qa_num_qa_pairs': 5, 'qa_source_mode': 'Input Text'}
        ns.update({'freeze_task_state': lambda: deepcopy(frozen), 'api_key': 'key',
                   'model_choice': 'model', 'provider': 'OpenAI', 'enable_llm_judge': False,
                   'eval_temperature': 0.3, 'eval_top_p': 0.8, 'num_eval_runs': 1,
                   'qa_source_mode': 'Input Text', 'AnalysisCheckpointError': AnalysisCheckpointError,
                   'get_llm_completion': lambda *args, **kwargs: 'answer',
                   'calculate_aggregate_metrics': lambda rows: {'total_evaluations': 2},
                   'run_knowledge_qa_evaluation': lambda *args, **kwargs: [[{'answer': 'A1'}, {'answer': 'A2'}]]})
        captured = []
        ns['generate_open_ended_question_pdf_report'] = lambda *args: None
        def generate(task_ui, factory, *args):
            captured.append(args)
            return b'pdf'
        ns['_generate_task_report_safely'] = generate
        tree = ast.parse((ROOT / 'src/ui.py').read_text())
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_run_task' and any(isinstance(n, ast.Name)
                  and n.id == 'run_knowledge_qa_evaluation' for n in ast.walk(node)))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'qa-report-task', 'exec'), ns)
        result = ns['_run_task'](lambda *args: None)
        self.assertTrue(result.complete)
        self.assertEqual(captured[0][5], 2)
        self.assertEqual(result['qa_evaluation_metadata']['num_qa_pairs'], 2)


class KnowledgeBatchRuntime(unittest.TestCase):
    def run_evaluation(self, replies):
        calls = []
        def answer(*args, **kwargs):
            calls.append(args[0])
            return replies[len(calls)-1]
        ns = {"__name__": __name__, "Dict": Dict, "List": List, "Any": Any,
              "Callable": Callable, "Optional": Optional, "math": math,
              "report_analysis_progress": lambda *args, **kwargs: None,
              "checkpoint_scope": lambda *args: nullcontext(),
              "answer_question_with_llm": answer,
              "is_api_error_response": lambda value: isinstance(value,str) and value.startswith('Error:'),
              "evaluate_qa_comparison": lambda q,a,p: {'question': q, 'ground_truth': a, 'llm_answer': p, 'f1': 1.0}}
        load_nodes('src/direct_recall/knowledge_qa.py', {'run_knowledge_qa_evaluation'}, ns)
        rows = ns['run_knowledge_qa_evaluation'](
            [{'question': 'Q1', 'answer': 'A1'}, {'question': 'Q2', 'answer': 'A2'}],
            'provider-secret', 'model', 'OpenAI', num_runs=1)
        return calls, rows

    def test_auth_failure_stops_before_remaining_questions_and_redacts_key(self):
        calls, rows = self.run_evaluation(['Error: 401 invalid API key provider-secret'])
        self.assertEqual(calls, ['Q1'])
        self.assertEqual(len(rows[0]), 1)
        self.assertNotIn('provider-secret', rows[0][0]['error'])
        self.assertIn('[redacted]', rows[0][0]['error'])

    def test_transient_item_failure_does_not_cancel_other_questions(self):
        calls, rows = self.run_evaluation(['Error: temporary timeout', 'answer'])
        self.assertEqual(calls, ['Q1', 'Q2'])
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(rows[0][1]['f1'], 1.0)

    def test_model_error_classifier_does_not_confuse_invalid_answers(self):
        from src.analysis_outcomes import is_systemic_api_failure
        self.assertTrue(is_systemic_api_failure('Error: model_not_found'))
        self.assertFalse(is_systemic_api_failure('Error: Invalid model response. Choose one valid option.'))
        self.assertFalse(is_systemic_api_failure('Error: Read timeout'))


class ScalingRuntime(unittest.TestCase):
    def runtime(self):
        ns = {"__name__": __name__, "dataclass": dataclass, 'Any': Any, 'Dict': Dict, 'List': List,
              'Optional': Optional, 'Tuple': Tuple, 'Callable': Callable, 'Mapping': Mapping,
              'Real': Real, 'math': math, 're': re,
              'OPENAI_PROVIDER': 'OpenAI', 'KIMI_PROVIDER': 'Kimi',
              'FIXED_MODELS': {'OpenAI': 'locked'}, 'MAX_RUNS_PER_PROVIDER': 50, 'MAX_OUTPUT_TOKENS': 700,
              'GROUND_TRUTH': 'reference', 'TARGET_WORD_COUNT': 2,
              'build_challenge_prompt': lambda *args: 'prompt',
              'enforce_exact_word_count': lambda text, size: text,
              'is_api_error_response': lambda text: text.startswith('Error:')}
        return load_nodes('src/game_continuation.py', {'ContinuationValidationError', 'ContinuationRunError',
            'ContinuationAttempt', 'ScalingBatch', '_response_text', 'run_provider_scaling'}, ns)

    def run_batch(self, runtime, completion, count=3):
        return runtime['run_provider_scaling']('provider-secret', provider='OpenAI', model='locked', runs=count,
            temperature=0.7, top_p=0.9, prompt_method='method', prompt_mode='mode',
            completion_fn=completion, metrics_fn=lambda *args: {'rouge_l': 0.2})

    def test_checkpoint_failure_aborts_before_another_paid_call(self):
        runtime = self.runtime()
        calls = []
        def completion(*args, **kwargs):
            calls.append(1)
            raise AnalysisCheckpointError('Unable to save checkpoint')
        with self.assertRaises(AnalysisCheckpointError):
            self.run_batch(runtime, completion)
        self.assertEqual(len(calls), 1)

    def test_partial_scaling_batch_is_not_complete_and_redacts_key(self):
        runtime = self.runtime()
        calls = []
        def completion(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('temporary failure provider-secret')
            return 'response'
        batch = self.run_batch(runtime, completion)
        self.assertFalse(batch.complete)
        self.assertEqual(batch.completed_runs, 2)
        self.assertNotIn('provider-secret', batch.error)
        self.assertIn('[redacted]', batch.error)

    def test_full_scaling_batch_is_complete(self):
        batch = self.run_batch(self.runtime(), lambda *args, **kwargs: 'response')
        self.assertTrue(batch.complete)
        self.assertIsNone(batch.error)


class SubmissionIntent(unittest.TestCase):
    def runtime(self):
        state = {}
        ns = {"__name__": __name__, "st": SimpleNamespace(session_state=state),
              "RUN_ARMED_KEY": "armed"}
        load_nodes('src/job_guard.py', {'should_execute_detection_run', 'discard_unconsumed_detection_run'}, ns)
        return state, ns

    def test_switching_away_cannot_leave_a_paid_action_armed(self):
        state, ns = self.runtime()
        state.update({'armed': 'text_run', 'detection_run_text_run': True,
                      'detection_run_unrelated_restore': True})
        ns['discard_unconsumed_detection_run']()
        self.assertFalse(ns['should_execute_detection_run']('text_run'))
        self.assertTrue(state['detection_run_unrelated_restore'])

    def test_submitted_click_is_consumed_once(self):
        state, ns = self.runtime()
        state.update({'armed': 'text_run', 'detection_run_text_run': True})
        self.assertTrue(ns['should_execute_detection_run']('text_run'))
        self.assertNotIn('armed', state)
        ns['discard_unconsumed_detection_run']()
        self.assertFalse(ns['should_execute_detection_run']('text_run'))




class SleekBatchRuntime(unittest.TestCase):
    def runtime(self, responses):
        from src.analysis_outcomes import is_systemic_api_failure
        calls = []
        progress = []
        def complete(*args, **kwargs):
            calls.append(kwargs.get('prompt', args[0] if args else 'request'))
            response = responses[len(calls) - 1]
            if isinstance(response, BaseException):
                raise response
            return response
        ns = {'__name__': __name__, 'dataclass': dataclass, 'Any': Any,
              'Dict': Dict, 'List': List, 'Optional': Optional, 'Callable': Callable,
              'AnalysisCheckpointError': AnalysisCheckpointError,
              'checkpoint_scope': lambda *args: nullcontext(),
              'report_analysis_progress': lambda *args, **kwargs: progress.append(args),
              'is_api_error_response': lambda value: isinstance(value, str) and value.startswith('Error:'),
              'is_systemic_api_failure': is_systemic_api_failure,
              'get_llm_completion': complete}
        load_nodes('src/direct_recall/sleek_attack.py',
                   {'SLEEKQuestion', '_response_text', 'run_sleek_attack',
                    'run_sleek_qa_evaluation', 'run_sleek_evaluation'}, ns)
        return ns, calls, progress

    def qa_evaluation(self, responses, *, num_runs=2, fail_in_cot=False):
        ns, calls, progress = self.runtime(responses)
        def decompose(**kwargs):
            if not fail_in_cot:
                ns['_response_text'](ns['get_llm_completion'](kwargs['question']))
            return [{'question': 'sub-question', 'category': 'Direct'}]
        def reasoning(**kwargs):
            answer = ns['get_llm_completion'](kwargs['original_question']) if fail_in_cot else 'answer'
            return {'final_answer': answer, 'sub_question_answers': []}
        ns.update(decompose_question=decompose, run_cot_reasoning=reasoning)
        comparison = ModuleType('src.direct_recall.comparison')
        comparison.calculate_rouge_score = lambda *args: 0.5
        comparison.calculate_jaccard_index = lambda *args: 0.5
        levenshtein = ModuleType('Levenshtein')
        levenshtein.distance = lambda *args: 0
        with patch.dict(sys.modules, {'src.direct_recall.comparison': comparison, 'Levenshtein': levenshtein}):
            result = ns['run_sleek_qa_evaluation'](
                [{'question': 'Q1', 'answer': 'answer'}, {'question': 'Q2', 'answer': 'answer'}],
                'provider-secret', 'model', 'OpenAI', num_runs=num_runs)
        return result, calls, progress

    def test_probe_auth_failure_stops_and_redacts(self):
        ns, calls, progress = self.runtime(['Error: 401 invalid API key provider-secret'])
        questions = [ns['SLEEKQuestion'](str(i), 'Direct', 'point') for i in range(4)]
        result = ns['run_sleek_attack'](questions, 'provider-secret', 'model', 'OpenAI')
        self.assertEqual(len(calls), 1)
        self.assertIn('[redacted]', result[0].response)
        self.assertNotIn('provider-secret', result[0].response)
        self.assertTrue(all(q.response is None for q in result[1:]))
        self.assertEqual(progress[-1], (1, 4))

    def test_probe_transient_error_continues(self):
        ns, calls, _ = self.runtime([RuntimeError('Read timeout'), 'answer'])
        questions = [ns['SLEEKQuestion'](str(i), 'Direct', 'point') for i in range(2)]
        result = ns['run_sleek_attack'](questions, 'key', 'model', 'OpenAI')
        self.assertEqual(len(calls), 2)
        self.assertEqual(result[1].response, 'answer')

    def test_qa_auth_failure_stops_outer_pair_and_run_loops(self):
        result, calls, progress = self.qa_evaluation(['Error: 401 invalid API key provider-secret'], num_runs=3)
        self.assertEqual(calls, ['Q1'])
        self.assertEqual(result['requested_evaluations'], 6)
        self.assertEqual(result['attempted_evaluations'], 1)
        self.assertEqual(result['unattempted_evaluations'], 5)
        self.assertEqual(result['failed_evaluations'], 1)
        self.assertEqual(len(result['qa_pair_results']), 1)
        self.assertIn('[redacted]', result['qa_pair_results'][0]['runs'][0]['error'])
        self.assertNotIn('provider-secret', str(result))
        self.assertEqual(progress[-1], (1, 6))
        self.assertTrue(task_outcome_errors({'qa_sleek_results': result}))

    def test_qa_reasoning_failure_preserves_prior_scores_and_coverage(self):
        result, calls, _ = self.qa_evaluation(['answer', 'Error: model_not_found'], fail_in_cot=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['successful_evaluations'], 1)
        self.assertEqual(result['failed_evaluations'], 1)
        self.assertEqual(result['unattempted_evaluations'], 2)
        self.assertEqual(result['avg_rouge_score'], 0.5)
        self.assertEqual(result['qa_pair_results'][0]['runs'][0]['final_answer'], 'answer')

    def test_qa_transient_error_continues_other_runs_and_pairs(self):
        result, calls, _ = self.qa_evaluation([RuntimeError('Read timeout'), 'ok', 'ok', 'ok'])
        self.assertEqual(len(calls), 4)
        self.assertEqual(result['attempted_evaluations'], 4)
        self.assertEqual(result['unattempted_evaluations'], 0)
        self.assertEqual(result['successful_evaluations'], 3)
        self.assertEqual(result['failed_evaluations'], 1)

    def test_checkpoint_error_propagates_without_continuing(self):
        with self.assertRaises(AnalysisCheckpointError):
            self.qa_evaluation([AnalysisCheckpointError('storage failure')])

    def test_document_report_distinguishes_failed_and_unattempted(self):
        ns, calls, _ = self.runtime(['answer', 'Error: 403 permission denied'])
        ns.update(generate_forget_question=lambda *args: 'forget',
                  generate_support_response=lambda *args: 'support',
                  extract_knowledge_points_and_generate_questions=lambda *args: {},
                  categorize_questions=lambda *args: [ns['SLEEKQuestion'](str(i), 'Direct', 'point') for i in range(3)],
                  assess_leakage=lambda questions, *args: questions)
        result = ns['run_sleek_evaluation']('document', 'key', 'model', 'OpenAI')
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['requested_evaluations'], 3)
        self.assertEqual(result['attempted_evaluations'], 2)
        self.assertEqual(result['successful_evaluations'], 1)
        self.assertEqual(result['failed_evaluations'], 1)
        self.assertEqual(result['unattempted_evaluations'], 1)
        self.assertEqual(result['category_breakdown']['Direct'],
                         {'total': 1, 'leaked': 0, 'failed': 1, 'unattempted': 1})


    def test_partial_pair_averages_weight_each_successful_response(self):
        ns, calls, _ = self.runtime(['high', 'high', 'high', 'low', 'Error: 401 invalid API key'])
        ns['decompose_question'] = lambda **kwargs: [{'question': 'sub', 'category': 'Direct'}]
        ns['run_cot_reasoning'] = lambda **kwargs: {
            'final_answer': ns['get_llm_completion'](kwargs['original_question']), 'sub_question_answers': []}
        comparison = ModuleType('src.direct_recall.comparison')
        comparison.calculate_rouge_score = lambda truth, answer: 1.0 if answer == 'high' else 0.0
        comparison.calculate_jaccard_index = comparison.calculate_rouge_score
        levenshtein = ModuleType('Levenshtein')
        levenshtein.distance = lambda truth, answer: 4 if answer == 'low' else 0
        with patch.dict(sys.modules, {'src.direct_recall.comparison': comparison, 'Levenshtein': levenshtein}):
            result = ns['run_sleek_qa_evaluation'](
                [{'question': 'Q1', 'answer': 'truth'}, {'question': 'Q2', 'answer': 'truth'}],
                'key', 'model', 'OpenAI', num_runs=3)
        self.assertEqual(result['successful_evaluations'], 4)
        self.assertEqual(result['attempted_evaluations'], 5)
        self.assertEqual(result['unattempted_evaluations'], 1)
        self.assertEqual(result['avg_rouge_score'], 0.75)
        self.assertEqual(result['avg_jaccard_index'], 0.75)
        self.assertEqual(result['avg_levenshtein_distance'], 1.0)

    def test_progress_callback_follows_saved_success_or_failure_and_skips_checkpoint_fatal(self):
        cases = [
            (['answer', 'Error: 401 invalid API key'], [1, 2], False),
            ([RuntimeError('Read timeout'), 'answer', 'answer', 'answer'], [1, 2, 3, 4], False),
            (['answer', AnalysisCheckpointError('storage failed')], [1], True),
        ]
        for responses, expected_progress, fatal in cases:
            with self.subTest(fatal=fatal, expected_progress=expected_progress):
                ns, calls, _ = self.runtime(responses)
                ns['decompose_question'] = lambda **kwargs: [{'question': 'sub', 'category': 'Direct'}]
                ns['run_cot_reasoning'] = lambda **kwargs: {
                    'final_answer': ns['get_llm_completion'](kwargs['original_question']), 'sub_question_answers': []}
                comparison = ModuleType('src.direct_recall.comparison')
                comparison.calculate_rouge_score = lambda *args: 0.5
                comparison.calculate_jaccard_index = lambda *args: 0.5
                levenshtein = ModuleType('Levenshtein')
                levenshtein.distance = lambda *args: 0
                callbacks = []
                def evaluate():
                    return ns['run_sleek_qa_evaluation'](
                        [{'question': 'Q1', 'answer': 'truth'}, {'question': 'Q2', 'answer': 'truth'}],
                        'key', 'model', 'OpenAI', num_runs=2,
                        progress_callback=lambda *values: callbacks.append((values[0], len(calls))))
                with patch.dict(sys.modules, {'src.direct_recall.comparison': comparison, 'Levenshtein': levenshtein}):
                    if fatal:
                        with self.assertRaises(AnalysisCheckpointError):
                            evaluate()
                    else:
                        evaluate()
                self.assertEqual(callbacks, [(value, value) for value in expected_progress])


if __name__ == '__main__':
    unittest.main()
