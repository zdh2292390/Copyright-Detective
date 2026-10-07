"""Regression coverage for background, game, SLEEK and judge execution failures."""
import io
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PyPDF2 import PdfReader
from src import background_jobs as jobs
from src import game_continuation as scaling
from src.game import engine as game
from src.game2 import engine as knowledge_game
from src.direct_recall import sleek_attack as sleek
from src.common.metrics import logger
from src.adversarial_persuasion_detection import adversarial_prompting as mutations
from src import pdf_preview as reports


class ManualExecutor:
    def submit(self, runner):
        self.runner = runner
        self.future = Future()
        return self.future

    def run(self):
        if self.future.set_running_or_notify_cancel():
            self.runner()
            self.future.set_result(None)


class BackgroundReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.executor = ManualExecutor()
        self.patches = [patch.object(jobs, '_EXECUTOR', self.executor), patch.object(jobs, '_JOBS', {})]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_duplicate_submission_and_defensive_result_snapshot(self):
        source = {'values': [1]}
        runner = Mock(return_value=source)
        self.assertTrue(jobs.submit_background_job('job', 'run', runner))
        self.assertFalse(jobs.submit_background_job('job', 'run', runner))
        self.assertFalse(jobs.forget_background_job('job'))
        self.executor.run()
        source['values'].append(2)
        result = jobs.get_background_job('job')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['result'], {'values': [1]})
        result['result']['values'].append(3)
        self.assertEqual(jobs.get_background_job('job')['result'], {'values': [1]})
        self.assertFalse(jobs.background_job_running('job'))
        runner.assert_called_once()

    def test_submission_failure_is_terminal(self):
        self.executor.submit = Mock(side_effect=RuntimeError('executor unavailable'))
        self.assertTrue(jobs.submit_background_job('job', 'run', Mock()))
        state = jobs.get_background_job('job')
        self.assertEqual(state['status'], 'failed')
        self.assertIn('executor unavailable', state['error'])
        self.assertTrue(state['finished_at'])

    def test_result_copy_failure_is_terminal(self):
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise ValueError('cannot snapshot')
        jobs.submit_background_job('job', 'run', lambda report: Uncopyable())
        self.executor.run()
        self.assertEqual(jobs.get_background_job('job')['status'], 'failed')
        self.assertIn('cannot snapshot', jobs.get_background_job('job')['error'])

    def test_worker_system_exit_and_queued_cancellation_are_terminal(self):
        jobs.submit_background_job('exit', 'run', Mock(side_effect=SystemExit()))
        self.executor.run()
        self.assertEqual(jobs.get_background_job('exit')['error'], 'SystemExit')
        jobs.submit_background_job('cancel', 'run', Mock())
        self.executor.future.cancel()
        self.assertEqual(jobs.get_background_job('cancel')['status'], 'failed')

    def test_late_callback_does_not_change_finished_or_replaced_job(self):
        saved = []
        def runner(report):
            saved.append(report)
            report(8, 3, 'in progress')
            self.assertEqual(jobs.get_background_job('job')['current'], 3)
            return 'first'
        jobs.submit_background_job('job', 'first', runner)
        self.executor.run()
        saved[0](0, 1, 'late')
        self.assertEqual(jobs.get_background_job('job')['message'], 'Completed')
        jobs.submit_background_job('job', 'second', lambda report: 'second')
        saved[0](0, 50, 'old')
        self.assertEqual(jobs.get_background_job('job')['total'], 1)
        self.executor.run()
        self.assertEqual(jobs.get_background_job('job')['result'], 'second')


class ModelRunReliabilityTests(unittest.TestCase):
    def test_scaling_attempts_all_requested_runs_and_retains_original_indices(self):
        completion = Mock(side_effect=['Error: 429 temporarily unavailable', 'good answer', TimeoutError(), 'last answer'])
        result = scaling.run_provider_scaling('key', provider='OpenAI', model=scaling.OPENAI_MODEL, runs=4,
            temperature=0.7, top_p=0.9, prompt_method=scaling.DIRECT_PROBE_METHODS[0], prompt_mode='Direct Probing',
            completion_fn=completion, metrics_fn=lambda ref, output: {'rouge_l': .25})
        self.assertEqual(completion.call_count, 4)
        self.assertEqual([attempt.run for attempt in result.attempts], [2, 4])
        self.assertEqual(len(result.errors), 2)
        self.assertEqual(result.avg_rouge_l, .25)
        for call in completion.call_args_list:
            self.assertEqual(call.args[0], scaling.build_challenge_prompt(scaling.DIRECT_PROBE_METHODS[0]))
            self.assertEqual(call.kwargs['temperature'], .7)

    def test_permanent_scaling_configuration_error_does_not_repeat_requests(self):
        completion = Mock(return_value='Error calling API: 401 invalid API key')
        with self.assertRaises(scaling.ContinuationRunError):
            scaling.run_provider_scaling('key', provider='OpenAI', model=scaling.OPENAI_MODEL, runs=50,
                temperature=.7, top_p=.9, prompt_method=scaling.DIRECT_PROBE_METHODS[0], prompt_mode='Direct Probing',
                completion_fn=completion)
        self.assertEqual(completion.call_count, 1)

    def test_all_failed_scaling_has_no_score(self):
        with self.assertRaises(scaling.ContinuationRunError):
            scaling.run_provider_scaling('key', provider='OpenAI', model=scaling.OPENAI_MODEL, runs=2,
                temperature=.7, top_p=.9, prompt_method=scaling.DIRECT_PROBE_METHODS[0], prompt_mode='Direct Probing',
                completion_fn=lambda *a, **k: object())

    def test_invalid_scaling_metrics_cannot_be_successes(self):
        for metrics in ({}, {'rouge_l': float('nan')}, {'rouge_l': float('inf')}, {'rouge_l': True}):
            with self.subTest(metrics=metrics), self.assertRaises(scaling.ContinuationRunError):
                scaling.run_provider_scaling('key', provider='OpenAI', model=scaling.OPENAI_MODEL, runs=1,
                    temperature=.7, top_p=.9, prompt_method=scaling.DIRECT_PROBE_METHODS[0], prompt_mode='Direct Probing',
                    completion_fn=lambda *a, **k: 'answer', metrics_fn=lambda *args: metrics)

    def test_standalone_sleek_category_statistics_exclude_failed_responses(self):
        questions = [sleek.SLEEKQuestion('one?', 'Direct', 'point'), sleek.SLEEKQuestion('two?', 'Direct', 'point')]
        questions[0].response = 'usable answer'
        questions[0].has_leakage = True
        questions[1].response = 'Error: 503 failed'
        with patch.object(sleek, 'generate_forget_question', return_value='question'), \
             patch.object(sleek, 'generate_support_response', return_value='support'), \
             patch.object(sleek, 'extract_knowledge_points_and_generate_questions', return_value=[]), \
             patch.object(sleek, 'categorize_questions', return_value=questions), \
             patch.object(sleek, 'run_sleek_attack', return_value=questions), \
             patch.object(sleek, 'assess_leakage', return_value=questions):
            result = sleek.run_sleek_evaluation('document', 'key', 'model', 'OpenAI')
        self.assertEqual(result['leakage_rate'], 1)
        self.assertEqual(result['category_breakdown']['Direct'], {'total':1, 'leaked':1, 'failed':1})

    def test_official_game_success_keeps_count_parameters_and_scores(self):
        complete = Mock(return_value='generated answer')
        result = game.run_stage_one('key', attempts=2, temperature=.4, top_p=.8,
            completion_fn=complete, metrics_fn=lambda ref, answer: {'rouge_l': .7})
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.rouge_l, .7)
        self.assertEqual(complete.call_count, 2)
        for call in complete.call_args_list:
            self.assertEqual(call.kwargs['temperature'], .4)
            self.assertEqual(call.kwargs['top_p'], .8)

    def test_invalid_model_objects_cannot_be_official_scores(self):
        for engine in (game, knowledge_game):
            for value in ((), object(), None, ['text']):
                with self.subTest(engine=engine.__name__, value=type(value).__name__):
                    with self.assertRaises(engine.GameRunError):
                        engine._completion_text(value)

    def test_sleek_transport_failure_cannot_be_parsing_fallback(self):
        for value in ('Error: 503 failed', None, object()):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(RuntimeError):
                    sleek.decompose_question('question', 'key', 'model', 'OpenAI', completion_fn=lambda **kwargs: value)

    def test_sleek_keeps_existing_plaintext_and_malformed_json_fallbacks(self):
        sub = sleek.decompose_question('question', 'key', 'model', 'OpenAI', completion_fn=lambda **kwargs: '{"not": "a list"}')
        self.assertEqual(sub, [{'question': 'question', 'category': 'Direct'}])
        result = sleek.run_cot_reasoning('question', sub, 'key', 'model', 'OpenAI', completion_fn=lambda **kwargs: 'plain answer')
        self.assertEqual(result['final_answer'], 'plain answer')

    def test_sleek_partial_runs_do_not_include_failures_in_average(self):
        complete = Mock(side_effect=['Error: 429 unavailable', '[{"question":"sub?", "category":"Direct"}]',
            '{"final_answer":"reference answer", "sub_question_answers":[]}', TimeoutError('timed out')])
        result = sleek.run_sleek_qa_evaluation([{'question':'question?', 'answer':'reference answer'}],
            'key', 'model', 'OpenAI', num_runs=3, completion_fn=complete)
        self.assertEqual(result['successful_evaluations'], 1)
        self.assertEqual(result['failed_evaluations'], 2)
        self.assertEqual(result['avg_rouge_score'], 1)
        self.assertEqual(len(result['qa_pair_results'][0]['runs']), 3)
        self.assertNotIn('rouge_score', result['qa_pair_results'][0]['runs'][0])

    def test_judge_failures_and_nonfinite_values_have_no_score(self):
        for response in ('Error: 503', '{}', '[]', '{"score": NaN}', '{"score": 1e309}', '{"score": true}', '{"score": NaN, "reasoning": "example score: 0.8"}', '{"score": true, "reasoning": "example score: 0.9"}', ''):
            with self.subTest(response=response):
                result = logger.parse_llm_judge_response(response)
                self.assertIsNone(result['score'])
                self.assertTrue(result['error'])
        valid = logger.parse_llm_judge_response('{"score":0.7, "reasoning":"good"}')
        self.assertEqual(valid, {'score':.7, 'reasoning':'good'})
        record = logger.FactRecallLogger(llm_judge_fn=Mock(side_effect=TimeoutError('timeout')))
        record.log('prompt', 'same', 'same')
        self.assertEqual(record.report()['mean_f1'], 1)
        self.assertIsNone(record.report()['mean_llm_judge_score'])
        self.assertEqual(record.report()['failed_judge_evaluations'], 1)

    def test_sleek_invalid_question_metadata_is_normalized_before_requests(self):
        questions = sleek.categorize_questions([None, {'question':'question?', 'knowledge_point':None, 'category':[]}])
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].knowledge_point, '')
        self.assertEqual(questions[0].category, 'Unknown')
        questions[0].response = 'answer'
        self.assertEqual(sleek.assess_leakage(questions, 'document', 'question'), questions)

    def test_ambiguous_intention_votes_are_errors(self):
        for answer in ('unknown', 'yesterday', 'yes and no'):
            self.assertIsNone(mutations._parse_judge_vote(answer))
        with patch.object(mutations, 'get_llm_completion', return_value='unknown') as call:
            result = mutations.run_intention_judge('key', 'model', 'OpenAI', 'source', 'mutation')
        self.assertEqual(call.call_count, 3)
        self.assertTrue(result.error)
        with patch.object(mutations, 'get_llm_completion', return_value='yes'):
            self.assertFalse(mutations.run_intention_judge('', 'local', 'Local vLLM', 'source', 'mutation').error)


class ReportFailureScopeTests(unittest.TestCase):
    def text(self, data):
        return ' '.join(' '.join(page.extract_text() or '' for page in PdfReader(io.BytesIO(data)).pages).split())

    def test_qa_pdf_failures_and_actual_f1_are_visible(self):
        rows = [[{'question':'first?', 'ground_truth':'one', 'llm_answer':'', 'error':'429 unavailable'},
                 {'question':'second?', 'ground_truth':'two', 'llm_answer':'two', 'f1':1., 'precision':1., 'recall':1.}]]
        metrics = {'total_attempted':2, 'failed_evaluations':1, 'total_evaluations':1, 'avg_f1':1., 'avg_precision':1., 'avg_recall':1.}
        text = self.text(reports.generate_open_ended_question_pdf_report(rows, metrics,
            [{'question':'first?', 'answer':'one'}, {'question':'second?', 'answer':'two'}], 'model', 'Input Text', 2, 1, .7, .9))
        self.assertIn('1/2 successful', text)
        self.assertIn('429 unavailable', text)
        self.assertIn('Average Token F1: 1.0000', text)
        self.assertIn('successful responses only', text)

    def test_all_failed_single_choice_and_sleek_reports_have_no_low_risk_claim(self):
        text = self.text(reports.generate_single_choice_question_pdf_report(
            {'results':[], 'metrics':{'overall_accuracy':None, 'failed_attempts':1, 'successful_attempts':0, 'total_attempts':1}}, 'model', 'OpenAI', 'Input Text'))
        self.assertIn('No memorization assessment', text)
        self.assertNotIn('LOW memorization', text)
        sleek_text = self.text(reports.generate_sleek_attack_pdf_report(
            {'total_questions':0, 'successful_evaluations':0, 'failed_evaluations':1}, 'model', 'OpenAI'))
        self.assertIn('No leakage assessment', sleek_text)
        self.assertNotIn('LOW KNOWLEDGE LEAKAGE', sleek_text)


if __name__ == '__main__':
    unittest.main()
