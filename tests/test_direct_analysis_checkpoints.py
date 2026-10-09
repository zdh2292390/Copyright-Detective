"""Direct provider adapters journal successful calls without changing public results."""
import copy
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import torch
from src.direct_recall import single_choice as sc, knowledge_qa as qa, decop_analysis as decop
from src.pages import unlearning_detection as unlearning

class CheckpointError(RuntimeError):
    pass

class Journal(ModuleType):
    def __init__(self, records=None, fail_save=False):
        super().__init__('src.resumable_analysis')
        self.AnalysisCheckpointError = CheckpointError
        self.records = copy.deepcopy(records or {})
        self.ordinal = 0
        self.depth = 0
        self.fail_save = fail_save

    def checkpoint_call(self, operation, payload, invoke, *, is_success):
        if self.depth:
            return invoke()
        ordinal = self.ordinal
        self.ordinal += 1
        fingerprint = json.dumps([operation, payload], sort_keys=True)
        if ordinal in self.records:
            saved = self.records[ordinal]
            if saved['fingerprint'] != fingerprint:
                raise CheckpointError('saved request differs')
            return copy.deepcopy(saved['result'])
        self.depth += 1
        try:
            result = invoke()
        finally:
            self.depth -= 1
        if is_success(result):
            if self.fail_save:
                raise CheckpointError('storage unavailable')
            self.records[ordinal] = {'fingerprint': fingerprint, 'operation': operation,
                'payload': copy.deepcopy(payload), 'result': json.loads(json.dumps(result))}
        return result


def question(text='Question?'):
    return {'question': text, 'options': [{'label': 'A', 'text': 'correct'},
        {'label': 'B', 'text': 'wrong'}], 'correct_option': 'A', 'source_fragment': 'source'}

class DirectCheckpointTests(unittest.TestCase):
    def scope(self, journal):
        return patch.dict(sys.modules, {'src.resumable_analysis': journal})

    def test_single_choice_direct_replays_fresh_journal_with_complete_request_payload(self):
        journal = Journal()
        complete = Mock(return_value={'choice': 'A', 'option_probabilities': {'A': .8, 'B': .2}, 'raw_response': 'A', 'logit_mode': 'logprobs'})
        with self.scope(journal), patch.object(sc, '_try_openai_style_completion', complete):
            first = sc.evaluate_single_choice_question(question(), 'private-key', 'gpt-4o-mini', 'OpenAI', .4, .8)
        restarted = Journal(journal.records)
        with self.scope(restarted), patch.object(sc, '_try_openai_style_completion', complete):
            second = sc.evaluate_single_choice_question(question(), 'new-key', 'gpt-4o-mini', 'OpenAI', .4, .8)
        self.assertEqual(first, second)
        complete.assert_called_once()
        payload = journal.records[0]['payload']
        self.assertEqual(payload['question'], question())
        self.assertEqual((payload['provider'], payload['model'], payload['temperature'], payload['top_p']), ('OpenAI', 'gpt-4o-mini', .4, .8))
        self.assertEqual(payload['endpoint'], 'https://api.openai.com/v1')
        self.assertNotIn('private-key', json.dumps(journal.records))

    def test_single_choice_failed_result_is_retried_and_never_cached_as_success(self):
        journal = Journal()
        complete = Mock(side_effect=[sc._sc_error_result('Error: timed out'), {'choice': 'A', 'raw_response': 'A'}])
        with self.scope(journal), patch.object(sc, '_try_openai_style_completion', complete):
            failed = sc.evaluate_single_choice_question(question(), 'key', 'gpt-4o-mini', 'OpenAI')
        self.assertTrue(failed['error'])
        self.assertEqual(journal.records, {})
        with self.scope(Journal(journal.records)), patch.object(sc, '_try_openai_style_completion', complete):
            self.assertEqual(sc.evaluate_single_choice_question(question(), 'key', 'gpt-4o-mini', 'OpenAI')['choice'], 'A')
        self.assertEqual(complete.call_count, 2)

    def test_fallback_nested_completion_uses_only_outer_item_ordinal(self):
        journal = Journal()
        api = Mock(return_value='A')
        def shared_completion(*args, **kwargs):
            return journal.checkpoint_call('llm', {'prompt': args[0]}, api, is_success=lambda text: bool(text))
        with self.scope(journal), patch.object(sc, 'get_llm_completion', side_effect=shared_completion):
            first = sc.evaluate_single_choice_question(question(), 'key', 'unknown-model', 'Google Gemini')
        self.assertEqual(first['choice'], 'A')
        self.assertEqual([item['operation'] for item in journal.records.values()], ['single_choice.evaluate'])
        with self.scope(Journal(journal.records)), patch.object(sc, 'get_llm_completion', side_effect=shared_completion):
            self.assertEqual(sc.evaluate_single_choice_question(question(), 'key', 'unknown-model', 'Google Gemini'), first)
        api.assert_called_once()

    def test_single_choice_storage_failure_stops_batch_before_next_paid_call(self):
        journal = Journal(fail_save=True)
        complete = Mock(return_value={'choice': 'A', 'raw_response': 'A'})
        with self.scope(journal), patch.object(sc, '_try_openai_style_completion', complete):
            with self.assertRaisesRegex(CheckpointError, 'storage unavailable'):
                sc.run_single_choice_evaluation([question('one?'), question('two?')], 'key', 'gpt-4o-mini', 'OpenAI')
        complete.assert_called_once()

    def test_decop_openai_tensor_is_json_journaled_and_restored_exactly(self):
        logs = [SimpleNamespace(token='A', logprob=-.2), SimpleNamespace(token='B', logprob=-2.)]
        response = SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=logs)]))])
        create = Mock(return_value=response)
        client = SimpleNamespace(base_url='https://api.openai.com/v1/', chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        row = pd.Series({f'Example_{label}': label + ' text' for label in 'ABCD'})
        journal = Journal()
        with self.scope(journal):
            first = decop.query_llm_chatgpt(row, 'document', 'author', 'BookTection', client)
        with self.scope(Journal(journal.records)):
            second = decop.query_llm_chatgpt(row, 'document', 'author', 'BookTection', client)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(second.dtype, torch.float32)
        create.assert_called_once()
        self.assertIsInstance(journal.records[0]['result'], list)
        self.assertEqual(journal.records[0]['payload']['endpoint'], client.base_url)
        self.assertEqual(create.call_args.kwargs['max_tokens'], 1)

    def test_decop_invalid_answer_does_not_get_a_success_checkpoint(self):
        create = Mock(side_effect=[{'content': [{'type': 'text', 'text': 'unavailable'}]}, {'content': [{'type': 'text', 'text': 'B'}]}])
        client = SimpleNamespace(base_url='https://api.anthropic.com', messages=SimpleNamespace(create=create))
        row = pd.Series({f'Example_{label}': label for label in 'ABCD'})
        journal = Journal()
        with self.scope(journal), self.assertRaises(ValueError):
            decop.query_llm_claude(row, 'paper', '', 'arXivTection', client)
        self.assertEqual(journal.records, {})
        valid = Journal()
        with self.scope(valid):
            self.assertEqual(decop.query_llm_claude(row, 'paper', '', 'arXivTection', client), 'B')
        with self.scope(Journal(valid.records)):
            self.assertEqual(decop.query_llm_claude(row, 'paper', '', 'arXivTection', client), 'B')
        self.assertEqual(create.call_count, 2)

    def test_min_k_probe_replays_success_and_unsupported_errors(self):
        for response in (([], 'Completion API error'), ([-.5, -1.], None)):
            with self.subTest(response=response):
                journal = Journal()
                invoke = Mock(return_value=response)
                with self.scope(journal), patch.object(unlearning, '_get_completion_logprobs_uncached', invoke):
                    self.assertEqual(unlearning._get_completion_logprobs('prompt', 'key', 'model', progress_message='first label'), response)
                with self.scope(Journal(journal.records)), patch.object(unlearning, '_get_completion_logprobs_uncached', invoke):
                    self.assertEqual(unlearning._get_completion_logprobs('prompt', 'changed-key', 'model', progress_message='new label'), response)
                invoke.assert_called_once()
                self.assertEqual(journal.records[0]['operation'], 'min_k.completion_probe')
                self.assertNotIn('key', json.dumps(journal.records))

    def test_min_k_probe_rejects_invalid_logprob_cache_values(self):
        journal = Journal()
        with self.scope(journal), patch.object(unlearning, '_get_completion_logprobs_uncached', return_value=([float('nan')], None)):
            unlearning._get_completion_logprobs('prompt', 'key', 'model')
        self.assertEqual(journal.records, {})

    def test_min_k_successful_chat_fallback_completes_real_journal_and_replays_after_restart(self):
        from src import resumable_analysis as durable
        from uuid import uuid4
        # Use the real journal and codec with a small in-memory persistence adapter.
        task = {'id': str(uuid4()), 'status': 'queued', 'lease_token': str(uuid4())}
        rows = []
        store = Mock()
        store.claim.side_effect = lambda *args, **kwargs: copy.deepcopy(task)
        store.heartbeat.side_effect = lambda *args, **kwargs: copy.deepcopy(task)
        store.load_items.side_effect = lambda task_id, offset=0, limit=200: copy.deepcopy(rows[offset:offset + limit])
        def append(task_id, index, payload, **kwargs):
            self.assertEqual(index, len(rows))
            rows.append({'item_index': index, 'input': copy.deepcopy(payload), 'status': 'pending', 'attempts': 0})
            return copy.deepcopy(task)
        def save(task_id, index, **kwargs):
            rows[index].update({key: copy.deepcopy(value) for key, value in kwargs.items() if key != 'lease_token'})
            return copy.deepcopy(task)
        def release(task_id, lease_token, status='incomplete', **kwargs):
            task['status'] = status
            return copy.deepcopy(task)
        store.append_item.side_effect, store.save_item.side_effect, store.release.side_effect = append, save, release
        legacy = Mock(return_value=([], 'Completion API unsupported for this model'))
        chat = Mock(return_value=('generated', [{'token': 'a', 'logprob': -.5}, {'token': 'b', 'logprob': -1.}]))
        def completion(**kwargs):
            payload = {key: value for key, value in kwargs.items() if key != 'api_key'}
            return durable.checkpoint_call('llm.completion', payload, chat)
        first = durable.CallJournal(store, copy.deepcopy(task), heartbeat=False)
        with durable.journal_scope(first), patch.object(unlearning, '_get_completion_logprobs_uncached', legacy), patch.object(unlearning, 'get_llm_completion', side_effect=completion):
            result = unlearning.run_min_k_prob_analysis('Prompt', 'private-key', 'model', 'OpenAI')
        first.finish(True)
        self.assertEqual(task['status'], 'complete')
        self.assertEqual([row['status'] for row in rows], ['complete', 'complete'])
        self.assertEqual(rows[0]['input']['operation'], 'min_k.completion_probe')
        self.assertEqual(durable.decode(rows[0]['result']), ([], 'Completion API unsupported for this model'))
        resumed = durable.CallJournal(store, copy.deepcopy(task), heartbeat=False)
        with durable.journal_scope(resumed), patch.object(unlearning, '_get_completion_logprobs_uncached', legacy), patch.object(unlearning, 'get_llm_completion', side_effect=completion):
            replayed = unlearning.run_min_k_prob_analysis('Prompt', 'changed-key', 'model', 'OpenAI')
        resumed.finish(True)
        self.assertEqual(result, replayed)
        self.assertEqual(result['min_k_prob'], 1.)
        legacy.assert_called_once()
        chat.assert_called_once()
        self.assertNotIn('private-key', json.dumps(rows))

    def test_qa_can_generate_from_recovered_text_without_upload_and_preserves_original_cap(self):
        source = ' '.join('word' + str(index) for index in range(3001))
        with patch.object(qa, 'extract_text_from_document') as extract, patch.object(qa, 'generate_qa_pairs_from_text', return_value=[{'question': 'q?', 'answer': 'a'}]) as generate:
            pairs, saved = qa.generate_qa_pairs_from_document(None, 'key', 'model', 'Google Gemini', source_text=source)
        extract.assert_not_called()
        self.assertEqual(len(saved.split()), 3000)
        self.assertEqual(generate.call_args.args[0], saved)
        self.assertEqual(pairs, [{'question': 'q?', 'answer': 'a'}])
        self.assertEqual(generate.call_args.kwargs, {'num_pairs': 5, 'temperature': .7, 'top_p': .9})

    def test_qa_answer_does_not_convert_checkpoint_storage_error_to_a_provider_error(self):
        complete = Mock(side_effect=CheckpointError('storage unavailable'))
        with self.scope(Journal()), self.assertRaises(CheckpointError):
            qa.run_knowledge_qa_evaluation([{'question': 'one?', 'answer': 'a'}, {'question': 'two?', 'answer': 'b'}], 'key', 'model', 'Google Gemini', completion_fn=complete)
        complete.assert_called_once()

if __name__ == '__main__':
    unittest.main()
