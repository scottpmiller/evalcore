"""Runner tests: concurrency, lifecycle hooks, and error paths."""

import asyncio
import pathlib
import tempfile
import typing
import unittest
import uuid
from unittest import mock

from evalcore import loader, models, runner, store
from evalcore.adapters import base as adapters_base
from evalcore.graders import base as graders_base


@adapters_base.register('_rec')
class _RecordingAdapter:
    """Counts invocations and records aclose; mode set via set_mode."""

    instances: typing.ClassVar[list] = []

    def __init__(self, **_):
        self.calls = 0
        self.closed = False
        _RecordingAdapter.instances.append(self)

    async def invoke(self, case, variant):
        self.calls += 1
        return models.Output(fields={'text': f'{case.id}:{variant.name}'})

    async def aclose(self):
        self.closed = True


@graders_base.register('_modewatch')
class _ModeWatchGrader:
    """A per-case grader that records the run mode via set_mode."""

    seen_mode: str | None = None
    seen_retry: object = None

    def __init__(self, name='_modewatch'):
        self.name = name

    def set_mode(self, mode):
        _ModeWatchGrader.seen_mode = mode

    def set_retry(self, config):
        _ModeWatchGrader.seen_retry = config

    def grade(self, case, output):
        return [
            models.Score(
                grader=self.name,
                metric='len',
                value=float(len(output.fields['text'])),
                case_id=case.id,
            )
        ]


@graders_base.register('_pin')
class _PinGrader:
    """A grader exposing a judge_version pin, like the LLM judge does."""

    name = '_pin'

    def __init__(self, pin='judge@v9'):
        self._pin = pin

    @property
    def judge_version(self):
        return self._pin

    def grade(self, case, output):
        return []


def _suite(**over):
    base = {
        'project': 'p',
        'suite': 's',
        'dataset': 'd',
        'dataset_version': 'v1',
        'mode_default': 'live',
        'adapter': {'type': '_rec'},
        'graders': [{'type': '_modewatch'}],
        'variants': {'baseline': {'model': 'm1'}},
        'n_samples': 2,
    }
    base.update(over)
    return loader.SuiteConfig.model_validate(base)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        _RecordingAdapter.instances = []
        _ModeWatchGrader.seen_mode = None
        _ModeWatchGrader.seen_retry = None

    def _load_cases(self, monkey_cases):
        # Patch loader.load_cases so no dataset dir is needed.
        self._orig = loader.load_cases
        loader.load_cases = lambda _d: monkey_cases

    def tearDown(self):
        if hasattr(self, '_orig'):
            loader.load_cases = self._orig

    def test_serial_run_samples_and_aggregates_stdev(self):
        self._load_cases([models.Case(id='c1'), models.Case(id='c2')])
        run = runner.run_suite_sync(_suite(), 'baseline', mode='live')
        # 2 cases x 2 samples
        self.assertEqual(len(run.results), 4)
        self.assertEqual(_RecordingAdapter.instances[0].calls, 4)
        self.assertTrue(_RecordingAdapter.instances[0].closed)  # aclose ran
        self.assertEqual(_ModeWatchGrader.seen_mode, 'live')
        # 'len' is a per-case mean metric with 2+ obs -> stdev populated.
        self.assertEqual(run.scorecard.metrics['len'].kind, 'mean')
        self.assertIsNotNone(run.scorecard.metrics['len'].stdev)
        self.assertEqual(run.scorecard.model_id, 'm1')

    def test_concurrent_run(self):
        self._load_cases([models.Case(id='c1'), models.Case(id='c2')])
        run = runner.run_suite_sync(
            _suite(concurrency=4), 'baseline', mode='live'
        )
        self.assertEqual(len(run.results), 4)

    def test_grader_mode_decoupled_from_adapter(self):
        # Adapter runs live, judge/grader binds to a different mode.
        self._load_cases([models.Case(id='c1')])
        runner.run_suite_sync(
            _suite(), 'baseline', mode='live', grader_mode='replay'
        )
        self.assertEqual(_ModeWatchGrader.seen_mode, 'replay')
        self.assertTrue(_RecordingAdapter.instances[0].calls)  # adapter ran

    def test_unknown_variant_raises(self):
        with self.assertRaises(KeyError):
            runner.run_suite_sync(_suite(), 'ghost', mode='live')

    def test_suite_retry_plumbed_to_graders(self):
        self._load_cases([models.Case(id='c1')])
        runner.run_suite_sync(
            _suite(retry={'max_attempts': 4}), 'baseline', mode='live'
        )
        self.assertEqual(_ModeWatchGrader.seen_retry.max_attempts, 4)

    def test_judge_version_pin_collected_onto_scorecard(self):
        self._load_cases([models.Case(id='c1')])
        run = runner.run_suite_sync(
            _suite(graders=[{'type': '_modewatch'}, {'type': '_pin'}]),
            'baseline',
            mode='live',
        )
        self.assertEqual(run.scorecard.judge_version, 'judge@v9')

    def test_judge_version_none_without_judge_grader(self):
        self._load_cases([models.Case(id='c1')])
        run = runner.run_suite_sync(_suite(), 'baseline', mode='live')
        self.assertIsNone(run.scorecard.judge_version)

    def test_replay_mode_requires_fixtures(self):
        with self.assertRaises(ValueError):
            runner.run_suite_sync(_suite(), 'baseline', mode='replay')


class ResumeTests(unittest.TestCase):
    """Checkpoint + resume: an interrupted run continues instead of redoing."""

    def setUp(self):
        _RecordingAdapter.instances = []
        self.tmp = tempfile.TemporaryDirectory()
        self.ckpt = str(pathlib.Path(self.tmp.name) / 'ck.jsonl')
        self._orig = loader.load_cases
        loader.load_cases = lambda _d: [
            models.Case(id=c) for c in ('c1', 'c2', 'c3')
        ]

    def tearDown(self):
        loader.load_cases = self._orig
        self.tmp.cleanup()

    def _suite(self):
        return loader.SuiteConfig.model_validate(
            {
                'project': 'p',
                'suite': 's',
                'dataset': 'd',
                'dataset_version': 'v1',
                'mode_default': 'live',
                'adapter': {'type': '_rec'},
                'graders': [],
                'variants': {'baseline': {}},
                'n_samples': 1,
                'suite_hash': 'H',
            }
        )

    def _calls(self):
        return _RecordingAdapter.instances[-1].calls

    def test_checkpoint_records_meta_and_results(self):
        run = runner.run_suite_sync(
            self._suite(), 'baseline', mode='live', checkpoint=self.ckpt
        )
        meta = store.checkpoint_meta(self.ckpt)
        self.assertEqual(meta['run_id'], run.run_id)
        self.assertEqual(meta['dataset_hash'], run.scorecard.dataset_hash)
        self.assertEqual(len(store.read_checkpoint_results(self.ckpt)), 3)

    def test_resume_runs_only_missing(self):
        # Full run to populate the checkpoint, then chop it to 2 results to
        # simulate an interruption, then resume.
        first = runner.run_suite_sync(
            self._suite(), 'baseline', mode='live', checkpoint=self.ckpt
        )
        lines = pathlib.Path(self.ckpt).read_text().splitlines()
        pathlib.Path(self.ckpt).write_text('\n'.join(lines[:3]) + '\n')

        run = runner.run_suite_sync(
            self._suite(),
            'baseline',
            mode='live',
            checkpoint=self.ckpt,
            resume=True,
        )
        self.assertEqual(self._calls(), 1)  # only c3 re-invoked
        self.assertEqual([r.case.id for r in run.results], ['c1', 'c2', 'c3'])
        self.assertEqual(run.run_id, first.run_id)  # same logical run

    def test_resume_refuses_mismatched_eval(self):
        runner.run_suite_sync(
            self._suite(), 'baseline', mode='live', checkpoint=self.ckpt
        )
        changed = self._suite().model_copy(update={'suite_hash': 'OTHER'})
        with self.assertRaises(ValueError):
            runner.run_suite_sync(
                changed,
                'baseline',
                mode='live',
                checkpoint=self.ckpt,
                resume=True,
            )

    def test_no_resume_overwrites_checkpoint(self):
        runner.run_suite_sync(
            self._suite(), 'baseline', mode='live', checkpoint=self.ckpt
        )
        first_id = store.checkpoint_meta(self.ckpt)['run_id']
        # A fresh run (no --resume) starts the checkpoint over with a new id.
        runner.run_suite_sync(
            self._suite(), 'baseline', mode='live', checkpoint=self.ckpt
        )
        self.assertNotEqual(
            store.checkpoint_meta(self.ckpt)['run_id'], first_id
        )
        self.assertEqual(len(store.read_checkpoint_results(self.ckpt)), 3)


class _FlakyAdapter:
    """Returns a retryable error for the first ``fail_n`` calls, then ok."""

    def __init__(self, fail_n, retryable=True):
        self.calls = 0
        self.fail_n = fail_n
        self.retryable = retryable

    async def invoke(self, case, variant):
        self.calls += 1
        if self.calls <= self.fail_n:
            return models.Output(error='HTTP 429', retryable=self.retryable)
        return models.Output(fields={'ok': 1})


class RetryTests(unittest.TestCase):
    def _retry(self, adapter, retry):
        with mock.patch.object(runner.asyncio, 'sleep') as slept:
            out = asyncio.run(
                runner._invoke_with_retry(
                    adapter,
                    models.Case(id='c'),
                    models.Variant(name='v'),
                    retry,
                )
            )
        return out, slept

    def test_retries_then_succeeds(self):
        adapter = _FlakyAdapter(fail_n=2)
        out, slept = self._retry(
            adapter, loader.RetryConfig(max_attempts=5, jitter=0)
        )
        self.assertEqual(adapter.calls, 3)
        self.assertIsNone(out.error)
        self.assertEqual(slept.call_count, 2)  # backed off before each retry

    def test_gives_up_after_budget_keeps_error(self):
        adapter = _FlakyAdapter(fail_n=10)
        out, _ = self._retry(
            adapter, loader.RetryConfig(max_attempts=3, jitter=0)
        )
        self.assertEqual(adapter.calls, 3)
        self.assertEqual(out.error, 'HTTP 429')

    def test_non_retryable_error_is_not_retried(self):
        adapter = _FlakyAdapter(fail_n=10, retryable=False)
        self._retry(adapter, loader.RetryConfig(max_attempts=5))
        self.assertEqual(adapter.calls, 1)

    def test_default_config_does_not_retry(self):
        adapter = _FlakyAdapter(fail_n=10)
        _, slept = self._retry(adapter, loader.RetryConfig())
        self.assertEqual(adapter.calls, 1)  # max_attempts=1 default
        self.assertEqual(slept.call_count, 0)

    def test_backoff_is_exponential_and_capped(self):
        adapter = _FlakyAdapter(fail_n=10)
        _, slept = self._retry(
            adapter,
            loader.RetryConfig(
                max_attempts=6, backoff_base=1.0, backoff_max=4.0, jitter=0
            ),
        )
        delays = [c.args[0] for c in slept.call_args_list]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 4.0, 4.0])  # capped at 4

    def test_retry_config_defaults(self):
        cfg = loader.RetryConfig()
        self.assertEqual(cfg.max_attempts, 1)
        self.assertEqual(cfg.backoff_base, 0.5)


if __name__ == '__main__':
    unittest.main()


class Uuid7Tests(unittest.TestCase):
    """The run id is a time-ordered UUIDv7, dashed so ClickHouse parses it."""

    def _bits(self, value):
        return (value.version, (value.int >> 62) & 0b11)

    def test_version_and_variant(self):
        self.assertEqual(self._bits(runner._uuid7()), (7, 0b10))

    def test_hand_rolled_fallback_matches_rfc_9562(self):
        # The 3.11-3.13 path: uuid.uuid7 does not exist, so the bits are
        # assembled here rather than by the stdlib. Swap the module reference
        # for a stand-in without uuid7 rather than mutating the real module.
        class _Without:
            UUID = uuid.UUID

        with mock.patch.object(runner, 'uuid', _Without):
            value = runner._uuid7()
        self.assertEqual(self._bits(value), (7, 0b10))

    def test_time_ordered(self):
        earlier = runner._uuid7()
        later = runner._uuid7()
        # The leading 48 bits are a millisecond timestamp, so it never moves
        # backwards even when two ids land in the same millisecond.
        self.assertLessEqual(earlier.int >> 80, later.int >> 80)

    def test_dashed_string(self):
        run_id = str(runner._uuid7())
        self.assertEqual(len(run_id), 36)
        self.assertEqual(run_id.count('-'), 4)
