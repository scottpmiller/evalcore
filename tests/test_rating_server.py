"""Rating web-app tests: panel derivation, HTTP handler, and serve()."""

import contextlib
import http.server
import io
import json
import pathlib
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

from evalcore import models, rating, store


class HelperTests(unittest.TestCase):
    def test_kind_for_path(self):
        self.assertEqual(rating._kind_for_path('a/b.PNG'), 'image')
        self.assertEqual(rating._kind_for_path('x.pdf'), 'pdf')
        self.assertEqual(rating._kind_for_path('x.html'), 'html')
        self.assertEqual(rating._kind_for_path('x.json'), 'json')
        self.assertEqual(rating._kind_for_path('x.log'), 'text')

    def test_infer_kind(self):
        self.assertEqual(
            rating._infer_kind('artifacts.shot', '/a.png'), 'image'
        )
        self.assertEqual(rating._infer_kind('output.x', {'a': 1}), 'json')
        self.assertEqual(rating._infer_kind('output.x', '<div>'), 'html')
        self.assertEqual(rating._infer_kind('output.x', 'plain'), 'text')

    def test_default_view_specs(self):
        specs = rating._default_view_specs('output.html', 'artifacts.shot')
        self.assertEqual([s['kind'] for s in specs], ['html', 'image'])
        self.assertEqual(rating._default_view_specs(None, None), [])


def _run_with(fields=None, artifacts=None):
    return models.RunResult(
        run_id='R',
        scorecard=models.Scorecard(
            project='p',
            suite='s',
            variant=models.Variant(name='v'),
            dataset_version='v1',
        ),
        results=[
            models.CaseResult(
                case=models.Case(id='c1', input={'q': 1}),
                variant_name='v',
                sample_hash='h0',
                output=models.Output(
                    fields=fields or {}, artifacts=artifacts or {}
                ),
            )
        ],
    )


class PanelDerivationTests(unittest.TestCase):
    def _panels(self, **kw):
        run = _run_with(kw.pop('fields', None), kw.pop('artifacts', None))
        app = rating._RatingApp(run and [run], '/dev/null', ['d'], 5, **kw)
        return app.items[0]['panels'], app.items[0]['files']

    def test_text_only_fallback(self):
        panels, _ = self._panels(fields={'answer': 'Paris'})
        self.assertEqual(
            panels[0], {'label': 'Answer', 'kind': 'text', 'value': 'Paris'}
        )

    def test_json_fallback_for_multifield(self):
        panels, _ = self._panels(fields={'a': 1, 'b': 2})
        self.assertEqual(panels[0]['kind'], 'json')
        self.assertEqual(panels[0]['label'], 'Output')

    def test_visual_artifacts_become_panels(self):
        panels, files = self._panels(
            artifacts={
                'desktop': '/x/a.png',
                'report': '/x/r.pdf',
                'notes': '/x/n.txt',
            }
        )
        kinds = {p['label']: p['kind'] for p in panels}
        self.assertEqual(kinds.get('Desktop'), 'image')
        self.assertEqual(kinds.get('Report'), 'pdf')
        # Non-visual text artifact is not auto-added.
        self.assertNotIn('Notes', kinds)
        self.assertEqual(len(files), 2)  # image + pdf paths, order-aligned

    def test_explicit_views_infer_kind(self):
        panels, _ = self._panels(
            fields={'summary': 'ok', 'raw': {'k': 1}},
            views=[
                {'label': 'Summary', 'ref': 'output.summary'},
                {'label': 'Raw', 'kind': 'json', 'ref': 'output.raw'},
            ],
        )
        self.assertEqual([p['kind'] for p in panels], ['text', 'json'])


class HandlerTests(unittest.TestCase):
    """Drive the real handler over a loopback ThreadingHTTPServer."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        png = pathlib.Path(self.tmp.name) / 'shot.png'
        png.write_bytes(b'\x89PNG\r\n\x1a\n fake')
        run = _run_with(
            fields={'html': '<b>hi</b>'}, artifacts={'screenshot': str(png)}
        )
        self.ratings = str(pathlib.Path(self.tmp.name) / 'r.jsonl')
        rating._Handler.app = rating._RatingApp(
            [run],
            self.ratings,
            ['quality'],
            5,
            content_ref='output.html',
            screenshot_ref='artifacts.screenshot',
        )
        self.server = http.server.ThreadingHTTPServer(
            ('127.0.0.1', 0), rating._Handler
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _get(self, path):
        url = f'http://127.0.0.1:{self.port}{path}'
        with urllib.request.urlopen(url) as resp:
            return resp.status, resp.headers.get('Content-Type'), resp.read()

    def test_index_injects_config(self):
        status, ctype, body = self._get('/')
        self.assertEqual(status, 200)
        self.assertIn('text/html', ctype)
        self.assertIn(b'"scale": 5', body)

    def test_queue_and_artifact_and_rate(self):
        status, ctype, body = self._get('/api/queue?rater=x')
        queue = json.loads(body)
        self.assertEqual(len(queue), 1)
        views = queue[0]['views']
        image = next(v for v in views if v['kind'] == 'image')
        # artifact serves the file bytes with detected content-type.
        status, ctype, data = self._get(f'/api/artifact?id=0&v={image["v"]}')
        self.assertEqual(status, 200)
        self.assertEqual(ctype, 'image/png')
        self.assertTrue(data.startswith(b'\x89PNG'))
        # POST a rating; it persists and drops from the queue (resume).
        payload = json.dumps(
            {'id': queue[0]['id'], 'rater': 'x', 'scores': {'quality': 4}}
        ).encode()
        req = urllib.request.Request(
            f'http://127.0.0.1:{self.port}/api/rate',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
        self.assertEqual(len(store_ratings(self.ratings)), 1)
        self.assertEqual(json.loads(self._get('/api/queue?rater=x')[2]), [])

    def test_unknown_path_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._get('/nope')
        self.assertEqual(cm.exception.code, 404)


def store_ratings(path):
    from evalcore import store

    return store.read_ratings(path)


def _variant_run(name, cases, *, fields=None, artifacts=None):
    return models.RunResult(
        run_id=name,
        scorecard=models.Scorecard(
            project='p',
            suite='s',
            variant=models.Variant(name=name),
            dataset_version='v1',
        ),
        results=[
            models.CaseResult(
                case=models.Case(id=c, input={'q': c}),
                variant_name=name,
                sample_hash='h0',
                output=models.Output(
                    fields=(fields or {'html': f'<b>{name}:{c}</b>'}),
                    artifacts=artifacts or {},
                ),
            )
            for c in cases
        ],
    )


class RankAppTests(unittest.TestCase):
    def _app(
        self,
        prefs_path='/dev/null',
        cases_a=('c1', 'c2'),
        cases_b=('c1', 'c2'),
        **kw,
    ):
        run_a = _variant_run('baseline', cases_a)
        run_b = _variant_run('candidate', cases_b)
        return rating._RankApp(
            run_a,
            run_b,
            prefs_path,
            ['visual'],
            content_ref='output.html',
            **kw,
        )

    def test_only_common_cases_are_aligned(self):
        app = self._app(cases_a=('c1', 'c2'), cases_b=('c2', 'c3'))
        self.assertEqual([it['case_id'] for it in app.items], ['c2'])

    def test_queue_is_blind_two_columns_no_identity(self):
        app = self._app()
        payload = app.queue_for('rater-x')
        self.assertEqual(len(payload), 2)
        for item in payload:
            self.assertEqual(set(item), {'id', 'input', 'left', 'right'})
            self.assertNotIn('variant_a', item)
            self.assertNotIn('a', item)  # no raw variant columns leak
            for view in item['left'] + item['right']:
                self.assertIn('kind', view)

    def test_orientation_is_stable_per_rater_but_varies(self):
        app = self._app()
        first = [app._a_on_left('cv', idx) for idx in range(len(app.items))]
        again = [app._a_on_left('cv', idx) for idx in range(len(app.items))]
        self.assertEqual(first, again)  # stable -> resumable
        many = [app._a_on_left(f'r{n}', 0) for n in range(16)]
        self.assertTrue(any(many) and not all(many))  # counterbalanced

    def test_record_unblinds_left_pick_to_variant_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'prefs.jsonl'
            app = self._app(str(path))
            # Record a 'left' overall pick on item 0; the stored winner must
            # match whichever variant was actually on the left for this rater.
            a_left = app._a_on_left('cv', 0)
            app.record(0, 'cv', 'left', {'visual': 'left'})
            saved = store.read_preferences(path)
            self.assertEqual(len(saved), 1)
            expected = 'a' if a_left else 'b'
            self.assertEqual(saved[0].winner, expected)
            self.assertEqual(saved[0].dims['visual'], expected)
            self.assertEqual(saved[0].variant_a, 'baseline')
            self.assertEqual(saved[0].variant_b, 'candidate')

    def test_tie_stays_tie_regardless_of_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'p.jsonl'
            app = self._app(str(path))
            app.record(0, 'cv', 'tie', {'visual': 'tie'})
            self.assertEqual(store.read_preferences(path)[0].winner, 'tie')

    def test_ranked_items_drop_from_queue_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'p.jsonl'
            app = self._app(str(path))
            before = len(app.queue_for('cv'))
            app.record(
                app.queue_for('cv')[0]['id'], 'cv', 'left', {'visual': 'a'}
            )
            self.assertEqual(len(app.queue_for('cv')), before - 1)

    def test_file_path_matches_displayed_side(self):
        run_a = _variant_run(
            'baseline', ['c1'], fields={}, artifacts={'screenshot': '/x/a.png'}
        )
        run_b = _variant_run(
            'candidate',
            ['c1'],
            fields={},
            artifacts={'screenshot': '/x/b.png'},
        )
        app = rating._RankApp(
            run_a,
            run_b,
            '/dev/null',
            ['d'],
            screenshot_ref='artifacts.screenshot',
        )
        a_left = app._a_on_left('cv', 0)
        left_path = app.file_path(0, 'left', 'cv', 0)
        self.assertEqual(left_path, '/x/a.png' if a_left else '/x/b.png')
        right_path = app.file_path(0, 'right', 'cv', 0)
        self.assertEqual(right_path, '/x/b.png' if a_left else '/x/a.png')


class RankHandlerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prefs = str(pathlib.Path(self.tmp.name) / 'p.jsonl')
        rating._RankHandler.app = rating._RankApp(
            _variant_run('baseline', ['c1']),
            _variant_run('candidate', ['c1']),
            self.prefs,
            ['visual'],
            content_ref='output.html',
        )
        self.server = http.server.ThreadingHTTPServer(
            ('127.0.0.1', 0), rating._RankHandler
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _get(self, path):
        url = f'http://127.0.0.1:{self.port}{path}'
        with urllib.request.urlopen(url) as resp:
            return resp.status, resp.headers.get('Content-Type'), resp.read()

    def test_index_and_queue_and_rank(self):
        status, _, body = self._get('/')
        self.assertEqual(status, 200)
        self.assertIn(b'blind ranking', body)
        queue = json.loads(self._get('/api/queue?rater=x')[2])
        self.assertEqual(len(queue), 1)
        self.assertIn('left', queue[0])
        payload = json.dumps(
            {
                'id': queue[0]['id'],
                'rater': 'x',
                'winner': 'left',
                'dims': {'visual': 'right'},
            }
        ).encode()
        req = urllib.request.Request(
            f'http://127.0.0.1:{self.port}/api/rank',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
        self.assertEqual(len(store.read_preferences(self.prefs)), 1)
        # persisted -> drops from the queue (resume)
        self.assertEqual(json.loads(self._get('/api/queue?rater=x')[2]), [])

    def test_unknown_path_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._get('/nope')
        self.assertEqual(cm.exception.code, 404)


class ServeTests(unittest.TestCase):
    def test_serve_runs_until_interrupt(self):
        run = _run_with(fields={'answer': 'hi'})

        class _FakeServer:
            def __init__(self, *a, **k):
                pass

            def serve_forever(self):
                raise KeyboardInterrupt

            def shutdown(self):
                _FakeServer.shut = True

        with (
            mock.patch.object(http.server, 'ThreadingHTTPServer', _FakeServer),
            mock.patch.object(rating.threading, 'Timer') as timer,
            mock.patch.object(rating.webbrowser, 'open'),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rating.serve([run], '/tmp/x.jsonl', ['d'], open_browser=True)
        self.assertTrue(_FakeServer.shut)
        timer.assert_called_once()

    def test_serve_rank_runs_until_interrupt(self):
        run_a = _variant_run('baseline', ['c1'])
        run_b = _variant_run('candidate', ['c1'])

        class _FakeServer:
            def __init__(self, *a, **k):
                pass

            def serve_forever(self):
                raise KeyboardInterrupt

            def shutdown(self):
                _FakeServer.shut = True

        with (
            mock.patch.object(http.server, 'ThreadingHTTPServer', _FakeServer),
            mock.patch.object(rating.threading, 'Timer') as timer,
            mock.patch.object(rating.webbrowser, 'open'),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rating.serve_rank(
                run_a, run_b, '/tmp/p.jsonl', ['d'], open_browser=True
            )
        self.assertTrue(_FakeServer.shut)
        timer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
