"""Suite-declared plugin modules (``plugins:``) and the CLI flag.

The registry is written by ``@register`` at import time, so the only question
these tests answer is *who* does the importing and *when*. The when matters as
much as the who: loading a suite must not run a consumer's code, and starting a
run must.
"""

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from evalcore import errors, loader, models, plugins, runner


def _suite(**over):
    base = {
        'project': 'p',
        'suite': 's',
        'dataset': 'd',
        'dataset_version': 'v1',
        'mode_default': 'live',
        'adapter': {'type': '_plugin_probe'},
        'variants': {'baseline': {}},
    }
    base.update(over)
    return loader.SuiteConfig.model_validate(base)


class LoadTests(unittest.TestCase):
    def setUp(self):
        # The module-level cache is process state; keep tests independent of
        # each other and of whatever imported first.
        self._saved = set(plugins._loaded)
        plugins._loaded.clear()
        self.addCleanup(
            lambda: (
                plugins._loaded.clear(),
                plugins._loaded.update(self._saved),
            )
        )

    def test_imports_once_and_reports_only_fresh_names(self):
        with mock.patch.object(plugins.importlib, 'import_module') as imp:
            first = plugins.load(['json', 'json', ' csv '])
            second = plugins.load(['json'])
        self.assertEqual(first, ['json', 'csv'])
        self.assertEqual(second, [])
        self.assertEqual(
            [c.args[0] for c in imp.call_args_list], ['json', 'csv']
        )

    def test_blanks_and_none_are_skipped(self):
        # A comma-split CLI string with no value is the common case:
        # ''.split(',') is [''], not [].
        self.assertEqual(plugins.load(None), [])
        self.assertEqual(plugins.load(''.split(',')), [])
        self.assertEqual(plugins.load(['', '  ']), [])

    def test_unimportable_module_raises_config_error_naming_it(self):
        with self.assertRaises(errors.ConfigError) as ctx:
            plugins.load(['evalcore_no_such_plugin_module'])
        self.assertIn('evalcore_no_such_plugin_module', str(ctx.exception))
        # ConfigError is also a ValueError, so existing handlers keep working.
        self.assertIsInstance(ctx.exception, ValueError)

    def test_failed_import_is_not_cached_as_loaded(self):
        for _ in range(2):
            with self.assertRaises(errors.ConfigError):
                plugins.load(['evalcore_no_such_plugin_module'])
        self.assertNotIn('evalcore_no_such_plugin_module', plugins._loaded)

    def test_allow_cwd_imports_adds_cwd_once(self):
        saved = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved))
        cwd = str(pathlib.Path.cwd())
        # The test runner already has cwd on the path, so drop it first or the
        # insert branch never runs and this asserts nothing.
        sys.path[:] = [p for p in sys.path if p != cwd]

        plugins.allow_cwd_imports()
        self.assertEqual(sys.path[0], cwd)
        plugins.allow_cwd_imports()
        self.assertEqual(sys.path.count(cwd), 1)


class SuiteIntegrationTests(unittest.TestCase):
    def test_load_suite_accepts_plugins_and_defaults_to_empty(self):
        self.assertEqual(_suite().plugins, [])
        self.assertEqual(_suite(plugins=['a.b']).plugins, ['a.b'])

    def test_load_suite_does_not_import_them(self):
        """Reading a suite must stay side-effect free.

        A bad module name would raise if load_suite imported it; parsing and
        hashing a suite has to work on a file you have not decided to run.
        """
        body = (
            'project: p\nsuite: s\ndataset: d\n'
            'adapter: {type: http}\n'
            'plugins: [evalcore_no_such_plugin_module]\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'suite.yaml'
            path.write_text(body, encoding='utf-8')
            cfg = loader.load_suite(str(path))
        self.assertEqual(cfg.plugins, ['evalcore_no_such_plugin_module'])
        self.assertNotIn('evalcore_no_such_plugin_module', sys.modules)

    def test_run_suite_imports_them_before_building_the_adapter(self):
        """The registration has to land before the `type` lookup.

        The probe module registers `_plugin_probe`, which the suite names. If
        the import happened after build_adapter - or not at all - this raises
        an unknown-adapter ConfigError instead of running.
        """
        module = 'evalcore_plugin_probe'
        source = (
            'from evalcore import models\n'
            'from evalcore.adapters import base\n'
            '\n'
            '@base.register("_plugin_probe")\n'
            'class Probe:\n'
            '    async def invoke(self, case, variant):\n'
            '        return models.Output(fields={"text": "ok"})\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / f'{module}.py').write_text(
                source, encoding='utf-8'
            )
            sys.path.insert(0, tmp)
            self.addCleanup(lambda: sys.path.remove(tmp))
            self.addCleanup(lambda: sys.modules.pop(module, None))
            plugins._loaded.discard(module)
            self.addCleanup(lambda: plugins._loaded.discard(module))

            # No dataset dir needed; the point here is the registry lookup.
            with mock.patch.object(
                loader, 'load_cases', return_value=[models.Case(id='c1')]
            ):
                run = runner.run_suite_sync(
                    _suite(plugins=[module]), 'baseline'
                )

        self.assertEqual(run.results[0].output.fields['text'], 'ok')

    def test_run_suite_surfaces_a_bad_plugin_before_anything_else(self):
        suite = _suite(
            plugins=['evalcore_no_such_plugin_module'],
            variants={'baseline': {}},
        )
        with self.assertRaises(errors.ConfigError) as ctx:
            runner.run_suite_sync(suite, 'baseline')
        self.assertIn('evalcore_no_such_plugin_module', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
