"""Comment tolerance must never accept an algorithm or preserved-evidence change."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_source_comments import compare_source, verify_sources, digest, safe_path


class SourceComparisonTests(unittest.TestCase):
    def test_accepts_comments_and_blank_lines(self):
        compare_source(b'x = 1\ny = x + 2\n',
                       '# Giải thích biến\nx = 1  # hệ số\n\ny = x + 2\n'.encode())

    def test_accepts_line_endings(self):
        compare_source(b'x = 1\n', b'# note\r\nx = 1\r\n')

    def test_hash_inside_string_is_not_a_comment(self):
        with self.assertRaises(ValueError):
            compare_source(b'x = "# original"\n', b'x = "# changed"\n')

    def test_rejects_changed_constant_operator_and_name(self):
        for changed in (b'x = 2 + y\n', b'x = 1 - y\n', b'x = 1 + z\n'):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                compare_source(b'x = 1 + y\n', changed)

    def test_rejects_changed_indentation(self):
        with self.assertRaises(ValueError):
            compare_source(b'if ok:\n    x = 1\ny = 2\n', b'if ok:\n    x = 1\n    y = 2\n')

    def test_rejects_changed_docstring_and_multiline_string(self):
        for before, after in ((b'"""old"""\nx=1\n', b'"""new"""\nx=1\n'),
                              (b'x="""# old"""\n', b'x="""# new"""\n')):
            with self.subTest(before=before), self.assertRaises(ValueError):
                compare_source(before, after)

    def test_rejects_invalid_python_and_type_directive(self):
        for after in (b'x = (\n', b'x=1 # type: ignore\n'):
            with self.subTest(after=after), self.assertRaises(ValueError):
                compare_source(b'x=1\n', after)


class PreservedReferenceTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder) / 'working'
        ref = Path(folder) / 'reference'
        source = ref / 'source_snapshot'
        (root/'step7h').mkdir(parents=True)
        (source/'step7h').mkdir(parents=True)
        for name in ('run_step7h.py', 'step7h/example.py'):
            (source/name).write_text('x = 1\n', encoding='utf-8')
            (root/name).write_text('# explanation\nx = 1\n', encoding='utf-8')
        plan = {'format':'six_method_benchmark_7h_v1',
                'code_sha256':{name:digest(source/name) for name in ('run_step7h.py','step7h/example.py')}}
        (ref/'plan.json').write_text(json.dumps(plan), encoding='utf-8')
        (ref/'completion.json').write_text(json.dumps({'plan_sha256':digest(ref/'plan.json')}), encoding='utf-8')
        return root, ref, source

    @patch('verify_source_comments.READABLE_MODULES', ())
    def test_accepts_only_current_comments(self):
        with TemporaryDirectory() as folder:
            root, ref, source = self.fixture(folder)
            result = verify_sources(ref, root)
            self.assertEqual(len(result['comment_only_files']), 2)

    @patch('verify_source_comments.READABLE_MODULES', ())
    def test_rejects_tampered_snapshot_even_if_only_comment(self):
        with TemporaryDirectory() as folder:
            root, ref, source = self.fixture(folder)
            (source/'run_step7h.py').write_text('# changed evidence\nx = 1\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                verify_sources(ref, root)

    @patch('verify_source_comments.READABLE_MODULES', ())
    def test_rejects_added_source_file_and_changed_plan(self):
        with TemporaryDirectory() as folder:
            root, ref, source = self.fixture(folder)
            (root/'step7h/extra.py').write_text('x=1\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source-file set'):
                verify_sources(ref, root)
            (root/'step7h/extra.py').unlink()
            (ref/'plan.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'original 7h plan'):
                verify_sources(ref, root)

    def test_rejects_unsafe_manifest_paths(self):
        for name in ('../secret', '/secret', 'C:\\secret'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_path(Path.cwd(), name)


if __name__ == '__main__':
    unittest.main()
