"""PR classification and label preservation, without a GitHub token/network."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('triage',Path(__file__).resolve().parent.parent/'.github/triage.py')
triage=importlib.util.module_from_spec(spec); spec.loader.exec_module(triage)


class Triage(unittest.TestCase):
    def test_module_with_supporting_docs_and_tests(self):
        labels=triage.classify(['backends/modules/tiny/backend.c','docs/BACKEND_MODULES.md','tests/test_native.c'])
        self.assertEqual(labels, {'area:backend','backend:tiny','area:docs','area:tests'})

    def test_cross_module_and_unsafe_backend_name(self):
        labels=triage.classify(['src/head.c','src/plane.c','backends/modules/$(command)/backend.c'])
        self.assertIn('scope:cross-module',labels)
        self.assertFalse(any(x.startswith('backend:') for x in labels))

    def test_reconcile_preserves_manual_priority(self):
        class API:
            def __init__(self): self.writes=[]
            def request(self, path, method='GET', body=None):
                if path=='/pulls/7': return {'state':'open','changed_files':1}
                self.writes.append((path,method,body))
            def pages(self, path, **kwargs):
                if path=='/pulls/7/files': return [{'filename':'backends/modules/new/backend.c', 'previous_filename':'backends/modules/old/backend.c'}]
                if path=='/labels': return [{'name':n} for n in ['priority:high','priority:normal','priority:later']]
                if path=='/issues/7/labels': return [{'name':n} for n in ['priority:high','reviewed','area:head']]
                raise AssertionError(path)
        api=API(); triage.label_pr(api,7)
        self.assertIn(('/issues/7/labels/area%3Ahead','DELETE',None),api.writes)
        added=[x[2]['labels'] for x in api.writes if x[:2]==('/issues/7/labels','POST')][0]
        self.assertEqual(set(added),{'area:backend','backend:new','backend:old'})
        self.assertFalse(any('priority' in x[0] or x[1]=='PUT' for x in api.writes))


if __name__=='__main__': unittest.main()
