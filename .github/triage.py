"""Label PRs by changed paths. Run only the default branch's copy of this file.

No PR checkout, artifact, title or source is evaluated. File names are data.
Only our area/backend/scope labels are reconciled; maintainer priorities stay.
"""
import fnmatch
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.parse import quote

RULES = json.loads(Path(__file__).with_name('modules.json').read_text())
SUPPORT = {'docs', 'tests', 'build'}


def classify(paths):
    areas = {area for area, patterns in RULES.items()
             if any(fnmatch.fnmatchcase(path, pattern) for path in paths for pattern in patterns)}
    labels = {'area:'+a for a in areas}
    for path in paths:
        match = re.match(r'backends/modules/([a-z][a-z0-9_]{0,14})/', path)
        if match: labels.add('backend:'+match[1])
        for name in ['avx2', 'neon', 'metal', 'ref']:
            if path.startswith('backends/'+name+'.') or path.startswith('backends/'+name+'_'):
                labels.add('backend:'+name)
    if len(areas-SUPPORT)>1: labels.add('scope:cross-module')
    return labels


class GitHub:
    def __init__(self):
        repo = os.environ['GITHUB_REPOSITORY']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo): raise ValueError('Invalid repository')
        self.base = os.environ.get('GITHUB_API_URL', 'https://api.github.com')+'/repos/'+repo
        self.token = os.environ['GH_TOKEN']

    def request(self, path, method='GET', body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(self.base+path, data=data, method=method, headers={
            'Accept':'application/vnd.github+json', 'Authorization':'Bearer '+self.token,
            'X-GitHub-Api-Version':'2026-03-10', 'Content-Type':'application/json'})
        with urlopen(request, timeout=30) as response:
            raw=response.read()
            return json.loads(raw) if raw else None

    def pages(self, path, limit=100):
        result=[]
        for page in range(1, limit+1):
            chunk=self.request(path+('&' if '?' in path else '?')+f'per_page=100&page={page}')
            result.extend(chunk)
            if len(chunk)<100: return result
        raise ValueError('Pagination limit reached; classify this PR manually')


def label_pr(api, number):
    number=int(number)
    if number<=0: raise ValueError('Invalid pull request number')
    pr=api.request(f'/pulls/{number}')
    if pr['state']!='open': return
    files=api.pages(f'/pulls/{number}/files', limit=30)
    if len(files)!=pr['changed_files']: raise ValueError('Incomplete changed-file list')
    paths=[p for file in files for p in [file['filename'],file.get('previous_filename','')] if p]
    wanted=classify(paths)
    existing={x['name'] for x in api.pages('/labels')}
    for name in sorted(wanted|{'priority:high','priority:normal','priority:later'}):
        if name not in existing:
            api.request('/labels','POST',{'name':name,'color':'b60205' if name=='priority:high' else 'd4e8f7',
                'description':'Maintainer priority' if name.startswith('priority:') else 'Changed module; assigned from file paths'})
    current={x['name'] for x in api.pages(f'/issues/{number}/labels')}
    owned={name for name in current if name.startswith(('area:','backend:')) or name=='scope:cross-module'}
    # Add/remove our labels individually instead of replacing the whole set.
    # This preserves a maintainer's concurrent priority or status edits.
    for name in sorted(owned-wanted): api.request(f'/issues/{number}/labels/'+quote(name,safe=''),'DELETE')
    if wanted-current: api.request(f'/issues/{number}/labels','POST',{'labels':sorted(wanted-current)})
    print(f'PR #{number}: '+', '.join(sorted(wanted)))


def main():
    event=json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text()); api=GitHub()
    if os.environ['GITHUB_EVENT_NAME']=='workflow_dispatch':
        numbers=[int(event['inputs']['number'])]
    else:
        run=event['workflow_run']
        if run['event']!='pull_request': return
        # Fork events may omit pull_requests. Resolve against open PRs, then
        # read their current changed paths, never the workflow's downloaded files.
        numbers=[pr['number'] for pr in run.get('pull_requests',[])]
        if not numbers:
            numbers=[pr['number'] for pr in api.pages('/pulls?state=open') if pr['head']['sha']==run['head_sha']]
    for number in sorted(set(numbers)): label_pr(api, number)


if __name__=='__main__': main()
