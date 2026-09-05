"""Score persistence and path restrictions, without touching real annotations."""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import pytest


def test_review_save_reload_export_and_validation(tmp_path):
    asset = tmp_path/'model.gltf'
    asset.write_text('{"buffers": [], "images": []}')
    (tmp_path/'manifest.json').write_text(json.dumps({'records':[{
        'variant':'clean','asset_id':'test','gltf_path':str(asset),'content_digest':'test-digest'},
        {'variant':'duplicate','asset_id':'test','gltf_path':str(asset),'content_digest':'test-digest'}]}))
    (tmp_path/'machine_scores.json').write_text(json.dumps({'scores':{'clean':{
        'scale':5,'digest':'test-digest','method':'fixture'}}}))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    script = Path(__file__).resolve().parents[1]/'scripts/serve_quality_review.py'
    proc = subprocess.Popen([sys.executable,str(script),'--root',str(tmp_path),'--port',str(port)],stdout=subprocess.DEVNULL)
    base = f'http://127.0.0.1:{port}'
    try:
        for _ in range(50):
            try:
                session=json.load(urlopen(base+'/api/session'));break
            except URLError: time.sleep(.05)
        else: raise AssertionError('server did not start')
        fallback = json.load(urlopen(base+'/api/training-labels'))['records'][0]
        assert fallback['effective_scale']==5 and fallback['label_source']=='machine'
        duplicate=json.load(urlopen(base+'/api/training-labels'))['records'][1]
        assert duplicate['duplicate_of']=='clean' and duplicate['independent_sample'] is False
        def post(scale,digest='test-digest'):
            return urlopen(Request(base+'/api/score',data=json.dumps({'id':'clean','digest':digest,'scale':scale,'note':'测试备注','uncertain':True}).encode(),headers={'X-Review-Token':session['token'],'Content-Type':'application/json'}))
        assert json.load(post(4))['saved']
        overridden = json.load(urlopen(base+'/api/training-labels'))['records'][0]
        assert overridden['effective_scale']==4 and overridden['label_source']=='human'
        assert overridden['machine_scale']==5
        assert json.load(urlopen(base+'/api/session'))['scores']['clean']['scale']==4
        assert '测试备注' in urlopen(base+'/api/export').read().decode('utf-8-sig')
        for scale,digest in [(0,'test-digest'),(True,'test-digest'),(4,'wrong-digest')]:
            with pytest.raises(HTTPError) as exc:post(scale,digest)
            assert exc.value.code==400
        with pytest.raises(HTTPError):urlopen(base+'/../pyproject.toml')
        assert json.load(post(None))['score']['scale'] is None
        restored = json.load(urlopen(base+'/api/training-labels'))['records'][0]
        assert restored['effective_scale']==5 and restored['label_source']=='machine'
    finally:
        proc.terminate();proc.wait(timeout=5)
