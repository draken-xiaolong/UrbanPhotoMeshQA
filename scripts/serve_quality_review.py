#!/usr/bin/env python3
"""Loopback-only 3D review app, with durable SQLite scores bound to asset hashes."""
import argparse
import csv
import io
import json
import mimetypes
import secrets
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit('请连接移动硬盘，并确认数据目录存在。')
    records = json.loads((root/'manifest.json').read_text())['records']
    by_id = {r['variant']: r for r in records}
    machine_path = root/'machine_scores.json'
    machine = json.loads(machine_path.read_text())['scores'] if machine_path.exists() else {}
    machine = {key:value for key,value in machine.items() if key in by_id and value['digest']==by_id[key]['content_digest']}
    static = Path(__file__).resolve().parents[1]/'apps/quality-review'
    db_path = root/'review_scores.sqlite3'
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS scores (id TEXT PRIMARY KEY, digest TEXT, scale INTEGER, uncertain INTEGER, note TEXT, updated TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS history (id TEXT, digest TEXT, scale INTEGER, uncertain INTEGER, note TEXT, updated TEXT)')
    token = secrets.token_urlsafe(24)
    allowed = {}
    for record in records:
        path = Path(record['gltf_path']).resolve()
        prefix = '/assets/'+record['variant']+'/'
        allowed[prefix+'model.gltf'] = path
        document = json.loads(path.read_text())
        for entry in document.get('buffers', [])+document.get('images', []):
            uri = entry.get('uri','')
            dependency = (path.parent/uri).resolve()
            if not dependency.is_relative_to(path.parent):
                raise ValueError('Asset dependency outside package')
            allowed[prefix+uri] = dependency

    def scores():
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            return {row['id']: dict(row) for row in db.execute('SELECT * FROM scores')
                    if row['id'] in by_id and row['digest'] == by_id[row['id']]['content_digest']}

    def training_rows():
        human = scores()
        result = []
        first_by_digest = {}
        for r in records:
            h = human.get(r['variant'],{}); m = machine.get(r['variant'],{})
            effective = h.get('scale') if h.get('scale') is not None else m.get('scale')
            source = 'human' if h.get('scale') is not None else ('machine' if m.get('scale') is not None else 'unrated')
            duplicate_of = first_by_digest.get(r['content_digest'], '')
            first_by_digest.setdefault(r['content_digest'],r['variant'])
            result.append({'asset_id':r['asset_id'],'variant':r['variant'],'content_digest':r['content_digest'],
                'gltf_path':r['gltf_path'],'machine_scale':m.get('scale'),'human_scale':h.get('scale'),
                'effective_scale':effective,'label_source':source,'machine_method':m.get('method'),
                'machine_reason':m.get('reason',''),'machine_review_flags':';'.join(m.get('review_flags',[])),
                'duplicate_of':duplicate_of,'independent_sample':not bool(duplicate_of),
                'uncertain':h.get('uncertain',0),'note':h.get('note',''),'updated':h.get('updated','')})
        return result

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, mime='application/json; charset=utf-8'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == '/api/patches':
                variant = parse_qs(urlparse(self.path).query).get('variant',[''])[0]
                if variant not in by_id: return self.respond(404,{'error':'unknown asset'})
                try:
                    from review_patch_data import patch_data
                    return self.respond(200,patch_data(str(root),variant))
                except (OSError,ValueError,KeyError) as exc:
                    return self.respond(400,{'error':f'Patch数据不可用: {exc}'})
            if path == '/api/session':
                return self.respond(200, {'token':token,'records':records,'scores':scores(),
                                         'machine_scores':machine,
                                         'building':records[0]['asset_id'],'revision':root.name})
            if path == '/api/training-labels':
                return self.respond(200, {'policy':'human_if_present_else_machine','records':training_rows()})
            if path == '/api/export':
                buffer = io.StringIO(); writer = csv.writer(buffer)
                rows = training_rows()
                writer.writerow(list(rows[0]))
                for row in rows:
                    writer.writerow(list(row.values()))
                return self.respond(200, ('\ufeff'+buffer.getvalue()).encode(), 'text/csv; charset=utf-8')
            target = allowed.get(path)
            if target is None:
                relative = 'index.html' if path == '/' else path.lstrip('/')
                target = (static/relative).resolve()
                if not target.is_relative_to(static.resolve()) or target.suffix not in ('.html','.js','.css'):
                    return self.respond(404, {'error':'not found'})
            try:
                content = target.read_bytes()
            except OSError:
                return self.respond(404, {'error':'文件不可用，请检查移动硬盘'})
            return self.respond(200,content,mimetypes.guess_type(target.name)[0] or 'application/octet-stream')

        def do_POST(self):
            if self.path != '/api/score' or self.headers.get('X-Review-Token') != token:
                return self.respond(403, {'error':'invalid session'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 20000: raise ValueError('请求过大')
                data = json.loads(self.rfile.read(length))
                record = by_id[data['id']]
                scale = data.get('scale')
                if scale is not None and (type(scale) is not int or scale not in range(1,6)):
                    raise ValueError('scale必须为1～5')
                if data.get('digest') != record['content_digest']: raise ValueError('资产版本变化，请刷新')
                note = str(data.get('note',''))[:3000]
                row = (data['id'],record['content_digest'],scale,int(bool(data.get('uncertain'))),note,
                       datetime.now(timezone.utc).isoformat())
                with sqlite3.connect(db_path) as db:
                    db.execute('INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?)',row)
                    db.execute('INSERT INTO history VALUES (?,?,?,?,?,?)',row)
                self.respond(200, {'saved':True,'score':scores()[data['id']]})
            except (KeyError, ValueError, TypeError) as exc:
                self.respond(400, {'error':str(exc)})

        def log_message(self, *args):
            pass

    print(f'Quality review: http://127.0.0.1:{args.port}   scores: {db_path}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__': main()
