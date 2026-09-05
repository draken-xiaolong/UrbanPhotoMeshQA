"""Discover complete V3 buildings and proxy independent review stores on loopback."""
import argparse
import http.client
import json
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8765)
    args=p.parse_args();datasets=[];children=[]
    for manifest in sorted((args.root/'assets').glob('*/process_v*/manifest.json')):
        root=manifest.parent
        if not (root/'machine_scores.json').exists() or not (root/'visual_patch_audit.json').exists():continue
        records=json.loads(manifest.read_text())['records']
        key=records[0]['asset_id'] + ('__'+root.name if root.name!='process_v2' else '')
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('serve_quality_review.py')),'--root',str(root),'--port',str(port)])
        children.append(proc)
        for _ in range(100):
            try:
                c=http.client.HTTPConnection('127.0.0.1',port,timeout=1);c.request('GET','/api/session');r=c.getresponse();r.read();c.close();break
            except OSError:time.sleep(.05)
        else:raise RuntimeError(f'Worker failed: {key}')
        metadata_path=root.parent/'clean'/'metadata.json'
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        datasets.append({'id':key,'name':metadata.get('display_name','建筑')+' · '+root.name, 'count':len(records),'port':port})
    if not datasets:raise SystemExit('未找到完整数据集，请检查移动硬盘。')
    by_id={d['id']:d for d in datasets}
    default='B355201752601063A0' if 'B355201752601063A0' in by_id else datasets[0]['id']
    if 'B415722108801063A0__process_v3' in by_id:default='B415722108801063A0__process_v3'
    class Gateway(BaseHTTPRequestHandler):
        def do_GET(self):self.proxy()
        def do_POST(self):self.proxy()
        def proxy(self):
            if self.path=='/api/datasets':
                body=json.dumps({'default':default,'datasets':[{k:v for k,v in d.items() if k!='port'} for d in datasets]}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            path=self.path;dataset=by_id[default]
            if path.startswith('/d/'):
                parts=path.split('/',3)
                if len(parts)!=4 or parts[2] not in by_id:self.send_error(404);return
                dataset=by_id[parts[2]];path='/'+parts[3]
            n=int(self.headers.get('Content-Length',0))
            if n>20000:self.send_error(413);return
            payload=self.rfile.read(n) if n else None
            headers={k:v for k,v in self.headers.items() if k.lower() not in ('host','connection','content-length')}
            c=http.client.HTTPConnection('127.0.0.1',dataset['port'],timeout=120)
            try:
                c.request(self.command,path,body=payload,headers=headers);response=c.getresponse();body=response.read()
                self.send_response(response.status)
                for k,v in response.getheaders():
                    if k.lower() not in ('server','date','connection','transfer-encoding'):self.send_header(k,v)
                self.end_headers();self.wfile.write(body)
            except (OSError,http.client.HTTPException):self.send_error(502,'Dataset unavailable; check drive')
            finally:c.close()
        def log_message(self,*args):pass
    print(f'{len(datasets)} datasets: http://127.0.0.1:{args.port}',flush=True)
    try:ThreadingHTTPServer(('127.0.0.1',args.port),Gateway).serve_forever()
    finally:
        for proc in children:proc.terminate()
        for proc in children:proc.wait(timeout=5)


if __name__=='__main__':main()
