"""Temporary token-protected HTTP transfer of this task's generated archive only."""
import argparse
import http.server
import os
import shutil
import subprocess
import sys
from pathlib import Path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--token', required=True)
    p.add_argument('--port', type=int, default=18765)
    p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    target = Path(a.file).resolve()
    if not target.name.startswith('dic_review_') or target.suffix != '.zip':
        raise ValueError(target)
    if a.launch:
        args = [x for x in sys.argv if x != '--launch']
        with open('/tmp/dic_review_transfer_http.log', 'ab') as log:
            child = subprocess.Popen([sys.executable, *args], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
        print('HTTP_PID', child.pid)
        raise SystemExit(0)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/' + a.token:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Length', str(target.stat().st_size))
            self.send_header('Content-Type', 'application/zip')
            self.end_headers()
            with target.open('rb') as source:
                shutil.copyfileobj(source, self.wfile, length=1024*1024)

    http.server.ThreadingHTTPServer(('0.0.0.0', a.port), Handler).serve_forever()
