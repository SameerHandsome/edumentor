#!/usr/bin/env python3
"""
Simple dev server for EduMentor frontend.
Run: python serve.py
Then open: http://localhost:3000
"""
import http.server
import socketserver
import os

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Proper MIME types — critical for CSS to load correctly
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
    
    def guess_type(self, path):
        # Ensure correct MIME types are sent
        if path.endswith('.css'):
            return 'text/css'
        if path.endswith('.js'):
            return 'application/javascript'
        return super().guess_type(path)

    def log_message(self, format, *args):
        print(f"  {self.address_string()} - {format % args}")

print(f"╔══════════════════════════════════════╗")
print(f"║   EduMentor Dev Server               ║")
print(f"║   http://localhost:{PORT}              ║")
print(f"║   Ctrl+C to stop                     ║")
print(f"╚══════════════════════════════════════╝")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()