"""Run: python -B -m src.minecraft_service [--viewer]. Binds loopback only."""
import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading
from src.minecraft_brain import MinecraftSession, NeuralAdapter
from src.minecraft_learning import MinecraftLearner


def make_server(session, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *args):
            pass

        def send(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/status':
                self.send(404, {'error': 'Unknown endpoint'})
                return
            with session.lock:
                self.send(200, session.status)

        def do_POST(self):
            if self.path != '/step' or self.headers.get('Origin'):
                self.send(403, {'error': 'Local mod requests only'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 131072:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                self.send(200, session.step(data))
            except (ValueError, TypeError, KeyError) as error:
                self.send(400, {'error': str(error)})
            except RuntimeError as error:
                self.send(503, {'error': str(error)})
    return HTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--policy', type=Path, default=Path('data/cache/minecraft/policy.json'))
    parser.add_argument('--log', type=Path, default=Path('runs/minecraft/events.jsonl'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent services from silently overwriting the same memory.
    with args.policy.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from src.live_brain import load_live_graph
        graph = load_live_graph()
        adapter = NeuralAdapter(graph, device=args.device)
        learner = MinecraftLearner(args.policy, training=not args.evaluate)
        session = MinecraftSession(adapter, learner, args.log)
        server = make_server(session, args.port)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        print(f'FlyBrain listening on http://127.0.0.1:{args.port}; memory: {args.policy}', flush=True)
        print('In Minecraft use /flybrain spawn. F8 pauses/resumes; F9 follows the fly. Synthetic sensors; experimental learned readout.', flush=True)
        try:
            if args.viewer:
                from src.minecraft_viewer import run_viewer
                run_viewer(session, graph)
            else:
                while worker.is_alive():
                    worker.join(.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
            learner.save()


if __name__ == '__main__':
    main()
