"""Watch any match with its tracking overlaid."""

import argparse
import webbrowser

from jaicodyn.viz.server import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match", nargs="?", default="", help="for example idsse/J03WMX")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = "http://127.0.0.1:{}/#{}".format(args.port, args.match)
    server = serve(args.port)
    print("Viewer at {}, Ctrl+C to stop".format(url))
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
