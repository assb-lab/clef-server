"""Ask questions about an image (local path or http(s) URL).

    uv run client/examples/image.py receipt.jpg
    uv run client/examples/image.py receipt.jpg --question "Is the receipt total legible?"
"""

import argparse
import json

from clef_client import ClefClient, noul

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--question", default="Is the receipt total legible?")
parser.add_argument("--url", default="http://localhost:8000")
args = parser.parse_args()

client = ClefClient(args.url)
answers = client.answers(
    {"task": "Review the attached image."},
    {"answer": noul(args.question)},
    images=[args.image],
)
print(json.dumps(answers, indent=2, ensure_ascii=False))
