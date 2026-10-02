"""Route a support message: one choice, one score, and one true/false question.

    uv run client/examples/basic.py
    uv run client/examples/basic.py --url http://gpu-host:8000 "Please refund my last invoice."
"""

import argparse
import json

from clef_client import ClefClient, choice, noul, score

parser = argparse.ArgumentParser()
parser.add_argument("message", nargs="?", default="Our checkout started returning errors and orders are blocked.")
parser.add_argument("--url", default="http://localhost:8000")
args = parser.parse_args()

client = ClefClient(args.url)
response = client.systemone(
    args.message,
    {
        "department": choice(
            {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            "Which team should handle the message?",
        ),
        "urgency": score(["Can wait", "This week", "Today"]),
        "outage": noul("Is a service down?"),
    },
)
print(json.dumps(response, indent=2, ensure_ascii=False))
