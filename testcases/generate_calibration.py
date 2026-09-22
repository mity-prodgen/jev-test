#!/usr/bin/env python3
"""Generate testcases/calibration.yaml: a reproducible batch of synthetic
support tickets with known-correct department labels, for bucketing Jev's
reported confidence against empirical accuracy.

Ground truth is by construction (each template was written to clearly belong
to one department), so the label itself isn't in question - wording and
detail vary per instance so Jev's confidence still has room to move, and the
30 templates deliberately range from clear-cut to vocabulary that brushes
against a neighboring department.

Deterministic given SEED: rerun this script to regenerate an identical file.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

SEED = 20260921
N_PER_TEMPLATE = 3

DEPARTMENT_CRITERIA = {
    "billing": "Payment, subscription, invoicing, or refund issues",
    "technical": "Bugs, integration failures, or platform errors",
    "shipping": "Order fulfillment, carrier, or delivery problems",
}

PLANS = ["Starter", "Growth", "Pro", "Scale"]
AMOUNTS = [12, 19, 29, 49, 99, 149]
COUNTS = [3, 5, 7, 10, 14, 200]
DAYS = ["Monday", "Tuesday", "last Friday", "yesterday morning", "this week"]
ERROR_CODES = ["500", "502", "504", "429"]

BILLING_TEMPLATES = [
    "I was charged twice for my {plan} subscription this month, the extra ${amount} charge needs to be refunded.",
    "Can you explain why my invoice for order #{order_id} shows a ${amount} charge I don't recognize?",
    "My card on file expired and now I can't access my account, how do I update my payment method?",
    "I want to downgrade from the {plan} plan, but I'm still being billed at the higher rate.",
    "Your invoice from {day} has the wrong tax amount listed - can someone correct it?",
    "I canceled my subscription {days_ago} days ago but was still charged ${amount} yesterday.",
    "Do you offer annual billing instead of monthly? I'd like to switch my {plan} plan over.",
    "The proration on my upgrade to {plan} doesn't add up, I think I was overcharged by about ${amount}.",
    "I need a copy of my receipt for order #{order_id} for expense reporting, where can I download it?",
    "My payment failed but you still show my account as past due even though I already fixed my card info.",
]

TECHNICAL_TEMPLATES = [
    "The API keeps returning a {error_code} error whenever I try to sync inventory for more than {count} SKUs.",
    "Webhooks stopped firing after I regenerated my API key {days_ago} days ago, is there a known issue?",
    "Our storefront widget throws a blank screen in Safari but works fine in Chrome.",
    "We're getting rate-limited even though we're well under the documented request quota.",
    "The bulk import tool silently fails on CSVs larger than {count}MB with no error message.",
    "Order statuses aren't syncing back to our ERP through the integration since {day}.",
    "Can you tell me why the sandbox environment returns different data than production for the same query?",
    "Our checkout page has been throwing a JavaScript error since your platform update on {day}.",
    "The OAuth token refresh is failing intermittently, about 1 in {count} requests.",
    "Product images uploaded via the API aren't showing up on the live storefront for the {plan} plan.",
]

SHIPPING_TEMPLATES = [
    "Order #{order_id} has been stuck in 'label created' status for {count} days and hasn't moved.",
    "Several customers are asking why their tracking numbers show no updates since {day}.",
    "We printed the wrong shipping label size and now packages are getting rejected by the carrier.",
    "Can you tell me why {count} orders from {day} didn't generate shipping labels automatically?",
    "A customer's package for order #{order_id} was marked delivered but they say it never arrived.",
    "Our default carrier rates suddenly doubled overnight since {day}, did something change on your end?",
    "International orders aren't calculating customs fees correctly anymore as of {day}.",
    "We need to bulk-reprint shipping labels for {count} orders after a printer jam this morning.",
    "A shipment for order #{order_id} got returned to sender and I don't know how to process the refund.",
    "Why does the platform show 'out for delivery' for order #{order_id}, which was cancelled {days_ago} days ago?",
]


def fill(template: str, rng: random.Random) -> str:
    return template.format(
        plan=rng.choice(PLANS),
        amount=rng.choice(AMOUNTS),
        order_id=rng.randint(10000, 99999),
        count=rng.choice(COUNTS),
        day=rng.choice(DAYS),
        days_ago=rng.choice([2, 3, 5, 10, 21]),
        error_code=rng.choice(ERROR_CODES),
    )


def build_cases() -> list[dict]:
    rng = random.Random(SEED)
    cases = []
    groups = [
        ("billing", BILLING_TEMPLATES),
        ("technical", TECHNICAL_TEMPLATES),
        ("shipping", SHIPPING_TEMPLATES),
    ]
    for dept, templates in groups:
        for t_idx, template in enumerate(templates):
            for v in range(N_PER_TEMPLATE):
                state = fill(template, rng)
                cases.append(
                    {
                        "id": f"{dept}_{t_idx:02d}_{v}",
                        "state": state,
                        "tags": ["calibration", dept],
                        "questions": {
                            "department": {
                                "type": "choice",
                                "instructions": "Which team should handle this support ticket?",
                                "criteria": DEPARTMENT_CRITERIA,
                            }
                        },
                        "expected": {"department": dept},
                    }
                )
    rng.shuffle(cases)
    return cases


def main() -> None:
    cases = build_cases()
    batch = {
        "batch": "calibration",
        "description": (
            f"Synthetic ShopFlow support tickets ({len(cases)} cases, seed={SEED}) with "
            "known-correct department labels (billing/technical/shipping), for bucketing "
            "Jev's reported confidence against empirical accuracy. Regenerate with "
            "generate_calibration.py for an identical file."
        ),
        "cases": cases,
    }
    out_path = Path(__file__).parent / "calibration.yaml"
    with out_path.open("w") as f:
        yaml.dump(batch, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
