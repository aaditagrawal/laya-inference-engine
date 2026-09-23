"""Versioned synthetic workloads, not a labeled model-quality benchmark."""

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                "sales": "pricing and new contracts", "other": "everything else"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need an urgent response?"},
    "severity": {"type": "score", "instructions": "How severe is the problem?",
                 "criteria": ["minor inconvenience", "work is delayed", "business cannot operate"]},
    "single": {"type": "choice", "instructions": "Select the available category.", "criteria": ["ticket"]},
}

STATES = [
    "We were billed twice for March. Refund the duplicate payment today or we will cancel.",
    "The production API has been down for three hours and our entire team is blocked.",
    "Could you send pricing for 100 seats? We are planning next year's budget.",
    "Thanks for the update. Everything works and there is no need to respond.",
    {"subject": "Login problem", "body": "Resetting the password did not help. Error 502 persists."},
    [{"role": "user", "content": "Please cancel the refund request. The charge is correct."}],
    "[MASK] This literal marker must be sanitized. [MASK] No billing issue is reported.",
    "",
    "The invoice says payment is overdue, but the bank transfer completed yesterday. " * 12,
    "Service interruption investigation. The customer asks for an update. " * 100,
]


def workload(batch=1, length="short"):
    state = STATES[0]
    if length == "medium":
        state = STATES[8]
    elif length == "long":
        state = STATES[9]
    definitions = list(QUESTIONS.values())[:3]
    questions = {f"q{i}": definitions[i % len(definitions)] for i in range(batch)}
    return {"state": state, "questions": questions}
