"""Consolidated verb lexicon for artifact extraction.

Centralized output/input verb banks used by both artifact_extractor.py (Pass 0)
and dependency_tree.py (Pass 2). Single source of truth eliminates sync risk
between two modules.
"""

# Verbs indicating artifact production (output, generation, transmission).
OUTPUT_VERBS = (
    "provide", "generate", "produce", "output", "compute", "calculate",
    "emit", "return", "report", "transmit", "send", "create", "store",
    "record", "log", "measure", "detect", "raise", "issue", "publish",
    "broadcast", "write", "save", "derive", "forward",
)

# Verbs indicating artifact consumption (input, processing, receipt).
INPUT_VERBS = (
    "receive", "require", "use", "read", "accept", "consume", "depend",
    "retrieve", "obtain", "process", "monitor", "respond", "trigger",
    "subscribe", "listen", "load", "fetch", "import", "request", "poll",
)
