"""Veris context engine — the cloud half of the dual-LLM architecture.

Phone GPS fix in -> maps enrichment (Google Maps, OSM fallback) -> compact
context pre-prompt out, in exactly the format the pack VLM's `--context`
slot consumes. Runs on the dev box for the prototype; the API shape is what
a real cloud deployment would expose.
"""

__version__ = "0.1.0"
