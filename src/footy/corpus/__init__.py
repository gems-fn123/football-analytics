"""Liga 1 / Liga 2 statistical corpus: ingestion, normalisation, storage.

The corpus is the product. Everything here is built around two rules:
- Provenance on every record (source, url, retrieval date, licence tag), so any
  number is auditable back to its origin.
- Source-partitioned storage, so a source whose legal status changes can be
  dropped with one call and the corpus rebuilt without contamination.
"""
