# Latmay

Latmay is an open-source job recommendation engine. It scrapes job postings from company career sites, enriches them into structured, searchable data, and ranks them against a submitted resume using a mix of hard filters and semantic text matching.

## How it works

The system is a four-stage pipeline plus a web app, run as separate services:

1. **Ingestion** — scrapes job postings directly from company career pages across several applicant-tracking systems (Ashby, Greenhouse, Lever, Workday, iCIMS, Workable) and writes raw postings into PostgreSQL.
2. **Enrichment** — processes each posting to extract structured information: cleaned text, parsed requirements, years-of-experience signals, security-clearance requirements, normalized location, and sentence-embedding vectors for semantic matching.
3. **Export** — bundles recently enriched jobs into ranking artifacts (CSV/JSONL/NPZ) and uploads them to cloud storage for the web app to serve from.
4. **Web app** — a Flask/Gunicorn app where a user submits a resume. Pages include a home/submit form, ranked results, and standard about/privacy/terms pages.

### Ranking

Given a resume, the web app narrows and orders candidate jobs in stages:

- **Hard filters** remove jobs that are categorically ineligible — e.g. wrong country/location, years-of-experience mismatch, unmet security-clearance requirements.
- **Semantic ranking** scores the remaining jobs by similarity between the resume and each job's text, using sentence embeddings (MiniLM) compared at the word, phrase, and requirements level, with an optional cross-encoder re-ranking pass and an optional LLM-based bad-match screen.
- Additional statistical filters (seniority gap, technology-category overlap, outlier detection) remove jobs that pass the semantic score but don't actually fit the resume.

## License

MIT — see [`LICENSE`](LICENSE).
