from __future__ import annotations

"""Cloud Run Job entry point for MiniLM-backed job-side embedding enrichment."""

from data_pipeline.common.db import connect
from data_pipeline.common.export_eligibility import require_export_eligibility_schema
from data_pipeline.common.schema import initialize_schema
from data_pipeline.enrichment import add_job_features


def run_embedding_enrichment_job() -> None:
    print("embedding enrichment: using configured database", flush=True)
    with connect() as conn:
        print("embedding enrichment step 1/2: initializing schema", flush=True)
        initialize_schema(conn)
        print("embedding enrichment step 2/2: embedding prepared job ranking features", flush=True)
        require_export_eligibility_schema(conn)
        add_job_features.run_embeddings(conn)
    print("embedding enrichment: finished successfully", flush=True)


if __name__ == "__main__":
    run_embedding_enrichment_job()
