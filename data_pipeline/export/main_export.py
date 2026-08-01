from __future__ import annotations

"""Cloud Run Job entry point for ranking artifact export."""

from data_pipeline.common.db import connect
from data_pipeline.common.export_eligibility import require_export_eligibility_schema
from data_pipeline.common.schema import initialize_schema
from data_pipeline.export import export_recent_jobs_to_gcs


def run_export_job() -> None:
    print("export: using configured database", flush=True)
    with connect() as conn:
        print("export step 1/2: initializing schema", flush=True)
        initialize_schema(conn)
        print("export step 2/2: exporting ranking artifacts to GCS", flush=True)
        require_export_eligibility_schema(conn)
        export_recent_jobs_to_gcs.run(conn)
    print("export: finished successfully", flush=True)


if __name__ == "__main__":
    run_export_job()
