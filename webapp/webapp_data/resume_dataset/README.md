# Resume dataset

The resumes in this folder are sample data used for statistical comparison tests
(ranking pipeline evaluation/benchmarking), not real user submissions. They are
excluded from version control (see `.gitignore`) but stay on disk locally, so the
deployed container's `Dockerfile.web` (`COPY webapp /app`) still picks them up at
build time as long as they're present in the working directory the image is built from.
