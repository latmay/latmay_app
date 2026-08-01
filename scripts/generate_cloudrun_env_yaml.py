from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_PATH = PROJECT_ROOT / "cloudrun.env.yaml"
SKIP_KEYS = {
    "DATABASE_URL",
    "DB_PASSWORD",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "K_SERVICE",
    "LOCAL_DATABASE_URL",
    "OPENAI_API_KEY",
    "PORT",
    "POSTGRES_PASSWORD",
}
SECRET_KEY_PARTS = ("API_KEY", "CREDENTIAL", "DATABASE_URL", "PASSWORD", "SECRET", "TOKEN")
ALLOWED_PUBLIC_KEYS = {"FIREBASE_WEB_API_KEY"}


def strip_surrounding_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_env_file(path: Path) -> dict[str, str]:
    env_vars: dict[str, str] = {}

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: expected KEY=value.")

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid .env line {line_number}: key cannot be empty.")

        if key in SKIP_KEYS or (
            key not in ALLOWED_PUBLIC_KEYS
            and any(secret_part in key.upper() for secret_part in SECRET_KEY_PARTS)
        ):
            continue

        env_vars[key] = strip_surrounding_quotes(value)

    return env_vars


def quote_yaml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_cloudrun_yaml(path: Path, env_vars: dict[str, str]) -> None:
    lines = [f"{key}: {quote_yaml_string(value)}" for key, value in env_vars.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Could not find .env at {ENV_PATH}")

    env_vars = parse_env_file(ENV_PATH)
    write_cloudrun_yaml(OUTPUT_PATH, env_vars)
    print(f"Wrote {OUTPUT_PATH.name} with {len(env_vars)} environment variables.")


if __name__ == "__main__":
    main()
