"""Make standalone SQLite backups, including data still present in WAL files."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if not source.is_dir():
        parser.error(f"Source directory does not exist: {source}")
    if source == output:
        parser.error("Output directory must differ from the source directory")
    databases = sorted(source.glob("*.sqlite"))
    if not databases:
        parser.error(f"No .sqlite databases found in {source}")
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / path.name for path in databases if (output / path.name).exists()]
    if existing:
        parser.error(f"Refusing to overwrite existing snapshot: {existing[0]}")
    for path in databases:
        target = output / path.name
        print(f"Backing up {path.name}", flush=True)
        created_target = False
        try:
            # Reserve the filename without overwriting a file created by another
            # process after the initial existence check.
            with target.open("xb"):
                created_target = True
            # mode=ro includes committed WAL content without changing the source.
            # sqlite3's own context manager manages transactions, but does not
            # close connections. Explicitly close both before any cleanup.
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as src:
                with closing(sqlite3.connect(target)) as dst:
                    src.backup(dst)
                    result = dst.execute("PRAGMA quick_check").fetchone()[0]
                    if result != "ok":
                        raise RuntimeError(f"SQLite quick_check failed: {result}")
        except BaseException:
            if created_target:
                try:
                    target.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    # Keep the backup error as the primary exception. A locked
                    # or incomplete copy must not be treated as a snapshot.
                    print(f"Could not remove incomplete snapshot {target}: {cleanup_error}", file=sys.stderr)
            raise
        print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
