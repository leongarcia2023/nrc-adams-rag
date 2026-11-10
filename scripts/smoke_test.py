"""Verify import, chunking, retrieval and Markdown export without network access."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory(prefix="nrc-smoke-") as directory:
        root = Path(directory)
        records = root / "fixture.jsonl"
        records.write_text(json.dumps({"AccessionNumber": "DEMO-LOCAL-001",
            "DocumentTitle": "Synthetic installation fixture — not an NRC record",
            "content": "A fictional inspection found a leaking pump seal. The fictional crew replaced the seal and repeated the pump test successfully.",
            "Url": "https://example.invalid/synthetic-fixture"}) + "\n")
        command = [sys.executable, "-m", "adams_rag", "--db", str(root / "smoke.sqlite"), "--json"]
        def run(*arguments):
            result = subprocess.run(command + list(arguments), check=True, text=True, capture_output=True)
            return json.loads(result.stdout)
        assert run("import-jsonl", str(records))["imported"] == 1
        result = run("retrieve", "leaking pump seal", "--offline", "--no-semantic",
                     "--catalog", str(root / "absent.sqlite"), "--index-dir", str(root / "index"))
        assert result["sources"] and result["sources"][0]["accession"] == "DEMO-LOCAL-001"
        exported = root / "document.md"
        run("export-document", "DEMO-LOCAL-001", "--output", str(exported))
        assert "fictional" in exported.read_text()
        print("Passed: import, chunking, retrieval, source identity, Markdown export")


if __name__ == "__main__":
    main()
