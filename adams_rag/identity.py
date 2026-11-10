"""NRC record identity is distinct from a potentially shared accession."""


def record_id(document):
    value = document.get("IdT") or document.get("Id")
    if isinstance(value, str) and value.strip():
        return "id:" + value.strip().strip("{}").casefold()
    return "accession:" + document["AccessionNumber"].casefold()


def is_legacy(document):
    return str(document.get("IsLegacy", "")).strip().lower() in {"yes", "true", "1"}


def source_key(document):
    library = "legacy" if is_legacy(document) else "main"
    identity = record_id(document)
    if library == "legacy" and not identity.startswith("id:"):
        raise ValueError("Legacy records require an NRC record ID; accession alone is ambiguous")
    return f"nrc:{library}:{identity}"
