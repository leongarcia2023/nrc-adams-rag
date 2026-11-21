"""Bounded live retrieval, local hybrid search, context assembly and cited RAG."""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

from .client import BudgetStop, unpack_search
from .ingest import archive
from .semantic_index import matches_filters
from .identity import is_legacy, record_id, source_key
from .storage import StoragePause, require_space

ACCESSION = re.compile(r"\bML\d{5}[A-Z0-9]\d{3}\b", re.I)
STOP = set("a an the what which where when how why are is was were did does do to from of for on in at and or with please find show explain tell me all any can could would should documents document nrc adams database reports report about".split())


def fallback_queries(question):
    from .query_terms import subject_phrases, phrase_expression
    phrases = subject_phrases(question)
    if phrases:
        return [phrase_expression(phrases)]
    tokens = list(dict.fromkeys(t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]*", question) if t.lower() not in STOP))
    return [" ".join(tokens[:12])] if tokens else []


def validate_filters(docket=None, date_from=None, date_to=None):
    if docket and not re.fullmatch(r"[0-9-]{5,12}", docket):
        raise ValueError("Use an NRC docket number, for example 05000373")
    for value in (date_from, date_to):
        if value:
            dt.date.fromisoformat(value)
    if date_from and date_to and date_from > date_to:
        raise ValueError("Start date must not follow end date")


def aps_filters(docket=None, date_from=None, date_to=None):
    validate_filters(docket, date_from, date_to)
    result = []
    if docket:
        result.append({"field": "DocketNumber", "value": docket, "operator": "equals"})
    dates = []
    if date_from:
        dates.append(f"(DocumentDate ge '{date_from}')")
    if date_to:
        # Use the following midnight to include the complete requested end date.
        end = (dt.date.fromisoformat(date_to) + dt.timedelta(days=1)).isoformat()
        dates.append(f"(DocumentDate le '{end}')")
    if dates:
        result.append({"field": "DocumentDate", "value": " and ".join(dates)})
    return result


def related_accessions(document):
    found = []
    for field, relation in [("DocumentsFiledInPackage", "package_member"), ("PackagesFiledIn", "filed_in_package")]:
        for accession in ACCESSION.findall(json.dumps(document.get(field, []))):
            found.append({"accession": accession.upper(), "relation": relation})
    return list({(x["accession"], x["relation"]): x for x in found}.values())


def search_live(store, client, question, queries, *, docket=None, date_from=None, date_to=None,
                max_documents=8, progress=None, resolved_accessions=None, legacy_scope=False):
    summary = {"queries": [], "discovered": 0, "fetched": [], "missing_text": [], "relationships": [], "warnings": []}
    candidates, scores, legacy_records = {}, {}, {}
    direct = list(dict.fromkeys(a.upper() for a in ACCESSION.findall(question)))[:8]
    direct.extend(a for a in (resolved_accessions or []) if a not in direct)
    for accession in direct:
        candidates[accession] = {"AccessionNumber": accession}
        scores[accession] = 100
    filters = aps_filters(docket, date_from, date_to)
    from .query_terms import subject_phrases, contains_subject, topic_terms
    phrases = subject_phrases(question)
    if not direct and not legacy_scope:
        for query in queries[:2]:
            if progress:
                progress("Searching ADAMS", query)
            try:
                response = client.request(payload={"q": query, "content": False, "filters": filters, "anyFilters": [],
                    "mainLibFilter": True, "legacyLibFilter": not bool(phrases), "sort": "", "sortDirection": 0, "skip": 0})
                docs = unpack_search(response)
                summary["queries"].append({"query": query, "reported_matches": response.get("count"), "examined": len(docs)})
                # Search content is discovery metadata, never evidence.
                pending, pending_scores, pending_legacy = {}, {}, {}
                with store.db:
                    for rank, doc in enumerate(docs):
                        accession = doc["AccessionNumber"]
                        store.put(doc, "api_search_excerpt")
                        if is_legacy(doc):
                            key = source_key(doc)
                            pending_legacy[key] = {"source_key": key, "accession": accession,
                                "title": doc.get("DocumentTitle"), "content_scope": "catalog_metadata"}
                            continue
                        pending[accession] = doc
                        title_match = phrases and contains_subject(doc.get('DocumentTitle') or '', phrases)
                        pending_scores[accession] = pending_scores.get(accession, 0) + 1 / (30 + rank) + (.1 if title_match else 0)
                candidates.update(pending)
                legacy_records.update(pending_legacy)
                for accession, score in pending_scores.items():
                    scores[accession] = scores.get(accession, 0) + score
            except (RuntimeError, ValueError, BudgetStop, OSError) as error:
                summary["warnings"].append(str(error))
                break
    summary["discovered"] = len(candidates) + len(legacy_records)
    summary["legacy_records"] = list(legacy_records.values())
    if legacy_records:
        summary["warnings"].append(f"{len(legacy_records)} legacy records retained separately as metadata; their full text was not retrieved.")
    queue = sorted(candidates, key=lambda a: scores[a], reverse=True)
    attempted = set()
    while queue and len(attempted) < max_documents:
        accession = queue.pop(0)
        if accession in attempted:
            continue
        attempted.add(accession)
        try:
            require_space(store.path.parent)
        except StoragePause as error:
            summary["warnings"].append(str(error))
            break
        if progress:
            progress("Reading source documents", accession)
        try:
            response = client.request(accession=accession)
            document = response.get("document") if isinstance(response, dict) else None
            if not isinstance(document, dict) or str(document.get("AccessionNumber", "")).upper() != accession.upper():
                raise ValueError("ADAMS returned an unexpected document identifier")
            require_space(store.path.parent, extra_bytes=max(32 * 1024 * 1024, len(json.dumps(response).encode()) * 12))
            archive(store, response)
            with store.db:
                state = store.put(document, "api_document_unverified")
                store.db.execute("INSERT INTO retrieval_events(accession,outcome) VALUES(?,?)", (accession, state))
            if state == "indexed":
                summary["fetched"].append(accession)
            elif state != "package":
                summary["missing_text"].append(accession)
            relations = related_accessions(document)
            summary["relationships"].extend({"source": accession, **r} for r in relations)
            # A directly requested package may be a container. Follow its members,
            # within the same global document and API budgets.
            if state == "package":
                members = [r["accession"] for r in relations if r["relation"] == "package_member" and r["accession"] not in attempted]
                queue = members + queue
        except (RuntimeError, ValueError, BudgetStop, OSError) as error:
            summary["warnings"].append(f"{accession}: {error}")
            if isinstance(error, BudgetStop):
                break
    summary["attempted"] = len(attempted)
    summary["exhaustive"] = False
    return summary


def rank_candidates(store, question, semantic=None, docket=None, date_from=None, date_to=None, focus=None, exclusions=None, source_issues=None):
    validate_filters(docket, date_from, date_to)
    from .query_terms import subject_phrases, contains_subject, topic_terms
    phrases = subject_phrases(question) if not ACCESSION.search(question) else []
    topics = topic_terms(question, phrases) if phrases else set()
    lexical = store.lexical(question, 200, docket, date_from, date_to)
    rankings = [lexical]
    if phrases:
        # Independently retrieve named-title passages so thousands of ordinary
        # uses of the same words cannot displace the company's own records.
        rankings.append(store.lexical(question, 80, docket, date_from, date_to, title_only=True))
    if semantic:
        diagnostic_search = getattr(semantic, 'search_with_diagnostics', None)
        if callable(diagnostic_search):
            rankings.append(diagnostic_search(question, 80, docket, date_from, date_to,
                                             exclusions=exclusions, source_issues=source_issues))
        else:
            rankings.append(semantic.search(question, 80, docket, date_from, date_to))
    by_id, scores = {}, {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, 1):
            by_id[row["id"]] = row
            scores[row["id"]] = scores.get(row["id"], 0) + 1 / (60 + rank)
    # Identifier lookup is independent of bounded lexical/ANN candidate lists.
    # Exact technical references should survive a large number of broad matches.
    for row in store.identifier_matches(question, 100, docket, date_from, date_to):
        by_id[row["id"]] = row
        scores[row["id"]] = scores.get(row["id"], 0) + .05 * min(row["identifier_hits"], 4)
    # Explicitly requested accessions must not be displaced by broad semantic neighbors.
    direct = {a.upper() for a in ACCESSION.findall(question)}
    if direct and re.search(r"\b(summariz\w*|summary|overview|purpose)\b", question, re.I):
        # Accession matches alone favor short fragments. A named-document overview
        # also needs its opening passage, where cover letters state purpose and date.
        for row in store.opening_chunks(direct, docket, date_from, date_to):
            by_id[row["id"]] = row
            scores[row["id"]] = scores.get(row["id"], 0) + .04
    if phrases and focus and re.search(r'\b(status|progress|licens\w*|latest|current)\b|how far', question, re.I):
        # Short distribution sheets can beat a longer decision letter in BM25
        # because the same title is repeated on every chunk. Retain the opening
        # of freshly discovered records for status questions.
        for row in store.opening_chunks(focus, docket, date_from, date_to):
            by_id[row['id']] = row
            scores[row['id']] = scores.get(row['id'], 0) + .04
    for key, row in by_id.items():
        if row["accession"].upper() in direct:
            scores[key] += .1
        elif focus and row["accession"] in focus:
            scores[key] += .05 if phrases else .005
        if phrases and contains_subject(row['title'], phrases):
            scores[key] += .04
        if topics:
            title_terms = {w[:6] for w in re.findall(r'[a-z0-9]+', row['title'].casefold())}
            body_terms = {w[:6] for w in re.findall(r'[a-z0-9]+', row['text'].casefold())}
            scores[key] += .012 * len(topics & title_terms) + .003 * len(topics & body_terms)
    order = sorted(scores, key=scores.get, reverse=True)
    coverage = getattr(semantic, "covered_chunk_ids", None) if semantic else None
    if callable(coverage):
        covered = coverage([row["id"] for row in lexical])
        # A pending passage has only one available retrieval channel. Normalize
        # by available channels so missing embeddings are not negative evidence.
        # Indexed passages retain ordinary two-channel RRF, including those that
        # did not rank in the semantic candidate list. Unknown coverage is unchanged.
        for rank, row in enumerate(lexical, 1):
            if row["id"] not in covered:
                scores[row["id"]] += 1 / (60 + rank)
        # Preserve the previous fused order when the adjustment creates a tie.
        order.sort(key=scores.get, reverse=True)
    from .extraction_review import filter_rows
    rows = [{**by_id[key], "retrieval_score": scores[key]} for key in order]
    if phrases:
        # A semantic neighbor about a different company is not evidence about
        # the named subject. Never silently relax a missing name into OR words.
        proper = not re.search(r'["“]', question)
        rows = [r for r in rows if contains_subject(r['title'], phrases)
                or contains_subject(r['text'], phrases, proper_names=proper)]
    return filter_rows(store.db, rows, exclusions)


def hybrid_retrieve(store, question, semantic=None, top_k=10, docket=None, date_from=None, date_to=None, focus=None,
                    scope_accessions=None, exclusions=None, source_issues=None):
    from .evidence import read_snapshot
    with read_snapshot(store.db):
        return _hybrid_retrieve(store, question, semantic, top_k, docket, date_from, date_to, focus,
                                scope_accessions, exclusions, source_issues)


def _hybrid_retrieve(store, question, semantic, top_k, docket, date_from, date_to, focus,
                     scope_accessions, exclusions, source_issues):
    from .evidence import SourceBindings,SourceBindingError,report_binding_issue
    bindings=SourceBindings(store.db)
    ranked = rank_candidates(store, question, semantic, docket, date_from, date_to, focus, exclusions, source_issues)
    direct = {a.upper() for a in ACCESSION.findall(question)}
    if direct or scope_accessions is not None:
        allowed = {a.casefold() for a in (scope_accessions if scope_accessions is not None else direct | set(focus or []))}
        scoped = []
        for row in ranked:
            metadata = json.loads(store.db.execute("SELECT metadata FROM documents WHERE accession=?", (row["accession"],)).fetchone()[0])
            if row["accession"].casefold() in allowed or str(metadata.get("SourceAccession", "")).casefold() in allowed:
                scoped.append(row)
        ranked = scoped
    sources, per_doc, seen, words, anchors = [], {}, set(), 0, []
    for row in ranked:
        if per_doc.get(row["accession"], 0) >= 3:
            continue
        text = row["text"].strip()
        signature = re.sub(r"\s+", " ", text).lower()
        if signature in seen:
            continue
        if words + len(text.split()) > 4200:
            continue
        try:
            metadata,binding=bindings.bind(row)
        except SourceBindingError as error:
            report_binding_issue(source_issues,row,error)
            continue
        words += len(text.split())
        seen.add(signature)
        per_doc[row["accession"]] = per_doc.get(row["accession"], 0) + 1
        source = {k: row[k] for k in ("accession", "title", "url", "page", "ordinal", "text", "provenance", "document_date")}
        source["source_version"] = binding
        source.update(citation=len(sources) + 1, score=round(row["retrieval_score"], 6),
                      source_accession=metadata.get("SourceAccession", row["accession"]),
                      document_type=metadata.get("DocumentType"), dockets=metadata.get("DocketNumber", []),
                      section=metadata.get("section_id"), dataset=metadata.get("Dataset"))
        from .text_quality import get as text_quality
        source["text_quality"] = text_quality(store.db, row["accession"])
        from .legacy_links import references
        legacy_references = references(metadata)[0]
        if legacy_references:
            source["legacy_report_references"] = list(legacy_references)
        if metadata.get("AccessionNumber") and (metadata.get("IdT") or metadata.get("Id")):
            source.update(source_key=source_key(metadata), source_record_id=record_id(metadata),
                          source_library="legacy" if is_legacy(metadata) else "main")
        if source["page"] and "#" not in source["url"]:
            source["url"] += f"#page={source['page']}"
        sources.append(source)
        anchors.append(row)
        if len(sources) >= top_k:
            break
    # Keep the selected anchors and their order. Use only otherwise-unused
    # context space to finish cut-off paragraphs; do not add another source.
    for source, row in zip(sources, anchors):
        if words >= 4200:
            break
        try:
            completion = bindings.paragraph_completion(row, min(200, 4200 - words))
        except SourceBindingError as error:
            report_binding_issue(source_issues, row, error)
            continue
        if completion is None:
            continue
        text, binding, expansion = completion
        start, end = expansion['anchor_end_character'], binding['end_character']
        intervals = sorted((s['source_version']['start_character'], s['source_version']['end_character'])
            for s in sources if s['accession'] == source['accession']
            and s['source_version']['version'] == binding['version'] and s['page'] == source['page'])
        covered = start
        for left, right in intervals:
            if left <= covered:
                covered = max(covered, right)
        if covered >= end:
            continue  # The continuation already appears in another citation.
        source.update(text=text, source_version=binding, context_expansion=expansion)
        words += expansion['added_words']
    return sources


def catalog_coverage(path):
    if not path or not Path(path).exists():
        return None
    db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        from .catalog_lookup import index_ready
        return {"libraries": [dict(r) for r in db.execute("SELECT library,final_count,finished,state FROM libraries")],
                "records": db.execute("SELECT sum(records) FROM inventory_metrics").fetchone()[0],
                "lookup_index_ready": index_ready(db), "record_identity": "library and NRC record ID",
                "scope": "Public catalog metadata, not indexed full text"}
    finally:
        db.close()


def research(store, question, *, engine=None, semantic=None, client=None, catalog=None,
             live=True, docket=None, date_from=None, date_to=None, top_k=10, max_documents=8,
             generate=True, progress=None, evidence_ready=None):
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise ValueError("Enter a question between 1 and 2,000 characters")
    if not 1 <= top_k <= 20 or not 1 <= max_documents <= 20:
        raise ValueError("Retrieval limits must be between 1 and 20")
    validate_filters(docket, date_from, date_to)
    started = time.monotonic()
    warnings, queries = [], fallback_queries(question)
    legacy_resolutions, resolved_accessions, scope = [], [], None
    if catalog and Path(catalog).is_file():
        from .legacy_links import LEGACY_NUMBER, lookup as legacy_lookup
        for number in list(dict.fromkeys(LEGACY_NUMBER.findall(question)))[:8]:
            try:
                resolution = legacy_lookup(catalog, number)
                if not resolution["legacy_matched_records"] and not resolution["records"]:
                    continue
                legacy_resolutions.append(resolution)
                resolved_accessions.extend(r["accession"] for r in resolution["records"])
                if resolution["state"] != "complete":
                    warnings.append(f"Legacy links for {number} are {resolution['state']}; additional digitized records may exist.")
                if resolution["truncated"] or resolution["legacy_records_truncated"]:
                    warnings.append(f"Legacy resolution for {number} reached its bounded lookup limit.")
                if resolution["legacy_matched_records"] > 1:
                    warnings.append(f"Legacy accession {number} has multiple catalog entries. NUDOCS references do not establish which entry a digitized document represents.")
            except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
                warnings.append(f"Legacy lookup unavailable: {error}")
        if legacy_resolutions:
            resolved_accessions = list(dict.fromkeys(resolved_accessions))
            scope = set(resolved_accessions) | {a.upper() for a in ACCESSION.findall(question)}
            scope.update(key for r in legacy_resolutions for key in r["legacy_source_keys"])
    from .query_terms import subject_phrases
    if (engine and live and not ACCESSION.search(question) and not legacy_resolutions
            and not subject_phrases(question) and not getattr(engine, 'cpu_only', False)):
        if progress:
            progress("Planning retrieval", "Identifying search terms and alternative wording")
        try:
            queries = engine.plan(question) or queries
        except (ValueError, RuntimeError, KeyError, OSError) as error:
            warnings.append(f"Using keyword search planning: {error}")
    discovery = None
    if live and client:
        discovery = search_live(store, client, question, queries, docket=docket, date_from=date_from,
                                date_to=date_to, max_documents=max_documents, progress=progress,
                                resolved_accessions=resolved_accessions, legacy_scope=bool(legacy_resolutions))
        warnings.extend(discovery["warnings"])
    elif live:
        warnings.append("ADAMS is not connected; this run searches only the local collection.")
    if discovery and discovery['fetched']:
        if progress:
            progress('Checking source text', 'Validating the newly retrieved records')
        try:
            from .markdown_archive import prepare_fetched
            discovery['markdown_checked'] = prepare_fetched(store, discovery['fetched'])
        except (RuntimeError, ValueError, OSError) as error:
            warnings.append(f'New records remain subject to the extraction holdout policy: {error}')
    if semantic and discovery and discovery["fetched"]:
        if progress:
            progress("Updating the semantic index", "Indexing newly read documents")
        try:
            semantic.sync(limit=128, accessions=discovery["fetched"])
        except (RuntimeError, ValueError, OSError) as error:
            warnings.append(f"New text is available to keyword search; semantic indexing failed: {error}")
    if progress:
        progress("Retrieving evidence", "Combining exact terms and semantic matches")
    retrieval_question = question + (" " + " ".join(resolved_accessions) if resolved_accessions else "")
    if scope is not None and discovery:
        scope.update(discovery["fetched"])
    exclusions, source_issues = [], []
    semantic_failed = False
    try:
        evidence = hybrid_retrieve(store, retrieval_question, semantic, top_k, docket, date_from, date_to,
                                   discovery["fetched"] if discovery else None, scope_accessions=scope, exclusions=exclusions,
                                   source_issues=source_issues)
    except (ValueError, RuntimeError, OSError) as error:
        semantic_failed = True
        warnings.append(f"Using keyword retrieval: {error}")
        evidence = hybrid_retrieve(store, retrieval_question, None, top_k, docket, date_from, date_to,
                                   discovery["fetched"] if discovery else None,
                                   scope_accessions=scope, exclusions=exclusions, source_issues=source_issues)
    exclusions = list({(r["accession"],r["text_hash"],r["page"]):r for r in exclusions}.values())
    if exclusions:
        warnings.append(f"Known extraction errors or pending verification excluded {len(exclusions)} page/document entries from answer evidence; see the recorded holdout reasons.")
    if source_issues:
        warnings.append(f"Excluded {len(source_issues)} candidate passages that could not be matched to their retained source text; see provenance issues.")
    from .text_quality import MESSAGES
    source_quality = {s["accession"]: s["text_quality"] for s in evidence}
    for accession, quality in source_quality.items():
        if quality.get("flags"):
            warnings.append(accession + ": " + " ".join(MESSAGES.get(flag, "Additional extraction diagnostic: " + flag) for flag in quality["flags"]))
    result = {"id": str(uuid.uuid4()), "question": question, "mode": "evidence_only", "answer": None,
              "sources": evidence, "source_quality": source_quality, "excluded_evidence": exclusions,
              "provenance_issues": source_issues,
              "discovery": discovery, "coverage": store.status(),
              "semantic_retrieval": {"requested_model": getattr(semantic, "model", None),
                                     "keyword_fallback": semantic_failed},
              "catalog": catalog_coverage(catalog), "warnings": warnings, "queries": queries,
              "filters": {"docket": docket, "date_from": date_from, "date_to": date_to},
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "citation_check": "not_generated", "exhaustive": False}
    if legacy_resolutions:
        result["legacy_resolution"] = legacy_resolutions
        result["source_scope"] = "Named legacy records, explicit NRC NUDOCS report references, and named main-library accessions"
    if ACCESSION.search(question):
        result.setdefault("source_scope", "Named accessions and fetched package members")
    from .query_terms import subject_phrases
    if not ACCESSION.search(question) and subject_phrases(question):
        result['subject_phrases'] = subject_phrases(question)
    if not live:
        warnings.append('This run searched the downloaded collection only; current NRC updates were not checked.')
    if evidence_ready:
        evidence_ready(result)
    if not evidence:
        result.update(answer="No supporting passages were found in this bounded retrieval. This does not establish that ADAMS has no answer.", insufficient_evidence=True)
    elif generate and engine:
        if progress:
            progress("Generating a cited answer", "The answer model is reading the retrieved passages")
        try:
            generate_call = getattr(engine, 'generate_with_progress', None)
            answer = (generate_call(question, evidence, progress) if callable(generate_call)
                      else engine.generate(None, question, evidence))
            result.update(answer, mode="generated",
                          citation_check="citation_identifiers_validated; entailment_not_verified")
        except (ValueError, RuntimeError, KeyError, OSError) as error:
            warnings.append(f"Generated answer withheld: {error}")
    elif generate:
        warnings.append("The answer model is not ready. Retrieved context is returned without generation.")
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    store.db.execute("CREATE TABLE IF NOT EXISTS research_runs(id TEXT PRIMARY KEY, created TEXT, result TEXT)")
    with store.db:
        store.db.execute("INSERT INTO research_runs VALUES(?,?,?)", (result["id"], result["created_utc"], json.dumps(result)))
    return result
