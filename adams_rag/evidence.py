"""Bind retrieved passages to immutable retained text, without asserting OCR fidelity."""
from .source_content import read_text
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import re
from urllib.parse import urlencode
import zlib
from .page_boundaries import PageBoundaryError, retained_pdf_pages


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def metadata_digest(metadata):
    return sha256(json.dumps(metadata,sort_keys=True,ensure_ascii=False))


@contextmanager
def read_snapshot(db):
    own=not db.in_transaction
    if own:
        db.execute('BEGIN')
    try:
        yield
    finally:
        if own:
            db.rollback()


class SourceBindingError(ValueError):
    pass


class SourceBindings:
    def __init__(self,db):
        self.db=db
        self.cache={}

    def document(self,accession):
        if accession in self.cache:
            return self.cache[accession]
        document=self.db.execute('SELECT * FROM documents WHERE accession=?',(accession,)).fetchone()
        if not document or document['status']!='indexed':
            raise SourceBindingError('source_not_indexed')
        metadata=json.loads(document['metadata'])
        # Metadata-only refreshes intentionally retain the old text. Bind its
        # immutable version separately from the current discovery metadata.
        params=[accession,document['text_hash'],document['provenance']]
        condition=''
        if document['provenance']=='pdf':
            condition=" AND json_extract(metadata,'$.LocalFileSHA256') IS ?"
            params.append(metadata.get('LocalFileSHA256'))
        retained=self.db.execute('''SELECT * FROM document_versions
            WHERE accession=? AND text_hash=? AND provenance=?'''+condition+
            ' ORDER BY captured DESC,rowid DESC LIMIT 1',params).fetchone()
        if not retained:
            raise SourceBindingError('retained_version_missing')
        if sha256(retained['text_hash']+retained['metadata'])!=retained['version']:
            raise SourceBindingError('retained_version_checksum_mismatch')
        try:
            body=read_text(retained['content'], self.db)
        except (zlib.error,UnicodeDecodeError,ValueError):
            raise SourceBindingError('retained_text_unreadable') from None
        if sha256(body)!=retained['text_hash']:
            raise SourceBindingError('retained_text_checksum_mismatch')
        original_metadata=json.loads(retained['metadata'])
        try:
            pages=retained_pdf_pages(body,original_metadata) if retained['provenance']=='pdf' else None
        except PageBoundaryError as error:
            raise SourceBindingError(str(error)) from None
        self.cache[accession]={'document':dict(document),'metadata':metadata,'retained':dict(retained),
            'original_metadata':original_metadata,'body':body,
            'pages':pages}
        return self.cache[accession]

    def bind(self,row):
        value=self.document(row['accession'])
        current=self.db.execute('SELECT accession,ordinal,page,text FROM chunks WHERE id=?',(row['id'],)).fetchone()
        if not current or any(current[key]!=row[key] for key in ('accession','ordinal','page','text')):
            raise SourceBindingError('passage_changed_during_retrieval')
        page=row['page']
        if page is not None:
            if type(page) is not int or not value['pages'] or not 1<=page<=len(value['pages']):
                raise SourceBindingError('retained_page_missing')
            context=value['pages'][page-1]
        else:
            if value['pages'] is not None:
                raise SourceBindingError('pdf_passage_has_no_page')
            context=value['body']
        offset=context.find(row['text'])
        if offset<0 or not row['text'].strip():
            raise SourceBindingError('passage_does_not_match_retained_text')
        retained=value['retained']
        binding={'state':'matched_retained_text','version':retained['version'],
                 'text_sha256':retained['text_hash'],'passage_sha256':sha256(row['text']),
                 'captured_utc':retained['captured'],'page':page,'start_character':offset,
                 'end_character':offset+len(row['text']),
                 'offset_scope':'physical_pdf_page' if page is not None else 'document_text',
                 'retrieval_metadata_sha256':metadata_digest(value['metadata']),
                 'retained_metadata_sha256':metadata_digest(value['original_metadata']),
                 'source_file_sha256':value['original_metadata'].get('LocalFileSHA256'),
                 'source_fidelity_verified':False,
                 'url':'/api/document/version?'+urlencode({'accession':row['accession'],'version':retained['version']})}
        return value['metadata'],binding

    def paragraph_completion(self, row, max_extra_words=200):
        """Return a bounded, exact retained span when a chunk cuts a paragraph.

        A PDF span stays on its physical page, so the anchor's extraction review
        still applies. Nothing is repaired, synthesized or written to the index.
        """
        metadata, binding = self.bind(row)
        if type(max_extra_words) is not int or not 1 <= max_extra_words <= 200:
            return None
        terminal = r'[.!?][\"\'’”\)\]]*$'
        if re.search(terminal, row['text'].rstrip()):
            return None
        value = self.document(row['accession'])
        context = value['pages'][row['page'] - 1] if row['page'] is not None else value['body']
        if context.find(row['text'], binding['start_character'] + 1) >= 0:
            return None  # Identical repeated chunks do not establish one continuation.
        end = binding['end_character']
        remaining = context[end:end + 6001]
        boundary = re.search(r'\n[ \t\r]*\n', remaining)
        if boundary:
            tail = remaining[:boundary.start()]
        elif end + len(remaining) == len(context):
            tail = remaining
        else:
            return None
        tail = tail.rstrip()
        text = row['text'] + tail
        extra_words = len(text.split()) - len(row['text'].split())
        if (not tail.strip() or not 0 < extra_words <= max_extra_words
                or not re.search(terminal, text.rstrip())):
            return None
        expansion = {'kind': 'paragraph_completion', 'anchor_chunk_id': row['id'],
                     'anchor_ordinal': row['ordinal'], 'anchor_passage_sha256': binding['passage_sha256'],
                     'anchor_end_character': end, 'added_words': extra_words}
        expanded = {**binding, 'end_character': end + len(tail), 'passage_sha256': sha256(text)}
        assert context[expanded['start_character']:expanded['end_character']] == text
        return text, expanded, expansion


def report_binding_issue(issues,row,error):
    if issues is not None:
        issue = {'accession':row['accession'],'page':row.get('page'),
                 'ordinal':row.get('ordinal'),'reason':str(error)}
        if issue not in issues:
            issues.append(issue)


def saved_run_view(store,run_id):
    """Keep the original answer/excerpts intact and append current review findings."""
    from .extraction_review import for_version
    with read_snapshot(store.db):
        exists=store.db.execute("SELECT 1 FROM sqlite_master WHERE name='research_runs'").fetchone()
        row=store.db.execute('SELECT result FROM research_runs WHERE id=?',(run_id,)).fetchone() if exists else None
        if not row:
            return None
        result=json.loads(row['result'])
        updates,warnings,cache=[],[],{}
        for source in result.get('sources',[]):
            accession=source['accession'];citation=source['citation']
            binding=source.get('source_version')
            entry={'citation':citation,'accession':accession,'page':source.get('page')}
            if not isinstance(binding,dict) or binding.get('state')!='matched_retained_text':
                entry['state']='unversioned_saved_evidence'
                warnings.append('This older run did not pin retained source versions; current review status cannot be matched reliably.')
                updates.append(entry)
                continue
            key=(accession,binding.get('version'))
            if key not in cache:
                retained=store.db.execute('SELECT * FROM document_versions WHERE accession=? AND version=?',key).fetchone()
                body=None
                if retained:
                    try:
                        candidate=read_text(retained['content'], store.db)
                        if (sha256(candidate)==retained['text_hash']
                                and sha256(retained['text_hash']+retained['metadata'])==retained['version']):
                            body=candidate
                    except (zlib.error,UnicodeDecodeError,ValueError):
                        pass
                cache[key]=(retained,body)
            retained,body=cache[key]
            valid=(retained is not None and body is not None and retained['text_hash']==binding.get('text_sha256')
                   and isinstance(source.get('text'),str) and sha256(source['text'])==binding.get('passage_sha256')
                   and source.get('page')==binding.get('page'))
            if valid:
                original_metadata=json.loads(retained['metadata'])
                valid=(metadata_digest(original_metadata)==binding.get('retained_metadata_sha256')
                       and original_metadata.get('LocalFileSHA256')==binding.get('source_file_sha256'))
                page=binding.get('page');start=binding.get('start_character');end=binding.get('end_character')
                context=body
                if page is not None:
                    try:
                        pages=retained_pdf_pages(body,original_metadata) if retained['provenance']=='pdf' else []
                    except PageBoundaryError:
                        pages=[]
                    valid=valid and type(page) is int and 1<=page<=len(pages)
                    context=pages[page-1] if valid else ''
                elif retained['provenance']=='pdf':
                    valid=False
                valid=(valid and type(start) is int and type(end) is int and 0<=start<end<=len(context)
                       and context[start:end]==source['text'])
            if not valid:
                entry['state']='retained_binding_unavailable_or_mismatched'
                warnings.append(f'Saved source [{citation}] could not be revalidated against its retained text; the original answer and excerpt are preserved for inspection.')
                updates.append(entry)
                continue
            reviews=[r for r in for_version(store.db,accession,retained['text_hash']) if r['page']==0 or r['page']==source.get('page')]
            excluded=any(r['disposition']=='exclude' for r in reviews)
            original=source.get('text_quality',{}).get('evidence_reviews',[])
            changed=any(r not in original for r in reviews)
            current=store.db.execute('SELECT text_hash,metadata,provenance,status FROM documents WHERE accession=?',(accession,)).fetchone()
            metadata=json.loads(current['metadata']) if current else {}
            same_text=bool(current and current['text_hash']==retained['text_hash'])
            same_source=same_text and current['provenance']==retained['provenance'] and metadata.get('LocalFileSHA256')==binding.get('source_file_sha256')
            entry.update(state='excluded_by_review' if excluded else ('review_warning' if reviews else 'no_matching_review'),
                         retained_version=binding['version'],text_sha256=retained['text_hash'],
                         current_source_matches=same_source,
                         current_metadata_matches=bool(current and metadata_digest(metadata)==binding.get('retrieval_metadata_sha256')),
                         current_document_status=current['status'] if current else 'not_in_collection',
                         review_changed_since_run=changed,evidence_reviews=reviews)
            if excluded:
                warnings.append(f'Saved source [{citation}] is now excluded by an extraction review. Its old answer is preserved; rerun the query before relying on that passage.')
            elif changed:
                warnings.append(f'Saved source [{citation}] has an additional extraction warning; inspect its current review before relying on the old answer.')
            if not same_source:
                warnings.append(f'The locally indexed source for [{citation}] has changed since this run; the saved answer still refers to its retained version.')
            if same_source and store.db.execute("SELECT 1 FROM sqlite_master WHERE name='markdown_checks'").fetchone():
                check = store.db.execute('SELECT state,report FROM markdown_checks WHERE accession=? AND text_hash=?',
                                         (accession, retained['text_hash'])).fetchone()
                if check and check['state'] == 'holdout':
                    entry['state'] = 'held_out_by_document_check'
                    entry['document_check'] = json.loads(check['report'])
                    warnings.append(f'Saved source [{citation}] is now held out by a document conversion check. Its old answer is preserved; inspect the holdout reasons before using it.')
            updates.append(entry)
        result['review_updates']={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'saved_result_unchanged':True,'sources':updates,'warnings':list(dict.fromkeys(warnings)),
            'claim_support_verified':False}
        result['warnings']=list(dict.fromkeys(result.get('warnings',[])+warnings))
        return result
