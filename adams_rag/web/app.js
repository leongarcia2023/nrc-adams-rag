"use strict";
const $ = id => document.getElementById(id);
const SESSION_KEY = 'nrc-current-question:' + location.pathname;
const appURL = path => new URL(path.replace(/^\/+/, ''), new URL('./', location.href)).href;
let active = false;
let pending = null;
let current = null;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(appURL(path), {...options, signal: controller.signal});
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(response.status === 409 ? 'Another question is being answered. Please try again shortly.' : (data.error || 'The request could not be completed.'));
      error.status = response.status;
      throw error;
    }
    return data;
  } finally { clearTimeout(timer); }
}
function remember(job) {
  pending = job;
  try { if (job) sessionStorage.setItem(SESSION_KEY, JSON.stringify(job)); else sessionStorage.removeItem(SESSION_KEY); } catch {}
}
function resizeQuestion() {
  const field = $('question');
  field.style.height = 'auto';
  field.style.height = Math.min(240, Math.max(60, field.scrollHeight)) + 'px';
}
function busy(value) {
  active = value;
  $('submit').disabled = value;
  $('submit-label').textContent = value ? 'Asking' : pending ? 'Reconnect' : 'Ask';
  $('question').readOnly = value || Boolean(pending);
  $('query-form').setAttribute('aria-busy', String(value));
  $('progress').hidden = !value;
}
function showError(text) {
  $('error').textContent = text;
  $('error').hidden = !text;
}
function openCitation(number) {
  $('source-details').open = true;
  const source = $('source-' + number);
  if (source) { source.focus({preventScroll: true}); source.scrollIntoView({block: 'start', behavior: 'auto'}); }
}
function inline(parent, text, citations) {
  // Source/model text is always inserted as text, never interpreted as HTML.
  for (const part of String(text).split(/(\[\d+\]|\*\*[^*\n]+\*\*|`[^`\n]+`)/g)) {
    const match = part.match(/^\[(\d+)\]$/);
    if (match && citations.has(Number(match[1]))) {
      const link = node('a', part, 'citation');
      link.href = '#source-' + match[1];
      link.setAttribute('aria-label', 'Read source ' + match[1]);
      link.addEventListener('click', event => { event.preventDefault(); openCitation(match[1]); });
      parent.append(link);
    } else if (part.startsWith('**') && part.endsWith('**')) parent.append(node('strong', part.slice(2, -2)));
    else if (part.startsWith('`') && part.endsWith('`')) parent.append(node('code', part.slice(1, -1)));
    else parent.append(document.createTextNode(part));
  }
}
function renderAnswer(text, citations) {
  const lines = text.split('\n');
  let paragraph = [], list = null;
  const flush = () => { if (paragraph.length) { const p = node('p'); inline(p, paragraph.join('\n'), citations); $('answer').append(p); paragraph = []; } list = null; };
  const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(value => value.trim());
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      flush(); const wrap = node('div', undefined, 'table-scroll'); wrap.tabIndex = 0; wrap.setAttribute('role', 'region'); wrap.setAttribute('aria-label', 'Answer table');
      const table = node('table'), head = node('thead'), row = node('tr');
      for (const value of cells(line)) { const cell = node('th'); cell.scope = 'col'; inline(cell, value, citations); row.append(cell); }
      head.append(row); table.append(head); const body = node('tbody'); i += 2;
      for (; i < lines.length && lines[i].includes('|') && lines[i].trim(); i++) {
        const tr = node('tr'); for (const value of cells(lines[i])) { const td = node('td'); inline(td, value, citations); tr.append(td); } body.append(tr);
      }
      i--; table.append(body); wrap.append(table); $('answer').append(wrap); continue;
    }
    const heading = line.match(/^#{1,4}\s+(.+)$/);
    const item = line.match(/^\s*(?:([-*])|\d+\.)\s+(.+)$/);
    if (!line.trim()) flush();
    else if (heading) { flush(); const h = node('h2'); inline(h, heading[1], citations); $('answer').append(h); }
    else if (item) {
      if (paragraph.length) flush();
      const tag = item[1] ? 'UL' : 'OL';
      if (!list || list.tagName !== tag) { list = node(tag.toLowerCase()); $('answer').append(list); }
      const li = node('li'); inline(li, item[2], citations); list.append(li);
    } else { if (list) list = null; paragraph.push(line); }
  }
  flush();
}
function safeLink(label, url) {
  const link = node('a', label);
  try { const parsed = new URL(url); if (parsed.protocol === 'https:') { link.href = parsed.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; } } catch {}
  return link;
}
async function render(result, preview = false) {
  if (!preview) current = result;
  const sources = Array.isArray(result.sources) ? result.sources : [];
  const citations = new Set(sources.map(source => Number(source.citation)));
  $('answer').replaceChildren(); $('sources').replaceChildren(); $('search-notes').replaceChildren();
  $('answer-note').hidden = true; $('source-details').open = false;
  $('source-details').hidden = !sources.length;
  $('source-summary').textContent = 'Sources · ' + sources.length;
  if (result.answer) renderAnswer(result.answer, citations);
  else if (preview) {
    $('answer').append(node('p', 'The matching sources are ready below while the answer is being prepared.'));
  } else {
    let message = 'I found relevant records, but could not generate an answer. You can read the sources below and try again.';
    if ((result.warnings || []).some(w => /exceeded its request timeout/i.test(w))) {
      message = 'The answer took too long on this server. The matching sources are available below.';
    } else if ((result.warnings || []).some(w => /generation paused for disk headroom/i.test(w))) {
      message = 'I found relevant records, but this server needs more free space to generate an answer. You can read the sources below.';
      try {
        const status = await api('/api/status');
        const g = status.generation;
        if (g?.state === 'headroom_needed') {
          const extra = Math.max(0, (g.required_free_bytes - g.free_bytes) / 1e9);
          message = `I found relevant records, but this server needs about ${extra.toFixed(1)} GB more free space to generate an answer. You can read the sources below.`;
        }
      } catch {}
    } else if ((result.warnings || []).some(w => /model is not ready/i.test(w))) message = 'The answer model is not ready yet. You can read the matching sources below.';
    $('answer').append(node('p', message));
  }
  if (result.insufficient_evidence && result.mode === 'generated') {
    $('answer-note').textContent = 'The available records do not fully answer this question.';
    $('answer-note').hidden = false;
  }
  if (result.review_updates?.warnings?.length) {
    $('answer-note').textContent = 'This saved answer has source warnings. Review the sources before using it.';
    $('answer-note').hidden = false;
  }
  for (const source of sources) {
    const article = node('section', undefined, 'source'); article.id = 'source-' + source.citation; article.tabIndex = -1;
    const heading = node('h2'); heading.append(document.createTextNode('[' + source.citation + '] '), safeLink(source.title || source.accession, source.url)); article.append(heading);
    const location = source.page ? 'PDF page ' + source.page : 'Passage ' + source.ordinal + ' (page unavailable)';
    article.append(node('p', [source.source_accession || source.accession, location, source.document_date, source.dataset ? 'Community text' : null].filter(Boolean).join(' · '), 'metadata'));
    article.append(node('blockquote', source.text));
    const links = node('p', undefined, 'source-links'); links.append(safeLink('Original document', source.url));
    const download = node('a', 'Markdown'); download.href = appURL('/api/document/markdown') + '?' + new URLSearchParams({accession: source.accession, ...(source.source_version?.version ? {version:source.source_version.version} : {})});
    download.download = source.accession.replace(/[^A-Za-z0-9_.-]/g, '_') + '.md'; links.append(download); article.append(links); $('sources').append(article);
  }
  $('search-notes').append(node('p', 'This search covers part of the NRC archive. It cannot establish that every relevant document was found.'));
  for (const warning of [...new Set([...(result.gaps || []), ...(result.warnings || [])])]) $('search-notes').append(node('p', warning));
  $('result').hidden = false;
  if (preview) return;
  $('result').focus({preventScroll:true});
  const url = new URL(location.href); url.searchParams.set('run', result.id); url.hash = ''; history.replaceState(null, '', url);
}
function stageLabel(stage) {
  if (/queued/i.test(stage)) return 'Waiting for the previous question…';
  if (/reading source context|generat/i.test(stage)) return 'The model is reading the selected sources…';
  if (/drafting/i.test(stage)) return 'Writing your answer…';
  if (/retriev|evidence|semantic/i.test(stage)) return 'Reading the matching records…';
  if (/search|ADAMS|discover|fetch/i.test(stage)) return 'Searching the NRC records…';
  return 'Reading the records…';
}
async function follow(job) {
  let previewShown = false;
  let interruptions = 0;
  while (true) {
    let state;
    try {
      state = await api('/api/jobs/' + encodeURIComponent(job.id));
      interruptions = 0;
    } catch (error) {
      if (error.status === 404 || ++interruptions >= 3) throw error;
      $('stage').textContent = 'Reconnecting to your answer…';
      await new Promise(resolve => setTimeout(resolve, 2000));
      continue;
    }
    const elapsed = state.created_at ? Math.max(0, Math.floor(Date.now() / 1000 - state.created_at)) : 0;
    const duration = elapsed >= 60 ? ` · ${Math.floor(elapsed / 60)}m ${elapsed % 60}s` : elapsed ? ` · ${elapsed}s` : '';
    $('stage').textContent = stageLabel(state.stage || '') + duration;
    if (state.preview && !previewShown) { await render(state.preview, true); previewShown = true; }
    if (state.state === 'failed') { remember(null); throw new Error(state.detail || 'The answer could not be completed. Please try again.'); }
    if (state.state === 'finished') { await render(state.result); remember(null); return; }
    await new Promise(resolve => setTimeout(resolve, 1200));
  }
}
async function run() {
  if (active) return;
  const question = $('question').value.trim();
  if (!question) { $('question').focus(); return; }
  busy(true); showError(''); $('stage').textContent = 'Reading the records…';
  if (!pending) { current = null; $('result').hidden = true; }
  try {
    if (!pending) {
      const job = await api('/v1/rag', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question, live:true})});
      remember({id:job.id, question});
      const url = new URL(location.href); url.search = ''; url.hash = ''; history.replaceState(null, '', url);
    }
    await follow(pending);
  } catch (error) {
    if (error.status === 404) remember(null);
    showError(pending ? 'The connection was interrupted. Reconnect to pick up this answer.' : error.name === 'AbortError' || error instanceof TypeError ? 'The local service is not responding. Your question is still here; please try again.' : error.message);
  } finally { busy(false); }
}
$('query-form').addEventListener('submit', event => { event.preventDefault(); run(); });
$('question').addEventListener('input', resizeQuestion);
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!active) $('query-form').requestSubmit(); }
});
async function restore() {
  try { const job = JSON.parse(sessionStorage.getItem(SESSION_KEY)); if (job && /^[\da-f-]{36}$/i.test(job.id) && typeof job.question === 'string') pending = job; } catch {}
  if (pending) { $('question').value = pending.question; resizeQuestion(); await run(); return; }
  const runId = new URL(location.href).searchParams.get('run');
  if (!runId || !/^[\da-f-]{36}$/i.test(runId)) return;
  busy(true); $('stage').textContent = 'Opening your answer…';
  try { const result = await api('/api/runs/' + runId); $('question').value = result.question; resizeQuestion(); await render(result); }
  catch { showError('This saved answer could not be opened. You can ask the question again.'); }
  finally { busy(false); }
}
resizeQuestion(); restore();
