"""Conservative phrase anchors for names in otherwise ordinary questions."""
import re

QUESTION_WORDS = set('a an the what which where when how why is are was were does do did can could would should tell me show explain summarize compare and or with for of on in at their its about latest current far along to get attempting please history timeline overview give describe has have had happens happened difference differences between versus vs list find all any information documents records document record there this that these those from into regarding want know looking understand'.split())

LICENSING = re.compile(r'\b(?:licens\w*|permit\w*)\b', re.I)
HISTORY = re.compile(r'\b(?:history|historical|timeline|chronolog\w*|over time)\b', re.I)
COMPARISON = re.compile(r'\b(?:compare|compar\w*|difference\w*|versus|vs\.?|differ)\b', re.I)
CURRENT = re.compile(r'\b(?:latest|current|today|recent|status|progress)\b|how far', re.I)
MILESTONE = re.compile(r'\b(?:accept\w*|deni\w*|reject\w*|withdraw\w*|resubmit\w*|issu\w*|approv\w*|application|pre.application|preapplication|readiness|licens\w*|permit\w*)\b', re.I)


def history_question(question):
    return bool(HISTORY.search(question))


def inquiry_kind(question):
    if COMPARISON.search(question):
        return 'comparison'
    if history_question(question):
        return 'history'
    if CURRENT.search(question):
        return 'status'
    if re.search(r'\b(?:how many|count|total number|every|list all)\b', question, re.I):
        return 'enumeration'
    if re.search(r'\b(?:require\w*|must|shall|obligations|criteria|deadline\w*)\b', question, re.I):
        return 'requirements'
    if re.search(r'\b(?:summari\w*|summary|overview|purpose)\b', question, re.I):
        return 'summary'
    return 'explanation'


def content_words(question):
    return list(dict.fromkeys(w.lower() for w in re.findall(r'[A-Za-z0-9]+', question)
                             if len(w) > 1 and w.lower() not in QUESTION_WORDS))[:24]


def inquiry_order(rows, question):
    """Reserve evidence for both sides of a comparison and recent status records."""
    kind, chosen = inquiry_kind(question), []
    if kind == 'comparison':
        for phrase in subject_phrases(question):
            match = next((r for r in rows if contains_subject(r['title'], [phrase])
                          or contains_subject(r['text'], [phrase], proper_names=True)), None)
            if match is not None and match not in chosen:
                chosen.append(match)
    elif kind == 'status' and rows:
        best = rows[0].get('retrieval_score', 0)
        relevant = [r for r in rows if r.get('document_date') and r.get('retrieval_score', 0) >= best * .65]
        if relevant:
            chosen.append(max(relevant, key=lambda r: r['document_date']))
    return chosen + [r for r in rows if r not in chosen]


def decision_title(title):
    """Prefer an actual decision over an audit of an acceptance review."""
    return bool(re.search(r'\b(?:deni\w*|reject\w*|withdraw\w*|accept\w*)\b', title, re.I)
                and not re.search(r'\b(?:audit|plan|meeting|draft)\b', title, re.I))


def history_order(rows):
    """Keep relevant milestones from distinct years ahead of repeated events."""
    years, representatives, remaining = set(), [], []
    best_score = max((r.get('retrieval_score', 0) for r in rows), default=0)
    for row in rows:
        year = str(row.get('document_date') or '')[:4]
        if (year.isdigit() and year not in years and MILESTONE.search(row['title'])
                and row.get('retrieval_score', 0) >= best_score * .6):
            years.add(year); representatives.append(row)
        else:
            remaining.append(row)
    chosen = representatives[:4]
    if representatives:
        latest = max(representatives, key=lambda r: str(r.get('document_date') or ''))
        if latest not in chosen:
            chosen[-1] = latest
    return chosen + [r for r in representatives if r not in chosen] + remaining


def topic_terms(question, phrases):
    names = {w.casefold() for p in phrases for w in re.findall(r'\w+', p)}
    terms = {w[:6] for w in re.findall(r'[a-z0-9]+', question.casefold())
            if len(w) > 2 and w not in QUESTION_WORDS and w not in names}
    if LICENSING.search(question):
        terms.update({'licens', 'permit', 'applic', 'accept', 'denied', 'denial',
                      'reject', 'withdr', 'submit', 'submis', 'approv', 'review', 'readin'})
    return terms


def topic_expression(question, phrases):
    return ' OR '.join('"' + word + '"*' for word in sorted(topic_terms(question, phrases)))


def subject_phrases(question):
    explicit = re.findall(r'["“]([^"”\n]{2,100})["”]', question)
    names = []
    for match in re.finditer(r'\b(?:[A-Z][a-zA-Z0-9-]*)(?:\s+[A-Z][a-zA-Z0-9-]*)+\b', question):
        words = match.group().split()
        while words and words[0].lower() in QUESTION_WORDS:
            words.pop(0)
        while words and words[-1].lower() in QUESTION_WORDS:
            words.pop()
        if len(words) >= 2 and any(not w.isupper() for w in words):
            names.append(' '.join(words))
    # A single proper name in a normal sentence is also an anchor. Do not
    # mistake question headings or technical acronyms for company names.
    covered = {w.casefold() for p in explicit + names for w in p.split()}
    generic = QUESTION_WORDS | set('nrc adams reactor reactors nuclear licensing license permits permit safety application regulatory status progress requirements regulations'.split())
    for match in re.finditer(r'\b[A-Z][a-zA-Z0-9-]*\b', question):
        word = match.group()
        if (word.casefold() not in generic | covered and not word.isupper()
                and len(word) > 2 and not re.match(r'ML\d', word)):
            names.append(word)
    return list(dict.fromkeys(' '.join(p.split()) for p in explicit + names))[:3]


def contains_subject(text, phrases, proper_names=False):
    normalized = ' '.join(re.findall(r'\w+', text))
    for phrase in phrases:
        phrase = ' '.join(re.findall(r'\w+', phrase))
        forms = [phrase, phrase.upper()] if proper_names else [phrase]
        flags = 0 if proper_names else re.I
        if any(re.search(r'(?<!\w)' + re.escape(form) + r'(?!\w)', normalized, flags) for form in forms):
            return True
    return False


def phrase_expression(phrases):
    return ' OR '.join('"' + p.replace('"', '""') + '"' for p in phrases)
