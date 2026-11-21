"""Conservative phrase anchors for names in otherwise ordinary questions."""
import re

QUESTION_WORDS = set('a an the what which where when how why is are was were does do did can could would should tell me show explain summarize compare and or with for of on in at their its about latest current far along to get attempting please history timeline overview give describe has have had'.split())

LICENSING = re.compile(r'\b(?:licens\w*|permit\w*)\b', re.I)
HISTORY = re.compile(r'\b(?:history|historical|timeline|chronolog\w*|over time)\b', re.I)
MILESTONE = re.compile(r'\b(?:accept\w*|deni\w*|reject\w*|withdraw\w*|resubmit\w*|issu\w*|approv\w*|application|pre.application|preapplication|readiness|licens\w*|permit\w*)\b', re.I)


def history_question(question):
    return bool(HISTORY.search(question))


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
