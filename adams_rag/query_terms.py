"""Conservative phrase anchors for names in otherwise ordinary questions."""
import re

QUESTION_WORDS = set('a an the what which where when how why is are was were does do did can could would should tell me show explain summarize compare and or with for of on in at their its about latest current'.split())


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
    return list(dict.fromkeys(' '.join(p.split()) for p in explicit + names))[:3]


def contains_subject(text, phrases):
    normalized = ' '.join(re.findall(r'\w+', text.casefold()))
    return any(re.search(r'(?<!\w)' + re.escape(' '.join(re.findall(r'\w+', p.casefold()))) + r'(?!\w)', normalized)
               for p in phrases)


def phrase_expression(phrases):
    return ' OR '.join('"' + p.replace('"', '""') + '"' for p in phrases)
