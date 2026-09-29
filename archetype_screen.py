"""Archetype screening for job postings.

Discovered in the Scam Signatures analysis (DSC680 Project 1) by clustering
15,872 EMSCAD postings with the fraud label withheld. This module ships the
findings that transfer: exact campaign signatures, and the structural shape of
postings that are recognizable by what they omit.

Deliberately stdlib only. No model, no vectorizer, no framework. It runs the
same in a Streamlit app, a web service, a CLI or a notebook, and it returns an
answer with no warm-up, which means it can respond while a language model is
still loading.

What this module does NOT do, and why:

  The Common Form archetype is not screened for. It runs at 43.1% fraud and
  every cluster within it contains real employers, so membership indicates how
  much scrutiny a posting warrants and never a verdict.

  The Borrowed Brand is not screened for. It is identified by corporate text
  copied from a real company, which requires comparison against a corpus this
  module does not carry, and by platform metadata (logo present, screening
  questions absent) that a pasted posting does not include.

  The fitted TF-IDF vectorizer and archetype centroids from the study are not
  shipped. Research question three found that current postings sit close to the
  ordinary-posting baseline and that 96.4% of them land nearest the Common
  Form, so centroid matching would report a meaningless result with apparent
  confidence. Exact phrases and structural absence were chosen because they
  either match or stay silent.

  Nothing here returns a fraud verdict. It reports the shape a posting
  resembles and the evidence for saying so.
"""

import re

CAMPAIGN_SIGNATURES = (
    {
        'name': 'The Ghost Agency',
        'source': 'EMSCAD 2012 to 2014',
        'postings_matched': 99,
        'precision': 1.0,
        'note': ('One operator running 60 job titles across 22 locations under '
                 'two brand names, reusing a single opening paragraph.'),
        'phrases': ('leveraging your career',
                    'redesigned the recruiting wheel',
                    'have partnered up in an effort to streamline',
                    'represented candidates'),
    },
)

# Empirical basis: in the source corpus the Bait Ad ran a median of 81 words
# against 366 for the corpus as a whole. The cutoff is set near twice the
# archetype median rather than at it, so a wordier bait ad still trips while an
# ordinary posting stays well clear. The value is a judgment call, not a
# measured optimum, and it has not been tuned against a false positive rate.
BAIT_AD_MAX_WORDS = 150

# Reasoned, not measured. The Bait Ad description in the source analysis notes
# guaranteed wages spanning an implausible range, but the analysis never
# operationalized it. This threshold was added during integration. Treat it as
# the weakest signal here and see the project notebook for its hit rate.
IMPLAUSIBLE_RANGE_RATIO = 3.0

MIN_WORDS_TO_SCREEN = 5

_EMPLOYER_DESCRIBED = (
    'about us', 'about the company', 'about our', 'our company', 'we are a',
    'we are an', 'our team', 'our mission', 'founded in', 'established in',
    'headquartered', 'who we are', 'company overview',
)

_REQUIREMENTS_STATED = (
    'requirement', 'qualification', 'must have', 'must be able',
    'you will need', 'years of experience', 'bachelor', 'associate degree',
    'high school diploma', 'proficien', 'skills needed', 'skills required',
    'certification',
)

_REMOTE_OFFER = ('work from home', 'work at home', 'remote position',
                 'remote work', 'telecommute', 'work anywhere', 'from anywhere')

_NO_EXPERIENCE = ('no experience', 'no prior experience', 'no degree',
                  'training provided', 'we will train', 'will train you')

_IMMEDIATE_START = ('start immediately', 'immediate start', 'start today',
                    'start right away', 'start asap', 'hiring immediately')

_GUARANTEED_PAY = ('guaranteed pay', 'guaranteed income', 'guaranteed weekly',
                   'guarantee wages', 'guaranteed salary', 'guaranteed earnings')

ARCHETYPE_REFERENCE = {
    'The Ghost Agency': {
        'postings_in_study': 99,
        'fraud_rate_in_study': 1.0,
        'plain': ('A single operator posting many different jobs under one or '
                  'more invented agency names, reusing the same opening '
                  'paragraph every time. These postings look professional.'),
    },
    'The Bait Ad': {
        'postings_in_study': 71,
        'fraud_rate_in_study': 0.929,
        'plain': ('A very short posting that never says who is hiring and never '
                  'says what you need, offering remote work, no experience and '
                  'immediate pay. It is recognizable by what it leaves out.'),
    },
}

_WHITESPACE = re.compile(r'\s+')
PAY_RANGE_RE = re.compile(r'\$\s?(\d[\d,]*)\s*(?:-|–|to)\s*\$?\s?(\d[\d,]*)')


def normalize(text):
    """Lowercase and flatten whitespace so phrases match across line breaks."""
    return _WHITESPACE.sub(' ', str(text)).strip().lower()


def match_campaigns(posting_text):
    """Return every known campaign whose signature phrases appear in the posting.

    Exact substring matching on normalized text. A campaign that does not match
    stays silent, which is the point: a phrase rule cannot be confidently wrong
    the way a similarity score can.
    """
    flat = normalize(posting_text)
    hits = []
    for campaign in CAMPAIGN_SIGNATURES:
        found = [p for p in campaign['phrases'] if p in flat]
        if found:
            hits.append({'name': campaign['name'],
                         'matched_phrases': found,
                         'source': campaign['source'],
                         'note': campaign['note'],
                         'precision': campaign['precision']})
    return hits


def _any_present(flat_text, terms):
    return [t for t in terms if t in flat_text]


def find_wide_pay_range(posting_text):
    """Return the widest advertised pay range if its spread is implausible.

    The Bait Ad advertises guaranteed wages across a range no real role pays,
    such as $500 to $5,000 a week. Returns (low, high, ratio) or None.
    """
    widest = None
    for low_raw, high_raw in PAY_RANGE_RE.findall(str(posting_text)):
        try:
            low = float(low_raw.replace(',', ''))
            high = float(high_raw.replace(',', ''))
        except ValueError:
            continue
        if low <= 0 or high <= low:
            continue
        ratio = high / low
        if ratio >= IMPLAUSIBLE_RANGE_RATIO and (widest is None or ratio > widest[2]):
            widest = (low, high, ratio)
    return widest


def screen_structure(posting_text):
    """Score a posting against the Bait Ad shape, which is defined by absence.

    In the source corpus the Bait Ad ran a median of 81 words against 366 for
    the corpus, carried a company profile in 0.0% of cases against 80.7%, and
    screening questions in 5.6% against 49.1%. It is the one archetype a plain
    structural filter catches reliably.

    Brevity alone is never enough. Short, plainly written, entirely legitimate
    postings exist, and in the source data the two most representative members
    of one 90% fraudulent cluster were real gas station job advertisements.
    Both absences are therefore required before the shape is reported at all.
    """
    flat = normalize(posting_text)
    words = len(flat.split())

    employer_described = _any_present(flat, _EMPLOYER_DESCRIBED)
    requirements_stated = _any_present(flat, _REQUIREMENTS_STATED)

    offer = {
        'remote work': _any_present(flat, _REMOTE_OFFER),
        'no experience needed': _any_present(flat, _NO_EXPERIENCE),
        'immediate start': _any_present(flat, _IMMEDIATE_START),
        'guaranteed pay': _any_present(flat, _GUARANTEED_PAY),
    }
    offer_present = [label for label, hits in offer.items() if hits]

    pay_range = find_wide_pay_range(posting_text)
    if pay_range:
        offer_present.append('an implausibly wide pay range')

    qualifies = (words <= BAIT_AD_MAX_WORDS
                 and not employer_described
                 and not requirements_stated
                 and len(offer_present) >= 2)

    return {
        'shape': 'The Bait Ad' if qualifies else None,
        'word_count': words,
        'employer_described': bool(employer_described),
        'requirements_stated': bool(requirements_stated),
        'offer_markers': offer_present,
        'pay_range': pay_range,
    }


def screen_archetype(posting_text):
    """Report which known shape a posting resembles, and the evidence for it.

    Returns a dict whose 'shape' is an archetype name or None. A campaign
    phrase match takes precedence over a structural match, because campaign
    phrases identified 99 postings with no false positives in the source study
    while the structural shape ran at 92.9%.

    This never returns a fraud verdict and the caller should never present it
    as one. It says what a posting looks like and why.
    """
    text = str(posting_text or '')

    if len(text.split()) < MIN_WORDS_TO_SCREEN:
        return {'shape': None, 'basis': None, 'evidence': [],
                'explanation': 'Not enough text to screen.',
                'caution': None, 'reference': None, 'details': {}}

    campaigns = match_campaigns(text)
    structure = screen_structure(text)
    details = {'campaigns': campaigns, 'structure': structure}

    if campaigns:
        hit = campaigns[0]
        shape, basis = hit['name'], 'campaign phrase'
        evidence = [f'matched the phrase "{p}"' for p in hit['matched_phrases']]
        caution = ('Phrase matches come from postings collected in 2012 to 2014. '
                   'A match is strong evidence. No match means nothing.')
    elif structure['shape']:
        shape, basis = structure['shape'], 'structure'
        evidence = [f'only {structure["word_count"]} words long',
                    'never describes the employer',
                    'never states any requirements']
        evidence += [f'offers {m}' for m in structure['offer_markers']
                     if not m.startswith('an implausibly')]
        if structure['pay_range']:
            low, high, ratio = structure['pay_range']
            evidence.append(f'advertises ${low:,.0f} to ${high:,.0f}, '
                            f'a spread of {ratio:.1f} times')
        caution = ('Short, plainly written, legitimate postings exist and can '
                   'look like this. Treat it as a reason to check, not a verdict.')
    else:
        return {'shape': None, 'basis': None, 'evidence': [],
                'explanation': ('This posting does not match any shape this tool '
                                'recognizes. That is not reassurance, only an '
                                'absence of one specific kind of evidence.'),
                'caution': None, 'reference': None, 'details': details}

    reference = ARCHETYPE_REFERENCE[shape]
    return {
        'shape': shape,
        'basis': basis,
        'evidence': evidence,
        'explanation': reference['plain'],
        'caution': caution,
        'reference': {'postings_in_study': reference['postings_in_study'],
                      'fraud_rate_in_study': reference['fraud_rate_in_study']},
        'details': details,
    }


SELF_TEST_CASES = (
    ('campaign signature',
     'Administrative Assistant. Our recruiters have redesigned the recruiting '
     'wheel and are leveraging your career goals to place you with an employer '
     'who values what you bring. Represented candidates receive priority '
     'consideration throughout the placement process.',
     'The Ghost Agency'),
    ('bait ad shape',
     'Data Entry Clerk. Work from home, no experience needed, start immediately. '
     'Guaranteed pay of $600 to $4,500 weekly. Reply to be considered.',
     'The Bait Ad'),
    ('ordinary short posting',
     'Cashier. Part time, 20 hours weekly, evenings and weekends. High school '
     'diploma preferred. Must be able to lift 30 pounds and stand for a full '
     'shift. Apply in person at the service desk.',
     None),
    ('ordinary full posting',
     'Staff Accountant. About us: we are a regional firm founded in 1994 serving '
     'small businesses across the southwest. Requirements include a bachelor '
     'degree in accounting and three years of experience in general ledger work. '
     'You will need proficiency with reconciliations and month end close.',
     None),
)


def _self_test():
    print('Archetype screen self-test')
    failures = 0
    for label, posting, expected in SELF_TEST_CASES:
        result = screen_archetype(posting)
        ok = result['shape'] == expected
        failures += not ok
        print(f"\n  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"        expected: {expected}")
        print(f"        returned: {result['shape']}"
              + (f" by {result['basis']}" if result['basis'] else ''))
        for item in result['evidence']:
            print(f"          - {item}")
    print(f"\n  {len(SELF_TEST_CASES) - failures} of {len(SELF_TEST_CASES)} passed")
    return failures


if __name__ == '__main__':
    raise SystemExit(_self_test())
