# Loading Libraries/imports
import base64
import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from archetype_screen import screen_archetype

# Colorblind-safe project palette (carried over from Milestone 3)
ROYAL_BLUE = '#4169E1'
PURPLE = '#4B0082'
FOREST_GREEN = '#228B22'
AMBER = '#E69F00'
DARK_GRAY = '#333333'

# Run configuration
MODEL_PATH = 'TH69426a/emscad-fraud-ft-final'  # public Hugging Face Hub repo
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 180

SYSTEM_PROMPT = (
    'You are a fraud analyst who protects job seekers from employment scams. '
    'Classify the posting and explain the red flags in plain language a '
    'non-technical job seeker can understand.'
)

st.set_page_config(
    page_title='Verify This Job',
    page_icon='\U0001F575',
    layout='centered',
)


def set_background(image_path):
    """Set the app's background image and float the content in a white card.

    Args:
        image_path (str): Path to the background image file.
    """
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f'''
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stAppViewContainer"] .main .block-container,
        [data-testid="stMainBlockContainer"] {{
            background-color: rgba(255, 255, 255, 0.96);
            border-radius: 12px;
            padding: 2rem 2.5rem;
            margin-top: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
        }}
        </style>
    ''', unsafe_allow_html=True)


set_background('assets/background3.png')


@st.cache_resource(show_spinner='Loading the fine-tuned model...')
def load_model():
    """Load the fine-tuned model and tokenizer once per app session.

    Returns:
        tuple: (model, tokenizer), tokenizer padded on the left for generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32)
    model.config.use_cache = True
    model.eval()

    return model, tokenizer


def parse_verdict(text):
    """Extract the verdict from a generated response.

    Args:
        text (str): The model's generated output.

    Returns:
        str: 'FRAUDULENT', 'LEGITIMATE', or 'UNPARSEABLE'.
    """
    match = re.search(r'VERDICT:\s*(FRAUDULENT|LEGITIMATE)', text.upper())
    return match.group(1) if match else 'UNPARSEABLE'


def parse_confidence(text):
    """Extract the confidence level from a generated response.

    Args:
        text (str): The model's generated output.

    Returns:
        str: 'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'.
    """
    match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', text.upper())
    return match.group(1) if match else 'UNKNOWN'


def check_grounding(text, posting):
    """Flag which quoted phrases in the response do not appear in the posting.

    Args:
        text (str): The generated response.
        posting (str): The posting the response describes.

    Returns:
        list: Quoted phrases that were not found in the posting text.
    """
    quotes = re.findall(r'"([^"]+)"', text)
    lowered = posting.lower()
    return [q for q in quotes if q.lower() not in lowered]

# Verify if missing information is actually missing
EMPLOYER_LABEL_RE = re.compile(
    r'(?:company(?: profile| name)?|employer|organization)\s*:\s*(\S.{5,})',
    re.IGNORECASE)
REQUIREMENTS_LABEL_RE = re.compile(
    r'(?:requirements?|qualifications?|duties|responsibilities)\s*:\s*(\S.{15,})',
    re.IGNORECASE)
MISSING_EMPLOYER_PHRASES = ('never names', 'never describes the employer',
                             'does not name', 'does not describe the employer')
MISSING_REQUIREMENTS_PHRASES = ('no qualifications', 'not listed',
                                 'unclear what the work involves',
                                 'not part of the job req')


def check_claim_consistency(red_flags, posting_text):
    """Flag red-flag bullets that claim something is missing when it isn't.

    Args:
        red_flags (list): Flags extracted from the model's response.
        posting_text (str): The posting text the flags describe.

    Returns:
        list: Red-flag bullets whose "missing" claim looks contradicted.
    """
    has_employer = bool(EMPLOYER_LABEL_RE.search(posting_text))
    has_requirements = bool(REQUIREMENTS_LABEL_RE.search(posting_text))

    contradicted = []
    for flag in red_flags:
        lowered = flag.lower()
        if has_employer and any(p in lowered for p in MISSING_EMPLOYER_PHRASES):
            contradicted.append(flag)
        elif has_requirements and any(
                p in lowered for p in MISSING_REQUIREMENTS_PHRASES):
            contradicted.append(flag)

    return contradicted

def extract_red_flags(text):
    """Pull the bullet lines listed under RED FLAGS: out of a response.

    Args:
        text (str): The generated response.

    Returns:
        list: One string per red-flag bullet, in the order the model wrote them.
    """
    match = re.search(r'RED FLAGS:\s*(.*)', text, re.DOTALL)
    if not match:
        return []
    lines = match.group(1).strip().splitlines()
    return [line.lstrip('-').strip() for line in lines if line.strip()]


def generate_response(model, tokenizer, posting_text):
    """Generate a fraud verdict and explanation for one job posting.

    Args:
        model: The loaded fine-tuned model.
        tokenizer: The model's tokenizer, padded on the left.
        posting_text (str): The raw job posting text pasted by the user.

    Returns:
        str: The model's generated response.
    """
    prompt = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': posting_text}],
        tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        prompt, return_tensors='pt', truncation=True,
        max_length=MAX_LENGTH - MAX_NEW_TOKENS)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)

    prompt_length = inputs['input_ids'].shape[1]
    return tokenizer.decode(
        generated[0][prompt_length:], skip_special_tokens=True)


# Rule-based screening
MONEY_TERMS = ('registration fee', 'start-up fee', 'startup fee',
               'processing fee', 'wire transfer', 'western union',
               'money order', 'bank account', 'social security number',
               'credit card')
PAY_TERMS = ('guaranteed income', 'guarantee wages', 'no experience needed',
             'earn up to', 'six-figure income', 'unlimited income',
             'easy money', 'daily pay', 'work from home', 'data entry')
URGENCY_TERMS = ('act now', 'apply immediately', 'limited time',
                  'immediate start', 'positions filling fast', 'hurry',
                  'start today')
UPSELL_TERMS = ('strategy call', 'strategy session', 'consultation call',
                 'consultation fee', 'clarity call', 'discovery call',
                 'discovery session', 'coaching call', 'coaching fee',
                 'enrollment fee', 'enrollment call', 'program fee',
                 'course fee', 'certification fee', 'training fee',
                 'invest in yourself', 'unlock your potential')
FREE_EMAIL_RE = re.compile(r'[\w.-]+@(gmail|yahoo|hotmail|outlook|aol)\.com')


def quote_list(items, limit=2):
    """Join matched phrases into a readable quoted list for an explanation.

    Args:
        items (list): Matched vocabulary terms.
        limit (int): Maximum number of terms to name individually.

    Returns:
        str: The terms joined as a quoted, readable phrase.
    """
    picked = [f'"{item}"' for item in items[:limit]]
    return picked[0] if len(picked) == 1 else ' and '.join(picked)


def screen_posting_rules(text):
    """Scan raw posting text for known scam vocabulary, independent of the model.

    Args:
        text (str): Raw job posting text as pasted by the user.

    Returns:
        list: Plain-language flags for whatever vocabulary matched.
    """
    lowered = text.lower()
    flags = []

    money_hits = [t for t in MONEY_TERMS if t in lowered]
    if money_hits:
        flags.append(
            f'Mentions {quote_list(money_hits)}, and a real employer never '
            f'asks for payment or your financial details before you start')

    pay_hits = [t for t in PAY_TERMS if t in lowered]
    if pay_hits:
        flags.append(
            f'Uses pay language like {quote_list(pay_hits)}, which is not '
            f'how real employers describe compensation')

    urgency_hits = [t for t in URGENCY_TERMS if t in lowered]
    if urgency_hits:
        flags.append(
            f'Pressures you with phrases like {quote_list(urgency_hits)}, '
            f'a tactic used to rush your decision')

    upsell_hits = [t for t in UPSELL_TERMS if t in lowered]
    if upsell_hits:
        flags.append(
            f'References a paid {quote_list(upsell_hits)} as a step toward '
            f'the job - legitimate employers do not charge applicants for '
            f'coaching, enrollment, or "strategy" calls')

    if FREE_EMAIL_RE.search(lowered):
        flags.append(
            'Lists a free personal email domain rather than a company '
            'address, so you cannot verify who you would actually be '
            'working for')

    return flags


def combine_signals(model_verdict, red_flags, rule_flags, archetype=None):
    """Combine the model's verdict with its red flags, the rule scan and the shape check.

    A campaign phrase match escalates to DANGER on its own, overriding a
    LEGITIMATE model verdict. Those phrases identified 99 postings with no
    false positives in the source study, and the same study found that a
    supervised model catches only 14.1% of a campaign it has not been trained
    on, which is exactly the gap this signal exists to close.

    A structural shape match only ever raises CAUTION. Short legitimate
    postings can carry that shape, so it is a reason to look closer.

    Args:
        model_verdict (str): 'FRAUDULENT', 'LEGITIMATE', or 'UNPARSEABLE'.
        red_flags (list): Flags extracted from the model's own response.
        rule_flags (list): Flags returned by screen_posting_rules.
        archetype (dict): Result from screen_archetype, or None.

    Returns:
        str: 'DANGER', 'CAUTION', or 'CLEAR'.
    """
    if archetype and archetype.get('basis') == 'campaign phrase':
        return 'DANGER'
    if model_verdict == 'FRAUDULENT':
        return 'DANGER'

    hedge_flags = [f for f in red_flags if f.lower().startswith('worth verifying')]
    if rule_flags or hedge_flags:
        return 'CAUTION'
    if archetype and archetype.get('shape'):
        return 'CAUTION'
    if model_verdict == 'UNPARSEABLE':
        return 'CAUTION'
    return 'CLEAR'


def md_safe(text):
    """Escape dollar signs so Streamlit does not read them as math delimiters.

    Streamlit renders markdown with KaTeX enabled, so two dollar signs on one
    line open and close an inline math span. A pay range such as $600 to
    $4,500 renders its first figure in a serif math font and the rest in body
    text. Apply this to any string built from posting text or model output.
    """
    return str(text).replace('$', r'\$')


# Example postings
EXAMPLE_POSTINGS = {
    'Select an example...': '',
    'Known campaign: Administrative Assistant': (
        'Title: Administrative Assistant\n'
        'Location: Phoenix, AZ\n'
        'Company profile: Our recruiters have redesigned the recruiting '
        'wheel. We have partnered up in an effort to streamline placement '
        'for professionals across a range of industries.\n'
        'Description: We are leveraging your career goals to place you with '
        'an employer who values what you bring. Represented candidates '
        'receive priority consideration throughout the placement process.\n'
        'Requirements: Two years of administrative experience, proficiency '
        'with scheduling, correspondence and calendar management.\n'
        'Benefits: Competitive salary, health coverage through the placing '
        'employer, paid time off'
    ),
    'Bare bones: Package Processor': (
        'Title: Package Processor - Remote\n'
        'Location: Work from anywhere\n'
        'Company profile: \n'
        'Description: Work from home, no experience needed, start '
        'immediately. Guaranteed pay of $600 to $4,500 weekly. Reply to be '
        'considered.'
    ),
    'Suspicious: Data Entry Clerk': (
        'Title: Data Entry Clerk - Remote\n'
        'Location: Work from home\n'
        'Company profile: \n'
        'Description: Earn up to $500 daily working from home! No '
        'experience needed. We provide all equipment. Apply immediately, '
        'positions filling fast!\n'
        'Requirements: Must have a bank account for direct deposit setup. '
        'A small $75 processing fee is required to activate your account '
        'and ship your equipment.\n'
        'Benefits: Flexible hours, unlimited income potential'
    ),
        'Ordinary: Graphic Designer': (
        'Title: Graphic Designer\n'
        'Location: Denver, CO\n'
        'Company profile: Bright Path Design Studio has served regional '
        'retail clients since 2011, with a team of 12 full-time '
        'designers.\n'
        'Description: We\'re looking for a Graphic Designer to develop '
        'print and digital campaigns for local retail clients under the '
        'direction of our Creative Director. This is a full-time, '
        'in-office position.\n'
        'Requirements: Portfolio required, 3+ years of experience with '
        'Adobe Creative Suite, bachelor\'s degree preferred but not '
        'required.\n'
        'Benefits: Health insurance, dental, paid holidays, professional '
        'development stipend'
    ),
}


# Main app layout
st.title('\U0001F575 Verify This Job')
st.write(
    'Paste a job posting below and this tool will tell you whether it looks '
    'like a scam, and which specific details led it there.'
)
st.warning(
    'This is a research project, not an official fraud check. In testing it '
    'caught about seven of every eight scam postings, which means it misses '
    'some. Use it as a second opinion rather than a final answer, especially '
    'when the result says caution.')

with st.expander('How this works, and where it falls short'):
    st.markdown(f'''
Two things read your posting.

The first learned what scam postings sound like from about 18,000 real job
ads, roughly 800 of them confirmed scams, collected and published by
researchers (Vidros et al., 2017). The second is a separate check for
specific shapes that scam postings tend to take, found by studying that
same collection. The second one needs no model at all, so it answers
immediately while the first is still starting up.

**How well it does.** Tested on 441 postings it had never seen before, it
caught about seven of every eight scams. When it called something a scam,
it was right about five times out of six. Those numbers come from a fixed
research collection rather than from live job boards, so read them as a
rough sense of how far to trust it, not a promise.

**Something to watch for.** The model sometimes quotes wording the posting
never actually used, in testing about one quote in every eight. This app
checks every quote against what you pasted and marks any it cannot find.
A quote marked unverified is the model's mistake, not something the
posting said.

**Which way it leans.** It is built to speak up rather than stay quiet,
because missing a real scam costs a job seeker far more than a false alarm
costs a second look. A clean result is not a guarantee, and a scam result
is not proof. Read the reasons it gives and decide for yourself.

The postings used to build this were collected between 2012 and 2014.
Scam wording moves, so an exact phrase match is strong evidence, while no
match tells you nothing at all.

---

**Technical details.** Full fine-tune of `{MODEL_PATH.split('/')[-1]}`
(base: Qwen2.5-0.5B-Instruct) on the EMSCAD corpus (Vidros et al., 2017),
trained to classify postings and cite the red flags behind each verdict.
Held-out evaluation on 441 postings: fraud recall 0.878, fraud precision
0.843, F1 0.860, 100% format compliance. Quote fabrication rate
approximately 13%. The pattern check is an independent module carrying no
model, derived from an unsupervised clustering study of the same corpus
([Scam Signatures](https://github.com/TTHollis/DataSciencePortfolio/tree/main/projects/ScamSignatures));
an exact campaign phrase match sets the result on its own, a structural
match only ever raises a caution.
''')

example_choice = st.selectbox(
    'Try an example posting', options=list(EXAMPLE_POSTINGS.keys()))

default_text = EXAMPLE_POSTINGS[example_choice]
posting_text = st.text_area(
    'Job posting text', value=default_text, height=220,
    placeholder='Paste the full job posting here...')

analyze_clicked = st.button('Analyze Posting', type='primary')

if analyze_clicked:
    if not posting_text.strip():
        st.warning('Paste a job posting first.')
    else:
        archetype = screen_archetype(posting_text)

        st.subheader('Known pattern check')
        if archetype['shape']:
            st.markdown(f"This posting matches **{archetype['shape']}**, "
                        f"identified by {archetype['basis']}.")
            st.markdown(f"_{archetype['explanation']}_")
            st.markdown('**Why it matched:**')
            for item in archetype['evidence']:
                st.markdown(f'- {md_safe(item)}')
            ref = archetype['reference']
            st.markdown(
                f"In the 2012 to 2014 study corpus this pattern covered "
                f"{ref['postings_in_study']} postings, of which "
                f"{ref['fraud_rate_in_study']:.0%} were fraudulent. That is "
                f"historical context, not a probability for this posting.")
            # A callout, not a caption. This sentence is what keeps a pattern
            # match from reading as a verdict, so it has to carry the same
            # visual weight as the match itself.
            st.info(f"**What this does not mean.** {archetype['caution']}")
        else:
            st.markdown(archetype['explanation'])

        model, tokenizer = load_model()
        with st.spinner('Analyzing...'):
            response = generate_response(model, tokenizer, posting_text)

        verdict = parse_verdict(response)
        confidence = parse_confidence(response)
        red_flags = extract_red_flags(response)
        fabricated = check_grounding(response, posting_text)
        contradicted = check_claim_consistency(red_flags, posting_text)
        rule_flags = screen_posting_rules(posting_text)
        overall = combine_signals(verdict, red_flags, rule_flags, archetype)

        st.subheader('Result')
        if overall == 'DANGER':
            if archetype and archetype.get('basis') == 'campaign phrase':
                if verdict == 'LEGITIMATE':
                    model_line = (
                        f'The language model read the posting as LEGITIMATE '
                        f'(confidence: {confidence}), which is the kind of '
                        f'miss the phrase check exists to catch.')
                elif verdict == 'FRAUDULENT':
                    model_line = (
                        f'The language model reached the same verdict '
                        f'independently (confidence: {confidence}).')
                else:
                    model_line = (
                        'The language model did not return a usable verdict, '
                        'so the phrase match is carrying this result alone.')
                st.error(
                    f'\U0001F6A8 High risk - this posting reuses wording from '
                    f'a known fraud campaign. That match sets this result on '
                    f'its own. {model_line}')
            else:
                st.error(
                    f'\U0001F6A8 High risk - the model flagged this as '
                    f'FRAUDULENT (confidence: {confidence})')
        elif overall == 'CAUTION':
            if verdict == 'UNPARSEABLE':
                st.warning(
                    '⚠️ Caution - the model could not produce a '
                    'clear verdict for this posting. Read the raw output '
                    'below and use your own judgment.')
            elif rule_flags:
                st.warning(
                    f'⚠️ Caution - the model called this posting '
                    f'{verdict} (confidence: {confidence}), but an '
                    f'independent rule-based check found signals it may '
                    f'have missed. Review before proceeding.')
            else:
                st.warning(
                    f'⚠️ Caution - the model called this posting '
                    f'{verdict} (confidence: {confidence}), but it noted '
                    f'its own details worth verifying below. Review before '
                    f'proceeding.')
        else:
            st.success(
                f'✅ No red flags detected (model verdict: {verdict}, '
                f'confidence: {confidence})')

        if red_flags:
            st.markdown('**Red flags the model cited:**')
            for flag in red_flags:
                st.markdown(f'- {md_safe(flag)}')

        if rule_flags:
            st.markdown('**Additional signals from the independent '
                         'rule-based check:**')
            for flag in rule_flags:
                st.markdown(f'- {md_safe(flag)}')

        if not red_flags and not rule_flags:
            st.markdown('Neither check found anything to flag in this posting.')

        # These report the model contradicting the posting you pasted, which
        # is the failure this app exists to catch. They render as warnings
        # rather than captions so they cannot be read past.
        if fabricated:
            noun = 'quote' if len(fabricated) == 1 else 'quotes'
            quoted = '; '.join(f'"{q}"' for q in fabricated)
            st.warning(
                f'**Check this: {len(fabricated)} {noun} above did not come '
                f'from your posting.** The model put wording in quotation '
                f'marks that does not appear in the text you pasted, so '
                f'ignore those quotes as evidence either way. '
                f'Flagged: {md_safe(quoted)}')
        if contradicted:
            noun = 'reason' if len(contradicted) == 1 else 'reasons'
            st.warning(
                f'**Check this: {len(contradicted)} {noun} above may be '
                f'wrong.** The model says something is missing from the '
                f'posting, but it appears to actually be there. Read those '
                f'reasons against the posting yourself before weighing them.')

        with st.expander('Raw model output'):
            st.text(response)