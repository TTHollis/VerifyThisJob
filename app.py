# Loading Libraries/imports
import base64
import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    page_title='Job Posting Fraud Detector',
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


set_background('background3.png')


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


def combine_signals(model_verdict, rule_flags):
    """Combine the model's verdict with the independent rule-based scan.

    Args:
        model_verdict (str): 'FRAUDULENT', 'LEGITIMATE', or 'UNPARSEABLE'.
        rule_flags (list): Flags returned by screen_posting_rules.

    Returns:
        str: 'DANGER', 'CAUTION', or 'CLEAR'.
    """
    if model_verdict == 'FRAUDULENT':
        return 'DANGER'
    if rule_flags:
        return 'CAUTION'
    if model_verdict == 'UNPARSEABLE':
        return 'CAUTION'
    return 'CLEAR'


# Example postings
EXAMPLE_POSTINGS = {
    'Select an example...': '',
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
    'Ordinary: Staff Accountant': (
        'Title: Staff Accountant\n'
        'Location: Columbus, OH\n'
        'Company profile: Founded in 1998, Meridian Manufacturing is a '
        'family-owned industrial parts supplier with 140 employees across '
        'two facilities.\n'
        'Description: We are hiring a Staff Accountant to manage accounts '
        'payable, assist with month-end close, and support the '
        'Controller during the annual audit. This is an in-office role '
        'reporting to the Finance Director.\n'
        'Requirements: Bachelor\'s degree in Accounting, 2+ years of '
        'experience, proficiency in QuickBooks and Excel.\n'
        'Benefits: Health insurance, 401(k) match, paid time off'
    ),
}


# Main app layout
st.title('\U0001F575 Job Posting Fraud Detector')
st.write(
    'Paste a job posting below and this tool will flag whether it looks '
    'fraudulent and explain, in plain language, which specific details '
    'triggered that call.'
)
st.warning(
    'This is a research prototype, not a certified fraud determination. '
    'On 441 held-out test postings the fine-tuned model caught 87.8% of '
    'fraud and missed the rest, so results here are a first-pass screening '
    'aid - use your own judgment alongside them, especially on a result '
    'marked CAUTION below.')

with st.expander('About this tool and its limitations'):
    st.markdown(f'''
This tool runs a full fine-tune of `{MODEL_PATH.split('/')[-1]}`
(base model Qwen2.5-0.5B-Instruct), trained on the EMSCAD job posting
dataset (Vidros et al., 2017) to classify postings and cite the specific
red flags behind each verdict, it also uses independent and more recent
scam pattern terminology to catch anything the model may have missed.

**On 441 held-out postings never seen during training:** fraud recall
0.878, fraud precision 0.843, F1 0.860, 100% format compliance.

**Known limitation:** the model fabricates citations at a measurable rate
(about 13% of quoted phrases in evaluation were not actually present in
the source posting). This app checks every quote against the pasted
posting and flags anything it cannot verify - treat an unverified quote
as the model's error, not evidence.

This tool is biased toward flagging anything questionable rather than
staying quiet, on the theory that missing a real scam costs a job seeker
far more than a false alarm costs a second look. A LEGITIMATE verdict is
not a guarantee, and a FRAUDULENT verdict is not a certainty - read the
listed red flags and judge for yourself.
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
        model, tokenizer = load_model()
        with st.spinner('Analyzing...'):
            response = generate_response(model, tokenizer, posting_text)

        verdict = parse_verdict(response)
        confidence = parse_confidence(response)
        red_flags = extract_red_flags(response)
        fabricated = check_grounding(response, posting_text)
        rule_flags = screen_posting_rules(posting_text)
        overall = combine_signals(verdict, rule_flags)

        st.subheader('Result')
        if overall == 'DANGER':
            st.error(
                f'\U0001F6A8 High risk - the model flagged this as '
                f'FRAUDULENT (confidence: {confidence})')
        elif overall == 'CAUTION':
            if verdict == 'UNPARSEABLE':
                st.warning(
                    '⚠️ Caution - the model could not produce a '
                    'clear verdict for this posting. Read the raw output '
                    'below and use your own judgment.')
            else:
                st.warning(
                    f'⚠️ Caution - the model called this posting '
                    f'{verdict} (confidence: {confidence}), but an '
                    f'independent rule-based check found signals it may '
                    f'have missed. Review before proceeding.')
        else:
            st.success(
                f'✅ No red flags detected (model verdict: {verdict}, '
                f'confidence: {confidence})')

        if red_flags:
            st.markdown('**Red flags the model cited:**')
            for flag in red_flags:
                st.markdown(f'- {flag}')

        if rule_flags:
            st.markdown('**Additional signals from the independent '
                         'rule-based check:**')
            for flag in rule_flags:
                st.markdown(f'- {flag}')

        if not red_flags and not rule_flags:
            st.markdown('Neither check found anything to flag in this posting.')

        if fabricated:
            st.caption(
                f'⚠️ {len(fabricated)} quoted phrase(s) above could '
                f'not be verified against the posting text and may be '
                f'fabricated: {"; ".join(fabricated)}')

        with st.expander('Raw model output'):
            st.text(response)