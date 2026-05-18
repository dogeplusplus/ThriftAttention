"""RULER benchmark generators — port of NVIDIA/RULER/scripts/data/synthetic.

Produces deterministic-by-seed samples under any HF tokenizer, so every
attention configuration (fp16, fp4, topk) sees identical inputs.

Covers all 13 RULER tasks: NIAH (8 variants), VT, CWE, FWE, QA_1 (SQuAD),
QA_2 (HotpotQA).

External data (fetched once, cached under ~/.cache/ruler_data):
  * PaulGrahamEssays.json — NIAH/VT haystack; built from NVIDIA's
    URL list via a small scrape. Deps: requests, beautifulsoup4, html2text.
  * english_words.json     — CWE fallback vocabulary; tiny (~130 B).
  * qa_squad.json          — SQuAD validation set (QA_1), via HF datasets.
  * qa_hotpotqa.json       — HotpotQA distractor val (QA_2), via HF datasets.

Runtime deps: wonderwords (word lists), nltk (sentence tokenizer,
needs punkt_tab: `python -c "import nltk; nltk.download('punkt_tab')"`),
datasets (QA tasks).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import string
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

RULER_CACHE_DIR = Path(
    os.environ.get("RULER_CACHE_DIR", Path.home() / ".cache" / "ruler_data")
)

# ─────────────────────────────────────────────────────────────────────
# Task templates (copied verbatim from NVIDIA/RULER constants.py)
# ─────────────────────────────────────────────────────────────────────
TASK_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "niah": {
        "tokens_to_generate": 128,
        "template": (
            "Some special magic {type_needle_v} are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
            "{context}\n"
            "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        ),
        "answer_prefix": (
            " The special magic {type_needle_v} for {query} mentioned in the provided text are"
        ),
    },
    "variable_tracking": {
        "tokens_to_generate": 30,
        "template": (
            "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
            "{context}\n"
            "Question: Find all variables that are assigned the value {query} in the text above."
        ),
        "answer_prefix": (
            " Answer: According to the chain(s) of variable assignment in the text above, "
            "{num_v} variables are assigned the value {query}, they are: "
        ),
    },
    "common_words_extraction": {
        "tokens_to_generate": 120,
        "template": (
            "Below is a numbered list of words. In these words, some appear more often than others. "
            "Memorize the ones that appear most often.\n"
            "{context}\n"
            "Question: What are the 10 most common words in the above list?"
        ),
        "answer_prefix": " Answer: The top 10 words that appear most often in the list are:",
    },
    "freq_words_extraction": {
        "tokens_to_generate": 50,
        "template": (
            "Read the following coded text and track the frequency of each coded word. "
            "Find the three most frequently appeared coded words. {context}\n"
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?"
        ),
        "answer_prefix": (
            " Answer: According to the coded text above, the three most frequently appeared words are:"
        ),
    },
    "qa": {
        "tokens_to_generate": 32,
        "template": (
            "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\n"
            "The following are given documents.\n\n{context}\n\n"
            "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\n"
            "Question: {query}"
        ),
        "answer_prefix": " Answer:",
    },
}

# ─────────────────────────────────────────────────────────────────────
# Task parametrization (from NVIDIA/RULER synthetic.yaml)
# ─────────────────────────────────────────────────────────────────────
TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "niah_single_1":   {"kind": "niah", "type_haystack": "noise",  "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_single_2":   {"kind": "niah", "type_haystack": "essay",  "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_single_3":   {"kind": "niah", "type_haystack": "essay",  "type_needle_k": "words", "type_needle_v": "uuids",   "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_multikey_1": {"kind": "niah", "type_haystack": "essay",  "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 4, "num_needle_v": 1, "num_needle_q": 1},
    "niah_multikey_2": {"kind": "niah", "type_haystack": "needle", "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_multikey_3": {"kind": "niah", "type_haystack": "needle", "type_needle_k": "uuids", "type_needle_v": "uuids",   "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_multivalue": {"kind": "niah", "type_haystack": "essay",  "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 4, "num_needle_q": 1},
    "niah_multiquery": {"kind": "niah", "type_haystack": "essay",  "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 4},
    "vt":   {"kind": "variable_tracking", "type_haystack": "noise", "num_chains": 1, "num_hops": 4, "add_fewshot": True},
    "cwe":  {"kind": "common_words_extraction", "freq_cw": 30, "freq_ucw": 3, "num_cw": 10, "num_fewshot": 1},
    "fwe":  {"kind": "freq_words_extraction", "alpha": 2.0, "coded_wordlen": 6},
    "qa_1": {"kind": "qa", "dataset": "squad"},
    "qa_2": {"kind": "qa", "dataset": "hotpotqa"},
}
TASK_NAMES = list(TASK_SPECS.keys())

NOISE_HAYSTACK = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NIAH_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))
DOCUMENT_PROMPT = "Document {i}:\n{document}"


# ─────────────────────────────────────────────────────────────────────
# Tokenizer shim — NVIDIA uses .text_to_tokens; HF uses .encode.
# ─────────────────────────────────────────────────────────────────────
def _ntokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# ─────────────────────────────────────────────────────────────────────
# Lazy corpus / word-list loaders
# ─────────────────────────────────────────────────────────────────────
_ESSAY_WORDS: Optional[List[str]] = None
_NIAH_WORDS: Optional[List[str]] = None
_CWE_WORDS: Optional[List[str]] = None
_ENGLISH_WORDS: Optional[List[str]] = None
_QA_DATA: Dict[str, List[Dict[str, Any]]] = {}


def _essay_cache_path() -> Path:
    return RULER_CACHE_DIR / "PaulGrahamEssays.json"


def _download_essays() -> None:
    """Fetch Paul Graham essays (paulgraham.com + gkamradt needle repo)
    and write the concatenated text to ~/.cache/ruler_data/PaulGrahamEssays.json.

    This mirrors NVIDIA/RULER's download_paulgraham_essay.py.
    """
    try:
        import html2text
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise RuntimeError(
            "PaulGrahamEssays.json not cached and auto-download needs "
            "`pip install beautifulsoup4 html2text`. Alternatively, pre-cache "
            f"the file at {_essay_cache_path()}."
        ) from e

    url_list = (
        "https://raw.githubusercontent.com/NVIDIA/RULER/main/"
        "scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"
    )
    LOGGER.info("Fetching Paul Graham essay URL list from NVIDIA/RULER")
    with urllib.request.urlopen(url_list) as r:
        urls = [ln.strip() for ln in r.read().decode().splitlines() if ln.strip()]

    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_tables = True
    h.escape_all = True
    h.reference_links = False
    h.mark_code = False

    chunks: List[str] = []
    n_ok = n_fail = 0
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=30) as w:
                raw = w.read()
            if url.endswith(".html"):
                soup = BeautifulSoup(raw.decode("unicode_escape", "utf-8"), "html.parser")
                tag = soup.find("font")
                if tag is None:
                    n_fail += 1
                    continue
                chunks.append(h.handle(str(tag)))
            else:
                chunks.append(raw.decode("utf-8"))
            n_ok += 1
        except Exception as e:
            LOGGER.warning("essay fetch failed (%s): %s", url, e)
            n_fail += 1

    if not chunks:
        raise RuntimeError("No essays downloaded — check network")
    text = "".join(chunks)

    RULER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_essay_cache_path(), "w") as f:
        json.dump({"text": text}, f)
    LOGGER.info("Cached %d essay chunks (%d failed) → %s", n_ok, n_fail, _essay_cache_path())


def _load_essays() -> List[str]:
    """Return the essay corpus as a list of whitespace-split tokens
    (matches NVIDIA's `haystack = re.sub(...).split(" ")` form)."""
    global _ESSAY_WORDS
    if _ESSAY_WORDS is not None:
        return _ESSAY_WORDS

    path = _essay_cache_path()
    if not path.exists():
        _download_essays()

    with open(path) as f:
        text = json.load(f)["text"]
    _ESSAY_WORDS = re.sub(r"\s+", " ", text).split(" ")
    return _ESSAY_WORDS


def _load_niah_words() -> List[str]:
    """adj-noun composite words used by NIAH 'words' needles.

    Matches NVIDIA niah.py exactly: `[f"{adj}-{noun}" for adj in adjs for noun in nouns]`.
    Uses wonderwords's public RandomWord.filter — no private symbol dependence.
    """
    global _NIAH_WORDS
    if _NIAH_WORDS is not None:
        return _NIAH_WORDS
    try:
        from wonderwords import RandomWord
    except ImportError as e:
        raise RuntimeError("RULER 'words' needles need `pip install wonderwords`") from e
    rw = RandomWord()
    nouns = rw.filter(include_parts_of_speech=["noun"])
    adjs = rw.filter(include_parts_of_speech=["adjective"])
    _NIAH_WORDS = sorted({f"{a}-{n}" for a in adjs for n in nouns})
    return _NIAH_WORDS


def _load_cwe_words() -> List[str]:
    """Flat word list for CWE: nouns + adjectives + verbs, matches NVIDIA."""
    global _CWE_WORDS
    if _CWE_WORDS is not None:
        return _CWE_WORDS
    try:
        from wonderwords import RandomWord
    except ImportError as e:
        raise RuntimeError("RULER CWE needs `pip install wonderwords`") from e
    rw = RandomWord()
    nouns = rw.filter(include_parts_of_speech=["noun"])
    adjs = rw.filter(include_parts_of_speech=["adjective"])
    verbs = rw.filter(include_parts_of_speech=["verb"])
    _CWE_WORDS = sorted(set(nouns + adjs + verbs))
    return _CWE_WORDS


def _load_english_words() -> List[str]:
    """Fallback large-vocabulary list for CWE when num_words exceeds wonderwords."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is not None:
        return _ENGLISH_WORDS
    path = RULER_CACHE_DIR / "english_words.json"
    if not path.exists():
        url = (
            "https://raw.githubusercontent.com/NVIDIA/RULER/main/"
            "scripts/data/synthetic/json/english_words.json"
        )
        RULER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r, open(path, "wb") as f:
            f.write(r.read())
    with open(path) as f:
        data = json.load(f)
    _ENGLISH_WORDS = list(data.values())
    return _ENGLISH_WORDS


def _sent_tokenize(text: str) -> List[str]:
    """Wrap nltk.sent_tokenize with a clear error if punkt_tab is missing."""
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError as e:
        raise RuntimeError("RULER needs `pip install nltk`") from e
    try:
        return sent_tokenize(text.strip())
    except LookupError as e:
        raise RuntimeError(
            "nltk punkt_tab missing — run "
            "`python -c \"import nltk; nltk.download('punkt_tab')\"`"
        ) from e


# ─────────────────────────────────────────────────────────────────────
# Random needle helpers (seeded via `rng: random.Random`)
# ─────────────────────────────────────────────────────────────────────
def _rand_number(rng: random.Random, num_digits: int = 7) -> str:
    lo, hi = 10 ** (num_digits - 1), 10 ** num_digits - 1
    return str(rng.randint(lo, hi))


def _rand_word(rng: random.Random) -> str:
    return rng.choice(_load_niah_words())


def _rand_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _rand_needle(rng: random.Random, type_: str) -> str:
    if type_ == "numbers":
        return _rand_number(rng)
    if type_ == "words":
        return _rand_word(rng)
    if type_ == "uuids":
        return _rand_uuid(rng)
    raise ValueError(f"unknown needle type: {type_}")


# ─────────────────────────────────────────────────────────────────────
# NIAH
# ─────────────────────────────────────────────────────────────────────
def _niah_input(
    spec: Dict[str, Any],
    rng: random.Random,
    num_haystack: int,
) -> Tuple[str, List[str]]:
    """Build one NIAH input (context + question), returning (input_text, answers).

    Direct port of generate_input_output() from NVIDIA niah.py.
    """
    type_haystack = spec["type_haystack"]
    type_needle_k = spec["type_needle_k"]
    type_needle_v = spec["type_needle_v"]
    num_k, num_v, num_q = spec["num_needle_k"], spec["num_needle_v"], spec["num_needle_q"]
    num_k = max(num_k, num_q)  # matches NVIDIA niah.py argparse adjustment

    keys, values, needles = [], [], []
    for _ in range(num_k):
        keys.append(_rand_needle(rng, type_needle_k))
        vals = []
        for _ in range(num_v):
            vals.append(_rand_needle(rng, type_needle_v))
            needles.append(NIAH_NEEDLE.format(
                type_needle_v=type_needle_v, key=keys[-1], value=vals[-1],
            ))
        values.append(vals)

    # NVIDIA shuffles needles with a fresh Random(seed); we use the sample rng.
    rng.shuffle(needles)

    # Build context
    if type_haystack == "essay":
        haystack = _load_essays()
        if num_haystack <= len(haystack):
            text = " ".join(haystack[:num_haystack])
        else:
            reps = (num_haystack + len(haystack) - 1) // len(haystack)
            text = " ".join((haystack * reps)[:num_haystack])
        sents = _sent_tokenize(text)
        insert_points = (
            [0]
            + sorted(int(len(sents) * (d / 100)) for d in rng.sample(DEPTHS, len(needles)))
            + [len(sents)]
        )
        out_parts: List[str] = []
        for i in range(1, len(insert_points)):
            a, b = insert_points[i - 1], insert_points[i]
            out_parts.append(" ".join(sents[a:b]))
            if i - 1 < len(needles):
                out_parts.append(needles[i - 1])
        context = " ".join(out_parts)
    else:
        if type_haystack == "noise":
            sentences = [NOISE_HAYSTACK] * num_haystack
        elif type_haystack == "needle":
            sentences = [
                NIAH_NEEDLE.format(
                    type_needle_v=type_needle_v,
                    key=_rand_needle(rng, type_needle_k),
                    value=_rand_needle(rng, type_needle_v),
                )
                for _ in range(num_haystack)
            ]
        else:
            raise ValueError(f"unknown type_haystack: {type_haystack}")

        positions = sorted(rng.sample(range(num_haystack), len(needles)), reverse=True)
        for idx, needle in zip(positions, needles):
            sentences.insert(idx, needle)
        context = "\n".join(sentences)

    # Query + answer
    q_idx = rng.sample(range(num_k), num_q)
    queries = [keys[i] for i in q_idx]
    answers = [a for i in q_idx for a in values[i]]
    query = (
        ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]
    )

    template = TASK_TEMPLATES["niah"]["template"] + TASK_TEMPLATES["niah"]["answer_prefix"]
    type_needle_v_eff = type_needle_v
    if num_q * num_v == 1:
        template = template.replace("Some", "A").replace("are all", "is").replace("are", "is").replace("answers", "answer")
        type_needle_v_eff = type_needle_v[:-1]
    input_text = template.format(
        type_needle_v=type_needle_v_eff, context=context, query=query,
    )
    return input_text, answers


# ─────────────────────────────────────────────────────────────────────
# VT
# ─────────────────────────────────────────────────────────────────────
def _vt_chains(rng: random.Random, num_chains: int, num_hops: int) -> Tuple[List[List[str]], List[List[str]]]:
    k = 5
    vars_all = [
        "".join(rng.choices(string.ascii_uppercase, k=k))
        for _ in range((num_hops + 1) * num_chains)
    ]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append("".join(rng.choices(string.ascii_uppercase, k=k)))

    vars_ret, chains_ret = [], []
    for i in range(0, len(vars_all), num_hops + 1):
        this = vars_all[i : i + num_hops + 1]
        vars_ret.append(this)
        chain = [f"VAR {this[0]} = {rng.randint(10000, 99999)}"]
        for j in range(num_hops):
            chain.append(f"VAR {this[j + 1]} = VAR {this[j]} ")
        chains_ret.append(chain)
    return vars_ret, chains_ret


def _vt_shuffle_interleave(rng: random.Random, chains: List[List[str]]) -> List[str]:
    # NVIDIA uses a heap with random priorities so chain order is preserved
    # within each chain but chains are interleaved.
    import heapq
    heap = [(rng.random(), i, 0) for i in range(len(chains))]
    heapq.heapify(heap)
    out = []
    while heap:
        _, i, j = heapq.heappop(heap)
        out.append(chains[i][j])
        if j + 1 < len(chains[i]):
            heapq.heappush(heap, (rng.random(), i, j + 1))
    return out


def _vt_input(
    spec: Dict[str, Any],
    rng: random.Random,
    num_noises: int,
) -> Tuple[str, List[str]]:
    vars, chains = _vt_chains(rng, spec["num_chains"], spec["num_hops"])
    value = chains[0][0].split("=")[-1].strip()

    if spec["type_haystack"] == "essay":
        haystack = _load_essays()
        text = " ".join(haystack[:num_noises])
        sents = _sent_tokenize(text)
        flat = _vt_shuffle_interleave(rng, chains)
        insert_points = (
            [0]
            + sorted(int(len(sents) * (d / 100)) for d in rng.sample(DEPTHS, len(flat)))
            + [len(sents)]
        )
        parts: List[str] = []
        for i in range(1, len(insert_points)):
            a, b = insert_points[i - 1], insert_points[i]
            parts.append(" ".join(sents[a:b]))
            if i - 1 < len(flat):
                parts.append(flat[i - 1].strip() + ".")
        context = " ".join(parts)
    elif spec["type_haystack"] == "noise":
        sentences = [NOISE_HAYSTACK] * num_noises
        for chain in chains:
            positions = sorted(rng.sample(range(len(sentences)), len(chain)))
            for p, j in zip(positions, range(len(chain))):
                sentences.insert(p + j, chain[j])
        context = "\n".join(sentences)
    else:
        raise ValueError(f"vt type_haystack: {spec['type_haystack']}")

    context = context.replace(". \n", ".\n")
    template = TASK_TEMPLATES["variable_tracking"]["template"] + TASK_TEMPLATES["variable_tracking"]["answer_prefix"]
    input_text = template.format(context=context, query=value, num_v=spec["num_hops"] + 1)
    return input_text, vars[0]


# ─────────────────────────────────────────────────────────────────────
# CWE
# ─────────────────────────────────────────────────────────────────────
def _cwe_example(
    rng: random.Random,
    num_words: int,
    common_repeats: int,
    uncommon_repeats: int,
    num_cw: int,
) -> Tuple[str, List[str]]:
    words = _load_cwe_words()
    if num_words <= len(words):
        pool = rng.sample(words, num_words)
    else:
        pool = rng.sample(_load_english_words(), num_words)
    common, uncommon = pool[:num_cw], pool[num_cw:]
    word_list = common * common_repeats + uncommon * uncommon_repeats
    rng.shuffle(word_list)
    context = " ".join(f"{i+1}. {w}" for i, w in enumerate(word_list))
    return context, common


def _cwe_input(
    spec: Dict[str, Any],
    rng: random.Random,
    num_words: int,
    max_seq_length: int,
) -> Tuple[str, List[str]]:
    few_shots: List[Tuple[str, List[str]]] = []
    if max_seq_length < 4096:
        for _ in range(spec["num_fewshot"]):
            few_shots.append(_cwe_example(rng, 20, 3, 1, spec["num_cw"]))
        context, answer = _cwe_example(rng, num_words, 6, 1, spec["num_cw"])
    else:
        for _ in range(spec["num_fewshot"]):
            few_shots.append(_cwe_example(rng, 40, 10, 3, spec["num_cw"]))
        context, answer = _cwe_example(rng, num_words, spec["freq_cw"], spec["freq_ucw"], spec["num_cw"])

    template = TASK_TEMPLATES["common_words_extraction"]["template"] + TASK_TEMPLATES["common_words_extraction"]["answer_prefix"]
    few_shot_blocks = [
        template.format(context=fs_ctx, query="")
        + " "
        + " ".join(f"{i+1}. {w}" for i, w in enumerate(fs_ans))
        for fs_ctx, fs_ans in few_shots
    ]
    input_text = template.format(context=context, query="")
    return ("\n".join(few_shot_blocks) + "\n" + input_text), answer


# ─────────────────────────────────────────────────────────────────────
# FWE
# ─────────────────────────────────────────────────────────────────────
def _fwe_input(
    spec: Dict[str, Any],
    rng: random.Random,
    num_words: int,
    vocab_size: int,
) -> Tuple[str, List[str]]:
    alpha, coded_wordlen = spec["alpha"], spec["coded_wordlen"]
    # Build a deterministic coded vocab
    vocab = list({
        "".join(rng.choices(string.ascii_lowercase, k=coded_wordlen))
        for _ in range(vocab_size * 2)
    })[:vocab_size]
    while len(vocab) < vocab_size:
        vocab.append("".join(rng.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(set(vocab))
    rng.shuffle(vocab)
    vocab[0] = "..."  # noise token, treated as ignored

    # Zipf weights, truncated to integer counts (matches NVIDIA's behavior)
    k = np.arange(1, len(vocab) + 1)
    weights = k.astype(np.float64) ** -alpha
    counts = (num_words * weights / weights.sum()).astype(int)  # approximation of ·/zeta(alpha)

    words: List[str] = []
    for w, c in zip(vocab, counts):
        words.extend([w] * int(c))
    rng.shuffle(words)

    template = TASK_TEMPLATES["freq_words_extraction"]["template"] + TASK_TEMPLATES["freq_words_extraction"]["answer_prefix"]
    input_text = template.format(context=" ".join(words), query="")
    answer = vocab[1:4]  # top-3 non-noise
    return input_text, answer


# ─────────────────────────────────────────────────────────────────────
# QA (SQuAD / HotpotQA)
# ─────────────────────────────────────────────────────────────────────
def _load_qa_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """Return SQuAD/HotpotQA validation set as a stable list of
    {context: [paragraph, ...], question: str, outputs: [str, ...]}.

    Cached as JSON under RULER_CACHE_DIR; sorted by question text so the
    sample-index → question mapping is stable across HF dataset versions.
    """
    if dataset_name in _QA_DATA:
        return _QA_DATA[dataset_name]

    cache_path = RULER_CACHE_DIR / f"qa_{dataset_name}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        _QA_DATA[dataset_name] = data
        return data

    # Load parquet directly via huggingface_hub to avoid `datasets` feature-type
    # incompatibilities (e.g. `List` missing in older `datasets` versions).
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("RULER QA tasks need `pip install huggingface_hub pandas`") from e

    if dataset_name == "squad":
        pq = hf_hub_download(
            repo_id="rajpurkar/squad",
            filename="plain_text/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(pq)
        seen_q = set()
        data = []
        for row in df.itertuples(index=False):
            answers = sorted(set(row.answers["text"]))
            if not answers or row.question in seen_q:
                continue
            seen_q.add(row.question)
            data.append({
                "context": [row.context],
                "question": row.question,
                "outputs": answers,
            })
    elif dataset_name == "hotpotqa":
        pq = hf_hub_download(
            repo_id="hotpot_qa",
            filename="distractor/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(pq)
        data = []
        for row in df.itertuples(index=False):
            sf_titles = set(row.supporting_facts["title"])
            gold = []
            for title, sentences in zip(row.context["title"], row.context["sentences"]):
                if title in sf_titles:
                    gold.append(" ".join(sentences))
            if not gold or not row.answer:
                continue
            data.append({
                "context": gold,
                "question": row.question,
                "outputs": [row.answer],
            })
    else:
        raise ValueError(f"unknown QA dataset: {dataset_name}")

    data.sort(key=lambda x: x["question"])
    RULER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    LOGGER.info("Cached %d %s QA samples → %s", len(data), dataset_name, cache_path)
    _QA_DATA[dataset_name] = data
    return data


def _qa_input(
    spec: Dict[str, Any],
    rng: random.Random,
    num_docs: int,
    sample_index: int,
    qa_data: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """Build one QA prompt: gold paragraphs + distractors from other questions.

    sample_index picks the question deterministically; rng controls distractor
    selection and shuffle. Mirrors NVIDIA RULER's qa.py: total docs is num_docs,
    gold paragraphs are mixed in among distractors, then all are shuffled.
    """
    n = len(qa_data)
    q_idx = sample_index % n
    item = qa_data[q_idx]
    gold = list(item["context"])

    n_distractors = max(num_docs - len(gold), 0)
    n_distractors = min(n_distractors, n - 1)
    if n_distractors > 0:
        other_indices = list(range(n))
        other_indices.pop(q_idx)
        sampled = rng.sample(other_indices, n_distractors)
        distractors = [qa_data[i]["context"][0] for i in sampled]
    else:
        distractors = []

    docs = gold + distractors
    rng.shuffle(docs)
    context = "\n\n".join(
        DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(docs)
    )
    template = TASK_TEMPLATES["qa"]["template"] + TASK_TEMPLATES["qa"]["answer_prefix"]
    input_text = template.format(context=context, query=item["question"])
    return input_text, item["outputs"]


# ─────────────────────────────────────────────────────────────────────
# Binary search — find the largest haystack size whose prompt fits.
# ─────────────────────────────────────────────────────────────────────
def _binary_search_size(
    eval_fn,
    tokenizer,
    budget: int,
    lo: int,
    hi: int,
    tokens_to_generate: int,
) -> int:
    """eval_fn(n) -> input_text. Return the largest n such that
    ntokens(input_text) + tokens_to_generate <= budget."""
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        text = eval_fn(mid)
        total = _ntokens(tokenizer, text) + tokens_to_generate
        if total <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ─────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────
def _seed_for(seed: int, task: str, target_length: int) -> int:
    h = hashlib.sha256(f"{seed}|{task}|{target_length}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _cache_path(
    cache_dir: Path,
    tokenizer_id: str,
    task: str,
    target_length: int,
    seed: int,
    num_samples: int,
) -> Path:
    tid = re.sub(r"[^A-Za-z0-9._-]", "_", tokenizer_id)
    return cache_dir / "samples" / tid / f"{task}_{target_length}_{seed}_{num_samples}.jsonl"


def _tokenizer_id(tokenizer) -> str:
    return getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__


def generate_samples(
    tokenizer,
    task: str,
    target_length: int,
    num_samples: int = 100,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Generate (or load from cache) RULER samples for one (task, length).

    Returns a list of sample dicts:
        {task, target_length, input, outputs, max_gen, length, answer_prefix}

    'input' is the prompt *without* the answer_prefix appended; 'answer_prefix'
    is the continuation fed to the model before it starts generating. Call
    sites should concatenate `input + answer_prefix` to build the actual prompt
    (mirrors NVIDIA's output format).
    """
    if task not in TASK_SPECS:
        raise ValueError(f"unknown RULER task: {task}")
    spec = TASK_SPECS[task]
    kind = spec["kind"]

    cache_dir = Path(cache_dir) if cache_dir else RULER_CACHE_DIR
    cache_file = _cache_path(
        cache_dir, _tokenizer_id(tokenizer), task, target_length, seed, num_samples,
    )
    if cache_file.exists():
        with open(cache_file) as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    tpl = TASK_TEMPLATES[kind]
    tokens_to_generate = tpl["tokens_to_generate"]
    budget = target_length  # NVIDIA convention: prompt + gen <= target_length

    task_seed = _seed_for(seed, task, target_length)
    LOGGER.info(
        "Generating RULER %s @ %d tokens (n=%d, seed=%d)",
        task, target_length, num_samples, task_seed,
    )

    # ── Pick an initial search range + eval fn per task ──
    if kind == "niah":
        incremental = 500 if spec["type_haystack"] == "essay" else 25
        if target_length < 4096 and spec["type_haystack"] != "essay":
            incremental = 5

        def eval_fn(n, _rng=random.Random(task_seed)):
            return _niah_input(spec, _rng, n)[0]

        sample_text = eval_fn(incremental)
        t_per = _ntokens(tokenizer, sample_text) / incremental
        upper = max(int(budget / t_per * 3), incremental * 2)
        num_haystack = _binary_search_size(
            eval_fn, tokenizer, budget, incremental, upper, tokens_to_generate,
        )

        samples = []
        for i in range(num_samples):
            used = num_haystack
            while True:
                # Fresh RNG per sample so samples are independent but deterministic
                sample_rng = random.Random(task_seed + i)
                input_text, answer = _niah_input(spec, sample_rng, used)
                length = _ntokens(tokenizer, input_text) + tokens_to_generate
                if length <= budget or used <= incremental:
                    break
                used -= incremental
            samples.append(_split_prefix(task, kind, input_text, answer, length))

    elif kind == "variable_tracking":
        incremental = 50 if spec["type_haystack"] == "essay" else 5

        def eval_fn(n, _rng=random.Random(task_seed)):
            return _vt_input(spec, _rng, n)[0]

        sample_text = eval_fn(incremental)
        t_per = _ntokens(tokenizer, sample_text) / incremental
        upper = max(int(budget / t_per * 3), incremental * 2)
        num_noises = _binary_search_size(
            eval_fn, tokenizer, budget, incremental, upper, tokens_to_generate,
        )

        samples = []
        for i in range(num_samples):
            used = num_noises
            while True:
                sample_rng = random.Random(task_seed + i)
                input_text, answer = _vt_input(spec, sample_rng, used)
                length = _ntokens(tokenizer, input_text) + tokens_to_generate
                if length <= budget or used <= incremental:
                    break
                used -= incremental
            samples.append(_split_prefix(task, kind, input_text, answer, length))

    elif kind == "common_words_extraction":
        incremental = 10

        def eval_fn(n, _rng=random.Random(task_seed)):
            return _cwe_input(spec, _rng, n, target_length)[0]

        sample_text = eval_fn(4096)
        t_per = _ntokens(tokenizer, sample_text) / 4096
        upper = max(int(budget / t_per * 2), incremental * 2)
        num_words = _binary_search_size(
            eval_fn, tokenizer, budget, incremental, upper, tokens_to_generate,
        )

        samples = []
        for i in range(num_samples):
            used = num_words
            while True:
                sample_rng = random.Random(task_seed + i)
                input_text, answer = _cwe_input(spec, sample_rng, used, target_length)
                length = _ntokens(tokenizer, input_text) + tokens_to_generate
                if length <= budget or used <= incremental:
                    break
                used -= incremental
            samples.append(_split_prefix(task, kind, input_text, answer, length))

    elif kind == "freq_words_extraction":
        vocab_size = max(target_length // 50, 4)
        inner_budget = budget - tokens_to_generate

        def eval_fn(n, _rng=random.Random(task_seed)):
            return _fwe_input(spec, _rng, n, vocab_size)[0]

        incremental = max(inner_budget // 32, 1)
        num_words_init = inner_budget // spec["coded_wordlen"]
        upper = max(num_words_init * 3, incremental * 2)
        num_words = _binary_search_size(
            eval_fn, tokenizer, budget, incremental, upper, tokens_to_generate,
        )

        samples = []
        for i in range(num_samples):
            sample_rng = random.Random(task_seed + i)
            input_text, answer = _fwe_input(spec, sample_rng, num_words, vocab_size)
            length = _ntokens(tokenizer, input_text) + tokens_to_generate
            # FWE rarely overshoots but trim if it does
            trim = num_words
            while length > budget and trim > incremental:
                trim -= incremental
                sample_rng = random.Random(task_seed + i)
                input_text, answer = _fwe_input(spec, sample_rng, trim, vocab_size)
                length = _ntokens(tokenizer, input_text) + tokens_to_generate
            samples.append(_split_prefix(task, kind, input_text, answer, length))

    elif kind == "qa":
        qa_data = _load_qa_dataset(spec["dataset"])
        incremental = 10

        def eval_fn(n, _rng=random.Random(task_seed)):
            return _qa_input(spec, _rng, n, 0, qa_data)[0]

        sample_text = eval_fn(incremental)
        t_per = _ntokens(tokenizer, sample_text) / incremental
        upper = max(int(budget / t_per * 3), incremental * 2)
        num_docs = _binary_search_size(
            eval_fn, tokenizer, budget, incremental, upper, tokens_to_generate,
        )

        samples = []
        for i in range(num_samples):
            used = num_docs
            while True:
                sample_rng = random.Random(task_seed + i)
                input_text, answer = _qa_input(spec, sample_rng, used, i, qa_data)
                length = _ntokens(tokenizer, input_text) + tokens_to_generate
                if length <= budget or used <= incremental:
                    break
                used -= incremental
            samples.append(_split_prefix(task, kind, input_text, answer, length))

    else:
        raise ValueError(f"unknown task kind: {kind}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    LOGGER.info("Cached %d samples → %s", len(samples), cache_file)
    return samples


def _split_prefix(
    task: str, kind: str, full_input: str, outputs: Any, length: int,
) -> Dict[str, Any]:
    """Split the full prompt into (input, answer_prefix) on the task's prefix marker.

    Mirrors NVIDIA's output format — downstream code concatenates them back
    before tokenizing, or tokenizes the prefix separately for last-position
    scoring.
    """
    prefix_tpl = TASK_TEMPLATES[kind]["answer_prefix"]
    marker = prefix_tpl[:10]
    idx = full_input.rfind(marker)
    if idx == -1:
        # Should never happen — means the template was changed without updating marker.
        raise RuntimeError(f"answer_prefix marker not found for task {task}")
    input_text = full_input[:idx]
    answer_prefix = full_input[idx:]
    max_gen = TASK_TEMPLATES[kind]["tokens_to_generate"]
    return {
        "task": task,
        "target_length": length,
        "input": input_text,
        "answer_prefix": answer_prefix,
        "outputs": list(outputs) if isinstance(outputs, (list, tuple)) else [outputs],
        "max_gen": max_gen,
        "length": length,
    }
