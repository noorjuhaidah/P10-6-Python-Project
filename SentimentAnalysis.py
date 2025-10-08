"""
sentiment_backend.py
Plain-Python helpers for dictionary-based sentiment analysis.
Designed to work both from the CLI and inside Streamlit.
"""

from typing import List, Tuple, Dict, Optional, Set
import re


# ---------- Loading & tokenizing ----------

def load_afinn(path_or_file) -> Dict[str, int]:
    """
    Load the AFINN dictionary.
    - Accepts a filesystem path (str) OR a file-like object (Streamlit uploader).
    - Returns: dict mapping word -> integer score.
    """
    afinn: Dict[str, int] = {}

    # Allow either a path or an uploaded file (which has .read()).
    if hasattr(path_or_file, "read"):
        raw = path_or_file.read()
        # Uploaded files are bytes; decode to text.
        text = raw.decode("utf-8", errors="ignore")
        lines = text.splitlines()
    else:
        with open(path_or_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    for line in lines:
        if not line.strip():
            continue
        # Each line looks like: word<TAB>score
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        word, score = parts
        try:
            afinn[word] = int(score)
        except ValueError:
            # Skip malformed rows safely
            continue
    return afinn


def tokenize(sentence: str) -> List[str]:
    """
    This is our 'cleaning' stage: it drops punctuation/emoji/numbers.
    Convert a sentence into lowercase word tokens (letters + apostrophes). 
    Making it case-insensitive (e.g. 'Happy' and 'happy')
    """
    return re.findall(r"[a-zA-Z']+", sentence.lower()) 
    #''I LOVE this phone!' → ['i', 'love', 'this', 'phone'].


# ---------- Sentence scoring ----------

def score_sentence(sentence: str, afinn: Dict[str, int]) -> int:
    """
    Sum the AFINN scores of tokens in a sentence.
    Tokens not in the dictionary contribute 0.
    """
    return sum(afinn.get(w, 0) for w in tokenize(sentence))
    # love = +3, phone = 0 → total score = +3.


def split_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter:
    - Splits on '.', '!' or '?' followed by whitespace OR on line breaks.
    - Splits on new lines (\n+) to handle multi-line input.
    - Trims whitespace and drops empty chunks.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]
    # ‘This phone is great! Battery lasts long. Camera could be better.’ → Split into 3 separate sentences.



def score_paragraph(text: str, afinn: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """
    Given a paragraph/review, return:
      - sentences: list of sentence strings
      - scores:    list of integer sentiment scores, aligned by index
    """
    sentences = split_sentences(text)
    scores = [score_sentence(s, afinn) for s in sentences]
    return sentences, scores
    # One for the sentences and one for their sentiment scores

def most_extreme_sentence(sentences: List[str], scores: List[int]):
    """
    Return (max_sentence, max_score, min_sentence, min_score).
    If there are no sentences, returns (None, None, None, None).
    """
    if not sentences:
        return None, None, None, None
    max_i = max(range(len(scores)), key=lambda i: scores[i])
    min_i = min(range(len(scores)), key=lambda i: scores[i])
    return sentences[max_i], scores[max_i], sentences[min_i], scores[min_i]
    #If our sentences have scores [3, -2, 1] → ‘most positive’ = sentence 1 (score 3), ‘most negative’ = sentence 2 (score -2).

# ---------- Fixed-size sliding window ----------

def sliding_window_segments(scores: List[int], k: int):
    """
    Fixed window analysis over sentence scores.
    Returns:
      best_pos: (sum, (start_idx, end_idx))  # largest window sum
      best_neg: (sum, (start_idx, end_idx))  # smallest window sum
    If k <= 0 or not enough sentences, returns (None, None).
    """
    n = len(scores)
    if k <= 0 or n < k:
        return None, None

    window_sum = sum(scores[:k])
    max_sum, max_rng = window_sum, (0, k - 1)
    min_sum, min_rng = window_sum, (0, k - 1)

    for i in range(k, n):
        window_sum += scores[i] - scores[i - k]
        if window_sum > max_sum:
            max_sum, max_rng = window_sum, (i - k + 1, i)
        if window_sum < min_sum:
            min_sum, min_rng = window_sum, (i - k + 1, i)

    return (max_sum, max_rng), (min_sum, min_rng)


# ---------- Arbitrary-length segments ----------

def kadane_best_segment(scores: List[int]) -> Tuple[int, Tuple[int, int]]:
    """
    Kadane's algorithm (max subarray):
    Returns (best_sum, (start_idx, end_idx)) for the most positive contiguous segment.
    """
    best_sum = float("-inf")
    best_rng = (0, 0)

    cur_sum = 0
    cur_start = 0

    for i, x in enumerate(scores):
        if cur_sum <= 0:
            cur_sum = x
            cur_start = i
        else:
            cur_sum += x

        if cur_sum > best_sum:
            best_sum = cur_sum
            best_rng = (cur_start, i)

    return best_sum, best_rng


def kadane_worst_segment(scores: List[int]) -> Tuple[int, Tuple[int, int]]:
    """
    Min subarray (most negative contiguous segment).
    We reuse Kadane by flipping signs.
    """
    inv = [-x for x in scores]
    best_inv, (s, e) = kadane_best_segment(inv)
    return -best_inv, (s, e)


# ---------- Word-break (space reinsertion) ----------

def word_break_one(s: str, dictionary: Set[str]) -> Optional[List[str]]:
    n = len(s)
    dp: List[Optional[List[str]]] = [None] * (n + 1)
    dp[0] = []
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] is not None and s[j:i] in dictionary:
                dp[i] = dp[j] + [s[j:i]]
                break
    return dp[n]

def word_break_all(s: str, dictionary: Set[str], max_paths: int = 20) -> List[List[str]]:
    from functools import lru_cache
    @lru_cache(None)
    def backtrack(i: int) -> List[List[str]]:
        if i == len(s):
            return [[]]
        ans: List[List[str]] = []
        for j in range(i + 1, len(s) + 1):
            w = s[i:j]
            if w in dictionary:
                tails = backtrack(j)
                for t in tails:
                    if len(ans) >= max_paths:
                        break
                    ans.append([w] + t)
        return ans
    return backtrack(0)[:max_paths]

def word_break_with_punct(s: str, dictionary: Set[str], all_solutions: bool = False, max_paths: int = 20):
    tokens = re.findall(r"[a-zA-Z0-9]+|[^a-zA-Z0-9]", s)
    results = []

    for tok in tokens:
        if tok.isalnum():
            if all_solutions:
                segs = word_break_all(tok.lower(), dictionary, max_paths=max_paths)
                results.append([" ".join(seg) for seg in segs] if segs else [tok])
            else:
                seg = word_break_one(tok.lower(), dictionary)
                results.append([" ".join(seg)] if seg else [tok])
        else:
            results.append([tok])  # keep punctuation

    def combine(seq):
        out = seq[0]
        for chunk in seq[1:]:
            new_out = []
            for a in out:
                for b in chunk:
                    if re.match(r"^[a-zA-Z0-9]", b):
                        new_out.append(a + " " + b)
                    else:
                        new_out.append(a + b)
            out = new_out
        return out

    return combine(results)
