# app.py  (or WebInterface.py)
# Streamlit web interface + backend for AFINN-based sentiment analysis
# Run:  streamlit run app.py

from typing import List, Tuple, Dict, Optional, Set
import csv
import io
import re
import streamlit as st


# ---------- Backend helpers ----------

def load_afinn(path_or_file) -> Dict[str, int]:
    """Load AFINN word->score dictionary from a path or an uploaded file."""
    afinn: Dict[str, int] = {}

    # Accept both file-like (uploaded) and str path
    if hasattr(path_or_file, "read"):
        raw = path_or_file.read()
        text = raw.decode("utf-8", errors="ignore")
        lines = text.splitlines()
    else:
        # try the exact path; user can set data/AFINN-en-165.txt in sidebar
        with open(path_or_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        word, score = parts
        try:
            afinn[word] = int(score)
        except ValueError:
            continue
    return afinn


def tokenize(sentence: str) -> List[str]:
    """Lowercase, keep letters/apostrophes, drop punctuation/numbers."""
    return re.findall(r"[a-zA-Z']+", sentence.lower())


def score_sentence(sentence: str, afinn: Dict[str, int]) -> int:
    """Sum AFINN scores for tokens in a sentence."""
    return sum(afinn.get(w, 0) for w in tokenize(sentence))


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter on . ! ? or line breaks."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def score_paragraph(text: str, afinn: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """Return sentences and their sentiment scores for a paragraph/review."""
    sentences = split_sentences(text)
    scores = [score_sentence(s, afinn) for s in sentences]
    return sentences, scores


def most_extreme_sentence(sentences: List[str], scores: List[int]):
    """Return (most_pos_sentence, score, most_neg_sentence, score)."""
    if not sentences:
        return None, None, None, None
    max_i = max(range(len(scores)), key=lambda i: scores[i])
    min_i = min(range(len(scores)), key=lambda i: scores[i])
    return sentences[max_i], scores[max_i], sentences[min_i], scores[min_i]


def sliding_window_segments(scores: List[int], k: int):
    """
    Fixed-size window over sentence scores.
    Returns ((max_sum, (start,end)), (min_sum, (start,end))) or (None, None)
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


def kadane_best_segment(scores: List[int]) -> Tuple[int, Tuple[int, int]]:
    """Max subarray sum (most positive contiguous segment)."""
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
    """Min subarray sum (most negative contiguous segment) via sign flip."""
    inv = [-x for x in scores]
    best_inv, (s, e) = kadane_best_segment(inv)
    return -best_inv, (s, e)


def word_break_one(s: str, dictionary: Set[str]) -> Optional[List[str]]:
    """Return one valid segmentation of s using DP, or None if impossible."""
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
    """Return up to max_paths segmentations using memoized backtracking."""
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

def word_break_with_punct(
    s: str,
    dictionary: Set[str],
    all_solutions: bool = False,
    max_paths: int = 20
) -> List[str]:
    """
    Segment alphanumeric chunks with word-break; keep punctuation; 
    recombine with no space before punctuation. Returns a list of strings.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+|[^a-zA-Z0-9]", s)  # words + punctuation
    parts: List[List[str]] = []

    for tok in tokens:
        if tok.isalnum():
            if all_solutions:
                segs = word_break_all(tok.lower(), dictionary, max_paths=max_paths)
                parts.append([" ".join(seg) for seg in segs] if segs else [tok])
            else:
                seg = word_break_one(tok.lower(), dictionary)
                parts.append([" ".join(seg)] if seg else [tok])
        else:
            parts.append([tok])  # keep punctuation as-is

    # recombine → add space before words, not before punctuation
    out = parts[0] if parts else [""]
    for chunk in parts[1:]:
        new_out = []
        for a in out:
            for b in chunk:
                if re.match(r"^[a-zA-Z0-9]", b):   # next part starts with letter/number
                    new_out.append(a + " " + b)
                else:  # punctuation
                    new_out.append(a + b)
        out = new_out
    return out

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Sentiment Analysis (AFINN)", page_icon="🪄", layout="wide")
st.title("🪄 Sentiment Analysis — AFINN Dictionary")

with st.sidebar:
    st.header("Settings")

    # Choose dictionary: upload or default path
    dict_file = st.file_uploader("AFINN dictionary (.txt)", type=["txt"])
    if dict_file is None:
        # You can change this default to "data/AFINN-en-165.txt" if you keep it in /data
        default_path = "AFINN-en-165.txt"
        st.caption(f"Using local file: {default_path}")
        dict_path_or_file = default_path
    else:
        dict_path_or_file = dict_file

    # Load AFINN
    try:
        AFINN = load_afinn(dict_path_or_file)
        st.success(f"AFINN loaded with {len(AFINN):,} entries.")
    except Exception as e:
        st.error(f"Failed to load AFINN: {e}")
        st.stop()

    k = st.number_input("Sliding window size (sentences)", min_value=1, max_value=20, value=3)

tab1, tab2, tab3 = st.tabs(["Batch Review Analysis (CSV)", "Analyze Free Text", "Fix Spaceless Text (Word-Break)"])

# ---------- Tab 1: CSV ----------
with tab1:
    st.subheader("Upload CSV of Reviews")
    csv_file = st.file_uploader("**Select a CSV file containing your reviews**", type=["csv"], key="csv_upl")

    if csv_file:
        # Read header with DictReader so we can pick the text column by name
        try:
            csv_bytes = csv_file.read()
            text_io = io.StringIO(csv_bytes.decode("utf-8", errors="ignore"))
            reader = csv.DictReader(text_io)
            fieldnames = reader.fieldnames or []
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        if not fieldnames:
            st.error("No header detected in CSV. Please include a header row.")
            st.stop()

        colname = st.selectbox("Select the column that contains the review text:", fieldnames)

        if st.button("Analyze CSV", type="primary"):
            overall_max: Tuple[str, int] = ("", -10**9)
            overall_min: Tuple[str, int] = ("",  10**9)

            best_fixed = None   # (sum, (a,b), seg_text, row_idx)
            worst_fixed = None

            best_any = None     # (sum, (a,b), seg_text, row_idx)
            worst_any = None

            # rewind for the real pass
            text_io.seek(0)
            reader = csv.DictReader(text_io)

            for row_idx, row in enumerate(reader):
                review = (row.get(colname) or "").strip()
                if not review:
                    continue

                sentences, scores = score_paragraph(review, AFINN)
                if not sentences:
                    continue

                # global extreme sentences
                mx_s, mx_v, mn_s, mn_v = most_extreme_sentence(sentences, scores)
                if mx_s is not None and mx_v > overall_max[1]:
                    overall_max = (mx_s, mx_v)
                if mn_s is not None and mn_v < overall_min[1]:
                    overall_min = (mn_s, mn_v)

                # fixed-size segment
                fixed_pos, fixed_neg = sliding_window_segments(scores, int(k))
                if fixed_pos:
                    s, (a, b) = fixed_pos
                    txt = " ".join(sentences[a:b+1])
                    if (best_fixed is None) or (s > best_fixed[0]):
                        best_fixed = (s, (a, b), txt, row_idx)
                if fixed_neg:
                    s, (a, b) = fixed_neg
                    txt = " ".join(sentences[a:b+1])
                    if (worst_fixed is None) or (s < worst_fixed[0]):
                        worst_fixed = (s, (a, b), txt, row_idx)

                # arbitrary-length segments
                best_sum, (a, b) = kadane_best_segment(scores)
                worst_sum, (c, d) = kadane_worst_segment(scores)
                best_txt  = " ".join(sentences[a:b+1])
                worst_txt = " ".join(sentences[c:d+1])
                if (best_any is None) or (best_sum > best_any[0]):
                    best_any = (best_sum, (a, b), best_txt, row_idx)
                if (worst_any is None) or (worst_sum < worst_any[0]):
                    worst_any = (worst_sum, (c, d), worst_txt, row_idx)

            # ----- Display -----
            st.markdown("### 🌟 Sentence-level Extremes (across all reviews)")
            st.markdown(
                f"<span style='color:green; font-weight:bold'>Most Positive Sentence:</span> "
                f"<span style='color:green'>{overall_max[0]}</span> "
                f"<br><b>(score {overall_max[1]})</b>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<span style='color:red; font-weight:bold'>Most Negative Sentence:</span> "
                f"<span style='color:red'>{overall_min[0]}</span> "
                f"<br><b>(score {overall_min[1]})</b>",
                unsafe_allow_html=True
            )

            st.markdown(f"### 📏 Fixed-size Segment (window = {int(k)})")
            if best_fixed:
                s, (a, b), txt, ridx = best_fixed
                st.write(f"**Most Positive Segment:** score **{s}**, review #{ridx}, sentences {a}–{b}")
                st.info(txt)
            else:
                st.write("_Not enough sentences for a positive fixed-size segment._")

            if worst_fixed:
                s, (a, b), txt, ridx = worst_fixed
                st.write(f"**Most Negative Segment:** score **{s}**, review #{ridx}, sentences {a}–{b}")
                st.warning(txt)
            else:
                st.write("_Not enough sentences for a negative fixed-size segment._")

            st.markdown("### 🧲 Arbitrary-length Segment (Kadane)")
            if best_any:
                s, (a, b), txt, ridx = best_any
                st.write(f"**Most Positive Segment:** score **{s}**, review #{ridx}, sentences {a}–{b}")
                st.info(txt)
            if worst_any:
                s, (a, b), txt, ridx = worst_any
                st.write(f"**Most Negative Segment:** score **{s}**, review #{ridx}, sentences {a}–{b}")
                st.warning(txt)

    else:
        st.caption("*Your CSV must have a header row (e.g., ‘review’). Each row should contain **one review** or **paragraph**.*")


# ---------- Tab 2: Free text ----------
with tab2:
    st.subheader("Paste or Type a Review Below")
    text = st.text_area("**Your Review**", height=180, placeholder="e.g., The movie was fantastic but a little too long.")
    st.markdown("**Tip: The more detail you provide, the more accurate the analysis will be.**")

    if st.button("Analyze My Sentiment", type="primary", key="an_text"):
        sentences, scores = score_paragraph(text, AFINN)
        if not sentences:
            st.warning("No sentences detected.")
        else:
            st.markdown("#### Sentence Scores")
            # simple table for quick inspection
            rows = [{"i": i, "sentence": s, "score": v} for i, (s, v) in enumerate(zip(sentences, scores))]
            st.dataframe(rows, use_container_width=True)
            st.line_chart(scores)

            mx_s, mx_v, mn_s, mn_v = most_extreme_sentence(sentences, scores)
            st.write(f"**Most Positive Sentence:** `{mx_s}` (score **{mx_v}**)")
            st.write(f"**Most Negative Sentence:** `{mn_s}` (score **{mn_v}**)")

            fixed_pos, fixed_neg = sliding_window_segments(scores, int(k))
            st.markdown(f"#### Fixed-size Segment (k = {int(k)})")
            if fixed_pos:
                s, (a, b) = fixed_pos
                st.write(f"Most Positive Segment: score **{s}**, sentences {a}–{b}")
                st.info(" ".join(sentences[a:b+1]))
            if fixed_neg:
                s, (a, b) = fixed_neg
                st.write(f"Most Negative Segment: score **{s}**, sentences {a}–{b}")
                st.warning(" ".join(sentences[a:b+1]))

            best_sum, (a, b) = kadane_best_segment(scores)
            worst_sum, (c, d) = kadane_worst_segment(scores)
            st.markdown("#### Arbitrary-length Segment (Kadane)")
            st.write(f"Most Positive: score **{best_sum}**, sentences {a}–{b}")
            st.info(" ".join(sentences[a:b+1]))
            st.write(f"Most Negative: score **{worst_sum}**, sentences {c}–{d}")
            st.warning(" ".join(sentences[c:d+1]))


# ---------- Tab 3: Word-break ----------
with tab3:
    st.subheader("Smart Word Separator")
    st.markdown("By default, we use the AFINN dictionary plus common English words to detect natural breaks. You can also upload your own word list below.")

    # Optional custom dictionary upload
    custom_dict = st.file_uploader("Add Your Own Dictionary (Optional)", type=["txt"], key="dict_upl")
    st.caption("***Tip: Upload a `.txt` file with **one word per line** to improve segmentation accuracy.***")


    if custom_dict:
        text = custom_dict.read().decode("utf-8", errors="ignore")
        DICT = {w.strip().lower() for w in text.splitlines() if w.strip()}
    else:
        # fallback = AFINN + some common English words
        COMMON_WORDS = {
        # Articles / determiners
        "a", "an", "the", "this", "that", "these", "those",

        # Pronouns
        "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "our", "their",

        # Auxiliaries / linking verbs
        "is", "am", "are", "was", "were", "be", "been", "being",

        # Modals
        "will", "would", "can", "could", "shall", "should", "may", "might", "must",

        # Common verbs
        "do", "does", "did", "have", "has", "had", "go", "went", "gone", "make", "made",
        "say", "said", "get", "got", "see", "saw", "seen",

        # Connectors
        "and", "or", "but", "so", "because", "if", "when", "then",

        # Negatives
        "not", "no", "never", "nothing", "none", "nobody",

        # Intensifiers
        "very", "really", "too", "quite", "just",

        # Common adverbs
        "up", "down", "out", "in", "on", "off", "over", "under",

        # Common adjectives
        "good", "bad", "new", "old", "big", "small", "great",

        # Misc fillers
        "yes", "yeah", "okay", "ok", "oh", "well", "hmm"
        }

        DICT = set(AFINN.keys()) | COMMON_WORDS


    # --- Streamlit form (pressing Enter triggers submit) ---
    with st.form(key="wordbreak_form"):
        s = st.text_input("**Paste Your Text Below**")
        st.caption("***We will automatically figure out where the spaces should go.***")
        col1, col2 = st.columns(2)
        with col1:
            submit_one = st.form_submit_button("Find Best Segmentation")
        with col2:
            submit_many = st.form_submit_button("Explore All Possible Breaks (up to 20)")

    # --- handle buttons ---
    if s:
        if submit_one:
            outs = word_break_with_punct(s, DICT, all_solutions=False)
            if outs:
                st.success(outs[0])
            else:
                st.error("No valid segmentation found.")

        if submit_many:
            outs = word_break_with_punct(s, DICT, all_solutions=True, max_paths=20)
            if outs:
                for i, seg in enumerate(outs, 1):
                    st.write(f"{i}. {seg}")
            else:
                st.error("No valid segmentation found.")