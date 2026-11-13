import re

from langdetect import DetectorFactory, LangDetectException, detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentiment.lexicon import FINANCE_LEXICON

DetectorFactory.seed = 0  # make langdetect deterministic

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(FINANCE_LEXICON)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MARKDOWN_RE = re.compile(r"[*_~`^]|\[([^\]]*)\]\([^)]*\)")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def clean_text(raw: str) -> str:
    # Markdown links must be unwrapped before bare URLs are stripped, otherwise
    # greedy URL matching eats the link's closing paren and leaves "[text](" behind.
    text = _MARKDOWN_RE.sub(r"\1", raw)
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def score_sentiment(text: str) -> float:
    """VADER compound score in [-1, 1]; -1 bearish, +1 bullish."""
    return _analyzer.polarity_scores(text)["compound"]
