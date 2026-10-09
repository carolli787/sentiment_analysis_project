"""Text normalization applied to every tweet, both during training and at inference time.

Keeping this in one function guarantees the model sees the same kind of text in
production as it saw during training.
"""

import html
import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
# Three or more of the same character in a row ("loooove", "!!!!")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
APOSTROPHE_PATTERN = re.compile(r"['’`]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Placeholders keep the fact that a URL or mention was present (a weak signal)
# without letting the model memorize specific usernames or links.
URL_TOKEN = " urltoken "
MENTION_TOKEN = " mentiontoken "


def normalize_text(text: str) -> str:
    """Normalize a raw tweet into the form the model is trained on.

    Steps, in order:
      1. Decode HTML entities ("&amp;" -> "&", "&quot;" -> '"').
      2. Lowercase.
      3. Replace URLs and @mentions with placeholder tokens.
      4. Shorten runs of 3+ repeated characters to 2 ("soooo" -> "soo"), so
         stretched spellings share one feature while still differing from "so".
      5. Remove apostrophes, so "don't" and "dont" (both common in tweets)
         become the same word.
      6. Collapse whitespace.

    Negation words ("not", "no", "never") and stopwords are deliberately kept:
    they change the meaning of a sentence ("not good").
    """
    text = html.unescape(text).lower()
    text = URL_PATTERN.sub(URL_TOKEN, text)
    text = MENTION_PATTERN.sub(MENTION_TOKEN, text)
    text = REPEATED_CHAR_PATTERN.sub(r"\1\1", text)
    text = APOSTROPHE_PATTERN.sub("", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()
