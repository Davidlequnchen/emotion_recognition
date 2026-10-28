"""
Turn an audio LLM's answer into one of the labels, or None when it cannot be read as exactly one label.

The prompts (prompts.yaml) ask for exactly one lowercase word from the list, but models sometimes add capitals or
punctuation, use another form of the word ("Anger", "Surprise!") or answer with a short sentence ("The speaker sounds
angry."). parse_label lowercases the answer, splits it into words (spaces and punctuation are dropped) and maps each
word to a label: a label itself, or one of its VARIANTS (anger -> angry, happiness -> happy, ...). It returns that
label when exactly one label appears, the answer has at most MAX_WORDS words and no word negates it ("not",
"isn't", ...). Anything else gives None: no label ("I can't tell"), two or more ("happy or sad"), a negation, a
longer answer, an emotion that is not in the list. The runner counts None as an invalid answer.

Usage (from the emotion_recognition/ folder):
    python scripts/llm_eval/parsing.py     checks the examples in parse_label's docstring and prints a few more
"""
import doctest
import re

MAX_WORDS = 15  # a longer answer (an explanation) is not searched for a label
# Other forms and close synonyms of the labels; a variant counts only when its label is in the list given
VARIANTS = {
    "anger": "angry", "angered": "angry", "angrily": "angry",
    "disgust": "disgusted", "disgusting": "disgusted", "disgustedly": "disgusted",
    "fear": "fearful", "fearfully": "fearful", "afraid": "fearful", "scared": "fearful", "frightened": "fearful",
    "happiness": "happy", "happily": "happy", "joy": "happy", "joyful": "happy",
    "neutrality": "neutral", "neutrally": "neutral",
    "sadness": "sad", "sadly": "sad", "unhappy": "sad",
    "surprise": "surprised", "surprising": "surprised", "surprisedly": "surprised",
}
NEGATIONS = {"not", "no", "never", "neither", "nor", "cannot"}  # and every word ending in n't
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")  # a word, with apostrophes inside it ("isn't", "speaker's")


def parse_label(text, labels):
    """The one label in a model's answer, or None.

    >>> labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    >>> parse_label("Neutral.", labels)
    'neutral'
    >>> parse_label("The speaker sounds angry.", labels)
    'angry'
    >>> parse_label("Surprise!", labels)
    'surprised'
    >>> parse_label("**Sadness**", labels)
    'sad'
    >>> print(parse_label("happy or sad", labels))
    None
    >>> print(parse_label("I can't tell", labels))
    None
    >>> print(parse_label("Not happy.", labels))
    None
    """
    if not text:
        return None
    words = WORD.findall(text.lower().replace("’", "'"))  # a typographic apostrophe counts as '
    if not words or len(words) > MAX_WORDS or any(w in NEGATIONS or w.endswith("n't") for w in words):
        return None
    # Every emotion word counts, also one whose label is not in the list, so that "surprised or sad" stays two answers
    found = {VARIANTS.get(w, w) for w in words} & (set(labels) | set(VARIANTS.values()))
    return found.pop() if len(found) == 1 and found <= set(labels) else None


def main():
    # Step 1: the examples in the docstring of parse_label
    failed, tried = doctest.testmod()
    print(f"Step 1 | docstring examples: {tried - failed} of {tried} passed")
    # Step 2: more answers, as a model might give them
    labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    answers = ["angry", " Fearful\n", "`disgusted`", "Anger", "It sounds like happiness to me.", "afraid",
               "neutral (no clear emotion)", "angry, disgusted, fearful, happy, neutral, sad, surprised", "",
               "Based on the rising pitch, the fast tempo and the loud voice of the speaker in this clip: surprised"]
    print("Step 2 | more answers:")
    for answer in answers:
        print(f"  {answer!r:<70} -> {parse_label(answer, labels)}")
    if failed:
        raise SystemExit(f"ERROR: {failed} docstring example(s) failed")


if __name__ == "__main__":
    main()
