"""Alignment copy of the text, the mapping back to the original words, and the language check."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmenter.language import detect_language  # noqa: E402
from segmenter.textnorm import WILDCARD, has_punctuation, number_words, spell, tokenize  # noqa: E402


class Normalisation(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(number_words(0), ["ZERO"])
        self.assertEqual(number_words(1250), ["ONE", "THOUSAND", "TWO", "HUNDRED", "FIFTY"])
        self.assertEqual(spell("$50"), ["FIFTY", "DOLLARS"])
        self.assertEqual(spell("24/7"), ["TWENTY", "FOUR", "SEVEN"])
        self.assertEqual(spell("20%"), ["TWENTY", "PERCENT"])
        self.assertEqual(spell("21st"), ["TWENTY", "FIRST"])
        self.assertEqual(spell("3:05"), ["THREE", "OH", "FIVE"])
        self.assertEqual(spell("1999"), ["NINETEEN", "NINETY", "NINE"])
        self.assertEqual(spell("2005"), ["TWO", "THOUSAND", "FIVE"])
        self.assertEqual(spell("3.5"), ["THREE", "POINT", "FIVE"])

    def test_letters(self):
        self.assertEqual(spell("don't"), ["DON'T"])
        self.assertEqual(spell("Café"), ["CAFE"])
        self.assertEqual(spell("well-known"), ["WELL", "KNOWN"])

    def test_unspellable_tokens_get_a_wildcard(self):
        for token in ["[NAME]", "<unk>", "***", "XXXX", "4420013", "007", "A12B", "(inaudible)"]:
            self.assertEqual(spell(token), [WILDCARD], token)

    def test_numbers_with_several_readings_get_a_wildcard(self):
        for token in ["-50", "1/2", "3/4", "10-15", "555-1234", "+1", "COVID-19"]:
            self.assertEqual(spell(token), [WILDCARD], token)
        self.assertEqual(spell("24/7"), ["TWENTY", "FOUR", "SEVEN"])     # fixed idiom stays spelled
        self.assertEqual(spell("and/or"), ["AND", "OR"])                 # no digits: plain words

    def test_multi_word_annotations_are_one_wildcard(self):
        for text in ["(inaudible speech) okay", "[background noise] okay", "{laughs loudly} okay"]:
            words = tokenize(text)
            self.assertEqual(len(words), 2, text)
            self.assertTrue(words[0].wildcard, text)
            self.assertEqual(words[1].units, ["OKAY"], text)

    def test_mapping_back_to_original_text(self):
        text = "Hi — I was charged $50 twice!  My number is [NAME], (laughs) okay..."
        words = tokenize(text)
        self.assertEqual([w.text for w in words],
                         ["Hi —", "I", "was", "charged", "$50", "twice!", "My", "number", "is", "[NAME],",
                          "(laughs)", "okay..."])
        for w in words:                                   # each word is an exact slice of the original
            self.assertEqual(text[w.char_start:w.char_end], w.text)
        self.assertEqual(words[4].units, ["FIFTY", "DOLLARS"])
        self.assertTrue(words[9].wildcard and words[10].wildcard)
        self.assertEqual(text[words[3].char_start:words[5].char_end], "charged $50 twice!")

    def test_text_cue(self):
        q = {w.text: w.punct for w in tokenize("Well, Mr. Smith said no. Really? Yes; fine")}
        self.assertEqual((q["Well,"], q["Mr."], q["no."], q["Really?"], q["Yes;"], q["fine"]),
                         (0.5, 0.0, 1.0, 1.0, 0.5, 0.0))

    def test_has_punctuation(self):
        self.assertTrue(has_punctuation("yes, okay"))
        self.assertFalse(has_punctuation("yes okay i was charged twice"))


class Language(unittest.TestCase):
    def test_english(self):
        self.assertEqual(detect_language("Hi, I was charged twice on my account and I want to know why.")[0], "en")
        self.assertEqual(detect_language("José said the payment is fine.")[0], "en")   # a name is not Spanish
        self.assertEqual(detect_language("Gracias, thank you so much for your help.")[0], "en")

    def test_spanish(self):
        self.assertEqual(detect_language("Hola, me cobraron dos veces en mi cuenta y quiero saber por qué.")[0], "es")
        self.assertEqual(detect_language("¿Cómo está usted?")[0], "es")

    def test_mixed(self):
        text = "Yes I called yesterday pero nadie me ayudó con el problema, and now the bill is wrong."
        self.assertEqual(detect_language(text)[0], "mixed")

    def test_default_when_no_cues(self):
        self.assertEqual(detect_language("[NAME]")[0], "en")


if __name__ == "__main__":
    unittest.main()
