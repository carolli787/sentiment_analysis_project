import unittest

from sentiment.preprocessing import normalize_text


class NormalizeTextTests(unittest.TestCase):
    def test_lowercases_and_collapses_whitespace(self):
        self.assertEqual(normalize_text("  Hello   WORLD \n"), "hello world")

    def test_replaces_urls_and_mentions_with_placeholders(self):
        result = normalize_text("@bob look http://twitpic.com/2y1zl and www.example.com")
        self.assertEqual(result, "mentiontoken look urltoken and urltoken")

    def test_decodes_html_entities(self):
        self.assertEqual(normalize_text("rock &amp; roll &quot;yes&quot;"), 'rock & roll "yes"')

    def test_shortens_repeated_characters_to_two(self):
        self.assertEqual(normalize_text("sooooo goood!!!!"), "soo good!!")

    def test_keeps_two_repeated_characters(self):
        self.assertEqual(normalize_text("good"), "good")

    def test_removes_apostrophes_so_contractions_match(self):
        self.assertEqual(normalize_text("don't"), normalize_text("dont"))
        self.assertEqual(normalize_text("can’t"), "cant")

    def test_keeps_negation_words(self):
        self.assertEqual(normalize_text("not good, no fun, never again"), "not good, no fun, never again")


if __name__ == "__main__":
    unittest.main()
