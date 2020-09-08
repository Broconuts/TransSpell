import re
import copy
import collections
import torch
import enchant
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer
from nltk.corpus import stopwords


class TransSpell:
    """
    A context-sensitive spelling correction tool that uses pretrained transformer models to detect errors and supply
    corrections for them.
    """
    def __init__(self, minimum_token_length: int = 3, maximum_frequency: int = 10, corpus_path: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        self.model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
        self.stopwords = stopwords.words("english")
        self.char_minimum = minimum_token_length
        self.frequency_maximum = maximum_frequency
        self.dict_us = None
        self.dict_gb = None
        self.frequency_list = None
        if corpus_path:
            self.frequency_list = self.generate_frequency_list(corpus_path)

    def is_error(self, token: str) -> bool:
        """
        Uses a rule-based approach to evaluate whether a given token could be considered an error. This way of error
        detection is insensitive to context and will therefore only detect non-word errors.
        :param token: The word that is to be checked for being an error.
        :returns: True if token can be considered an error. False otherwise.
        """
        token = self.clean_token(token)
        # load required components if they are not loaded yet
        if not self.dict_us:
            self.dict_us = enchant.Dict("en_US")
        if not self.dict_gb:
            self.dict_gb = enchant.Dict("en_GB")
        
        # step 1: check if token is long enough to be considered an error
        if len(token) <= self.char_minimum:
            return False

        # step 2: check token frequency in corpus (if possible) to see if it is rare enough
        if self.frequency_list:
            if self.frequency_list[token] > self.frequency_maximum:
                return False

        # step 3: check whether token is contained in in English dictionaries (approximations of them)
        if self.dict_us.check(token) or self.dict_gb.check(token):
            return False

        return True

    def correct_errors(self, sequence: str) -> str:
        """
        Detects words that do not fit the context of the words surrounding it. If an error is detected, it is replaced
        by a word that better fits the context.

        :param sequence: The sentence that is to be checked for errors.
        :return: The input sequence, altered by replacing errors with their most likely suggestions.
        """
        temp_sent = sequence.split(" ")
        replacement_token = self.tokenizer.mask_token
        for i, token in enumerate(sequence.split(" ")):
            # don't check stopwords and the first token of a sentence to reduce false positives
            if i == 0 or token in self.stopwords:
                continue
            sent = copy.deepcopy(temp_sent)
            sent[i] = replacement_token
            sent = " ".join(sent)
            results = self.generate_candidates(sent, topn=25)
            contained_in_suggestions = False
            for j, suggestion in enumerate(results):
                # decode suggestion
                suggestion = self.tokenizer.decode([suggestion])
                # overwrite code with string
                results[j] = suggestion
                if suggestion.lower() == token.lower():
                    contained_in_suggestions = True
                    break
            if not contained_in_suggestions:
                temp_sent[i] = results[0]
        return " ".join(temp_sent)

    def generate_candidates(self, sequence: str, topn: int = 5) -> list:
        """
        Given a sequence with one masked token, generate a list of likely candidates for that token based on the
        context of the surrounding words.
        :param sequence: The input sentence with one masked word.
        :param topn: The amount of candidates that is to be supplied.
        :return: A list of candidates that could be in the masked position.
        """
        input_str = self.tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input_str == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(input_str)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]

        return torch.topk(mask_token_logits, topn, dim=1).indices[0].tolist()

    def generate_frequency_list(self, path: str) -> None:
        """
        Generate a Counter dict containing the frequency of every word in the corpus that spelling correction is to be 
        used on.

        :param path: The path to the file containing the texts that are to be analyzed.
        """
        # load data
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except FileNotFoundError:
            print("Corpus could not be found in provided path. Proceeding without frequency list.")
            return None
        self.frequency_list = collections.Counter()
        # tokenize each answer into a list of words
        for text in df["answers"]:
            answer = []
            for token in text.split():
                answer.append(self.clean_token(token))
            self.frequency_list.update(answer)

    def clean_token(self, token: str) -> str:
        """
        Cleans a given token by removing special characters and making them lowercase.

        :param token: The token that is to be cleaned.
        :returns: A cleaned version of the given token.
        """
        clean_token = ""
        for i, char in enumerate(list(token)):
            if re.match(r"\W", char):
                # if character is not one that might carry meaning for the word, do not add it to the clean token
                if char not in ["'", "-"]:
                    continue
                # if apostrophe is not in a valid position (i.e. the second to last token in the word), do not add it
                elif char == "'" and i != len(list(token)) - 2:
                    continue
            clean_token += char.lower()
        return clean_token


if __name__ == '__main__':
    ts = TransSpell()
    test_sent = "We made ensure to meet the customer requirements in a consistent manner."
    results = ts.correct_errors(test_sent)
    print(results)
