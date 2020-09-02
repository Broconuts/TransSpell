# TransSpell - Contextual Text Error Correction
## 1. Introduction and Problem Definition
When working on non-quality-controlled, human-generated text data, lexical errors can be problematic for automatic analysis.
Automatically correcting these errors is a non-trivial task, however. 
The task itself can be split into two sub-tasks - error detection (1) and error correction (2) - with their own sets of challenges.

**Error detection** is the task of finding incorrect words in a given sentence.
Errors can be categorized in two major categories: non-word errors and word errors.
Non-word errors consitute errors that are not real words, e.g. when someone writes "thre" instead of "three".
Word errors, on the other hand, are errors that are correct words used in the incorrect context, e.g. when someone writes "tree" instead of "three".
Word error detection is even more complex to solve than non-word errors, which are difficult to automatically detect in their own right.

**Error correction** is the task of finding the correct word for a given sentence, given a (non-)word error.
Like error detection, this task can also be done with or without contextual information.
An example for non-contextual error correction would be the spelling algorithm introduced by Peter Norvig [include source].
The downside of these solutions, however, becomes apparent in the example of the typo "thre".
Did the author mean to write "there" or did they mean to write "three"? 
This information can only be obtained through the context, i.e. the words surrounding the text error.

TransSpell is an attempt to solve the problem of contextual text error correction.
It is the successor to [FastSpell](https://github.com/Broconuts/FastSpell/), a different spelling correction approach that was developed for this Study Project; it was abandoned due to performance issues.
It is, however, well-documented and offers some interesting insights into some more specific issues of spelling correction.

---
## 2. The Approach
Bidirectional Encoder Representations from Transformers (BERT) models [reference] provide a workflow for using models that were pre-trained in a masking task on large amounts of data.
Simply put, in masking tasks random tokens of a sequence are "masked" and the model is required to predict which token is most likely to occur in the position of the masked token.
BERT models have yielded promising results in diverse NLP tasks by continuing training the massive pre-trained models on a smaller dataset for a downstream task;
they can, however, still be used for masking tasks.
We can therefore use these pre-trained models to generate candidates for a token in a sentence given its context.
While this approach is computationally expensive, meaning that it does not scale very well, this project can be considered a proof of concept and we can dismiss the issue of computational efficiency for now.

### 2.1 The Model
Since the release of the original paper and models in 2019, multiple derivates of the BERT architecture have been developed.
Some aim to reduce computational demand at the cost of some performance, such as [DistilBERT](https://arxiv.org/abs/1910.01108) and [ALBERTA](https://arxiv.org/abs/1909.11942).
Others go the other way, improving the performance of the original models but being more computationally expensive to use, such as [XLNet](https://arxiv.org/abs/1906.08237) and [RoBERTa](https://arxiv.org/abs/1907.11692).

We have opted to use a distillated version of a RoBERTa model (i.e. a more lightweight model based on a more complex architecture) with 82M parameters.
The model we are using is a pyTorch-based implementation through [HuggingFace's transformers library](https://huggingface.co/transformers/).

---
## 3. Error Correction
We will first describe error correction, as the mechanism is quite straightforward and will actually play a part in error detection as well.
Given a word error and its surrounding sentence, we simply let the pre-trained RoBERTa model predict, which token should be in the given position based on its concept.
Sentence length is an important factor in how well this approach will perform. For one-word answers, for example, it will not work at all.
Furthermore, the pre-trained models seem to unproportionally weigh the positional context in the first and last token of a sequence, e.g. by suggesting replacement of the first token with tokens like bullet points or similar beginning-of-line indicators.
Because of this, this error correction approach will only work properly on tokens that are preceded by at least one token and followed by at least one token.

One could easily supply this model with an auxiliary candidate generation algorithm for these cases; 
as this is not necessary for the proof of concept and would increase the scope of this project even further, this auxiliary algorithm is omitted from this project.

---
## 4. Error Detection
In order to take full advantage of the context-sensitive error correction, our spelling corrector should be capable of detecting word errors (such as "tree" instead of "three").
This, in turn, requires our error detection to be context-sensitive as well.
In this section, a concept for a context-sensitive approach will be introduced.
FUrthermore, challenges that arise with this approach will be discussed, as well as the reasons for why we might have to rely on context-insensitive error detection for now.
### 4.1 Context-Sensitive
For context-sensitive error detection, we use the same technology as for error correction.
Given a sentence consisting of *n* tokens, we create *n* substrings where one token is masked; for example, in substring 2, the second token of the sentence is masked.
We then let the model determine 25 likely lexical candidates for the masked position.
If the original token is not among the generated tokens, we consider the original token to be a text error.
Due to the peculiar behavior regarding tokens in the first and last position of a sentence described in the previous section, the first and last token in a sentence need to be excluded from context-sensitive error detection.

#### The Issue
Unfortunately, this approach comes with two major flaws, not to mention high computational costs, reducing the scalability of this approach.
These will be described below.

Firstly, by masking a random word and letting the model determine a likely candidate for that word, we assert that the context is correct.
Let us assume a sentence with a word-error in the third token, e.g. "We made ensure that our customers get the products on time" when the author really meant "We made sure[...]".
Error detection iterates through the sentence, starting with the second token, "made". Given the context, i.e. "we [mask] ensure that [...]", the model will not generate the *correct* "made" but rather deem the actual token erroneous and supply something along the lines of "must".
The sentence now reads "we must ensure that [...]", which is now contextually coherent but not what the original author had in mind.
If the contextual error does not occur in the first checked position (i.e. the second token) of a sentence, TransSpell will make the context fit the original error, rather than make the error fit the context.

Secondly, this approach to error-detection can be boiled down to "making sure that a given sentence is a sequence of words that is statistically likely".
This entails that, rather than "just" looking for errors, we remodel sentences according to what the pre-trained model has learned sentences to look like.
For example, the sentence "We work hard to meet **consumer** requirements" is changed to "We work hard to meet **minimum** requirements". 
This is very problematic because it changes the meaning of the sentence.

### 4.2 Context-Insensitive
For the reasons described above, we cannot rely on context-sensitive error correction for now.
Unfortunately, context-insensitive error detection solutions are error-prone as well.
Usually, these approaches utilize finite lists of correct words, like lexica, and checking whether a token is contained in this finite list.
As is the norm with assets like that, these lists are not exhaustive. While there are -potentially - exhaustive options out there (e.g. the [Merriam-Webster API](https://dictionaryapi.com/)), these options are not for free and therefore not feasible for this project.
We therefore introduce a list of additional conditions besides "membership in a lexicon", in an attempt to reduce the amount of False Positives in error detection.

The complete process of context-insentive error detection for a given token is as follows (in order):
1. Check whether the token is longer than three characters. If it is not, do not check further and assume it is not an error. This aims to protect acronyms and similar domain-specific terms from being considered an error.
2. Check whether the token occurs more than 10 times in the entire corpus. If it does, do not consider it an error.
3. Check whether the token is contained in a lexicon for American English and British English, respectively.
If none of these conditions were met, consider the word an error. 
Please note that the second criterion requires knowledge of the entire corpus (as well as a certain size of the corpus for the measure to be effective).

---
## 5. Evaluation

---
## 5. Discussion