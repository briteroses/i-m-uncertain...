SAVE_PREFIX = ""
# SAVE_PREFIX = "/content/uncertainty/"

DATASET_LABEL_REGISTRY = {
    "imdb": ["negative", "positive"], # This is for normal IMDB
    "amazon_polarity": ["negative", "positive"],
    # "ag_news": ["politics", "sports", "business", "technology"],
    # "dbpedia_14": ["company", "educational institution", "artist", "athlete", "office holder", "mean of transportation", "building", "natural place", "village", "animal",  "plant",  "album",  "film",  "written work"],
    "copa": ["choice 1", "choice 2"],
    "rte": ["yes", "no"],   # whether entail
    "boolq": ["false", "true"],
    "qnli": ["yes", "no"],  # represent whether entail
    "piqa": ["solution 1", "solution 2"],
    # "story_cloze": ["choice 1", "choice 2"],
}

MODEL_TYPE_REGISTRY = {
    "gpt2": "decoder",
    "gpt2-medium": "decoder",
    "gpt2-large": "decoder",
    "gpt2-xl": "decoder",
    "gpt-neo": "decoder",
    "gpt-j": "decoder",
    "T0pp": "encoder_decoder",
    "unifiedqa": "encoder_decoder",
    "T5": "encoder_decoder",
    "deberta-mnli": "encoder",
    "deberta": "encoder",
    "roberta-mnli": "encoder",
}

get_label_name_for_dataset = lambda dataset_name: "label" if dataset_name != "story-cloze" else "answer_right_ending"

use_train_or_test = lambda dataset_name: 'train' if dataset_name in ('copa', 'rte', 'boolq', 'qnli', 'piqa') else 'test'

PROMPT_DICT = {
    "imdb": [
        ["Consider the following example: ''' {} '''\nBetween {} and {}, the sentiment of this example is", [
            "text", "neg_label", "pos_label"]],
        ["Consider the following example: ''' {} '''\nBetween {} and {}, which is the sentiment of this example?", [
            "text", "neg_label", "pos_label"]],
        ["Consider the following example: ''' {} '''\nBetween {} and {}, is {} the sentiment of this example?", [
            "text", "neg_label", "pos_label", "pos_label"]],
    ],
    "amazon_polarity": [
        ["Consider the following example: ''' {} '''\nBetween {} and {}, the sentiment of this example is", [
            "content", "neg_label", "pos_label"]],
        ["Consider the following example: ''' {} '''\nBetween {} and {}, which is the sentiment of this example?", [
            "content", "neg_label", "pos_label"]],
        ["Consider the following example: ''' {} '''\nBetween {} and {}, is {} the sentiment of this example?", [
            "text", "neg_label", "pos_label", "pos_label"]],
    ],
    "ag_news": [
        ["Consider the following example: ''' {} '''\nChoice 1: {}. Choice 2: {}. Between choice 1 and choice 2, the topic of this example is ", [
            "text", "neg_label", "pos_label"]],
        ["Consider the following example: ''' {} '''\nChoice 1: {}. Choice 2: {}. Between choice 1 and choice 2, what is the topic of this example?", [
            "text", "neg_label", "pos_label"]],
        ["Consider the following example: ''' {} '''\nWhich is the topic of this example, choice 1: {}, or choice 2: {}?", [
            "text", "neg_label", "pos_label"]],
        ["{}\nChoice 1: {}. Choice 2: {}. Between choice 1 and choice 2, the topic of this example is ",
         ["text", "neg_label", "pos_label"]],
        ["{}\nChoice 1: {}. Choice 2: {}. Between choice 1 and choice 2, what is the topic of this example?",
         ["text", "neg_label", "pos_label"]],
        ["{}\nWhich is the topic of this example, choice 1: {}, or choice 2: {}?", ["text", "neg_label", "pos_label"]],
        ["{}\nWhat label best describes this news article, choice 1: {}, or choice 2: {}?", ["text", "neg_label", "pos_label"]],
        ["{}\nWhich section of a newspaper would this article likely appear in, choice 1: {}, or choice 2: {}?", [
            "text", "neg_label", "pos_label"]],
    ],
    "dbpedia_14": [
        ["Consider the following example: ''' {} '''\nChoice 1: {}. Choice 2: {}. Between choice 1 and choice 2, the topic of this example is ", [
            "content", "neg_label", "pos_label"]],
        ["Consider the following example: ''' {} '''\nChoice 1: {}. Choice 2: {}. Between choice 1 and choice 2, what is the topic of this example?", [
            "content", "neg_label", "pos_label"]],
        ["Consider the following example: ''' {} '''\nWhich is the topic of this example, choice 1: {}, or choice 2: {}?", [
            "content", "neg_label", "pos_label"]],
        ["{}\nChoice 1: {}. Choice 2: {}. Between choice 1 and choice 2, the topic of this example is ",
         ["content", "neg_label", "pos_label"]],
        ["{}\nChoice 1: {}. Choice 2: {}. Between choice 1 and choice 2, what is the topic of this example?",
         ["content", "neg_label", "pos_label"]],
        ["{}\nWhich is the topic of this example, choice 1: {}, or choice 2: {}?", ["content", "neg_label", "pos_label"]],
        ["{}\nWhat category does the paragraph belong to, choice 1: {}, or choice 2: {}?",
         ["content", "neg_label", "pos_label"]],
        ["{}\nWhat label best describes this paragraph, choice 1: {}, or choice 2: {}?", ["content", "neg_label", "pos_label"]],
    ],
    "story_cloze": [
        ["Consider the following story: ''' {} {} {} {} '''\nChoice 1: {}\nChoice 2: {}\nWhich is the more plausible ending of this story, choice 1 or choice 2?",
         ["input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4", "sentence_quiz1", "sentence_quiz2"]],
        ["Consider the following story: ''' {} {} {} {} '''\nChoice 1: {}\nChoice 2: {}\nWhich is the more plausible ending of this story?",
         ["input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4", "sentence_quiz1", "sentence_quiz2"]],
        ["{} {} {} {}\nChoice 1: {}\nChoice 2: {}\nWhich is the more plausible ending of this story, choice 1 or choice 2?",
         ["input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4", "sentence_quiz1", "sentence_quiz2"]],
    ],
    "copa": [
        ["Consider the following premise: ''' {} ''' Choice 1: {}\nChoice 2: {}\nQ: Which one is more likely to be the {}, choice 1 or choice 2?",
            ["premise", "choice1", "choice2", "question"]],
    ],
    "rte": [
        ["{}\nQuestion: Does this imply that \"{}\", yes or no?",
            ["premise", "hypothesis"]],
    ],
    "piqa": [
        ["Consider the following task: ''' {} '''\nSolution 1: {}. Solution 2: {}. Between solution 1 and solution 2, which one best solves this task?",
            ["goal", "sol1", "sol2"]],
    ]
}