"""IFEval subset checkers (loose mode), graded through the ifeval grader."""

import pytest
from eval.graders import grade_deterministic


def ifeval(ids, kwargs, prompt=""):
    return {"type": "ifeval", "instruction_id_list": ids, "kwargs": kwargs, "prompt": prompt}


@pytest.mark.parametrize("text,grader,expected", [
    ("no commas here at all", ifeval(["punctuation:no_comma"], [{}]), True),
    ("one, two", ifeval(["punctuation:no_comma"], [{}]), False),
    ("all lower case", ifeval(["change_case:english_lowercase"], [{}]), True),
    ("Not lower", ifeval(["change_case:english_lowercase"], [{}]), False),
    ("SHOUTING ONLY", ifeval(["change_case:english_capital"], [{}]), True),
    ("one two three four five",
     ifeval(["length_constraints:number_words"], [{"num_words": 5, "relation": "at least"}]), True),
    ("one two three",
     ifeval(["length_constraints:number_words"], [{"num_words": 5, "relation": "at least"}]), False),
    ("<<My Title>>\nbody", ifeval(["detectable_format:title"], [{}]), True),
    ("# My Title\nbody", ifeval(["detectable_format:title"], [{}]), False),
    ("* a\n* b\n* c", ifeval(["detectable_format:number_bullet_lists"], [{"num_bullets": 3}]), True),
    ("* a\n* b", ifeval(["detectable_format:number_bullet_lists"], [{"num_bullets": 3}]), False),
    ('```json\n{"a": 1}\n```', ifeval(["detectable_format:json_format"], [{}]), True),
    ("first\n***\nsecond", ifeval(["length_constraints:number_paragraphs"], [{"num_paragraphs": 2}]), True),
    ("first *** second *** third",
     ifeval(["length_constraints:number_paragraphs"], [{"num_paragraphs": 2}]), False),
    ("It works. My answer is yes.", ifeval(["detectable_format:constrained_response"], [{}]), True),
    ("Hello there\n\nP.S. bye", ifeval(["detectable_content:postscript"], [{"postscript_marker": "P.S."}]), True),
    ("Hello there", ifeval(["detectable_content:postscript"], [{"postscript_marker": "P.S."}]), False),
    ("Talk about Rome and Paris.", ifeval(["keywords:existence"], [{"keywords": ["rome", "paris"]}]), True),
    ("I love cats.", ifeval(["keywords:forbidden_words"], [{"forbidden_words": ["cat"]}]), True),
    ("I love cat food.", ifeval(["keywords:forbidden_words"], [{"forbidden_words": ["cat"]}]), False),
    ("Is there anything else I can help with?",
     ifeval(["startend:end_checker"], [{"end_phrase": "Is there anything else I can help with?"}]), True),
    ('"quoted whole"', ifeval(["startend:quotation"], [{}]), True),
    ("Resp A\n******\nResp B", ifeval(["combination:two_responses"], [{}]), True),
    ("Resp A\n******\nResp A", ifeval(["combination:two_responses"], [{}]), False),
    ("Say hi. Hi!", ifeval(["combination:repeat_prompt"], [{"prompt_to_repeat": "Say hi."}]), True),
    ("Section 1 x\nSection 2 y",
     ifeval(["detectable_format:multiple_sections"], [{"section_spliter": "Section", "num_sections": 2}]), True),
    ("Use [name] and [address].",
     ifeval(["detectable_content:number_placeholders"], [{"num_placeholders": 2}]), True),
    # loose mode: a chatty first line does not break "wrap in quotes"
    ('Sure, here you go:\n"quoted whole"', ifeval(["startend:quotation"], [{}]), True),
    # every instruction must hold
    ("all lower, with comma",
     ifeval(["change_case:english_lowercase", "punctuation:no_comma"], [{}, {}]), False),
])
def test_ifeval_checks(text, grader, expected):
    assert grade_deterministic(text, grader).correct is expected
