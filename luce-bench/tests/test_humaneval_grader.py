"""Tests for the humaneval area's parse-pass grader.

The 2026-05-30 gemma full bench surfaced the trailing-garbage failure
mode: gemma produced valid Python function bodies but appended
chat-template artifacts (``return Falsestring``, leaked stop tokens,
hallucinated tails). ``ast.parse`` on the prompt + full completion
choked on the garbage and the grader marked the row failed even
though the actual code was correct. Pass rate on the snapshot lifts
1/10 → 8/10 once the grader tries progressive trim-from-end before
giving up.
"""

from __future__ import annotations

from lucebench.areas.humaneval import grade_completion


def test_clean_completion_passes():
    """Standard code-completion: ``prompt + completion`` parses straight."""
    prompt = "def add(a, b):\n    "
    completion = "return a + b\n"
    g = grade_completion(prompt, completion)
    assert g["graded_pass"] is True
    assert g["strict_pass"] is True


def test_completion_with_trailing_garbage_passes_after_trim():
    """Regression for the 2026-05-30 gemma run: valid function body
    followed by junk (``return Falsestring\n`` — a leaked stop token /
    chat-template artifact). The grader must trim from the end and
    pass on the prefix that parses."""
    prompt = (
        "def has_close_elements(numbers, threshold):\n"
        "    "
    )
    completion = (
        "for i in range(len(numbers)):\n"
        "        for j in range(i + 1, len(numbers)):\n"
        "            if abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False\n"
        "  return Falsestring\n"  # trailing garbage with bad indent
    )
    g = grade_completion(prompt, completion)
    assert g["graded_pass"] is True
    assert g["semantic_hint"] is True


def test_completely_broken_code_still_fails():
    """If no prefix of the completion parses, we still report fail."""
    prompt = "def f(x):\n    "
    completion = "}{][!!!nonsense$$$ Lorem ipsum dolor sit amet\n"
    g = grade_completion(prompt, completion)
    assert g["graded_pass"] is False


def test_empty_completion_fails():
    """Below the 8-char threshold → stub → fail."""
    g = grade_completion("def f():\n    ", "")
    assert g["graded_pass"] is False
    g = grade_completion("def f():\n    ", "  \n")
    assert g["graded_pass"] is False


def test_trailing_chat_template_tag_passes():
    """Real gemma artifact: function body + a leaked ``thought\\n`` tag."""
    prompt = "def square(n):\n    "
    completion = "return n * n\nthought\n探寻中...完成任务。thought\n"
    g = grade_completion(prompt, completion)
    assert g["graded_pass"] is True
