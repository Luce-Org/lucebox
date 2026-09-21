import unittest

from evaluation_helpers import same_json_type_and_value, score_message


class EvaluationHelpersTest(unittest.TestCase):
    def test_recursive_types_are_exact(self):
        self.assertTrue(same_json_type_and_value({"x": [1, True]}, {"x": [1, True]}))
        self.assertFalse(same_json_type_and_value({"x": [1.0, True]}, {"x": [1, True]}))
        self.assertFalse(same_json_type_and_value({"x": [1, 1]}, {"x": [1, True]}))

    def test_json_contract_is_raw_and_requires_stop(self):
        expected = {"parts": 527}
        self.assertTrue(score_message("arithmetic", {"content": '{"parts":527}'}, "stop", expected))
        self.assertFalse(score_message("arithmetic", {"content": '```json\n{"parts":527}\n```'}, "stop", expected))
        self.assertFalse(score_message("arithmetic", {"content": '<think>x</think>{"parts":527}'}, "stop", expected))
        self.assertFalse(score_message("arithmetic", {"content": '{"parts":527}'}, "length", expected))
        self.assertFalse(score_message("arithmetic", {"content": '{"parts":527.0}'}, "stop", expected))

    def test_tool_contract_requires_one_call_and_tool_finish(self):
        call = {"function": {"name": "lookup_weather", "arguments": '{"city":"Toronto","unit":"celsius"}'}}
        expected = {"city": "Toronto", "unit": "celsius"}
        self.assertTrue(score_message("tool_call", {"tool_calls": [call]}, "tool_calls", expected))
        self.assertFalse(score_message("tool_call", {"tool_calls": [call, call]}, "tool_calls", expected))
        self.assertFalse(score_message("tool_call", {"tool_calls": [call]}, "stop", expected))


if __name__ == "__main__":
    unittest.main()
