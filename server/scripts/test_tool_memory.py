from tool_memory import ToolMemory


def test_tool_memory_remembers_shared_raw_text_for_multiple_ids():
    mem = ToolMemory(max_entries=8, max_bytes=4096)
    raw = '<tool_call><function=read_file><parameter=path>a.py</parameter></function></tool_call>'
    mem.remember(["call_a", "call_b"], raw)

    assert mem.lookup_message([{"id": "call_a"}, {"id": "call_b"}]) == raw
    assert len(mem.by_id) == 2
    assert len(mem.by_block) == 1


def test_tool_memory_eviction_drops_oldest_entry_and_unique_block():
    mem = ToolMemory(max_entries=1, max_bytes=4096)
    mem.remember(["call_old"], "<tool_call>old</tool_call>")
    mem.remember(["call_new"], "<tool_call>new</tool_call>")

    assert mem.lookup_message([{"id": "call_old"}]) is None
    assert mem.lookup_message([{"id": "call_new"}]) == "<tool_call>new</tool_call>"
    assert len(mem.by_block) == 1


def test_tool_memory_lookup_message_requires_same_raw_text():
    mem = ToolMemory(max_entries=8, max_bytes=4096)
    mem.remember(["call_a"], "<tool_call>a</tool_call>")
    mem.remember(["call_b"], "<tool_call>b</tool_call>")

    assert mem.lookup_message([{"id": "call_a"}, {"id": "call_b"}]) is None
