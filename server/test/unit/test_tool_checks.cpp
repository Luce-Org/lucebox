// Host-side unit test for the tool-use check decisions (server/tool_checks.h):
// which requests get checks, what the model is sent, and when a re-plan or a
// final self-check runs. Pure OpenAI chat JSON, no model.

#include "server/tool_checks.h"
#include "host_check.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace luce::common::tool_checks;

static int g_checks = 0;

static json tools() {
    return json::array({
        {{"type", "function"}, {"function", {{"name", "get_order"}}}},
        {{"type", "function"}, {"function", {{"name", "send_email"}}}},
    });
}

static json call(const std::string & id, const std::string & name, const std::string & arguments) {
    return {{"id", id}, {"type", "function"},
            {"function", {{"name", name}, {"arguments", arguments}}}};
}

static json assistant_calls(json calls) {
    return {{"role", "assistant"}, {"content", ""}, {"tool_calls", std::move(calls)}};
}

static json tool_result(const std::string & id, const std::string & content) {
    return {{"role", "tool"}, {"tool_call_id", id}, {"content", content}};
}

static json user(const std::string & content) {
    return {{"role", "user"}, {"content", content}};
}

static json answer(const std::string & content) {
    return {{"role", "assistant"}, {"content", content}};
}

static void test_requested() {
    json body = {{"messages", json::array({user("hi")})}};
    CHECK(!requested(body, true));
    body["tools"] = tools();
    CHECK(requested(body, true));
    CHECK(!requested(body, false));
    body["tool_choice"] = "none";
    CHECK(requested(body, true));
    body["tool_checks"] = false;
    CHECK(!requested(body, true));
    body.erase("tool_checks");
    body.erase("tool_choice");
    body["extra_body"] = {{"tool_checks", false}};
    CHECK(!requested(body, true));
    body["tool_checks"] = true;
    CHECK(requested(body, false));
}

static void test_agent_request() {
    json body = {{"model", "m"}, {"stream", true}, {"stream_options", {{"include_usage", true}}},
                 {"tools", tools()},
                 {"messages", json::array({{{"role", "system"}, {"content", "Be brief."}}, user("hi")})}};
    const json request = agent_request(body, kEffort);
    CHECK(request["stream"] == false);
    CHECK(!request.contains("stream_options"));
    CHECK(request["reasoning_effort"] == "high");
    CHECK(request["messages"].size() == 2);
    const std::string system = request["messages"][0]["content"];
    CHECK(system.rfind("Be brief.\n\n", 0) == 0);
    CHECK(system.find(kRules) != std::string::npos);
    // The inner request takes the plain path.
    CHECK(!requested(request, true));
    // The client is untouched.
    CHECK(body["messages"][0]["content"] == "Be brief.");

    // No system message: the rules become one, in front.
    body["messages"] = json::array({user("hi")});
    const json inserted = agent_request(body, kEffort);
    CHECK(inserted["messages"].size() == 2);
    CHECK(inserted["messages"][0]["role"] == "system");
    CHECK(inserted["messages"][0]["content"] == kRules);
    CHECK(inserted["messages"][1]["role"] == "user");

    // A client that chose thinking keeps its choice.
    body["reasoning_effort"] = "none";
    CHECK(agent_request(body, kEffort)["reasoning_effort"] == "none");
    body.erase("reasoning_effort");
    body["chat_template_kwargs"] = {{"enable_thinking", false}};
    CHECK(!agent_request(body, kEffort).contains("reasoning_effort"));

    // tool_choice "none" skips the normal rules and reasoning changes.
    body.erase("chat_template_kwargs");
    body["tool_choice"] = "none";
    const json tools_off_request = agent_request(body, kEffort);
    CHECK(!tools_off_request.contains("reasoning_effort"));
    CHECK(tools_off_request["messages"] == body["messages"]);
    CHECK(!requested(tools_off_request, true));
}

static void test_tool_free_reply() {
    json body = {{"tools", tools()}, {"messages", json::array({user("hi")})}};
    const json calls = assistant_calls(json::array({call("call_1", "get_order", "{}")}));
    CHECK(!tools_off(body));
    CHECK(!needs_tool_free_reply(body, calls));
    const json forced = {{"type", "function"}, {"function", {{"name", "get_order"}}}};
    for (const json & choice : {json("auto"), json("required"), forced}) {
        body["tool_choice"] = choice;
        CHECK(!tools_off(body));
        CHECK(!needs_tool_free_reply(body, calls));
    }
    body["tool_choice"] = "none";
    CHECK(tools_off(body));
    CHECK(needs_tool_free_reply(body, calls));
    CHECK(!needs_tool_free_reply(body, answer("Done.")));
    body["tool_choice"] = {{"type", "none"}};
    CHECK(tools_off(body));
    CHECK(needs_tool_free_reply(body, calls));

    // A reply that still calls tools goes out as its text alone.
    json response = {{"choices", json::array({{
        {"index", 0}, {"finish_reason", "tool_calls"}, {"message", calls}}})}};
    (*reply_message(response))["content"] = "Listing.";
    drop_calls(response);
    CHECK(!needs_tool_free_reply(body, *reply_message(response)));
    CHECK((*reply_message(response))["content"] == "Listing.");
    CHECK(response["choices"][0]["finish_reason"] == "stop");
    response["choices"][0]["message"] = calls;
    (*reply_message(response))["content"] = nullptr;
    drop_calls(response);
    CHECK((*reply_message(response))["content"] == "");
}

static void test_replan() {
    CHECK(!needs_replan(answer("done")));
    CHECK(!needs_replan(assistant_calls(json::array({
        call("call_1", "get_order", "{\"id\":1}"), call("call_2", "get_order", "{\"id\":2}")}))));
    CHECK(needs_replan(assistant_calls(json::array({
        call("call_1", "get_order", "{\"id\":1}"), call("call_2", "send_email", "{}")}))));
}

static void test_final_check() {
    CheckedCalls checked;
    json messages = json::array({user("Cancel order 1 and email Tom.")});
    CHECK(!needs_final_check(messages, answer("Done."), checked));  // no tool results yet

    messages.push_back(assistant_calls(json::array({call("call_1", "get_order", "{\"id\":1}")})));
    messages.push_back(tool_result("call_1", "{\"status\":\"open\"}"));
    CHECK(needs_final_check(messages, answer("Done."), checked));
    CHECK(!needs_final_check(messages, answer("Which email address should I use?"), checked));
    CHECK(!needs_final_check(messages, assistant_calls(json::array({call("call_2", "send_email", "{}")})), checked));

    // A check added work for this user request: the next answer goes out.
    const json added = json::array({call("call_2", "send_email", "{}")});
    checked.add(added);
    json after = messages;
    after.push_back(assistant_calls(added));
    after.push_back(tool_result("call_2", "sent"));
    CHECK(!needs_final_check(after, answer("Done."), checked));
    // A new user request gets a check again.
    after.push_back(answer("Done."));
    after.push_back(user("Now order 2."));
    after.push_back(assistant_calls(json::array({call("call_3", "get_order", "{\"id\":2}")})));
    after.push_back(tool_result("call_3", "{\"status\":\"open\"}"));
    CHECK(needs_final_check(after, answer("Done."), checked));

    // The set is bounded: the oldest ids leave first.
    CheckedCalls small(2);
    small.add(json::array({call("a", "x", "{}"), call("b", "x", "{}"), call("c", "x", "{}")}));
    CHECK(!small.since_last_user(json::array({assistant_calls(json::array({call("a", "x", "{}")}))})));
    CHECK(small.since_last_user(json::array({assistant_calls(json::array({call("c", "x", "{}")}))})));
}

static void test_fresh_calls() {
    json messages = json::array({
        user("Look up order 1."),
        assistant_calls(json::array({call("call_1", "get_order", "{\"id\":1,\"full\":true}")})),
        tool_result("call_1", "Error: service unavailable"),
    });
    // The same failed call, with its arguments in another order, is not new work.
    const json again = assistant_calls(json::array({
        call("call_2", "get_order", "{\"full\":true,\"id\":1}"),
        call("call_3", "send_email", "{\"to\":\"tom\"}"),
    }));
    const json fresh = fresh_calls(messages, again);
    CHECK(fresh.size() == 1);
    CHECK(fresh[0]["function"]["name"] == "send_email");

    // A result fails by its shape; one that only mentions errors does not.
    auto failed = [](const char * result) {
        const json history = json::array({
            user("Look up order 1."),
            assistant_calls(json::array({call("call_1", "get_order", "{\"id\":1}")})),
            tool_result("call_1", result),
        });
        const json retry = assistant_calls(json::array({call("call_2", "get_order", "{\"id\":1}")}));
        return fresh_calls(history, retry).empty();
    };
    CHECK(failed("{\"error\": \"not found\"}"));
    CHECK(failed("{\"errors\": [{\"message\": \"bad id\"}]}"));
    CHECK(failed("{\"success\": false, \"message\": \"timeout\"}"));
    CHECK(failed("{\"status\": \"FAILED\"}"));
    CHECK(failed("{\"isError\": true, \"content\": []}"));
    CHECK(failed("  error 503"));
    CHECK(!failed("{\"error\": null, \"value\": 5}"));
    CHECK(!failed("{\"errors\": 0, \"rows\": 3}"));
    CHECK(!failed("{\"success\": true, \"note\": \"no error occurred\"}"));
    CHECK(!failed("Order 1 shipped, no error occurred."));
    CHECK(!failed("errors: none"));
}

static void test_note() {
    const json request = {{"messages", json::array({user("hi")})}};
    const json reply = assistant_calls(json::array({call("call_1", "get_order", "{}")}));
    const json next = with_note(request, reply, kReplanNote);
    CHECK(request["messages"].size() == 1);
    CHECK(next["messages"].size() == 3);
    CHECK(next["messages"][1]["tool_calls"].size() == 1);
    CHECK(next["messages"][2]["role"] == "user");
    CHECK(next["messages"][2]["content"] == kReplanNote);
}

static void test_sse() {
    json response = {
        {"id", "chatcmpl-1"}, {"object", "chat.completion"}, {"created", 7}, {"model", "m"},
        {"choices", json::array({{
            {"index", 0}, {"finish_reason", "tool_calls"},
            {"message", {{"role", "assistant"}, {"content", "Checking."},
                         {"reasoning_content", "think"},
                         {"tool_calls", json::array({call("call_1", "get_order", "{}")})}}},
        }})},
        {"usage", {{"prompt_tokens", 3}, {"completion_tokens", 2}, {"total_tokens", 5}}},
    };
    CHECK(reply_message(response) != nullptr);
    const std::string sse = as_sse(response);
    std::vector<json> events;
    size_t at = 0;
    while ((at = sse.find("data: ", at)) != std::string::npos) {
        const size_t end = sse.find("\n\n", at);
        const std::string data = sse.substr(at + 6, end - at - 6);
        if (data != "[DONE]") events.push_back(json::parse(data));
        at = end;
    }
    CHECK(sse.size() > 14 && sse.compare(sse.size() - 14, 14, "data: [DONE]\n\n") == 0);
    CHECK(events.size() == 6);
    CHECK(events[0]["choices"][0]["delta"]["role"] == "assistant");
    CHECK(events[1]["choices"][0]["delta"]["reasoning_content"] == "think");
    CHECK(events[2]["choices"][0]["delta"]["content"] == "Checking.");
    CHECK(events[3]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0);
    CHECK(events[3]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1");
    CHECK(events[4]["choices"][0]["finish_reason"] == "tool_calls");
    CHECK(events[5]["usage"]["total_tokens"] == 5);
    for (const auto & event : events) {
        CHECK(event["id"] == "chatcmpl-1");
        CHECK(event["object"] == "chat.completion.chunk");
    }
}

int main() {
    test_requested();
    test_agent_request();
    test_replan();
    test_tool_free_reply();
    test_final_check();
    test_fresh_calls();
    test_note();
    test_sse();
    std::printf("test_tool_checks: %d checks passed\n", g_checks);
    return 0;
}
