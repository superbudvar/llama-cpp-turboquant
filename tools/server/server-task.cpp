#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//

// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (timings.prompt_n >= 0) {
        server_sent_events.back().at("data").push_back({"timings", timings.to_json()});
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (timings.prompt_n >= 0) {
            data.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            data.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "n_draft_tokens_total",            n_draft_tokens_total },
        { "n_draft_accepted_total",          n_draft_accepted_total },
        { "n_draft_verif_steps_total",       n_draft_verif_steps_total },
        { "n_accepted_per_pos_total",        n_accepted_per_pos_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // a state that does not fit the RAM budget still goes into the disk pool when
    // one is configured (write-through), otherwise it is skipped; the KV copy is
    // transient and released as soon as it is committed to disk
    if (limit_size > 0 && state_size_new > limit_size) {
        if (!disk_dir_.empty()) {
            SRV_INF(" - prompt state size %.3f MiB exceeds RAM cache size limit %.3f MiB, writing through to disk\n",
                    state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));

            disk_state_.reset();

            try {
                disk_state_ = std::make_unique<server_prompt_cache_state>();
                disk_state_->data.main.resize(state_size_tgt);
                disk_state_->data.drft.resize(state_size_dft);
                // rehydrate spilled checkpoints so the write-through copy keeps full payloads
                server_prompt copy = prompt.clone();
                server_prompt_checkpoints_rehydrate(copy);
                server_prompt_checkpoints_wipe(copy);
                disk_state_->prompt = std::move(copy);
            } catch (const std::bad_alloc & e) {
                SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

                disk_state_.reset();

                return nullptr;
            }

            return disk_state_.get();
        }

        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    // cache tiers never carry spill records: rehydrate then wipe the copy so a
    // cached checkpoint either holds full payloads (RAM) or is written fully (.dpc)
    server_prompt cache_prompt = prompt.clone();
    server_prompt_checkpoints_rehydrate(cache_prompt);
    server_prompt_checkpoints_wipe(cache_prompt);

    states.push_back({
        /*.prompt =*/ std::move(cache_prompt),
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        if (!restore_state(prompt, *it_best, ctx_tgt, ctx_dft, id_slot)) {
            return false;
        }

        states.erase(it_best);
        return true;
    }

    // fall back to the decoupled disk tier
    return load_from_disk_tier(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot);
}

bool server_prompt_cache::restore_state(server_prompt & prompt, server_prompt_cache_state & state, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    {
        auto & data = state.data.main;

        const size_t size = data.size();
        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
        if (n != size) {
            SRV_ERR("failed to restore state with size %zu\n", size);

            return false;
        }

        data.clear();
        data.shrink_to_fit();
    }

    {
        auto & data = state.data.drft;

        if (!data.empty()) {
            GGML_ASSERT(ctx_dft);

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_WRN("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }
    }

    prompt = std::move(state.prompt);

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // one line per update at info level: the cache is the largest host allocation the server makes,
    // and its size has to be visible in ordinary logs
    SRV_INF("prompt cache: %zu prompts, %.1f MiB (limit %.1f MiB, %zu tokens)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// disk prompt cache
//
// each cache entry is stored as <sha256-of-text-tokens>.dpc in the cache directory.
// only text-only prompts (no media chunks) are persisted - media placeholder tokens
// cannot be reconstructed from token ids alone. entry numbers are streamed with
// FILE* writes so large KV blobs are not copied into an extra buffer.
//


constexpr uint32_t DPC_MAGIC   = 0x31435044; // "DPC1"
constexpr uint32_t DPC_VERSION = 3;
constexpr uint32_t DPC_MAX_TOKENS = 1u << 24;
constexpr uint32_t DPC_MAX_CHECKPOINTS = 1024;

// compact SHA-256 (FIPS 180-4), single-shot. mirrors the core in common/fit-cache.cpp
uint32_t dpc_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

namespace {

uint32_t dpc_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

void dpc_block(uint32_t h[8], const uint8_t * p) {
    uint32_t w[64];
    for (size_t i = 0; i < 16; i++) {
        w[i] = ((uint32_t) p[i * 4 + 0] << 24) |
               ((uint32_t) p[i * 4 + 1] << 16) |
               ((uint32_t) p[i * 4 + 2] <<  8) |
               ((uint32_t) p[i * 4 + 3] <<  0);
    }
    for (size_t i = 16; i < 64; i++) {
        const uint32_t s0 = dpc_rotr(w[i - 15], 7) ^ dpc_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = dpc_rotr(w[i - 2], 17)  ^ dpc_rotr(w[i - 2], 19)   ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

    for (size_t i = 0; i < 64; i++) {
        const uint32_t S1 = dpc_rotr(e, 6) ^ dpc_rotr(e, 11) ^ dpc_rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = hh + S1 + ch + dpc_sha256_k[i] + w[i];
        const uint32_t S0 = dpc_rotr(a, 2) ^ dpc_rotr(a, 13) ^ dpc_rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

std::string dpc_sha256_hex(const uint8_t * data, size_t len) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    const uint64_t bitlen = (uint64_t) len * 8;
    std::vector<uint8_t> buf(data, data + len);
    buf.push_back(0x80);
    while (buf.size() % 64 != 56) {
        buf.push_back(0x00);
    }
    for (int i = 7; i >= 0; i--) {
        buf.push_back((uint8_t) (bitlen >> (i * 8)));
    }

    for (size_t off = 0; off < buf.size(); off += 64) {
        dpc_block(h, buf.data() + off);
    }

    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (size_t i = 0; i < 8; i++) {
        const uint32_t v = h[i];
        const uint8_t b[4] = {
            (uint8_t) (v >> 24), (uint8_t) (v >> 16), (uint8_t) (v >> 8), (uint8_t) (v >> 0),
        };
        for (int j = 0; j < 4; j++) {
            out.push_back(hex[b[j] >> 4]);
            out.push_back(hex[b[j] & 0x0f]);
        }
    }
    return out;
}

} // namespace

// spill segment layout (all little-endian):
//   u32 magic, i64 n_tokens, i32 pos_min, i32 pos_max,
//   u64 len_tgt, u64 len_dft, u64 len_spec, tgt[len_tgt], dft[len_dft], spec[len_spec]
constexpr uint32_t CSPILL_MAGIC   = 0x4B505343; // "CSPK"
constexpr uint32_t CSPILL_HDR_SIZE = 4 + 8 + 4 + 4 + 8 + 8 + 8;

namespace {

bool cspill_check_header(const int64_t n_tokens, const llama_pos pos_min, const llama_pos pos_max, const uint64_t len_tgt, const uint64_t len_dft, const uint64_t len_spec) {
    if (n_tokens < 0 || pos_min < 0 || pos_max < 0) {
        return false;
    }
    if (n_tokens > (int64_t) (1ull << 40) || len_tgt > (1ull << 34) || len_dft > (1ull << 34) || len_spec > (1ull << 34)) {
        return false;
    }
    return len_tgt + len_dft + len_spec < (1ull << 36);
}

} // namespace

// these live at file scope (declared in server-task.h) so both the live
// slot and the async/RAM cache tiers can spill and rehydrate

int64_t server_ckpt_spill_store(common_prompt_checkpoint & ckpt, const std::string & path, bool truncate) {
    if (path.empty()) {
        return -1;
    }

    const size_t len_tgt  = ckpt.data_tgt.size();
    const size_t len_dft  = ckpt.data_dft.size();
    const size_t len_spec = ckpt.data_spec.size();

    FILE * f = fopen(path.c_str(), truncate ? "wb" : "ab");
    if (f == nullptr) {
        return -1;
    }

    const int64_t off = (int64_t) ftell(f);

    bool ok = fwrite(&CSPILL_MAGIC,   4,            1, f) == 1;
    ok = ok && fwrite(&ckpt.n_tokens, sizeof(int64_t), 1, f) == 1;
    ok = ok && fwrite(&ckpt.pos_min,  sizeof(llama_pos), 1, f) == 1;
    ok = ok && fwrite(&ckpt.pos_max,  sizeof(llama_pos), 1, f) == 1;

    const uint64_t len_tgt_u  = (uint64_t) len_tgt;
    const uint64_t len_dft_u  = (uint64_t) len_dft;
    const uint64_t len_spec_u = (uint64_t) len_spec;

    ok = ok && fwrite(&len_tgt_u,  sizeof(uint64_t), 1, f) == 1;
    ok = ok && fwrite(&len_dft_u,  sizeof(uint64_t), 1, f) == 1;
    ok = ok && fwrite(&len_spec_u, sizeof(uint64_t), 1, f) == 1;

    ok = ok && (len_tgt  == 0 || fwrite(ckpt.data_tgt.data(),  1, len_tgt,  f) == len_tgt);
    ok = ok && (len_dft  == 0 || fwrite(ckpt.data_dft.data(),  1, len_dft,  f) == len_dft);
    ok = ok && (len_spec == 0 || fwrite(ckpt.data_spec.data(), 1, len_spec, f) == len_spec);

    const size_t seg = CSPILL_HDR_SIZE + len_tgt + len_dft + len_spec;

    if (fclose(f) != 0) {
        ok = false;
    }

    if (!ok) {
        return -1;
    }

    ckpt.spill_off = off;
    ckpt.spill_len = seg;

    ckpt.data_tgt.clear();
    ckpt.data_dft.clear();
    ckpt.data_spec.clear();

    return off;
}

bool server_ckpt_spill_load(common_prompt_checkpoint & ckpt, const std::string & path) {
    if (!ckpt.spilled()) {
        return true;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }

    bool ok = fseek(f, ckpt.spill_off, SEEK_SET) == 0;

    uint32_t magic = 0;
    ok = ok && fread(&magic, 4, 1, f) == 1;
    ok = ok && magic == CSPILL_MAGIC;

    std::vector<uint8_t> tgt, dft, spec;

    int64_t  n_tokens = 0;
    llama_pos pos_min = 0, pos_max = 0;
    uint64_t len_tgt = 0, len_dft = 0, len_spec = 0;

    ok = ok && fread(&n_tokens, sizeof(int64_t), 1, f) == 1;
    ok = ok && fread(&pos_min,   sizeof(llama_pos), 1, f) == 1;
    ok = ok && fread(&pos_max,   sizeof(llama_pos), 1, f) == 1;
    ok = ok && fread(&len_tgt,   sizeof(uint64_t), 1, f) == 1;
    ok = ok && fread(&len_dft,   sizeof(uint64_t), 1, f) == 1;
    ok = ok && fread(&len_spec,  sizeof(uint64_t), 1, f) == 1;

    ok = ok && cspill_check_header(n_tokens, pos_min, pos_max, len_tgt, len_dft, len_spec);

    // only accept data that matches the in-RAM metadata; anything else is a corrupt file
    ok = ok && n_tokens == ckpt.n_tokens && pos_min == ckpt.pos_min && pos_max == ckpt.pos_max;

    if (ok) {
        tgt.resize((size_t) len_tgt);
        dft.resize((size_t) len_dft);
        spec.resize((size_t) len_spec);
        ok = ok && (len_tgt  == 0 || fread(tgt.data(),  1, (size_t) len_tgt,  f) == len_tgt);
        ok = ok && (len_dft  == 0 || fread(dft.data(),  1, (size_t) len_dft,  f) == len_dft);
        ok = ok && (len_spec == 0 || fread(spec.data(), 1, (size_t) len_spec, f) == len_spec);
    }

    fclose(f);

    if (!ok) {
        return false;
    }

    ckpt.data_tgt = std::move(tgt);
    ckpt.data_dft = std::move(dft);
    ckpt.data_spec = std::move(spec);

    ckpt.spill_off = -1;
    ckpt.spill_len = 0;

    return true;
}

void server_ckpt_spill_compact(server_prompt & prompt, const std::string & path) {
    if (path.empty()) {
        return;
    }

    std::error_code ec;
    const uint64_t file_size = (uint64_t) std::filesystem::file_size(path, ec);
    if (ec) {
        return;
    }

    if (prompt.checkpoints.empty()) {
        return;
    }

    // live payload: full checkpoints count their blobs, spilled ones their segment
    size_t live = 0;
    for (const auto & cur : prompt.checkpoints) {
        live += cur.size();
    }

    if ((uint64_t) live * 2 >= file_size) {
        return;
    }

    // rewrite with only the still-referenced segments
    const std::string tmp = path + ".tmp";

    FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        return;
    }

    int64_t off = 0;
    for (const auto & cur : prompt.checkpoints) {
        if (!cur.spilled()) {
            continue;
        }

        const size_t len_tgt  = cur.data_tgt.size();
        const size_t len_dft  = cur.data_dft.size();
        const size_t len_spec = cur.data_spec.size();

        bool ok = fwrite(&CSPILL_MAGIC,   4,              1, f) == 1;
        ok = ok && fwrite(&cur.n_tokens, sizeof(int64_t), 1, f) == 1;
        ok = ok && fwrite(&cur.pos_min,  sizeof(llama_pos), 1, f) == 1;
        ok = ok && fwrite(&cur.pos_max,  sizeof(llama_pos), 1, f) == 1;

        const uint64_t len_tgt_u  = (uint64_t) len_tgt;
        const uint64_t len_dft_u  = (uint64_t) len_dft;
        const uint64_t len_spec_u = (uint64_t) len_spec;

        ok = ok && fwrite(&len_tgt_u,  sizeof(uint64_t), 1, f) == 1;
        ok = ok && fwrite(&len_dft_u,  sizeof(uint64_t), 1, f) == 1;
        ok = ok && fwrite(&len_spec_u, sizeof(uint64_t), 1, f) == 1;

        ok = ok && (len_tgt  == 0 || fwrite(cur.data_tgt.data(),  1, len_tgt,  f) == len_tgt);
        ok = ok && (len_dft  == 0 || fwrite(cur.data_dft.data(),  1, len_dft,  f) == len_dft);
        ok = ok && (len_spec == 0 || fwrite(cur.data_spec.data(), 1, len_spec, f) == len_spec);

        if (!ok) {
            fclose(f);
            std::filesystem::remove(tmp, ec);
            return;
        }

        off += CSPILL_HDR_SIZE + len_tgt + len_dft + len_spec;
    }

    const bool ok = fclose(f) == 0;

    if (!ok) {
        std::filesystem::remove(tmp, ec);
        return;
    }

    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return;
    }

    // keep every recorded offset valid: rewrite moves the live segments to the
    // start of the file; the stride matches the write loop above
    int64_t off_new = 0;
    for (auto & cur : prompt.checkpoints) {
        if (!cur.spilled()) {
            continue;
        }
        cur.spill_off = off_new;
        off_new += (int64_t) CSPILL_HDR_SIZE + (int64_t) cur.data_tgt.size() + (int64_t) cur.data_dft.size() + (int64_t) cur.data_spec.size();
    }
}

bool server_prompt_checkpoints_rehydrate(server_prompt & prompt) {
    bool ok = true;

    for (auto & cur : prompt.checkpoints) {
        if (cur.spilled()) {
            ok = server_ckpt_spill_load(cur, prompt.checkpoint_spill_path) && ok;
        }
    }

    return ok;
}

void server_prompt_checkpoints_wipe(server_prompt & prompt) {
    for (auto & cur : prompt.checkpoints) {
        cur.spill_off = -1;
        cur.spill_len = 0;
    }
    prompt.checkpoint_spill_path.clear();
}

// entry layout (all little-endian):
//   u32 magic, u32 version, u32 n_tokens,
//   u64 len_main, u64 len_drft, u32 n_checkpoints,
//   tokens[n_tokens], main[len_main], drft[len_drft],
//   per checkpoint: i64 n_tokens, i32 id_task, i32 pos_min, i32 pos_max,
//                   u64 n_tgt, u64 n_dft, u64 n_spec, tgt[n_tgt], dft[n_dft], spec[n_spec]
namespace {
bool dpc_write_state(FILE * f, const server_prompt_cache_state & state) {
    const llama_tokens tokens = state.prompt.tokens.get_text_tokens();

    bool ok = true;

    const uint32_t n_tokens = (uint32_t) tokens.size();
    const uint64_t len_main = state.data.main.size();
    const uint64_t len_drft = state.data.drft.size();
    const uint32_t n_ckpt   = (uint32_t) state.prompt.checkpoints.size();

    ok = ok && fwrite(&n_tokens,     4, 1, f) == 1;
    ok = ok && fwrite(&len_main,     8, 1, f) == 1;
    ok = ok && fwrite(&len_drft,     8, 1, f) == 1;
    ok = ok && fwrite(&n_ckpt,       4, 1, f) == 1;
    ok = ok && (n_tokens == 0 || fwrite(tokens.data(), sizeof(llama_token), n_tokens, f) == n_tokens);
    ok = ok && (len_main == 0 || fwrite(state.data.main.data(), 1, len_main, f) == len_main);
    ok = ok && (len_drft == 0 || fwrite(state.data.drft.data(), 1, len_drft, f) == len_drft);

    for (const auto & ckpt : state.prompt.checkpoints) {
        const int64_t  n_tokens = ckpt.n_tokens;
        const int32_t  id_task  = ckpt.id_task;
        const int32_t  pos_min  = ckpt.pos_min;
        const int32_t  pos_max  = ckpt.pos_max;
        const uint64_t n_tgt    = ckpt.data_tgt.size();
        const uint64_t n_dft    = ckpt.data_dft.size();
        const uint64_t n_spec   = ckpt.data_spec.size();

        ok = ok && fwrite(&n_tokens, 8, 1, f) == 1;
        ok = ok && fwrite(&id_task,  4, 1, f) == 1;
        ok = ok && fwrite(&pos_min,  4, 1, f) == 1;
        ok = ok && fwrite(&pos_max,  4, 1, f) == 1;
        ok = ok && fwrite(&n_tgt,    8, 1, f) == 1;
        ok = ok && fwrite(&n_dft,    8, 1, f) == 1;
        ok = ok && fwrite(&n_spec,   8, 1, f) == 1;
        ok = ok && (n_tgt == 0  || fwrite(ckpt.data_tgt.data(),  1, n_tgt,  f) == n_tgt);
        ok = ok && (n_dft == 0  || fwrite(ckpt.data_dft.data(),  1, n_dft,  f) == n_dft);
        ok = ok && (n_spec == 0 || fwrite(ckpt.data_spec.data(), 1, n_spec, f) == n_spec);
    }

    return ok;
}

bool dpc_read_state(FILE * f, uint32_t version, server_prompt_cache_state & state) {
    auto rd = [&](void * dst, size_t n) { return fread(dst, 1, n, f) == n; };

    uint32_t n_tokens = 0;
    uint64_t len_main = 0;
    uint64_t len_drft = 0;
    uint32_t n_ckpt   = 0;

    bool ok = rd(&n_tokens, 4) && rd(&len_main, 8) && rd(&len_drft, 8) && rd(&n_ckpt, 4) &&
              n_tokens <= DPC_MAX_TOKENS && n_ckpt <= DPC_MAX_CHECKPOINTS &&
              len_main <= (1ull << 34) && len_drft <= (1ull << 34);

    llama_tokens tokens(n_tokens);
    ok = ok && (n_tokens == 0 || fread(tokens.data(), sizeof(llama_token), n_tokens, f) == n_tokens);

    state.data.main.resize(len_main);
    state.data.drft.resize(len_drft);
    ok = ok && (len_main == 0 || fread(state.data.main.data(), 1, len_main, f) == len_main);
    ok = ok && (len_drft == 0 || fread(state.data.drft.data(), 1, len_drft, f) == len_drft);

    for (uint32_t i = 0; ok && i < n_ckpt; ++i) {
        auto & ckpt = state.prompt.checkpoints.emplace_back();

        uint64_t n_tgt  = 0;
        uint64_t n_dft  = 0;
        uint64_t n_spec = 0;

        ok = ok && rd(&ckpt.n_tokens, 8) && rd(&ckpt.id_task, 4) &&
              rd(&ckpt.pos_min, 4) && rd(&ckpt.pos_max, 4) &&
              rd(&n_tgt, 8) && rd(&n_dft, 8) && rd(&n_spec, 8);

        ok = ok && n_tgt <= (1ull << 34) && n_dft <= (1ull << 34) && n_spec <= (1ull << 34);

        if (ok) {
            ckpt.data_tgt.resize(n_tgt);
            ckpt.data_dft.resize(n_dft);
            ckpt.data_spec.resize(n_spec);
            ok = ok && (n_tgt == 0  || fread(ckpt.data_tgt.data(),  1, n_tgt,  f) == n_tgt);
            ok = ok && (n_dft == 0  || fread(ckpt.data_dft.data(),  1, n_dft,  f) == n_dft);
            ok = ok && (n_spec == 0 || fread(ckpt.data_spec.data(), 1, n_spec, f) == n_spec);
        }
    }

    state.prompt.tokens = server_tokens(tokens, false);

    // v2: a shared base section is appended after the checkpoints (dedup, now removed);
    //     read and discard it so stale v2 entries still load
    if (ok && version == 2) {
        int64_t  base_n_tokens = 0;
        int32_t  base_pos_max  = 0;
        uint64_t len_base_tgt  = 0;
        uint64_t len_base_dft  = 0;

        ok = ok && rd(&base_n_tokens, 8) && rd(&base_pos_max, 4) &&
              rd(&len_base_tgt, 8) && rd(&len_base_dft, 8) &&
              len_base_tgt <= (1ull << 34) && len_base_dft <= (1ull << 34);

        ok = ok && (len_base_tgt == 0 || fseek(f, len_base_tgt, SEEK_CUR) == 0);
        ok = ok && (len_base_dft == 0 || fseek(f, len_base_dft, SEEK_CUR) == 0);
    }

    return ok;
}


} // namespace


bool server_prompt_cache::save_to_disk(const std::string & dir, const std::string & fingerprint) const {
    namespace fs = std::filesystem;

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        fs::create_directories(dir, ec);
        if (ec) {
            SRV_WRN(" - disk prompt cache: failed to create directory '%s': %s\n", dir.c_str(), ec.message().c_str());
            return false;
        }
    }

    const std::string meta_path = (fs::path(dir) / "meta").string();
    const std::string meta_tmp  = (fs::path(dir) / "meta.tmp").string();

    std::string stored;
    {
        std::ifstream in(meta_path, std::ios::binary);
        stored.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        // strip trailing newlines (older saves wrote a trailing '\n')
        while (!stored.empty() && (stored.back() == '\n' || stored.back() == '\r')) {
            stored.pop_back();
        }
    }

    // a config change invalidates the whole directory
    if (stored != fingerprint) {
        SRV_INF(" - disk prompt cache: fingerprint changed, clearing '%s'\n", dir.c_str());
        for (auto & entry : fs::directory_iterator(dir, ec)) {
            if (entry.path().extension() == ".dpc" || entry.path().extension() == ".ccn") {
                fs::remove(entry.path(), ec);
            }
        }
    }

    bool ok = true;

    for (const auto & state : states) {
        // media placeholders cannot be rebuilt from token ids, skip such entries
        if (state.prompt.tokens.has_media()) {
            SRV_TRC("%s", " - disk prompt cache: skipping media prompt\n");
            continue;
        }

        const llama_tokens tokens = state.prompt.tokens.get_text_tokens();
        const std::string fname = dpc_sha256_hex((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token)) + ".dpc";
        const std::string path  = (fs::path(dir) / fname).string();
        const std::string tmp   = path + ".tmp";

        FILE * f = fopen(tmp.c_str(), "wb");
        if (f == nullptr) {
            SRV_WRN(" - disk prompt cache: cannot open '%s'\n", tmp.c_str());
            ok = false;
            continue;
        }

        const uint32_t magic   = DPC_MAGIC;
        const uint32_t version = DPC_VERSION;

        bool entry_ok = fwrite(&magic, 4, 1, f) == 1 && fwrite(&version, 4, 1, f) == 1 && dpc_write_state(f, state);
        fclose(f);

        if (entry_ok) {
            fs::rename(tmp, path, ec);
            if (ec) {
                SRV_WRN(" - disk prompt cache: rename to '%s' failed: %s\n", path.c_str(), ec.message().c_str());
                ok = false;
            }
        } else {
            SRV_WRN(" - disk prompt cache: write to '%s' failed\n", tmp.c_str());
            ok = false;
        }
    }

    // entries that left the RAM tier stay on disk: only a config change (handled
    // above) clears the directory. enforce the independent disk budget instead,
    // removing the oldest entries first.
    prune_disk_budget(dir);

    // the fingerprint is written last: an interrupted save leaves the old meta
    // in place and the directory is never treated as valid against the new config
    {
        std::ofstream out(meta_tmp, std::ios::binary);
        out.write(fingerprint.data(), (std::streamsize) fingerprint.size());
        out.close();
    }
    fs::rename(meta_tmp, meta_path, ec);
    if (ec) {
        SRV_WRN(" - disk prompt cache: failed to write meta: %s\n", ec.message().c_str());
        ok = false;
    }

    return ok;
}

void server_prompt_cache::prune_disk_budget(const std::string & dir) const {
    namespace fs = std::filesystem;

    if (disk_max_size == 0) {
        return;
    }

    struct dpc_file {
        fs::path path;
        fs::file_time_type mtime;
        uint64_t size;
    };

    std::error_code ec;
    std::vector<dpc_file> files;
    uint64_t used = 0;

    for (auto & entry : fs::directory_iterator(dir, ec)) {
        const auto ext = entry.path().extension();
        if (ext != ".dpc" && ext != ".ccn") {
            continue;
        }
        const auto size = fs::file_size(entry.path(), ec);
        if (ec) {
            continue;
        }
        used += (uint64_t) size;
        files.push_back({ entry.path(), fs::last_write_time(entry.path(), ec), (uint64_t) size });
    }

    if (used > disk_max_size) {
        std::sort(files.begin(), files.end(), [](const dpc_file & a, const dpc_file & b) {
            return a.mtime < b.mtime;
        });

        for (const auto & file : files) {
            if (used <= disk_max_size) {
                break;
            }
            fs::remove(file.path, ec);
            used -= file.size;
        }

        SRV_INF(" - disk prompt cache: pruned to %.3f MiB (budget %.3f MiB)\n",
                used / (1024.0 * 1024.0), disk_max_size / (1024.0 * 1024.0));
    }
}

void server_prompt_cache::disk_prune_dpc(const std::string & dir) const {
    namespace fs = std::filesystem;

    if (disk_max_size == 0) {
        return;
    }

    struct dpc_file {
        fs::path path;
        fs::file_time_type mtime;
        uint64_t size;
    };

    std::error_code ec;
    std::vector<dpc_file> files;
    uint64_t used = 0;

    for (auto & entry : fs::directory_iterator(dir, ec)) {
        if (entry.path().extension() != ".dpc") {
            continue;
        }
        const auto size = fs::file_size(entry.path(), ec);
        if (ec) {
            continue;
        }
        used += (uint64_t) size;
        files.push_back({ entry.path(), fs::last_write_time(entry.path(), ec), (uint64_t) size });
    }

    if (used > disk_max_size) {
        std::sort(files.begin(), files.end(), [](const dpc_file & a, const dpc_file & b) {
            return a.mtime < b.mtime;
        });

        for (const auto & file : files) {
            if (used <= disk_max_size) {
                break;
            }
            fs::remove(file.path, ec);
            used -= file.size;
        }

        SRV_INF(" - disk prompt cache: pruned .dpc to %.3f MiB (budget %.3f MiB)\n",
                used / (1024.0 * 1024.0), disk_max_size / (1024.0 * 1024.0));
    }
}

bool server_prompt_cache::commit_disk_state(server_prompt_cache_state * state) {
    // only the pending write-through entry gets committed here; RAM-tier entries
    // (and callers with no pending scratch) are passed through untouched
    if (state != disk_state_.get()) {
        return true;
    }

    // media placeholders cannot be rebuilt from token ids, skip such entries
    if (state->prompt.tokens.has_media()) {
        SRV_TRC("%s", " - disk prompt cache: skipping media prompt\n");
        disk_state_.reset();
        return true;
    }

    if (disk_dir_.empty()) {
        disk_state_.reset();
        return true;
    }

    // move state into the job and free RAM before the write is done
    disk_write_job job;
    job.kind        = disk_write_job::type::state;
    job.state       = std::move(disk_state_);
    job.dir         = disk_dir_;
    job.fingerprint = disk_fingerprint_;

    disk_state_.reset();

    start_disk_thread();

    {
        std::lock_guard<std::mutex> lock(disk_jobs_mtx_);
        disk_jobs_.push_back(std::move(job));
    }
    disk_jobs_cv_.notify_one();

    return true;
}

void server_prompt_cache::start_disk_thread() {
    if (disk_thread_.joinable()) {
        return;
    }
    disk_thread_stop_ = false;
    try {
        disk_thread_ = std::thread([this] { disk_thread_main(); });
    } catch (const std::system_error & e) {
        SRV_WRN(" - disk prompt cache: failed to start writer thread: %s\n", e.what());
    }
}

void server_prompt_cache::drain_disk_writes() {
    if (!disk_thread_.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(disk_jobs_mtx_);
        disk_thread_stop_ = true;
    }
    disk_jobs_cv_.notify_one();
    disk_thread_.join();
}

bool server_prompt_cache::enqueue_sidecar(const std::string & path, std::vector<uint8_t> bytes) {
    start_disk_thread();
    if (!disk_thread_.joinable()) {
        return false;
    }
    disk_write_job job;
    job.kind  = disk_write_job::type::sidecar;
    job.path  = path;
    job.bytes = std::move(bytes);
    {
        std::lock_guard<std::mutex> lock(disk_jobs_mtx_);
        disk_jobs_.push_back(std::move(job));
    }
    disk_jobs_cv_.notify_one();
    return true;
}

void server_prompt_cache::disk_thread_main() {
    while (true) {
        disk_write_job job;
        {
            std::unique_lock<std::mutex> lock(disk_jobs_mtx_);
            disk_jobs_cv_.wait(lock, [this] { return disk_thread_stop_ || !disk_jobs_.empty(); });
            if (disk_thread_stop_ && disk_jobs_.empty()) {
                return;
            }
            job = std::move(disk_jobs_.front());
            disk_jobs_.pop_front();
        }

        switch (job.kind) {
        case disk_write_job::type::state:
            disk_run_state_job(job);
            break;
        case disk_write_job::type::sidecar:
            disk_run_sidecar_job(job);
            break;
        }
    }
}

void server_prompt_cache::disk_run_state_job(disk_write_job & job) {
    namespace fs = std::filesystem;

    const auto & dir        = job.dir;
    const auto & fingerprint = job.fingerprint;
    auto & state             = *job.state;

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        fs::create_directories(dir, ec);
        if (ec) {
            SRV_WRN(" - disk prompt cache: failed to create directory '%s': %s\n", dir.c_str(), ec.message().c_str());
            return;
        }
    }

    const llama_tokens tokens = state.prompt.tokens.get_text_tokens();
    const std::string fname = dpc_sha256_hex((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token)) + ".dpc";
    const std::string path  = (fs::path(dir) / fname).string();
    const std::string tmp   = path + ".tmp";

    FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        SRV_WRN(" - disk prompt cache: cannot open '%s'\n", tmp.c_str());
        return;
    }

    const uint32_t magic   = DPC_MAGIC;
    const uint32_t version = DPC_VERSION;

    bool ok = fwrite(&magic, 4, 1, f) == 1 && fwrite(&version, 4, 1, f) == 1 && dpc_write_state(f, state);
    fclose(f);

    if (!ok) {
        SRV_WRN(" - disk prompt cache: write to '%s' failed\n", tmp.c_str());
        std::error_code ec_rm;
        fs::remove(tmp, ec_rm);
        return;
    }

    fs::rename(tmp, path, ec);
    if (ec) {
        SRV_WRN(" - disk prompt cache: rename to '%s' failed: %s\n", path.c_str(), ec.message().c_str());
        return;
    }

    // append to the index and flag it valid so later matches do not need a reload
    {
        std::lock_guard<std::mutex> lock(disk_index_mtx_);
        disk_entries.push_back({ fname, path, state.prompt.tokens.clone(), (size_t) fs::file_size(path, ec) });
        disk_index_valid = true;
    }

    // .dpc-only prune; .ccn files are owned by the main thread and budgeted at exit
    disk_prune_dpc(dir);

    // keep the in-memory index in sync with any entries the prune removed
    {
        std::error_code ec_x;
        std::lock_guard<std::mutex> lock(disk_index_mtx_);
        disk_entries.erase(
            std::remove_if(disk_entries.begin(), disk_entries.end(),
                [&](const disk_entry & e) {
                    return !fs::exists(e.path, ec_x);
                }),
            disk_entries.end());
    }

    // keep the meta file current so a later shutdown save does not treat this
    // directory as a config mismatch and clear the freshly written entries
    {
        const std::string meta_path = (fs::path(dir) / "meta").string();
        const std::string meta_tmp  = meta_path + ".tmp";

        std::ofstream out(meta_tmp, std::ios::binary);
        out.write(fingerprint.data(), (std::streamsize) fingerprint.size());
        out.close();

        fs::rename(meta_tmp, meta_path, ec);
        if (ec) {
            SRV_WRN(" - disk prompt cache: failed to update meta: %s\n", ec.message().c_str());
        }
    }

    const size_t state_bytes = state.size();
    SRV_INF(" - disk prompt cache: wrote through state '%s' (%.3f MiB)\n", fname.c_str(), state_bytes / (1024.0 * 1024.0));
}

void server_prompt_cache::disk_run_sidecar_job(disk_write_job & job) {
    namespace fs = std::filesystem;

    const std::string tmp = job.path + ".tmp";

    FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        SRV_WRN(" - disk prompt cache: cannot open sidecar '%s'\n", tmp.c_str());
        return;
    }
    const bool ok = !job.bytes.empty() && fwrite(job.bytes.data(), 1, job.bytes.size(), f) == job.bytes.size();
    fclose(f);

    if (!ok) {
        SRV_WRN(" - disk prompt cache: write to sidecar '%s' failed\n", tmp.c_str());
        std::error_code ec;
        fs::remove(tmp, ec);
        return;
    }

    std::error_code ec;
    fs::rename(tmp, job.path, ec);
    if (ec) {
        SRV_WRN(" - disk prompt cache: rename sidecar to '%s' failed: %s\n", job.path.c_str(), ec.message().c_str());
    }
}

bool server_prompt_cache::load_from_disk_tier(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    // scan the index under lock; file I/O happens outside the lock so the writer
    // thread can keep pushing entries while a slow restore runs
    std::string best_path;
    std::string best_fname;
    float f_keep_best = 0.0f;
    float f_sim_best  = 0.0f;

    {
        std::lock_guard<std::mutex> lock(disk_index_mtx_);
        if (!disk_index_valid || disk_entries.empty()) {
            return true;
        }

        // selection mirrors the in-RAM search in load()
        const int lcp_base = prompt.tokens.get_common_prefix(tokens_new);

        f_keep_best = prompt.tokens.size() > 0 ? float(lcp_base) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
        f_sim_best  = float(lcp_base) / tokens_new.size();

        size_t idx_best = SIZE_MAX;

        for (size_t i = 0; i < disk_entries.size(); ++i) {
            const auto & entry = disk_entries[i];

            const int lcp_cur = entry.tokens.get_common_prefix(tokens_new);

            const float f_keep_cur = entry.tokens.size() > 0 ? float(lcp_cur) / entry.tokens.size() : -1.0f;
            const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

            SRV_TRC(" - disk prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n",
                    entry.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

            // don't trash large prompts
            if (f_keep_cur < 0.25f) {
                continue;
            }

            if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
                f_keep_best = f_keep_cur;
                f_sim_best  = f_sim_cur;

                idx_best = i;
            }
        }

        if (idx_best == SIZE_MAX) {
            return true;
        }

        best_path  = disk_entries[idx_best].path;
        best_fname = disk_entries[idx_best].fname;
    }

    SRV_TRC(" - disk prompt cache: found entry '%s' with f_keep = %.3f, f_sim = %.3f\n", best_fname.c_str(), f_keep_best, f_sim_best);

    server_prompt_cache_state state;
    {
        FILE * f = fopen(best_path.c_str(), "rb");
        if (f == nullptr) {
            SRV_WRN(" - disk prompt cache: cannot open '%s'\n", best_path.c_str());

            return true;
        }

        uint32_t magic = 0;
        uint32_t version = 0;
        const bool ok = fread(&magic, 4, 1, f) == 1 && fread(&version, 4, 1, f) == 1 &&
                        magic == DPC_MAGIC && (version == DPC_VERSION || version == 2 || version == 1) && dpc_read_state(f, version, state);
        fclose(f);

        if (!ok) {
            SRV_ERR("failed to load prompt state from disk ('%s')\n", best_fname.c_str());

            return false;
        }
    }

    // promote into the RAM tier when it fits the budget, otherwise restore directly from disk
    const size_t state_bytes = state.size();
    const bool can_promote = limit_size == 0 || size() + state_bytes <= limit_size;

    if (can_promote) {
        states.push_back(std::move(state));

        if (!restore_state(prompt, states.back(), ctx_tgt, ctx_dft, id_slot)) {
            states.pop_back();

            return false;
        }

        // consume like a RAM hit: the state now lives in the slot
        states.pop_back();

        SRV_INF(" - disk prompt cache: promoted entry '%s' into RAM (%.3f MiB)\n", best_fname.c_str(), state_bytes / (1024.0 * 1024.0));
    } else {
        if (!restore_state(prompt, state, ctx_tgt, ctx_dft, id_slot)) {
            return false;
        }

        SRV_INF(" - disk prompt cache: restored entry '%s' directly from disk (%.3f MiB, RAM budget exhausted)\n", best_fname.c_str(), state_bytes / (1024.0 * 1024.0));
    }

    return true;
}

size_t server_prompt_cache::load_disk_index(const std::string & dir, const std::string & fingerprint) {
    namespace fs = std::filesystem;

    disk_dir_         = dir;
    disk_fingerprint_ = fingerprint;

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return 0;
    }

    std::string stored;
    {
        std::ifstream in((fs::path(dir) / "meta").string(), std::ios::binary);
        if (!in) {
            SRV_INF(" - disk prompt cache: no meta in '%s', skipping load\n", dir.c_str());
            return 0;
        }
        stored.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        // strip trailing newlines (older saves wrote a trailing '\n')
        while (!stored.empty() && (stored.back() == '\n' || stored.back() == '\r')) {
            stored.pop_back();
        }
    }

    if (stored != fingerprint) {
        SRV_INF(" - disk prompt cache: fingerprint mismatch for '%s', ignoring cached prompts\n", dir.c_str());
        return 0;
    }

    std::vector<fs::path> files;
    for (auto & entry : fs::directory_iterator(dir, ec)) {
        if (entry.path().extension() == ".dpc") {
            files.push_back(entry.path());
        }
    }

    // deterministic order
    std::sort(files.begin(), files.end());

    // read the token prefix and tracked sizes only; the KV blobs stay on disk
    // until a matching entry is actually restored. build the index locally and
    // swap it in under the lock so the writer thread never sees a partial index
    std::vector<disk_entry> new_entries;

    size_t n_indexed = 0;
    for (const auto & path : files) {
        FILE * f = fopen(path.string().c_str(), "rb");
        if (f == nullptr) {
            continue;
        }

        uint32_t magic    = 0;
        uint32_t version  = 0;
        uint32_t n_tokens = 0;
        uint64_t len_main = 0;
        uint64_t len_drft = 0;
        uint32_t n_ckpt   = 0;

        bool ok =
            fread(&magic, 4, 1, f) == 1 && fread(&version, 4, 1, f) == 1 &&
            fread(&n_tokens, 4, 1, f) == 1 && fread(&len_main, 8, 1, f) == 1 &&
            fread(&len_drft, 8, 1, f) == 1 && fread(&n_ckpt, 4, 1, f) == 1 &&
            magic == DPC_MAGIC && (version == DPC_VERSION || version == 2 || version == 1) &&
            n_tokens <= DPC_MAX_TOKENS && n_ckpt <= DPC_MAX_CHECKPOINTS;

        llama_tokens tokens(n_tokens);
        ok = ok && (n_tokens == 0 || fread(tokens.data(), sizeof(llama_token), n_tokens, f) == n_tokens);
        fclose(f);

        if (!ok) {
            SRV_WRN(" - disk prompt cache: ignoring corrupt entry '%s'\n", path.filename().string().c_str());
            continue;
        }

        new_entries.push_back({
            /*.fname      =*/ path.filename().string(),
            /*.path       =*/ path.string(),
            /*.tokens     =*/ server_tokens(tokens, false),
            /*.byte_size  =*/ (size_t) fs::file_size(path, ec),
        });

        n_indexed++;
    }

    {
        std::lock_guard<std::mutex> lock(disk_index_mtx_);
        disk_entries     = std::move(new_entries);
        disk_index_valid = true;
    }

    return n_indexed;
}
