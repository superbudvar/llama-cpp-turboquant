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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define cc_fileno _fileno
#else
#include <unistd.h>
#include <sys/file.h>
#define cc_fileno fileno
#endif

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

size_t server_prompt_cache::select_cache_candidate(const std::vector<std::pair<size_t, size_t>> & cands, size_t lcp_base) {
    size_t idx_best = SIZE_MAX;
    size_t lcp_best = 0;
    size_t len_best = 0;

    for (size_t i = 0; i < cands.size(); ++i) {
        const size_t len = cands[i].first;
        const size_t lcp = cands[i].second;

        // don't trash large prompts: keep at least a quarter of the cached state
        if (len == 0 || float(lcp) / len < 0.25f) {
            continue;
        }

        // the pick must strictly extend the slot's own prefix
        if (lcp <= lcp_base) {
            continue;
        }

        // longest common prefix wins (least reprocessing), ties toward the
        // smaller state (preserves more of its own context)
        if (lcp > lcp_best || (lcp == lcp_best && len < len_best)) {
            idx_best = i;
            lcp_best = lcp;
            len_best = len;
        }
    }

    return idx_best;
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    // collect the candidate scores under the decision rule shared with the disk tier
    std::vector<std::pair<size_t, size_t>> cands;
    std::vector<std::list<server_prompt_cache_state>::iterator> cands_it;
    cands.reserve(states.size());
    cands_it.reserve(states.size());

    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        cands.push_back({ it->prompt.tokens.size(), (size_t) lcp_cur });
        cands_it.push_back(it);
    }

    const auto idx_best = select_cache_candidate(cands, (size_t) lcp_best);

    if (idx_best != SIZE_MAX) {
        const auto it_best = cands_it[idx_best];
        const float f_keep_cur = float(cands[idx_best].second) / cands[idx_best].first;
        const float f_sim_cur  = float(cands[idx_best].second) / tokens_new.size();

        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_cur, f_sim_cur);

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

// token-only state index: same header shape as DPC but len_main/len_drft/n_ckpt
// are always zero. the KV lives in cache.dcp as lineage-chained cells; the .dts
// sidecar exists only so boot-time load_disk_index can build a restore candidate
// without scanning the store log. a distinct magic means a legacy reader can
// never mistake it for a full state.
constexpr uint32_t DTS_MAGIC   = 0x31535444; // "DTS1"
constexpr uint32_t DTS_VERSION = 1;
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

// raw sha256 (32 bytes out); used for the chunk-store lineage chain
void dpc_sha256_raw(const uint8_t * data, size_t len, uint8_t * out) {
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

    for (size_t i = 0; i < 8; i++) {
        out[i * 4 + 0] = (uint8_t) (h[i] >> 24);
        out[i * 4 + 1] = (uint8_t) (h[i] >> 16);
        out[i * 4 + 2] = (uint8_t) (h[i] >>  8);
        out[i * 4 + 3] = (uint8_t) (h[i] >>  0);
    }
}

} // namespace

//
// TurboQuant chunked disk prompt cache
//
// replaces the "<sha256>.dpc" per-state files with one append-only cache.dcp
// record log shared by every state in the directory:
//
//   cache.dcp   record log (chunks + state refs + tombstones), generation
//               tagged so a torn tail - a partial append or an uncommitted
//               batch - is detected and truncated at open
//   cache.dci   index snapshot (chunk offsets + state references); a cache of
//               the authoritative .dcp, always rebuildable from a full scan
//   cache.lock  advisory single-writer lock (flock)
//
// a state blob is sliced by the save path into fixed-cell (region, layer)
// sheets; identical sheets dedup to an existing chunk. state records map a
// sha256 token fingerprint to the chunk ids that rebuild it in byte order.
//
// record layout (all little-endian):
//   u16 kind   (1 = chunk, 2 = state, 3 = tombstone)
//   u16 region (REGION_*)
//   u32 layer
//   u32 cell_lo
//   u32 cell_hi
//   u64 lineage_epoch
//   u64 gen    (commit generation the record belongs to)
//   u64 len    (payload size)
//   u32 crc32c (of the record header bytes [0..40) + payload)
//   u32 pad
// then payload, then zero-padded to the next 8-byte boundary.
//
// the cache.dcp header is a fixed 24 bytes (magic, version, base lineage,
// generation); a commit rewrites the generation after fsyncing the records.
// records tagged gen > header gen belong to a batch that never committed and
// are discarded at open. the base lineage is a fingerprint-derived seed; a
// different base resets the store (same semantics as the .dpc/.ccn wipe).

namespace {

constexpr uint32_t CHUNK_REC_HDR = 48;
constexpr uint16_t CHUNK_REC_CHUNK    = 1;
constexpr uint16_t CHUNK_REC_STATE    = 2;   // payload: fp[32] + n_tokens u64 + ids u64[n]
constexpr uint16_t CHUNK_REC_TOMB     = 3;   // payload: fp[32]

// crc32c = Castagnoli, reflected
uint32_t g_crc32c_table[256];
bool     g_crc32c_ready = false;

void crc32c_init_table() {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int b = 0; b < 8; b++) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0x82F63B78u : 0);
        }
        g_crc32c_table[i] = crc;
    }
    g_crc32c_ready = true;
}

uint32_t crc32c(const uint8_t * data, size_t len) {
    if (!g_crc32c_ready) {
        crc32c_init_table();
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ g_crc32c_table[(crc ^ data[i]) & 0xff];
    }
    return crc ^ 0xFFFFFFFFu;
}

} // namespace

struct prompt_cache_chunk_store::impl {
    std::string dir_;
    std::string dcp_path_;
    std::string dci_path_;
    std::string lock_path_;

    FILE * f_       = nullptr;   // cache.dcp
    int    lock_fd_ = -1;        // cache.lock

    uint64_t base_lineage_    = 0;
    uint64_t committed_gen_   = 0;   // generation in the .dcp header
    uint64_t queued_gen_      = 0;   // 0 = no open batch
    uint64_t batch_start_     = 0;   // file offset where the open batch appends
    size_t   batch_chunks_    = 0;   // chunks_.size() at batch start
    uint64_t batch_appended_  = 0;   // bytes this batch physically appended
    std::vector<std::array<uint8_t, 32>> batch_states_;
    std::vector<std::array<uint8_t, 32>> batch_tombs_;

    struct chunk {
        chunk_key key;
        uint64_t  offset = 0;   // payload offset in cache.dcp
        uint64_t  len    = 0;
        uint32_t  crc    = 0;
        uint32_t  refs   = 0;   // live state references
    };
    std::vector<chunk> chunks_;
    std::map<std::string, std::vector<uint64_t>> states_;  // fp-str -> ids, byte order
    std::map<std::string, uint64_t> state_tokens_;          // fp-str -> n_tokens
    std::map<chunk_key, uint64_t>  by_key_;                 // dedup oracle -> chunk id

    bool writable_   = false;
    bool want_dedup_ = true;
    bool dci_valid_  = false;
    bool close_log_  = false;   // header rewrite failed: stop writes

    std::mutex io_mtx_;

    impl() {
        want_dedup_ = store_enabled();
    }

    ~impl() = default;

    static size_t rec_size(size_t len) {
        return CHUNK_REC_HDR + ((len + 7) & ~(size_t) 7);
    }

    bool file_size_of(const std::string & path, uint64_t & out) {
        std::error_code ec;
        const uintmax_t sz = std::filesystem::file_size(path, ec);
        if (ec) {
            return false;
        }
        out = sz;
        return true;
    }

    // read len bytes at offset into buf (buffered, position-independent)
    bool pread_at(int64_t off, uint8_t * buf, size_t len) {
        if (cc_fileno(f_) < 0) {
            return false;
        }
#ifdef _WIN32
        if (_fseeki64(f_, off, SEEK_SET) != 0) {
            return false;
        }
        return fread(buf, 1, len, f_) == len;
#else
        const ssize_t n = pread(cc_fileno(f_), buf, len, (off_t) off);
        return n == (ssize_t) len;
#endif
    }

    // append payload at the current end of file; returns [0] logical offset
    bool append_at_end(const uint8_t * data, size_t len, uint64_t & off) {
        if (cc_fileno(f_) < 0) {
            return false;
        }
#ifdef _WIN32
        if (_fseeki64(f_, 0, SEEK_END) != 0) {
            return false;
        }
        off = (uint64_t) _ftelli64(f_);
        if (fwrite(data, 1, len, f_) != len) {
            return false;
        }
        return fflush(f_) == 0;
#else
        off = (uint64_t) lseek(cc_fileno(f_), 0, SEEK_END);
        const ssize_t n = pwrite(cc_fileno(f_), data, len, (off_t) off);
        return n == (ssize_t) len;
#endif
    }

    bool sync_file() {
#ifdef _WIN32
        return _commit(cc_fileno(f_)) == 0;
#else
        return fsync(cc_fileno(f_)) == 0;
#endif
    }

    bool truncate_to(uint64_t size) {
#ifdef _WIN32
        return _chsize_s(cc_fileno(f_), (long long) size) == 0;
#else
        return ftruncate(cc_fileno(f_), (off_t) size) == 0;
#endif
    }

    void set_header_gen(uint64_t gen) {
        uint8_t hdr[24];
        memset(hdr, 0, sizeof(hdr));
        cc_io::put_u32(hdr,      0x54514331); // "TQC1"
        cc_io::put_u32(hdr + 4,  1);
        cc_io::put_u64(hdr + 8,  base_lineage_);
        cc_io::put_u64(hdr + 16, gen);
        append_rewrite_hdr(hdr);
    }

    void append_rewrite_hdr(const uint8_t * hdr) {
        if (cc_fileno(f_) < 0) {
            return;
        }
#ifdef _WIN32
        _fseeki64(f_, 0, SEEK_SET);
        fwrite(hdr, 1, 24, f_);
        fflush(f_);
#else
        const ssize_t wrote = pwrite(cc_fileno(f_), hdr, 24, 0);
        if (wrote != 24) {
            close_log_ = true;   // header rewrite failed, stop writes
        }
#endif
    }

    // needs io_mtx_; writes one record at the end of the file
    bool append_record(uint16_t kind, const chunk_key & key, uint64_t gen,
                       const uint8_t * payload, size_t len, uint64_t & payload_off) {
        if (close_log_) {
            return false;
        }
        std::vector<uint8_t> hdr(CHUNK_REC_HDR);
        cc_io::put_u16(hdr.data(),       kind);
        cc_io::put_u16(hdr.data() + 2,   key.region);
        cc_io::put_u32(hdr.data() + 4,   key.layer);
        cc_io::put_u32(hdr.data() + 8,   key.cell_lo);
        cc_io::put_u32(hdr.data() + 12,  key.cell_hi);
        cc_io::put_u64(hdr.data() + 16,  key.lineage_epoch);
        cc_io::put_u64(hdr.data() + 24,  gen);
        cc_io::put_u64(hdr.data() + 32,  len);
        cc_io::put_u32(hdr.data() + 40,  crc32c(hdr.data(), 40) ^ crc32c(payload, len));
        cc_io::put_u32(hdr.data() + 44,  0);

        std::vector<uint8_t> rec(CHUNK_REC_HDR + ((len + 7) & ~(size_t) 7));
        memcpy(rec.data(), hdr.data(), CHUNK_REC_HDR);
        memcpy(rec.data() + CHUNK_REC_HDR, payload, len);
        memset(rec.data() + CHUNK_REC_HDR + len, 0, rec.size() - CHUNK_REC_HDR - len);

        uint64_t off = 0;
        if (!append_at_end(rec.data(), rec.size(), off)) {
            return false;
        }
        payload_off = off + CHUNK_REC_HDR;
        return true;
    }

    std::string fp_str(const std::array<uint8_t, 32> & fp) {
        return std::string((const char *) fp.data(), 32);
    }

    void recompute_refs() {
        for (auto & c : chunks_) {
            c.refs = 0;
        }
        for (const auto & it : states_) {
            for (uint64_t id : it.second) {
                if (id < chunks_.size()) {
                    chunks_[id].refs++;
                }
            }
        }
    }

    // needs io_mtx_; rebuild all in-memory state from a full .dcp scan.
    // returns false when the file is corrupt beyond a resettable state.
    bool rebuild_from_scan() {
        chunks_.clear();
        states_.clear();
        state_tokens_.clear();
        by_key_.clear();

        // read and verify the header first
        uint8_t hdr[24];
        if (!pread_at(0, hdr, 24)) {
            return false;
        }
        if (cc_io::get_u32(hdr)      != 0x54514331 || cc_io::get_u32(hdr + 4) != 1) {
            return false;
        }

        uint64_t filesz = 0;
        if (!file_size_of(dcp_path_, filesz)) {
            return false;
        }

        int64_t off = 24;
        while ((uint64_t) off + CHUNK_REC_HDR <= filesz) {
            uint8_t rh[CHUNK_REC_HDR];
            if (!pread_at(off, rh, CHUNK_REC_HDR)) {
                break;
            }
            const uint16_t kind   = cc_io::get_u16(rh);
            const uint16_t region = cc_io::get_u16(rh + 2);
            const uint32_t layer  = cc_io::get_u32(rh + 4);
            const uint32_t cell_lo = cc_io::get_u32(rh + 8);
            const uint32_t cell_hi = cc_io::get_u32(rh + 12);
            const uint64_t lineage = cc_io::get_u64(rh + 16);
            const uint64_t gen     = cc_io::get_u64(rh + 24);
            const uint64_t len     = cc_io::get_u64(rh + 32);
            const uint32_t crc     = cc_io::get_u32(rh + 40);

            if (kind != CHUNK_REC_CHUNK && kind != CHUNK_REC_STATE && kind != CHUNK_REC_TOMB) {
                break;   // torn tail
            }
            if (len > MAX_REC_LEN) {
                break;
            }
            const int64_t p_off = off + CHUNK_REC_HDR;
            if ((uint64_t) p_off + len > filesz) {
                break;   // short payload
            }
            std::vector<uint8_t> payload(len);
            if (len > 0 && !pread_at(p_off, payload.data(), len)) {
                break;
            }
            if ((crc32c(rh, 40) ^ crc32c(payload.data(), len)) != crc) {
                break;   // torn/corrupt record
            }

            // records from a batch that never committed (gen > header gen)
            if (gen > committed_gen_) {
                break;
            }

            if (kind == CHUNK_REC_CHUNK) {
                chunk c;
                c.key.region        = (uint8_t) region;
                c.key.layer         = (uint16_t) layer;
                c.key.cell_lo       = cell_lo;
                c.key.cell_hi       = cell_hi;
                c.key.lineage_epoch = lineage;
                c.offset            = (uint64_t) p_off;
                c.len               = len;
                c.crc               = crc;
                c.refs              = 0;
                chunks_.push_back(c);
                by_key_[c.key] = chunks_.size() - 1;
            } else if (kind == CHUNK_REC_STATE) {
                if (payload.size() < 32 + 8 + 8) {
                    break;
                }
                std::array<uint8_t, 32> fp;
                memcpy(fp.data(), payload.data(), 32);
                const uint64_t n_tokens = cc_io::get_u64(payload.data() + 32);
                const uint64_t n        = cc_io::get_u64(payload.data() + 40);
                if (n > (payload.size() - 48) / 8) {
                    break;
                }
                std::vector<uint64_t> ids(n);
                for (uint64_t i = 0; i < n; i++) {
                    ids[i] = cc_io::get_u64(payload.data() + 48 + 8 * i);
                    if (ids[i] >= chunks_.size()) {
                        break;   // reference to a later/absent chunk
                    }
                }
                const std::string fp_s = fp_str(fp);
                states_[fp_s] = std::move(ids);
                state_tokens_[fp_s] = n_tokens;
            } else { // tombstone
                if (payload.size() < 32) {
                    break;
                }
                std::array<uint8_t, 32> fp;
                memcpy(fp.data(), payload.data(), 32);
                states_.erase(fp_str(fp));
                state_tokens_.erase(fp_str(fp));
            }

            off += (int64_t) rec_size(len);
        }

        // a clean scan ends exactly on a record boundary; anything past the
        // last good record is garbage (torn append or torn uncommitted batch)
        if (writable_ && (uint64_t) off != filesz) {
            truncate_to((uint64_t) off);
            sync_file();
        }

        recompute_refs();
        return true;
    }

    // index snapshot on disk; valid when header + sizes match the current .dcp
    bool load_index_fast() {
        std::ifstream in(dci_path_, std::ios::binary);
        if (!in) {
            return false;
        }
        uint8_t hdr[48];
        in.read((char *) hdr, 48);
        if (!in || cc_io::get_u32(hdr) != 0x31494344 || cc_io::get_u32(hdr + 4) != 1) {
            return false;
        }
        if (cc_io::get_u64(hdr + 8)  != base_lineage_ ||
            cc_io::get_u64(hdr + 16) != committed_gen_) {
            return false;
        }
        const uint64_t dcp_size = cc_io::get_u64(hdr + 24);
        uint64_t real = 0;
        if (!file_size_of(dcp_path_, real) || real != dcp_size) {
            return false;
        }
        const uint32_t n_chunks = cc_io::get_u32(hdr + 32);
        const uint32_t n_states = cc_io::get_u32(hdr + 36);

        chunks_.clear();
        states_.clear();
        state_tokens_.clear();
        by_key_.clear();

        for (uint32_t i = 0; i < n_chunks; i++) {
            uint8_t e[48];
            in.read((char *) e, 48);
            if (!in) {
                return false;
            }
            chunk c;
            c.key.region        = (uint8_t) cc_io::get_u16(e + 2);
            c.key.layer         = (uint16_t) cc_io::get_u16(e + 4);
            c.key.cell_lo       = cc_io::get_u32(e + 8);
            c.key.cell_hi       = cc_io::get_u32(e + 12);
            c.key.lineage_epoch = cc_io::get_u64(e + 16);
            c.offset            = cc_io::get_u64(e + 24);
            c.len               = cc_io::get_u64(e + 32);
            c.crc               = cc_io::get_u32(e + 40);
            c.refs              = 0;
            chunks_.push_back(c);
            by_key_[c.key] = chunks_.size() - 1;
        }

        for (uint32_t i = 0; i < n_states; i++) {
            std::array<uint8_t, 32> fp;
            in.read((char *) fp.data(), 32);
            uint8_t meta[16];
            in.read((char *) meta, 16);
            if (!in) {
                return false;
            }
            const uint64_t n_tokens = cc_io::get_u64(meta);
            const uint64_t n        = cc_io::get_u64(meta + 8);
            std::vector<uint64_t> ids(n);
            for (uint64_t k = 0; k < n; k++) {
                uint8_t ib[8];
                in.read((char *) ib, 8);
                if (!in) {
                    return false;
                }
                ids[k] = cc_io::get_u64(ib);
            }
            states_[fp_str(fp)] = std::move(ids);
            state_tokens_[fp_str(fp)] = n_tokens;
        }
        recompute_refs();
        return true;
    }

    // needs io_mtx_; write cache.dci from in-memory state
    void write_index_locked() {
        const std::string tmp = dci_path_ + ".tmp";
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return;
        }
        uint8_t hdr[48];
        memset(hdr, 0, sizeof(hdr));
        cc_io::put_u32(hdr,      0x31494344); // "DCI1"
        cc_io::put_u32(hdr + 4,  1);
        cc_io::put_u64(hdr + 8,  base_lineage_);
        cc_io::put_u64(hdr + 16, committed_gen_);
        uint64_t dcp_size = 0;
        file_size_of(dcp_path_, dcp_size);
        cc_io::put_u64(hdr + 24, dcp_size);
        cc_io::put_u32(hdr + 32, (uint32_t) chunks_.size());
        cc_io::put_u32(hdr + 36, (uint32_t) states_.size());
        out.write((const char *) hdr, 48);

        for (size_t i = 0; i < chunks_.size(); i++) {
            uint8_t e[48];
            memset(e, 0, sizeof(e));
            const chunk & c = chunks_[i];
            cc_io::put_u32(e,       (uint32_t) i);
            cc_io::put_u16(e + 2,   c.key.region);
            cc_io::put_u16(e + 4,   c.key.layer);
            cc_io::put_u32(e + 8,   c.key.cell_lo);
            cc_io::put_u32(e + 12,  c.key.cell_hi);
            cc_io::put_u64(e + 16,  c.key.lineage_epoch);
            cc_io::put_u64(e + 24,  c.offset);
            cc_io::put_u64(e + 32,  c.len);
            cc_io::put_u32(e + 40,  c.crc);
            cc_io::put_u32(e + 44,  c.refs);
            out.write((const char *) e, 48);
        }
        for (const auto & it : states_) {
            std::array<uint8_t, 32> fp;
            memcpy(fp.data(), it.first.data(), 32);
            out.write((const char *) fp.data(), 32);
            uint8_t meta[16];
            cc_io::put_u64(meta,      state_tokens_[it.first]);
            cc_io::put_u64(meta + 8,  it.second.size());
            out.write((const char *) meta, 16);
            for (uint64_t id : it.second) {
                uint8_t ib[8];
                cc_io::put_u64(ib, id);
                out.write((const char *) ib, 8);
            }
        }
        out.flush();
        if (!out) {
            return;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, dci_path_, ec);
        dci_valid_ = !ec;
    }

    // reopen the store as read-only (lock lost later); keeps existing cache.dcp
    void reopen_readonly() {
        if (f_) {
            fclose(f_);
            f_ = nullptr;
        }
        f_ = fopen(dcp_path_.c_str(), "rb");
        writable_ = false;
    }

    void open_lock() {
#ifdef _WIN32
        lock_fd_ = -1; // no advisory lock on Windows
#else
        lock_fd_ = ::open(lock_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (lock_fd_ >= 0 && ::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
            ::close(lock_fd_);
            lock_fd_ = -1;
        }
#endif
    }

    void close_lock() {
#ifdef _WIN32
        if (lock_fd_ >= 0) {
            ::close(lock_fd_);
            lock_fd_ = -1;
        }
#else
        if (lock_fd_ >= 0) {
            ::flock(lock_fd_, LOCK_UN);
            ::close(lock_fd_);
            lock_fd_ = -1;
        }
#endif
    }

    // roll the open batch back to the committed state (caller holds io_mtx_)
    void abort_batch_locked();

    // clear the log and rebuild an empty store (caller holds io_mtx_)
    void reset_log_locked();
};

prompt_cache_chunk_store::prompt_cache_chunk_store()
    : p_(std::make_unique<impl>()) {
}

prompt_cache_chunk_store::~prompt_cache_chunk_store() {
    close();
}

bool prompt_cache_chunk_store::open(const std::string & dir, uint64_t base_lineage) {
    p_ = std::unique_ptr<impl>(new impl());
    impl & I = *p_;

    I.dir_          = dir;
    I.dcp_path_     = (std::filesystem::path(dir) / "cache.dcp").string();
    I.dci_path_     = (std::filesystem::path(dir) / "cache.dci").string();
    I.lock_path_    = (std::filesystem::path(dir) / "cache.lock").string();
    I.base_lineage_ = base_lineage;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    I.open_lock();
    const bool have_lock = I.lock_fd_ >= 0;
    const bool exists    = std::filesystem::exists(I.dcp_path_, ec);

    // open the log for read, and for write when we hold the lock
    I.f_ = fopen(I.dcp_path_.c_str(), have_lock ? "r+b" : "rb");
    if (!I.f_ && have_lock) {
        I.f_ = fopen(I.dcp_path_.c_str(), "w+b");
    }
    if (!I.f_) {
        SRV_ERR("chunk store: cannot open %s", I.dcp_path_.c_str());
        I.close_lock();
        p_.reset();
        return false;
    }
    I.writable_ = have_lock;

    // base lineage mismatch = config/fingerprint change: reset the store;
    // an unreadable header is treated the same
    bool base_ok = false;
    if (exists) {
        uint8_t hdr[24];
        if (I.pread_at(0, hdr, 24)) {
            base_ok = cc_io::get_u32(hdr)     == 0x54514331 &&
                      cc_io::get_u32(hdr + 4) == 1 &&
                      cc_io::get_u64(hdr + 8) == base_lineage;
            if (base_ok) {
                I.committed_gen_ = cc_io::get_u64(hdr + 16);
            }
        }
    }
    if (exists && !base_ok && !have_lock) {
        SRV_WRN("chunk store %s: foreign, open denied", I.dcp_path_.c_str());
        I.close_lock();
        p_.reset();
        return false;
    }
    if (I.writable_ && (!exists || !base_ok)) {
        I.reset_log_locked();
    }

    // index: fast path from cache.dci, else full scan of cache.dcp
    if (!I.load_index_fast()) {
        if (!I.rebuild_from_scan()) {
            SRV_WRN("chunk store %s: corrupt log, resetting under lock", I.dcp_path_.c_str());
            if (I.writable_) {
                I.reset_log_locked();
            } else {
                I.close_lock();
                p_.reset();
                return false;
            }
        }
    }
    return true;
}

void prompt_cache_chunk_store::close() {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    {
        std::lock_guard<std::mutex> lk(I.io_mtx_);
        if (I.writable_) {
            I.write_index_locked();
        }
        if (I.f_) {
            fclose(I.f_);
            I.f_ = nullptr;
        }
    }
    I.close_lock();
    p_.reset();
}

bool prompt_cache_chunk_store::store_enabled() {
    const char * v = getenv("TURBO_DISK_CHUNK_STORE");
    return !(v && strcmp(v, "0") == 0);
}

bool prompt_cache_chunk_store::is_open() const {
    return p_ && p_->f_;
}

bool prompt_cache_chunk_store::wants_dedup() const {
    return p_ ? p_->want_dedup_ : true;
}

uint64_t prompt_cache_chunk_store::batch_appended_bytes() const {
    return p_ ? p_->batch_appended_ : 0;
}

bool prompt_cache_chunk_store::begin_commit() {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (!I.writable_ || I.queued_gen_ != 0) {
        return false;
    }
    // append point is the true end of the log (a batch starts after every
    // committed record, including trailing state/tombstone records)
    uint64_t batch_start = 0;
    if (!I.file_size_of(I.dcp_path_, batch_start)) {
        return false;
    }
    I.batch_start_  = batch_start;
    I.batch_chunks_ = I.chunks_.size();
    I.batch_appended_ = 0;
    I.batch_states_.clear();
    I.batch_tombs_.clear();
    I.queued_gen_ = I.committed_gen_ + 1;
    return true;
}

int64_t prompt_cache_chunk_store::put_chunk(const chunk_key & key, const uint8_t * data, size_t len) {
    if (!p_) {
        return -1;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (I.queued_gen_ == 0 || len > MAX_REC_LEN) {
        return -1;
    }
    if (I.want_dedup_) {
        const auto it = I.by_key_.find(key);
        if (it != I.by_key_.end()) {
            return (int64_t) it->second;   // no append, no ref bump (pending state does)
        }
    }
    uint64_t p_off = 0;
    if (!I.append_record(CHUNK_REC_CHUNK, key, I.queued_gen_, data, len, p_off)) {
        // disk full guard: roll the batch back, caller retries on eviction
        I.abort_batch_locked();
        return -1;
    }
    I.batch_appended_ += I.rec_size(len);
    impl::chunk c;
    c.key   = key;
    c.offset = p_off;
    c.len    = (uint64_t) len;
    c.crc    = crc32c(data, len);
    c.refs   = 0;
    const uint64_t id = I.chunks_.size();
    I.chunks_.push_back(c);
    I.by_key_[key] = id;
    return (int64_t) id;
}

void prompt_cache_chunk_store::impl::abort_batch_locked() {
    if (queued_gen_ == 0) {
        return;
    }
    queued_gen_ = 0;
    batch_appended_ = 0;
    chunks_.resize(batch_chunks_);
    // drop any states queued this batch (tombs were never tracked in-memory)
    for (const auto & fp : batch_states_) {
        states_.erase(fp_str(fp));
        state_tokens_.erase(fp_str(fp));
    }
    batch_states_.clear();
    batch_tombs_.clear();
    truncate_to(batch_start_);
    // rebuild the key oracle from the surviving chunks
    by_key_.clear();
    for (size_t i = 0; i < chunks_.size(); i++) {
        by_key_[chunks_[i].key] = i;
    }
    recompute_refs();
}

void prompt_cache_chunk_store::impl::reset_log_locked() {
    if (f_) {
        fclose(f_);
        f_ = nullptr;
    }
    chunks_.clear();
    states_.clear();
    state_tokens_.clear();
    by_key_.clear();
    committed_gen_  = 0;
    queued_gen_     = 0;
    batch_start_    = 0;
    batch_chunks_   = 0;
    batch_appended_ = 0;
    batch_states_.clear();
    batch_tombs_.clear();
    dci_valid_ = false;
    f_ = fopen(dcp_path_.c_str(), "w+b");
    if (!f_) {
        return;
    }
    set_header_gen(0);
    sync_file();
    std::error_code ec;
    std::filesystem::remove(dci_path_, ec);   // stale index: next open rescans
}

bool prompt_cache_chunk_store::put_state(const state_ref & st) {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (I.queued_gen_ == 0) {
        return false;
    }
    // every referenced chunk must exist and predate this state
    for (uint64_t id : st.chunks) {
        if (id >= I.chunks_.size()) {
            I.abort_batch_locked();
            return false;
        }
    }
    std::vector<uint8_t> payload(32 + 8 + 8 + 8 * st.chunks.size());
    memcpy(payload.data(), st.fp.data(), 32);
    cc_io::put_u64(payload.data() + 32, st.n_tokens);
    cc_io::put_u64(payload.data() + 40, st.chunks.size());
    for (size_t i = 0; i < st.chunks.size(); i++) {
        cc_io::put_u64(payload.data() + 48 + 8 * i, st.chunks[i]);
    }
    uint64_t p_off = 0;
    chunk_key key;
    if (!I.append_record(CHUNK_REC_STATE, key, I.queued_gen_, payload.data(), payload.size(), p_off)) {
        I.abort_batch_locked();
        return false;
    }
    I.batch_appended_ += I.rec_size(payload.size());
    for (uint64_t id : st.chunks) {
        I.chunks_[id].refs++;
    }
    I.states_[I.fp_str(st.fp)] = st.chunks;
    I.state_tokens_[I.fp_str(st.fp)] = st.n_tokens;
    I.batch_states_.push_back(st.fp);
    return true;
}

bool prompt_cache_chunk_store::commit() {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (I.queued_gen_ == 0) {
        return false;
    }
    if (!I.sync_file()) {
        I.abort_batch_locked();
        return false;
    }
    I.set_header_gen(I.queued_gen_);
    if (!I.sync_file()) {
        I.abort_batch_locked();
        return false;
    }
    I.committed_gen_ = I.queued_gen_;
    I.queued_gen_    = 0;
    I.batch_start_   = 0;
    I.batch_states_.clear();
    I.batch_tombs_.clear();
    I.dci_valid_ = false;   // cache.dci is now stale until write_index()
    return true;
}

void prompt_cache_chunk_store::abort_commit() {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    I.abort_batch_locked();
}

bool prompt_cache_chunk_store::get_chunk(uint64_t id, std::vector<uint8_t> & out) const {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (id >= I.chunks_.size()) {
        return false;
    }
    const impl::chunk & c = I.chunks_[id];
    uint8_t rh[CHUNK_REC_HDR];
    if (!I.pread_at((int64_t) c.offset - CHUNK_REC_HDR, rh, CHUNK_REC_HDR)) {
        return false;
    }
    out.resize(c.len);
    if (c.len > 0 && !I.pread_at((int64_t) c.offset, out.data(), c.len)) {
        return false;
    }
    if (cc_io::get_u16(rh) != CHUNK_REC_CHUNK ||
        cc_io::get_u64(rh + 32) != c.len ||
        cc_io::get_u32(rh + 40) != (crc32c(rh, 40) ^ crc32c(out.data(), c.len))) {
        return false;
    }
    return true;
}

bool prompt_cache_chunk_store::get_state(const std::array<uint8_t, 32> & fp, state_ref & out) const {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    const std::string fp_s((const char *) fp.data(), 32);
    const auto it = I.states_.find(fp_s);
    if (it == I.states_.end()) {
        return false;
    }
    out.fp      = fp;
    out.chunks  = it->second;
    out.n_tokens = I.state_tokens_[fp_s];
    return true;
}

bool prompt_cache_chunk_store::has_state(const std::array<uint8_t, 32> & fp) const {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    return I.states_.find(std::string((const char *) fp.data(), 32)) != I.states_.end();
}

void prompt_cache_chunk_store::drop_state(const std::array<uint8_t, 32> & fp) {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    const std::string fp_s((const char *) fp.data(), 32);
    const auto it = I.states_.find(fp_s);
    if (it == I.states_.end()) {
        return;
    }
    if (!I.writable_) {
        return;
    }
    // tombstone must commit without a surrounding batch
    if (I.queued_gen_ != 0) {
        SRV_WRN("chunk store: drop_state ignored while a batch is open (gen %lu)", (unsigned long) I.queued_gen_);
        return;
    }
    std::vector<uint8_t> payload(32);
    memcpy(payload.data(), fp.data(), 32);
    chunk_key key;
    uint64_t p_off = 0;
    if (I.append_record(CHUNK_REC_TOMB, key, I.committed_gen_ + 1, payload.data(), payload.size(), p_off)) {
        I.set_header_gen(I.committed_gen_ + 1);
        I.sync_file();
        I.committed_gen_++;
        I.dci_valid_ = false;
        // unref and drop
        for (uint64_t id : it->second) {
            if (id < I.chunks_.size() && I.chunks_[id].refs > 0) {
                I.chunks_[id].refs--;
            }
        }
        I.states_.erase(fp_s);
        I.state_tokens_.erase(fp_s);
        I.recompute_refs();   // keep refs source-of-truth (tombstones may collapse)
    }
}

void prompt_cache_chunk_store::drop_evicted(const std::vector<std::array<uint8_t, 32>> & fps) {
    if (!p_ || fps.empty()) {
        return;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (!I.writable_ || I.queued_gen_ != 0) {
        return;
    }
    size_t n_dropped = 0;
    for (const auto & fp : fps) {
        const std::string fp_s((const char *) fp.data(), 32);
        if (I.states_.find(fp_s) == I.states_.end()) {
            continue;
        }
        // same tombstone as drop_state: durable so a crash-reload keeps the
        // eviction, and refcounts drop so compact can reclaim the chunks
        std::vector<uint8_t> payload(32);
        memcpy(payload.data(), fp.data(), 32);
        chunk_key key;
        uint64_t p_off = 0;
        if (I.append_record(CHUNK_REC_TOMB, key, I.committed_gen_ + 1, payload.data(), payload.size(), p_off)) {
            I.set_header_gen(I.committed_gen_ + 1);
            I.sync_file();
            I.committed_gen_++;
            I.dci_valid_ = false;
            for (uint64_t id : I.states_[fp_s]) {
                if (id < I.chunks_.size() && I.chunks_[id].refs > 0) {
                    I.chunks_[id].refs--;
                }
            }
            I.states_.erase(fp_s);
            I.state_tokens_.erase(fp_s);
            I.recompute_refs();
            n_dropped++;
        }
    }
    if (n_dropped > 0) {
        SRV_INF(" - disk prompt cache: chunk store dropped %zu evicted states\n", n_dropped);
    }
}

void prompt_cache_chunk_store::compact_if_stale() {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    // read stats without the lock (n_dead/n_chunks are only written under io_mtx_
    // by compact/drop_state, and stale reads are harmless here)
    uint64_t n_dead = 0, n_total = 0;
    {
        std::lock_guard<std::mutex> lk(I.io_mtx_);
        n_total = I.chunks_.size();
        for (const auto & c : I.chunks_) {
            if (c.refs == 0) {
                n_dead++;
            }
        }
    }
    const uint64_t n_live = n_total - n_dead;
    if (n_live == 0 || n_dead * 100 / n_live <= 35) {
        return;
    }
    // compact() acquires io_mtx_ internally; no lock held here
    compact();
}

uint64_t prompt_cache_chunk_store::compact() {
    if (!p_) {
        return 0;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (!I.writable_ || I.queued_gen_ != 0) {
        return 0;
    }
    uint64_t before = 0;
    if (!I.file_size_of(I.dcp_path_, before)) {
        return 0;
    }
    uint64_t after = 0;
    {
        // rewrite only refcount>0 chunks + states, in id order -> new ids
        std::vector<int64_t> remap(I.chunks_.size(), -1);
        std::vector<const impl::chunk *> keep;
        for (size_t i = 0; i < I.chunks_.size(); i++) {
            if (I.chunks_[i].refs > 0) {
                remap[i] = (int64_t) keep.size();
                keep.push_back(&I.chunks_[i]);
            }
        }
        std::vector<uint64_t> new_offs(keep.size());
        const std::string tmp = I.dcp_path_ + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) {
                return 0;
            }
            uint8_t hdr[24];
            cc_io::put_u32(hdr,      0x54514331);
            cc_io::put_u32(hdr + 4,  1);
            cc_io::put_u64(hdr + 8,  I.base_lineage_);
            cc_io::put_u64(hdr + 16, I.committed_gen_ + 1);
            out.write((const char *) hdr, 24);

            uint64_t new_off = 24;
            for (size_t i = 0; i < keep.size(); i++) {
                const impl::chunk * c = keep[i];
                std::vector<uint8_t> payload(c->len);
                if (c->len > 0 && !I.pread_at((int64_t) c->offset, payload.data(), c->len)) {
                    return 0;
                }
                std::vector<uint8_t> h(CHUNK_REC_HDR);
                cc_io::put_u16(h.data(), CHUNK_REC_CHUNK);
                cc_io::put_u16(h.data() + 2, c->key.region);
                cc_io::put_u32(h.data() + 4, c->key.layer);
                cc_io::put_u32(h.data() + 8, c->key.cell_lo);
                cc_io::put_u32(h.data() + 12, c->key.cell_hi);
                cc_io::put_u64(h.data() + 16, c->key.lineage_epoch);
                cc_io::put_u64(h.data() + 24, I.committed_gen_ + 1);
                cc_io::put_u64(h.data() + 32, c->len);
                cc_io::put_u32(h.data() + 40, crc32c(h.data(), 40) ^ crc32c(payload.data(), c->len));
                cc_io::put_u32(h.data() + 44, 0);
                std::vector<uint8_t> rec(CHUNK_REC_HDR + ((c->len + 7) & ~(size_t) 7));
                memcpy(rec.data(), h.data(), CHUNK_REC_HDR);
                memcpy(rec.data() + CHUNK_REC_HDR, payload.data(), c->len);
                out.write((const char *) rec.data(), rec.size());
                new_offs[i] = new_off + CHUNK_REC_HDR;
                new_off   += I.rec_size(c->len);
            }
            // states, remapped
            for (const auto & it : I.states_) {
                std::vector<uint64_t> ids;
                ids.reserve(it.second.size());
                bool ok = true;
                for (uint64_t id : it.second) {
                    const int64_t nid = id < remap.size() ? remap[id] : -1;
                    if (nid < 0) {
                        ok = false;
                        break;
                    }
                    ids.push_back((uint64_t) nid);
                }
                if (!ok) {
                    return 0;   // state references a dropped chunk: corrupt, bail
                }
                std::array<uint8_t, 32> fp;
                memcpy(fp.data(), it.first.data(), 32);
                std::vector<uint8_t> payload(32 + 8 + 8 + 8 * ids.size());
                memcpy(payload.data(), fp.data(), 32);
                cc_io::put_u64(payload.data() + 32, I.state_tokens_[it.first]);
                cc_io::put_u64(payload.data() + 40, ids.size());
                for (size_t k = 0; k < ids.size(); k++) {
                    cc_io::put_u64(payload.data() + 48 + 8 * k, ids[k]);
                }
                std::vector<uint8_t> h(CHUNK_REC_HDR);
                h[0] = (uint8_t) CHUNK_REC_STATE;
                cc_io::put_u16(h.data() + 2, 0);
                cc_io::put_u32(h.data() + 8, 0);
                cc_io::put_u32(h.data() + 12, 0);
                cc_io::put_u64(h.data() + 16, 0);
                cc_io::put_u64(h.data() + 24, I.committed_gen_ + 1);
                cc_io::put_u64(h.data() + 32, payload.size());
                cc_io::put_u32(h.data() + 40, crc32c(h.data(), 40) ^ crc32c(payload.data(), payload.size()));
                cc_io::put_u32(h.data() + 44, 0);
                std::vector<uint8_t> rec(CHUNK_REC_HDR + ((payload.size() + 7) & ~(size_t) 7));
                memcpy(rec.data(), h.data(), CHUNK_REC_HDR);
                memcpy(rec.data() + CHUNK_REC_HDR, payload.data(), payload.size());
                out.write((const char *) rec.data(), rec.size());
            }
            out.flush();
            if (!out) {
                return 0;
            }
        }
        std::error_code ec;
        std::filesystem::rename(tmp, I.dcp_path_, ec);
        if (ec) {
            return 0;
        }
        if (I.f_) {
            fclose(I.f_);
        }
        I.f_ = fopen(I.dcp_path_.c_str(), "r+b");
        if (!I.f_) {
            return 0;
        }
        I.sync_file();

        // commit the compacted generation
        I.committed_gen_++;
        // rebuild the in-memory index with the new ids (keep no longer valid)
        std::vector<impl::chunk> new_chunks;
        new_chunks.reserve(keep.size());
        for (size_t i = 0; i < keep.size(); i++) {
            impl::chunk c = *keep[i];
            c.offset = new_offs[i];
            c.refs   = 0;
            new_chunks.push_back(c);
        }
        I.chunks_.swap(new_chunks);
        I.by_key_.clear();
        for (size_t i = 0; i < I.chunks_.size(); i++) {
            I.by_key_[I.chunks_[i].key] = i;
        }
        // states still carry pre-compact ids: remap them to the new layout
        for (auto & it : I.states_) {
            for (uint64_t & id : it.second) {
                id = (uint64_t) remap[id];
            }
        }
        I.recompute_refs();

        if (!I.file_size_of(I.dcp_path_, after)) {
            after = before;
        }
        I.write_index_locked();
    }
    return before > after ? before - after : 0;
}

void prompt_cache_chunk_store::write_index() {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (I.writable_) {
        I.write_index_locked();
    }
}

prompt_cache_chunk_store::stats prompt_cache_chunk_store::get_stats() const {
    stats s;
    if (!p_) {
        return s;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    s.n_chunks = I.chunks_.size();
    s.n_states = I.states_.size();
    for (const auto & c : I.chunks_) {
        if (c.refs == 0) {
            s.n_dead++;
        }
        s.payload_bytes += c.len;
    }
    uint64_t dcp = 0, dci = 0;
    I.file_size_of(I.dcp_path_, dcp);
    I.file_size_of(I.dci_path_, dci);
    s.store_bytes = dcp + dci;
    return s;
}

uint64_t prompt_cache_chunk_store::generation() const {
    return p_ ? p_->committed_gen_ : 0;
}

uint32_t prompt_cache_chunk_store::refcount(uint64_t id) const {
    if (!p_ || id >= p_->chunks_.size()) {
        return 0;
    }
    return p_->chunks_[id].refs;
}

std::vector<uint64_t> prompt_cache_chunk_store::cell_lineages(uint64_t base, const int32_t * tokens, size_t n_tokens) {
    // chain over 128-token cells: h_0 = sha256(base), h_i = sha256(h_prev || cell tokens)
    std::vector<uint64_t> out;
    out.reserve((n_tokens + 127) / 128);
    uint8_t h[32];
    {
        uint8_t b[8];
        cc_io::put_u64(b, base);
        dpc_sha256_raw(b, 8, h);
    }
    for (size_t off = 0; off < n_tokens; off += 128) {
        const size_t n = std::min<size_t>(128, n_tokens - off);
        std::vector<uint8_t> buf(32 + 4 * n);
        memcpy(buf.data(), h, 32);
        for (size_t i = 0; i < n; i++) {
            cc_io::put_u32(buf.data() + 32 + 4 * i, (uint32_t) tokens[off + i]);
        }
        dpc_sha256_raw(buf.data(), buf.size(), h);
        out.push_back(cc_io::get_u64(h));
    }
    return out;
}

// slice a llama_state_seq blob into fixed 128-token cell sheets, in canonical
// (blob-order) slice order. the slices tile [0, blob_size) contiguously, so
// reassembly by concatenation is byte-exact for any accepted blob. acceptance
// requires the single-stream write-through layout the store persists; anything
// else - a draft stream, a non-consecutive or multi-sequence meta section, a
// transposed V, an out-of-range header, or an ambiguous meta stride - returns
// false and the caller keeps the legacy .dpc writer.
bool slice_state_blob(const uint8_t * blob, size_t blob_size, size_t n_tokens,
                      std::vector<state_blob_slice> & out) {
    constexpr uint32_t SEQ_MAGIC = 0xaf143cd8;
    constexpr uint64_t MAX_ROW   = 1ull << 24;
    constexpr uint32_t CELL      = 128;
    out.clear();

    if (blob_size < 24) {
        return false;
    }
    if (cc_io::get_u32(blob) != SEQ_MAGIC ||
        cc_io::get_u32(blob + 8) != 1 ||
        cc_io::get_u32(blob + 12) != (uint32_t) n_tokens) {
        return false;
    }

    // the single stream's tree section ([16, meta_start)) holds its own
    // n_tokens/n_cur; the KV meta rows begin after it
    const size_t meta_start = 16 + (size_t) 8 * 1;

    // meta stride: 12 bytes/cell without M-RoPE ext, 20 with ext. the right
    // stride makes every cell a single-sequence entry (n_seq_id == 1) with
    // pos == its row index, which is exactly the contiguous write-through
    // layout the store requires. acceptance is exclusive: two candidates that
    // validate are a layout the walk cannot prove, none is unsupported
    size_t meta_stride = 0;
    size_t d           = 0;
    const size_t strides[] = { 12, 20 };
    for (const size_t cand : strides) {
        size_t off = meta_start;
        bool   ok  = true;
        for (size_t i = 0; i < n_tokens && ok; ++i) {
            if (off + cand > blob_size) {
                ok = false;
                break;
            }
            // a write-through cell reads id == i, pos == i, single sequence
            if (cc_io::get_u32(blob + off) != (uint32_t) i ||
                cc_io::get_u32(blob + off + 4) != (uint32_t) i ||
                cc_io::get_u32(blob + off + 8) != 1) {
                ok = false;
                break;
            }
            off += cand;
        }
        if (ok) {
            if (meta_stride != 0) {
                return false;
            }
            meta_stride = cand;
            d           = off;
        }
    }
    if (meta_stride == 0) {
        return false;
    }

    if (d + 8 > blob_size) {
        return false;
    }
    const uint32_t v_trans = cc_io::get_u32(blob + d);
    const uint32_t n_layer = cc_io::get_u32(blob + d + 4);
    if (v_trans != 0 || n_layer == 0 || n_layer > 4096) {
        return false;
    }

    // K blocks: one 12 B header then n_tokens rows of row_size bytes, per layer.
    // the header must be a plausible ggml type and the data block must fit
    size_t k_end = d + 8;
    std::vector<uint64_t> k_off;
    std::vector<uint64_t> k_row;
    k_off.reserve(n_layer);
    k_row.reserve(n_layer);
    for (uint32_t l = 0; l < n_layer; ++l) {
        if (k_end + 12 > blob_size) {
            return false;
        }
        const int32_t type = (int32_t) cc_io::get_u32(blob + k_end);
        const uint64_t row = cc_io::get_u64(blob + k_end + 4);
        if (type < 0 || type >= (int32_t) GGML_TYPE_COUNT || row == 0 || row > MAX_ROW ||
            (uint64_t) n_tokens * row > blob_size - k_end - 12) {
            return false;
        }
        k_off.push_back(k_end);
        k_row.push_back(row);
        k_end += 12 + (uint64_t) n_tokens * row;
    }

    // V blocks: present for non-MLA models, absent for MLA. parse them
    // best-effort with the same header checks; any failure turns the whole
    // remainder into one opaque tail chunk, which reassembles equally
    // byte-exactly on restore
    std::vector<uint64_t> v_off;
    std::vector<uint64_t> v_row;
    size_t v_end = k_end;
    for (uint32_t l = 0; l < n_layer; ++l) {
        if (v_end + 12 > blob_size) {
            v_off.clear();
            v_row.clear();
            break;
        }
        const int32_t type = (int32_t) cc_io::get_u32(blob + v_end);
        const uint64_t row = cc_io::get_u64(blob + v_end + 4);
        if (type < 0 || type >= (int32_t) GGML_TYPE_COUNT || row == 0 || row > MAX_ROW ||
            (uint64_t) n_tokens * row > blob_size - v_end - 12) {
            v_off.clear();
            v_row.clear();
            break;
        }
        v_off.push_back(v_end);
        v_row.push_back(row);
        v_end += 12 + (uint64_t) n_tokens * row;
    }
    const size_t attn_end = v_off.size() == n_layer ? v_end : k_end;

    const uint32_t n_cells   = (uint32_t) ((n_tokens + CELL - 1) / CELL);
    const uint32_t n_tok32   = (uint32_t) n_tokens;

    // 1) fixed prefix [0, meta_start): magic, seq id, n_stream, stream tree
    out.push_back({ prompt_cache_chunk_store::REGION_HEADER, 0, 0, n_tok32, 0, (uint64_t) meta_start });

    // 2) meta rows [meta_start, d): one sheet per 128-token cell
    for (uint32_t c = 0; c < n_cells; ++c) {
        const uint32_t lo = c * CELL;
        const uint32_t hi = std::min(n_tok32, lo + CELL);
        out.push_back({ prompt_cache_chunk_store::REGION_META_ROW, 0, lo, hi,
                        (uint64_t) (meta_start + (size_t) lo * meta_stride),
                        (uint64_t) ((size_t) (hi - lo) * meta_stride) });
    }

    // 3) v_trans + n_layer table [d, d + 8); keyed apart from the prefix so the
    //    two header chunks never share a dedup identity
    out.push_back({ prompt_cache_chunk_store::REGION_HEADER, 0, 0, 0, (uint64_t) d, 8 });

    // 4) + 5) K and V cell sheets: the 12 B block header folds into cell 0 so
    //    the sheet ranges stay contiguous byte ranges of the blob
    auto emit_cells = [&](prompt_cache_chunk_store::region region, uint16_t layer, uint64_t block_off, uint64_t row) {
        const uint64_t cell_bytes = (uint64_t) CELL * row;
        for (uint32_t c = 0; c < n_cells; ++c) {
            const uint32_t lo = c * CELL;
            const uint32_t hi = std::min(n_tok32, lo + CELL);
            const uint64_t n  = (uint64_t) (hi - lo);
            out.push_back({ region, layer, lo, hi,
                            block_off + (c == 0 ? 0 : 12) + (uint64_t) c * cell_bytes,
                            n * row + (c == 0 ? 12 : 0) });
        }
    };
    for (uint32_t l = 0; l < n_layer; ++l) {
        emit_cells(prompt_cache_chunk_store::REGION_K, (uint16_t) l, k_off[l], k_row[l]);
    }
    for (uint32_t l = 0; l < v_off.size(); ++l) {
        emit_cells(prompt_cache_chunk_store::REGION_V, (uint16_t) l, v_off[l], v_row[l]);
    }

    // 6) recurrent-state tail [attn_end, blob_size); absent when the model has
    //    no recurrent head
    if (attn_end < blob_size) {
        out.push_back({ prompt_cache_chunk_store::REGION_TAIL, 0, 0, n_tok32,
                        (uint64_t) attn_end, (uint64_t) (blob_size - attn_end) });
    }
    return true;
}

// n_*(): not used by the store contract itself but kept for parity with the
// legacy per-state index that callers build from get_stats()

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
            if (entry.path().extension() == ".dpc" || entry.path().extension() == ".dts" ||
                entry.path().extension() == ".ccn" ||
                entry.path().extension() == ".dcp" || entry.path().extension() == ".dci") {
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

        // disk-full write guard: same rule as the write-through path. evict-first
        // from the shared pool, then skip the entry if the volume is still tight
        const llama_tokens tokens = state.prompt.tokens.get_text_tokens();
        const std::string fname = dpc_sha256_hex((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token)) + ".dpc";
        const std::string path  = (fs::path(dir) / fname).string();
        const std::string tmp   = path + ".tmp";

        {
            const fs::path root = fs::path(dir).parent_path();
            const uint64_t need = (uint64_t) state.size() + 64ull * 1024 * 1024;

            std::error_code ec_space;
            const auto free0 = fs::space(root, ec_space).available;
            if (!ec_space && free0 < need) {
                prune_disk_budget(dir);
                const auto free1 = fs::space(root, ec_space).available;
                if (ec_space || free1 < need) {
                    SRV_WRN(" - disk prompt cache: skipping save '%s' - %.3f MiB free (need %.3f MiB)\n",
                            fname.c_str(), (double) free1 / (1024.0 * 1024.0), (double) need / (1024.0 * 1024.0));
                    continue;
                }
            }
        }

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

    // the pool root is the parent of the per-preset dir (cache_disk_path/<name>)
    const fs::path root = fs::path(dir).parent_path();
    prune_pool(root.string(), dir, true);
}

void server_prompt_cache::disk_prune_dpc(const std::string & dir, std::vector<std::array<uint8_t, 32>> * evicted) const {
    namespace fs = std::filesystem;

    const fs::path root = fs::path(dir).parent_path();
    prune_pool(root.string(), dir, false, evicted);
}

void server_prompt_cache::prune_pool(const std::string & root, const std::string & own_dir, bool include_own_ccn,
                                    std::vector<std::array<uint8_t, 32>> * evicted) const {
    namespace fs = std::filesystem;

    if (disk_max_size == 0) {
        // 0 is documented as "no limit" (common.h); on a shared root that silently
        // unenforces the pool for every preset. warn once instead of asserting,
        // asserts are compiled out in Release
        if (!disk_no_limit_warned_ && fs::is_directory(root)) {
            std::error_code ec_w;
            for (const auto & sd : fs::directory_iterator(root, ec_w)) {
                if (sd.is_directory(ec_w)) {
                    disk_no_limit_warned_ = true;
                    SRV_WRN(" - disk prompt cache: disk_max_size == 0 (no limit) on a shared root '%s' - pool budget is not enforced\n", root.c_str());
                    break;
                }
            }
        }
        return;
    }

    struct pool_file {
        fs::path path;
        fs::file_time_type mtime;
        uint64_t size;
    };

    std::error_code ec;
    std::vector<pool_file> files;
    uint64_t used = 0;

    const fs::path own_path(own_dir);

    for (const auto & sd : fs::directory_iterator(root, ec)) {
        if (!sd.is_directory(ec)) {
            continue;
        }
        const bool is_own_dir = sd.path() == own_path;

        for (const auto & entry : fs::directory_iterator(sd.path(), ec)) {
            const auto ext = entry.path().extension();
            if (ext == ".dpc" || ext == ".dts") {
                const auto size = fs::file_size(entry.path(), ec);
                if (ec) {
                    continue;
                }
                used += (uint64_t) size;
                files.push_back({ entry.path(), fs::last_write_time(entry.path(), ec), (uint64_t) size });
            } else if (ext == ".ccn" && is_own_dir && include_own_ccn) {
                const auto size = fs::file_size(entry.path(), ec);
                if (ec) {
                    continue;
                }
                used += (uint64_t) size;
                files.push_back({ entry.path(), fs::last_write_time(entry.path(), ec), (uint64_t) size });
            } else if (ext == ".dcp" || ext == ".dci") {
                // chunk store files count toward the budget but are never evicted
                const auto size = fs::file_size(entry.path(), ec);
                if (!ec) {
                    used += (uint64_t) size;
                }
            }
        }
    }

    if (used > disk_max_size) {
        std::sort(files.begin(), files.end(), [](const pool_file & a, const pool_file & b) {
            return a.mtime < b.mtime;
        });

        for (const auto & file : files) {
            if (used <= disk_max_size) {
                break;
            }

            // extract the fingerprint from the .dts sidecar before removing it,
            // so the caller can drop the stale state from the chunk store
            if (evicted && file.path.extension() == ".dts") {
                FILE * f = fopen(file.path.string().c_str(), "rb");
                if (f) {
                    uint32_t magic = 0, version = 0, n_tokens = 0;
                    if (fread(&magic, 4, 1, f) == 1 && magic == DTS_MAGIC &&
                        fread(&version, 4, 1, f) == 1 && version == DTS_VERSION &&
                        fread(&n_tokens, 4, 1, f) == 1 && n_tokens > 0 && n_tokens <= 131072) {
                        // skip len_main(8) + len_drft(8) + n_ckpt(4) = 20 bytes
                        fseek(f, 20, SEEK_CUR);
                        std::vector<int32_t> tokens(n_tokens);
                        if (fread(tokens.data(), sizeof(int32_t), n_tokens, f) == n_tokens) {
                            std::array<uint8_t, 32> fp;
                            dpc_sha256_raw((const uint8_t *) tokens.data(), n_tokens * sizeof(int32_t), fp.data());
                            evicted->push_back(fp);
                        }
                    }
                    fclose(f);
                }
            }

            std::error_code ec_remove;
            fs::remove(file.path, ec_remove);
            if (ec_remove) {
                continue; // file still occupied budget, keep its size
            }
            used -= file.size;
        }

        SRV_INF(" - disk prompt cache: pruned shared pool '%s' to %.3f MiB (budget %.3f MiB)\n",
                root.c_str(), used / (1024.0 * 1024.0), disk_max_size / (1024.0 * 1024.0));
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

// base lineage for the chunk store: the low 8 bytes of sha256 of the prompt
// cache fingerprint. a config change changes the fingerprint, hence the
// lineage, which resets the store exactly like the .dpc/.ccn wipe
static uint64_t fingerprint_lineage(const std::string & fingerprint) {
    uint8_t h[32];
    dpc_sha256_raw((const uint8_t *) fingerprint.data(), fingerprint.size(), h);
    return cc_io::get_u64(h);
}

// keep the meta fingerprint current so later saves do not treat the directory
// as a config mismatch; shared by the legacy and chunk-store writers
static void disk_write_meta_file(const std::string & dir, const std::string & fingerprint) {
    namespace fs = std::filesystem;
    const std::string meta_path = (fs::path(dir) / "meta").string();
    const std::string meta_tmp  = meta_path + ".tmp";

    std::ofstream out(meta_tmp, std::ios::binary);
    out.write(fingerprint.data(), (std::streamsize) fingerprint.size());
    out.close();

    std::error_code ec;
    fs::rename(meta_tmp, meta_path, ec);
    if (ec) {
        SRV_WRN(" - disk prompt cache: failed to update meta: %s\n", ec.message().c_str());
    }
}

// write the <fp>.dts token-only index for a chunk-store state. called after the
// store batch commits, so a sidecar present always implies a durable state; a
// failed sidecar only costs the entry at restore time (state stays on disk).
static void disk_write_token_index(const std::string & dir, const std::string & fp_hex, const llama_tokens & tokens) {
    namespace fs = std::filesystem;
    const std::string dts_path = (fs::path(dir) / (fp_hex + ".dts")).string();
    const std::string dts_tmp  = dts_path + ".tmp";

    FILE * f = fopen(dts_tmp.c_str(), "wb");
    if (f == nullptr) {
        SRV_WRN(" - disk prompt cache: cannot create token index '%s': %s\n", dts_tmp.c_str(), strerror(errno));
        return;
    }

    const uint32_t n_tokens = (uint32_t) tokens.size();
    const uint64_t len_zero = 0;
    const uint32_t n_ckpt   = 0;

    bool ok = fwrite(&DTS_MAGIC,   4, 1, f) == 1;
    ok = ok && fwrite(&DTS_VERSION, 4, 1, f) == 1;
    ok = ok && fwrite(&n_tokens,    4, 1, f) == 1;
    ok = ok && fwrite(&len_zero,    8, 1, f) == 1;
    ok = ok && fwrite(&len_zero,    8, 1, f) == 1;
    ok = ok && fwrite(&n_ckpt,      4, 1, f) == 1;
    ok = ok && (n_tokens == 0 || fwrite(tokens.data(), sizeof(llama_token), n_tokens, f) == n_tokens);
    fclose(f);

    if (!ok) {
        SRV_WRN(" - disk prompt cache: failed to write token index '%s'\n", dts_path.c_str());
        std::error_code ec;
        fs::remove(dts_tmp, ec);
        return;
    }

    std::error_code ec;
    fs::rename(dts_tmp, dts_path, ec);
    if (ec) {
        SRV_WRN(" - disk prompt cache: failed to finalize token index '%s': %s\n", dts_path.c_str(), ec.message().c_str());
    }
}

bool server_prompt_cache::disk_store_write(disk_write_job & job, const llama_tokens & tokens) {
    namespace fs = std::filesystem;
    auto & state = *job.state;
    if (state.data.main.empty()) {
        return false;
    }

    const uint64_t base = fingerprint_lineage(job.fingerprint);

    // store disabled: stick to the legacy .dpc writer without ever touching
    // cache.dcp/.dci (one-pass rollback) or creating them on disk
    if (!prompt_cache_chunk_store::store_enabled()) {
        return false;
    }

    // open lazily (writer thread first, and later any restore on the reader
    // thread); a config change is a different base lineage and resets the store
    // like the .dpc/.ccn wipe
    {
        std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
        if (!disk_store_) {
            disk_store_ = std::make_unique<prompt_cache_chunk_store>();
        }
        if (!disk_store_->is_open() && !disk_store_->open(job.dir, base)) {
            SRV_WRN(" - disk prompt cache: chunk store unavailable in '%s', using .dpc writer\n", job.dir.c_str());
            return false;
        }
    }
    if (!disk_store_->wants_dedup()) {
        return false;
    }

    std::vector<state_blob_slice> slices;
    if (!slice_state_blob(state.data.main.data(), state.data.main.size(), tokens.size(), slices)) {
        return false;
    }

    const std::vector<uint64_t> lineages =
        prompt_cache_chunk_store::cell_lineages(base, tokens.data(), tokens.size());
    const uint64_t last_epoch = lineages.empty() ? base : lineages.back();

    // the epoch of a K/V/meta sheet is the token chain up to and including its
    // cell; the header/tail chunks span the whole state and carry the last one
    auto epoch_of = [&](const state_blob_slice & sl) -> uint64_t {
        switch (sl.region) {
        case prompt_cache_chunk_store::REGION_K:
        case prompt_cache_chunk_store::REGION_V:
        case prompt_cache_chunk_store::REGION_META_ROW:
            return lineages[sl.cell_lo / 128];
        default:
            return last_epoch;
        }
    };

    // one batch per save; abort leaves no trace. disk-full is checked on the
    // appended delta (dedup hits add nothing), then compacted and retried once
    auto write_batch = [&]() -> bool {
        if (!disk_store_->begin_commit()) {
            return false;
        }
        std::vector<uint64_t> ids;
        ids.reserve(slices.size());
        for (const auto & sl : slices) {
            prompt_cache_chunk_store::chunk_key key;
            key.region        = sl.region;
            key.layer         = sl.layer;
            key.cell_lo       = sl.cell_lo;
            key.cell_hi       = sl.cell_hi;
            key.lineage_epoch = epoch_of(sl);
            const int64_t id = disk_store_->put_chunk(key, state.data.main.data() + sl.byte_off, sl.byte_len);
            if (id < 0) {
                disk_store_->abort_commit();
                return false;
            }
            ids.push_back((uint64_t) id);
        }

        // never commit when the appended bytes would push the volume full
        {
            std::error_code ecs;
            const uint64_t free = fs::space(fs::path(job.dir).parent_path(), ecs).available;
            const uint64_t need = disk_store_->batch_appended_bytes() + 64ull * 1024 * 1024;
            if (!ecs && free < need) {
                disk_store_->abort_commit();
                return false;
            }
        }

        prompt_cache_chunk_store::state_ref st;
        dpc_sha256_raw((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token), st.fp.data());
        st.n_tokens = tokens.size();
        st.chunks   = std::move(ids);
        if (!disk_store_->put_state(st) || !disk_store_->commit()) {
            disk_store_->abort_commit();
            return false;
        }
        return true;
    };

    if (!write_batch()) {
        disk_store_->compact();
        if (!write_batch()) {
            SRV_WRN(" - disk prompt cache: chunk store write failed after compact in '%s', using .dpc writer\n", job.dir.c_str());
            return false;
        }
    }

    disk_write_meta_file(job.dir, job.fingerprint);

    // token-only index so a later boot can build a restore candidate for this
    // state without scanning the store log
    disk_write_token_index(job.dir, dpc_sha256_hex((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token)), tokens);

    const size_t state_bytes = state.size();
    SRV_INF(" - disk prompt cache: wrote through state '%zu' tokens to chunk store (%.3f MiB, %zu slices)\n",
            tokens.size(), (double) state_bytes / (1024.0 * 1024.0), slices.size());
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

    // chunk-store write-through: slice the KV blob into fixed cells and commit
    // one append-only batch. a sliceable state + usable store replaces the .dpc
    // write; a draft model, an unslicable blob, or a failed batch keeps the
    // legacy writer below (its own disk-full guard then decides)
    if (state.data.drft.empty() && disk_store_write(job, tokens)) {
        // enforce pool budget after a store write; track .dts evictions so the
        // chunk store can drop the corresponding states
        std::vector<std::array<uint8_t, 32>> evicted;
        disk_prune_dpc(dir, &evicted);
        {
            std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
            if (disk_store_) {
                disk_store_->drop_evicted(evicted);
                disk_store_->compact_if_stale();
            }
        }
        return;
    }

    const std::string fname = dpc_sha256_hex((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token)) + ".dpc";
    const std::string path  = (fs::path(dir) / fname).string();
    const std::string tmp   = path + ".tmp";

    // disk-full write guard: never attempt a write into a full volume. the pool
    // root is the parent of the per-preset dir; when the free space is below the
    // write need plus a safety margin, evict-first from the shared pool and only
    // then write; if still insufficient, skip the write entirely
    {
        const fs::path root = fs::path(dir).parent_path();
        const uint64_t need = (uint64_t) state.size() + 64ull * 1024 * 1024; // entry + 64 MiB headroom

        std::error_code ec_space;
        const auto free0 = fs::space(root, ec_space).available;
        if (!ec_space && free0 < need) {
            std::vector<std::array<uint8_t, 32>> evicted;
            disk_prune_dpc(dir, &evicted);
            if (!evicted.empty()) {
                std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
                if (disk_store_) {
                    disk_store_->drop_evicted(evicted);
                }
            }
            const auto free1 = fs::space(root, ec_space).available;
            if (ec_space || free1 < need) {
                SRV_WRN(" - disk prompt cache: skipping write-through '%s' - %.3f MiB free (need %.3f MiB)\n",
                        fname.c_str(), (double) free1 / (1024.0 * 1024.0), (double) need / (1024.0 * 1024.0));
                return;
            }
        }
    }

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
    {
        std::vector<std::array<uint8_t, 32>> evicted;
        disk_prune_dpc(dir, &evicted);
        if (!evicted.empty()) {
            std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
            if (disk_store_) {
                disk_store_->drop_evicted(evicted);
            }
        }
    }

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
    disk_write_meta_file(dir, fingerprint);

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
    namespace fs = std::filesystem;

    // scan the index under lock; file I/O happens outside the lock so the writer
    // thread can keep pushing entries while a slow restore runs. retry when the
    // best entry's file went stale so a dead LRU head cannot shadow live ones
    while (true) {
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

            std::vector<std::pair<size_t, size_t>> cands;
            cands.reserve(disk_entries.size());

            for (size_t i = 0; i < disk_entries.size(); ++i) {
                const auto & entry = disk_entries[i];

                const int lcp_cur = entry.tokens.get_common_prefix(tokens_new);

                const float f_keep_cur = entry.tokens.size() > 0 ? float(lcp_cur) / entry.tokens.size() : -1.0f;
                const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

                SRV_TRC(" - disk prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n",
                        entry.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

                cands.push_back({ entry.tokens.size(), (size_t) lcp_cur });
            }

            const size_t idx_best = select_cache_candidate(cands, (size_t) lcp_base);

            if (idx_best == SIZE_MAX) {
                return true;
            }

            const float f_keep_cur = float(cands[idx_best].second) / cands[idx_best].first;
            const float f_sim_cur  = float(cands[idx_best].second) / tokens_new.size();

            best_path  = disk_entries[idx_best].path;
            best_fname = disk_entries[idx_best].fname;
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
        }

        SRV_TRC(" - disk prompt cache: found entry '%s' with f_keep = %.3f, f_sim = %.3f\n", best_fname.c_str(), f_keep_best, f_sim_best);

        server_prompt_cache_state state;
        {
            FILE * f = fopen(best_path.c_str(), "rb");
            if (f == nullptr) {
                SRV_WRN(" - disk prompt cache: cannot open '%s', dropping stale entry\n", best_path.c_str());

                // prune the dead entry from the in-memory index and re-select
                std::lock_guard<std::mutex> lock(disk_index_mtx_);
                disk_entries.erase(
                    std::remove_if(disk_entries.begin(), disk_entries.end(),
                        [&](const disk_entry & e) { return e.path == best_path; }),
                    disk_entries.end());

                continue;
            }

            const auto drop_stale = [&]() {
                std::lock_guard<std::mutex> lock(disk_index_mtx_);
                disk_entries.erase(
                    std::remove_if(disk_entries.begin(), disk_entries.end(),
                        [&](const disk_entry & e) { return e.path == best_path; }),
                    disk_entries.end());
            };

            uint32_t magic   = 0;
            uint32_t version = 0;
            bool ok = fread(&magic, 4, 1, f) == 1 && fread(&version, 4, 1, f) == 1;

            // store-backed entry: the KV lives in cache.dcp as lineage-chained
            // cells. reassemble the blob from its chunks; a fresh full restore
            // starts at position 0 and reuses via LCP, so the token-only sidecar
            // needs no checkpoints
            if (ok && magic == DTS_MAGIC) {
                uint32_t n_tokens = 0;
                uint64_t len_main = 0;
                uint64_t len_drft = 0;
                uint32_t n_ckpt   = 0;

                ok = fread(&n_tokens, 4, 1, f) == 1 && fread(&len_main, 8, 1, f) == 1 &&
                     fread(&len_drft, 8, 1, f) == 1 && fread(&n_ckpt, 4, 1, f) == 1 &&
                     version == DTS_VERSION && n_tokens <= DPC_MAX_TOKENS &&
                     len_main == 0 && len_drft == 0 && n_ckpt == 0;

                llama_tokens dts_tokens(n_tokens);
                ok = ok && (n_tokens == 0 || fread(dts_tokens.data(), sizeof(llama_token), n_tokens, f) == n_tokens);
                fclose(f);

                if (!ok) {
                    SRV_ERR("failed to load prompt state from disk ('%s')\n", best_fname.c_str());

                    return false;
                }

                // ensure the chunk store is open; the same pool instance serves
                // the writer thread, and io_mtx_ guards the concurrent reads
                // (a .dts index entry with the store disabled is stale - the
                // boot rollback prune normally removes both, drop it here too)
                const uint64_t base = fingerprint_lineage(disk_fingerprint_);
                {
                    std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
                    if (!prompt_cache_chunk_store::store_enabled()) {
                        drop_stale();
                        continue;
                    }
                    if (!disk_store_) {
                        disk_store_ = std::make_unique<prompt_cache_chunk_store>();
                    }
                    if (!disk_store_->is_open() &&
                        !disk_store_->open((fs::path(best_path).parent_path()).string(), base)) {
                        SRV_WRN(" - disk prompt cache: chunk store unavailable, dropping stale entry '%s'\n", best_path.c_str());

                        drop_stale();
                        continue;
                    }
                }

                std::array<uint8_t, 32> fp;
                dpc_sha256_raw((const uint8_t *) dts_tokens.data(), dts_tokens.size() * sizeof(llama_token), fp.data());

                prompt_cache_chunk_store::state_ref st;
                if (!disk_store_->get_state(fp, st) || st.n_tokens != (uint64_t) dts_tokens.size()) {
                    SRV_WRN(" - disk prompt cache: entry '%s' no longer in the chunk store, dropping stale entry\n", best_path.c_str());

                    drop_stale();
                    continue;
                }

                // chunk ids list the slices in the order they were written
                // (disk_store_write), so concatenation is the canonical blob
                try {
                    state.data.main.clear();
                    for (uint64_t id : st.chunks) {
                        std::vector<uint8_t> cell;
                        if (!disk_store_->get_chunk(id, cell)) {
                            SRV_ERR("failed to restore chunk %llu for '%s'\n", (unsigned long long) id, best_fname.c_str());

                            return false;
                        }
                        state.data.main.insert(state.data.main.end(), cell.begin(), cell.end());
                    }
                } catch (const std::bad_alloc & e) {
                    SRV_ERR("failed to allocate memory for prompt state from disk ('%s'): %s\n", best_fname.c_str(), e.what());

                    return false;
                }

                state.data.drft.clear();
                state.prompt.tokens = server_tokens(dts_tokens, false);
            } else {
                ok = ok && (magic == DPC_MAGIC && (version == DPC_VERSION || version == 2 || version == 1)) && dpc_read_state(f, version, state);
                fclose(f);

                if (!ok) {
                    SRV_ERR("failed to load prompt state from disk ('%s')\n", best_fname.c_str());

                    return false;
                }
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

        // successful restore refreshes the LRU order so the prune prefers cold entries
        std::error_code ec;
        fs::last_write_time(best_path, fs::file_time_type::clock::now(), ec);
        if (ec) {
            SRV_WRN(" - disk prompt cache: cannot touch '%s': %s\n", best_path.c_str(), ec.message().c_str());
        }

        return true;
    }
}

size_t server_prompt_cache::load_disk_index(const std::string & dir, const std::string & fingerprint) {
    namespace fs = std::filesystem;

    disk_dir_         = dir;
    disk_fingerprint_ = fingerprint;

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return 0;
    }

    // store disabled: one-pass rollback on the next boot. remove the store
    // artifacts (.dcp/.dci/.dts/cache.lock) so the tree reverts to the legacy
    // .dpc/.ccn layout; never touch those. idempotent - a store-enabled boot
    // skips this entirely
    if (!prompt_cache_chunk_store::store_enabled()) {
        for (auto & entry : fs::directory_iterator(dir, ec)) {
            const std::string name = entry.path().filename().string();
            if (entry.path().extension() == ".dcp" || entry.path().extension() == ".dci" ||
                entry.path().extension() == ".dts" || name == "cache.lock") {
                fs::remove(entry.path(), ec);
            }
        }
        SRV_WRN(" - disk prompt cache: chunk store disabled, pruned store files in '%s' (rollback)\n", dir.c_str());
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
        const auto ext = entry.path().extension();
        if (ext == ".dpc" || ext == ".dts") {
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
            (magic == DPC_MAGIC ? (version == DPC_VERSION || version == 2 || version == 1)
                                : (magic == DTS_MAGIC && version == DTS_VERSION)) &&
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
