#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <regex>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// clip images are quantized with this tool, then loaded by llama-mtmd
// supported output types:
//   Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_1S, TQ4_1S
// legacy numeric ggml-type ids are also accepted (2=q4_0, 3=q4_1, 6=q5_0, 7=q5_1, 8=q8_0)

static bool supported_ftype(llama_ftype ftype, const char ** name_out) {
    const struct {
        llama_ftype ftype;
        const char * name;
        ggml_type type;
    } table[] = {
        { LLAMA_FTYPE_MOSTLY_Q4_0,    "Q4_0",    GGML_TYPE_Q4_0 },
        { LLAMA_FTYPE_MOSTLY_Q4_1,    "Q4_1",    GGML_TYPE_Q4_1 },
        { LLAMA_FTYPE_MOSTLY_Q5_0,    "Q5_0",    GGML_TYPE_Q5_0 },
        { LLAMA_FTYPE_MOSTLY_Q5_1,    "Q5_1",    GGML_TYPE_Q5_1 },
        { LLAMA_FTYPE_MOSTLY_Q8_0,    "Q8_0",    GGML_TYPE_Q8_0 },
        { LLAMA_FTYPE_MOSTLY_TQ3_1S,  "TQ3_1S",  GGML_TYPE_TQ3_1S },
        { LLAMA_FTYPE_MOSTLY_TQ4_1S,  "TQ4_1S",  GGML_TYPE_TQ4_1S },
    };

    for (const auto & it : table) {
        if (ftype == it.ftype) {
            *name_out = it.name;
            return true;
        }
    }
    return false;
}

static ggml_type ftype_to_ggml_type(llama_ftype ftype, const char ** name_out) {
    const struct {
        llama_ftype ftype;
        const char * name;
        ggml_type type;
    } table[] = {
        { LLAMA_FTYPE_MOSTLY_Q4_0,    "Q4_0",    GGML_TYPE_Q4_0 },
        { LLAMA_FTYPE_MOSTLY_Q4_1,    "Q4_1",    GGML_TYPE_Q4_1 },
        { LLAMA_FTYPE_MOSTLY_Q5_0,    "Q5_0",    GGML_TYPE_Q5_0 },
        { LLAMA_FTYPE_MOSTLY_Q5_1,    "Q5_1",    GGML_TYPE_Q5_1 },
        { LLAMA_FTYPE_MOSTLY_Q8_0,    "Q8_0",    GGML_TYPE_Q8_0 },
        { LLAMA_FTYPE_MOSTLY_TQ3_1S,  "TQ3_1S",  GGML_TYPE_TQ3_1S },
        { LLAMA_FTYPE_MOSTLY_TQ4_1S,  "TQ4_1S",  GGML_TYPE_TQ4_1S },
    };

    for (const auto & it : table) {
        if (ftype == it.ftype) {
            *name_out = it.name;
            return it.type;
        }
    }
    return GGML_TYPE_F32;
}

static bool striequals(const char * a, const char * b) {
    size_t len = strlen(a);
    if (strlen(b) != len) {
        return false;
    }
    for (size_t i = 0; i < len; ++i) {
        if (tolower((unsigned char) a[i]) != tolower((unsigned char) b[i])) {
            return false;
        }
    }
    return true;
}

static void print_usage(int argc, char ** argv) {
    (void) argc;

    fprintf(stderr, "usage: %s /path/to/ggml-model-f32.gguf /path/to/ggml-model-quantized.gguf type\n", argv[0]);
    fprintf(stderr, "  type = Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | TQ3_1S | TQ4_1S\n");
    fprintf(stderr, "  legacy numeric ids also work: 2 - q4_0, 3 - q4_1, 6 - q5_0, 7 - q5_1, 8 - q8_0\n");
}

static bool try_parse_type(const char * arg, llama_ftype & ftype, const char ** name_out) {
    const char * names[] = { "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "TQ3_1S", "TQ4_1S" };
    const llama_ftype ftypes[] = {
        LLAMA_FTYPE_MOSTLY_Q4_0, LLAMA_FTYPE_MOSTLY_Q4_1, LLAMA_FTYPE_MOSTLY_Q5_0,
        LLAMA_FTYPE_MOSTLY_Q5_1, LLAMA_FTYPE_MOSTLY_Q8_0,
        LLAMA_FTYPE_MOSTLY_TQ3_1S, LLAMA_FTYPE_MOSTLY_TQ4_1S,
    };

    for (int i = 0; i < (int) (sizeof(names) / sizeof(names[0])); ++i) {
        if (striequals(arg, names[i])) {
            ftype = ftypes[i];
            *name_out = names[i];
            return true;
        }
    }

    // legacy numeric ggml-type ids: 2=q4_0, 3=q4_1, 6=q5_0, 7=q5_1, 8=q8_0
    const int id = atoi(arg);
    if (id >= 2 && id <= 8) {
        const llama_ftype legacy[] = {
            LLAMA_FTYPE_MOSTLY_Q4_0, LLAMA_FTYPE_MOSTLY_Q4_1, LLAMA_FTYPE_MOSTLY_Q5_0,
            LLAMA_FTYPE_MOSTLY_Q5_1, LLAMA_FTYPE_MOSTLY_Q8_0,
        };
        (void) legacy;
        switch (id) {
            case 2: ftype = LLAMA_FTYPE_MOSTLY_Q4_0; *name_out = "Q4_0"; return true;
            case 3: ftype = LLAMA_FTYPE_MOSTLY_Q4_1; *name_out = "Q4_1"; return true;
            case 6: ftype = LLAMA_FTYPE_MOSTLY_Q5_0; *name_out = "Q5_0"; return true;
            case 7: ftype = LLAMA_FTYPE_MOSTLY_Q5_1; *name_out = "Q5_1"; return true;
            case 8: ftype = LLAMA_FTYPE_MOSTLY_Q8_0; *name_out = "Q8_0"; return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        print_usage(argc, argv);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    llama_ftype ftype;
    const char * ftype_name;
    if (!try_parse_type(argv[3], ftype, &ftype_name)) {
        fprintf(stderr, "%s: invalid type: '%s'\n", __func__, argv[3]);
        print_usage(argc, argv);
        return 1;
    }

    const ggml_type type = ftype_to_ggml_type(ftype, &ftype_name);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model through the public gguf API; no_alloc=false materializes
    // all tensor data into a ggml context (same contract as llama-model-loader)
    {
        const int64_t t_start_us = ggml_time_us();

        struct ggml_context * ctx_data = nullptr;
        struct gguf_init_params params = {
            /*.no_alloc =*/ false,
            /*.ctx      =*/ &ctx_data,
        };
        struct gguf_context * ctx_src = gguf_init_from_file(fname_inp.c_str(), params);
        if (ctx_src == nullptr || ctx_data == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        auto * ctx_out = gguf_init_empty();
        gguf_set_kv(ctx_out, ctx_src);
        gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
        gguf_set_val_u32(ctx_out, "general.file_type", ftype);

        auto fout = std::ofstream(fname_out, std::ios::binary);
        if (!fout) {
            fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
            return 1;
        }

        const int n_tensors = gguf_get_n_tensors(ctx_src);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_src, i);
            ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
            gguf_add_tensor(ctx_out, cur);
        }

        const size_t meta_size = gguf_get_meta_size(ctx_out);
        for (size_t i = 0; i < meta_size; ++i) {
            fout.put(0);
        }

        // regexes of tensor names to be quantized
        const std::vector<std::string> k_names = {
            ".*weight",
        };

        std::vector<uint8_t> work(512);
        std::vector<float> conv_buf(512);
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        for (int i = 0; i < n_tensors; ++i) {
            const std::string name = gguf_get_tensor_name(ctx_src, i);
            ggml_tensor * cur = ggml_get_tensor(ctx_data, name.c_str());

            enum ggml_type new_type;
            void * new_data;
            size_t new_size;

            bool quantize = false;
            for (const auto & s : k_names) {
                if (std::regex_match(name, std::regex(s))) {
                    quantize = true;
                    break;
                }
            }

            // quantize only 2D tensors, bigger than block size, and aligned
            // to the block (TQ and Q8_0 require ne[0] % blck_size == 0)
            quantize &= (ggml_n_dims(cur) == 2) && cur->ne[0] > ggml_blck_size(type)
                && cur->ne[0] % ggml_blck_size(type) == 0;

            if (quantize) {
                new_type = type;
                if (new_type >= GGML_TYPE_Q2_K && name.find("embd") != std::string::npos) {
                    new_type = GGML_TYPE_Q8_0; // ggml_get_rows needs non K/TQ type
                }
                const size_t n_elms = ggml_nelements(cur);
                float * f32_data;

                switch (cur->type) {
                case GGML_TYPE_F32:
                    f32_data = (float *)cur->data;
                    break;
                case GGML_TYPE_F16:
                    if (conv_buf.size() < n_elms) {
                        conv_buf.resize(n_elms);
                    }
                    for (size_t j = 0; j < n_elms; ++j) {
                        conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t *)cur->data)[j]);
                    }
                    f32_data = (float *)conv_buf.data();
                    break;
                case GGML_TYPE_BF16:
                    if (conv_buf.size() < n_elms) {
                        conv_buf.resize(n_elms);
                    }
                    ggml_bf16_to_fp32_row((ggml_bf16_t *)cur->data, (float *)conv_buf.data(), n_elms);
                    f32_data = (float *)conv_buf.data();
                    break;
                default:
                    LOG_ERR("%s: unsupported input tensor type %s for '%s'\n", __func__, ggml_type_name(cur->type), name.c_str());
                    gguf_free(ctx_out);
                    gguf_free(ctx_src);
                    return 1;
                }

                if (work.size() < n_elms * 4) {
                    work.resize(n_elms * 4);
                }
                new_data = work.data();

                new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, n_elms/cur->ne[0], cur->ne[0], nullptr);
            } else {
                new_type = cur->type;
                new_data = cur->data;
                new_size = ggml_nbytes(cur);
            }
            const size_t orig_size = ggml_nbytes(cur);
            total_size_org += orig_size;
            total_size_new += new_size;
            gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_out, gguf_find_tensor(ctx_out, name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_out, name.c_str(), new_data);
            fout.write((const char *)new_data, new_size);
            size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_out)) - new_size;
            for (size_t j = 0; j < pad; ++j) {
                fout.put(0);
            }

            LOG_INF("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
                   orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
        }

        // go back to beginning of file and write the updated metadata
        fout.seekp(0, std::ios::beg);
        std::vector<uint8_t> meta(meta_size);
        gguf_get_meta_data(ctx_out, meta.data());
        fout.write((const char *)meta.data(), meta_size);

        fout.close();

        ggml_free(ctx_data);
        gguf_free(ctx_out);
        gguf_free(ctx_src);

        {
            LOG_INF("%s: original  size = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
            LOG_INF("%s: quantized size = %8.2f MB (%s)\n", __func__, total_size_new / 1024.0 / 1024.0, ftype_name);
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    return 0;
}