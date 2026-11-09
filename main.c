#define _POSIX_C_SOURCE 200809L
#define PCRE2_CODE_UNIT_WIDTH 8
#include <math.h>
#include <stddef.h>
#include <locale.h>
#include <pcre2.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <wchar.h>

struct Qwen3Tokenizer {
    char** vocab;  // array of strings
    uint64_t vocab_size;

    char** merges;  // array of strings
    uint64_t merge_count;
};

struct GGUFTensorInfo {
    char* name;
    uint32_t ndim;
    uint64_t shape[4];
    uint32_t type;
    uint64_t offset;
};

struct GGUFHeader {
    char magic[4];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

struct Qwen3Config {
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t context_length;
    uint32_t vocab_size;

    float rope_theta;
    float rms_norm_eps;
};

struct Tensor {
    const void* data;
    int64_t shape[4];
    int ndim;
    int type;
};

struct Qwen3Layer {
    struct Tensor attn_q;
    struct Tensor attn_k;
    struct Tensor attn_v;
    struct Tensor attn_output;

    struct Tensor attn_norm;
    struct Tensor ffn_norm;

    struct Tensor ffn_up;
    struct Tensor ffn_gate;
    struct Tensor ffn_down;

    struct Tensor attn_q_norm;
    struct Tensor attn_k_norm;
};

struct Qwen3Model {
    void* mmap_base;
    size_t mmap_size;

    struct GGUFHeader header;
    struct Qwen3Config config;
    struct Qwen3Layer* layers;

    struct Tensor token_embedding;
    struct Tensor output_norm;
    struct Tensor output;
};

/*
Modern C Standard (_Float16)

Introduced formally in the ISO/IEC TS 18661-3 extension and incorporated into the standard
library, you can use the _Float16 type. • Compiler Support: Supported by GCC and Clang on
architectures that natively support it or can emulate it (such as AArch64/ARM64 and x86
with SSE2 enabled).

https://gcc.gnu.org/onlinedocs/gcc/Half-Precision.html

example:
       _Float16 half_float = 12.34f16; // 'f16' literal suffix

Now suppose you want:
    weight[0][0];
    weight[0][1];
    weight[0][35];

You can still retrieve those logical weights, but you must calculate which physical block
contains them.

For a weight at logical position [row, col]:
    size_t block_index = col / 32;
    size_t within_block = col % 32;

Then:
    block_q8_0 *block = &blocks[row * 32 + block_index];

    float value = fp16_to_fp32(block->scale) *
                  block->qs[within_block];

*/

typedef struct {
    __fp16 scale;  // FP16 scale: 2 bytes
    int8_t qs[32];   // 32 quantized weights
} block_q8_0;

// ============================================================
// GGUF MODEL FILE PARSER
// ============================================================

enum GGUF_TYPE {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12
};

void print_tensor_metadata(const struct Tensor* tensor) {
    if (!tensor) {
        printf("Tensor is NULL\n");
        return;
    }

    printf("Tensor metadata:\n");
    printf("  Number of dimensions: %d\n", tensor->ndim);
    printf("  Shape: [");
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%" PRId64, tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("  Data type: %d\n", tensor->type);
}

// read exactly n bytes from the file into dst.
static bool gguf_read(FILE* file, void* dst, size_t n) {
    return fread(dst, 1, n, file) == n;
}

static bool gguf_read_u32(FILE* file, uint32_t* value) {
    unsigned char b[4];

    if(!gguf_read(file, b, sizeof(b))) return false;

    *value = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
             ((uint32_t)b[3] << 24);

    return true;
}

static bool gguf_read_u64(FILE* file, uint64_t* value) {
    unsigned char b[8];

    if(!gguf_read(file, b, sizeof(b))) return false;

    *value = 0;

    for(int i = 0; i < 8; i++) *value |= (uint64_t)b[i] << (8 * i);

    return true;
}

static bool gguf_read_f32(FILE* file, float* value) {
    uint32_t bits;

    if(!gguf_read_u32(file, &bits)) return false;

    _Static_assert(sizeof(float) == 4, "GGUF requires 32-bit floats");

    memcpy(value, &bits, sizeof(bits));
    return true;
}

// Read a length-prefixed GGUF string.
// Caller must free(*out).
static bool gguf_read_string(FILE* file, char** out) {
    uint64_t len;

    *out = NULL;

    if(!gguf_read_u64(file, &len)) return false;

    if(len > SIZE_MAX - 1) return false;

    // Prevent absurd allocations from malformed files.
    // Increase this limit if a legitimate metadata string needs it.
    const uint64_t MAX_STRING = 64ULL * 1024 * 1024;

    if(len > MAX_STRING) return false;

    char* str = malloc((size_t)len + 1);

    if(!str) return false;

    if(!gguf_read(file, str, (size_t)len)) {
        free(str);
        return false;
    }

    str[len] = '\0';

    *out = str;
    return true;
}

// Skip bytes without requiring a seekable stream.
static bool gguf_skip_bytes(FILE* file, uint64_t count) {
    unsigned char buffer[4096];

    while(count > 0) {
        size_t n = count < sizeof(buffer) ? (size_t)count : sizeof(buffer);

        if(!gguf_read(file, buffer, n)) return false;

        count -= n;
    }

    return true;
}

// Skip a GGUF value, including arrays.
// GGUF v3 arrays have a single element type and cannot
// contain nested arrays.
static bool gguf_skip_value(FILE* file, uint32_t type) {
    uint64_t len;

    switch(type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return gguf_skip_bytes(file, 1);

        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return gguf_skip_bytes(file, 2);

        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return gguf_skip_bytes(file, 4);

        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return gguf_skip_bytes(file, 8);

        case GGUF_TYPE_STRING:
            if(!gguf_read_u64(file, &len)) return false;

            return gguf_skip_bytes(file, len);

        case GGUF_TYPE_ARRAY: {
            uint32_t element_type;
            uint64_t count;

            if(!gguf_read_u32(file, &element_type)) return false;

            if(!gguf_read_u64(file, &count)) return false;

            // Nested arrays are not valid in GGUF v3.
            if(element_type > GGUF_TYPE_FLOAT64 || element_type == GGUF_TYPE_ARRAY)
                return false;

            // Avoid enormous loops on malformed input.
            if(count > 1000000000ULL) return false;

            for(uint64_t i = 0; i < count; i++) {
                if(!gguf_skip_value(file, element_type)) return false;
            }

            return true;
        }

        default:
            return false;
    }
}

// read a UINT32 metadata value, or reject an unexpected type.
static bool gguf_config_u32(FILE* file, uint32_t type, uint32_t* out) {
    if(type != GGUF_TYPE_UINT32) return false;

    return gguf_read_u32(file, out);
}

// read a FLOAT32 metadata value.
static bool gguf_config_f32(FILE* file, uint32_t type, float* out) {
    if(type != GGUF_TYPE_FLOAT32) return false;

    return gguf_read_f32(file, out);
}

static bool gguf_read_string_array(FILE* file, uint32_t type, char*** out_strings,
                                   uint64_t* out_count) {
    if(type != GGUF_TYPE_ARRAY) return false;

    uint32_t element_type;
    uint64_t count;

    if(!gguf_read_u32(file, &element_type) || !gguf_read_u64(file, &count)) return false;

    if(element_type != GGUF_TYPE_STRING) return false;

    if(count > SIZE_MAX / sizeof(char*)) return false;

    char** strings = calloc((size_t)count, sizeof(*strings));

    if(!strings && count != 0) return false;

    for(uint64_t i = 0; i < count; i++) {
        if(!gguf_read_string(file, &strings[i])) {
            for(uint64_t j = 0; j < i; j++) free(strings[j]);

            free(strings);
            return false;
        }
    }

    *out_strings = strings;
    *out_count = count;

    return true;
}

bool read_gguf_header(FILE* model, struct GGUFHeader* header) {
    if(model == NULL || header == NULL) {
        return false;
    }

    _Static_assert(sizeof(struct GGUFHeader) == 24, "Unexpected GGUFHeader layout");

    if(fseek(model, 0, SEEK_SET) != 0) {
        return false;
    }

    size_t bytes_read = fread(header, 1, sizeof(struct GGUFHeader), model);

    if(bytes_read != sizeof(struct GGUFHeader)) {
        return false;
    }

    if(memcmp(header->magic, "GGUF", 4) != 0) {
        return false;
    }

    if(header->version != 3) {
        return false;
    }

    return true;
}

// parse metadata and populate the Qwen3 configuration.
// important reading:
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md?utm_source=chatgpt.com#file-structure
bool read_gguf_metadata(FILE* file, const struct GGUFHeader* header,
                        struct Qwen3Config* config, struct Qwen3Tokenizer* tokenizer) {
    if(!file || !header || !config) return false;

    memset(config, 0, sizeof(*config));
    memset(tokenizer, 0, sizeof(*tokenizer));

    for(uint64_t i = 0; i < header->metadata_kv_count; i++) {
        char* key = NULL;
        uint32_t type;
        bool ok = false;

        if(!gguf_read_string(file, &key)) return false;

        if(!gguf_read_u32(file, &type)) {
            free(key);
            return false;
        }

        if(strcmp(key, "qwen3.block_count") == 0) {
            ok = gguf_config_u32(file, type, &config->num_layers);
        } else if(strcmp(key, "qwen3.embedding_length") == 0) {
            ok = gguf_config_u32(file, type, &config->hidden_size);
        } else if(strcmp(key, "qwen3.feed_forward_length") == 0) {
            ok = gguf_config_u32(file, type, &config->intermediate_size);
        } else if(strcmp(key, "qwen3.attention.head_count") == 0) {
            ok = gguf_config_u32(file, type, &config->num_heads);
        } else if(strcmp(key, "qwen3.attention.head_count_kv") == 0) {
            ok = gguf_config_u32(file, type, &config->num_kv_heads);
        } else if(strcmp(key, "qwen3.attention.key_length") == 0) {
            ok = gguf_config_u32(file, type, &config->head_dim);
        } else if(strcmp(key, "qwen3.context_length") == 0) {
            ok = gguf_config_u32(file, type, &config->context_length);
        } else if(strcmp(key, "qwen3.rope.freq_base") == 0) {
            ok = gguf_config_f32(file, type, &config->rope_theta);
        } else if(strcmp(key, "qwen3.attention.layer_norm_rms_epsilon") == 0) {
            ok = gguf_config_f32(file, type, &config->rms_norm_eps);
        } else if(strcmp(key, "tokenizer.ggml.tokens") == 0) {
            ok = gguf_read_string_array(file, type, &tokenizer->vocab,
                                        &tokenizer->vocab_size);
        } else if(strcmp(key, "tokenizer.ggml.merges") == 0) {
            ok = gguf_read_string_array(file, type, &tokenizer->merges,
                                        &tokenizer->merge_count);
        } else if(strcmp(key, "tokenizer.ggml.pre") == 0) {
            char* pre_str = NULL;
            if(!gguf_read_string(file, &pre_str)) {
                free(key);
                return false;
            }
            printf("tokenizer.ggml.pre: %s\n", pre_str);
            free(pre_str);
            ok = true;
        } else {
            // Unknown or currently unused metadata.
            ok = gguf_skip_value(file, type);
        }

        free(key);

        if(!ok) return false;
    }

    // Minimal sanity checks.
    if(config->num_layers == 0 || config->hidden_size == 0 || config->num_heads == 0 ||
       config->num_kv_heads == 0 || config->num_heads % config->num_kv_heads != 0) {
        return false;
    }

    return true;
}

static bool read_gguf_tensor_info(FILE* file, struct GGUFTensorInfo* info) {
    memset(info, 0, sizeof(*info));

    if(!gguf_read_string(file, &info->name)) return false;

    if(!gguf_read_u32(file, &info->ndim)) goto fail;

    if(info->ndim == 0 || info->ndim > 4) goto fail;

    for(uint32_t i = 0; i < info->ndim; i++) {
        if(!gguf_read_u64(file, &info->shape[i])) goto fail;

        if(info->shape[i] == 0) goto fail;
    }

    if(!gguf_read_u32(file, &info->type)) goto fail;

    if(!gguf_read_u64(file, &info->offset)) goto fail;

    return true;

fail:
    free(info->name);
    info->name = NULL;
    return false;
}

bool read_gguf_tensor_infos(FILE* file, const struct GGUFHeader* header,
                            struct GGUFTensorInfo** out_infos) {
    if(!file || !header || !out_infos) return false;

    *out_infos = NULL;

    if(header->tensor_count == 0 ||
       header->tensor_count > SIZE_MAX / sizeof(struct GGUFTensorInfo))
        return false;

    struct GGUFTensorInfo* infos =
        calloc((size_t)header->tensor_count, sizeof(struct GGUFTensorInfo));

    if(!infos) return false;

    for(uint64_t i = 0; i < header->tensor_count; i++) {
        if(!read_gguf_tensor_info(file, &infos[i])) {
            for(uint64_t j = 0; j < i; j++) free(infos[j].name);

            free(infos);
            return false;
        }
    }

    *out_infos = infos;
    return true;
}

bool qwen3_map_file(const char* path, struct GGUFTensorInfo* infos,
                    uint64_t tensor_data_start, struct Qwen3Model* model) {
    FILE* file = fopen(path, "rb");
    if(!file) {
        return false;
    }

    int fd = fileno(file);
    if(fd == -1) {
        fclose(file);
        return false;
    }

    if(fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return false;
    }

    long file_size = ftell(file);
    if(file_size <= 0) {
        fclose(file);
        return false;
    }

    if(fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return false;
    }

    void* base = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(base == MAP_FAILED) {
        fclose(file);
        return false;
    }

    model->mmap_base = base;
    model->mmap_size = file_size;

    // Once mmap succeeds, the mapping remains valid after closing the file.
    // The mapping owns the model data from this point onward.
    fclose(file);
    file = NULL;

    // allocate layers array
    model->layers = calloc(model->config.num_layers, sizeof(struct Qwen3Layer));
    if(!model->layers) {
        goto fail_mapped;
    }

    bool found_output_norm = false;
    bool found_token_embedding = false;

    // The first version of the loader assumed output_norm.weight and
    // token_embd.weight were tensor descriptors 0 and 1. That was true in
    // our test GGUF, but descriptor ordering is not guaranteed by the format,
    // so we now find both tensors by name in the same loop as the layer data.

    for(uint64_t i = 0; i < model->header.tensor_count; i++) {
        char* name = infos[i].name;
        uint64_t offset = infos[i].offset;

        // Validate the tensor start before doing pointer arithmetic. This
        // prevents a malformed GGUF offset from producing a pointer outside
        // the memory-mapped file. Full tensor-size validation comes later.

        if(tensor_data_start > model->mmap_size ||
           offset > model->mmap_size - tensor_data_start) {
            fprintf(stderr, "Tensor '%s' starts outside the mapped file\n", name);
            goto fail_mapped;
        }

        struct Tensor tensor = {
            .data = (const uint8_t*)model->mmap_base + tensor_data_start + offset,
            .ndim = infos[i].ndim,
            .type = infos[i].type};

        memcpy(tensor.shape, infos[i].shape, sizeof(infos[i].shape));

        if(strcmp(name, "output_norm.weight") == 0) {
            model->output_norm = tensor;
            found_output_norm = true;
            continue;
        }

        if(strcmp(name, "token_embd.weight") == 0) {
            model->token_embedding = tensor;
            print_tensor_metadata(&tensor);
            found_token_embedding = true;
            continue;
        }

        // simple parsing: find layer index from name
        int layer_idx = -1;
        if(sscanf(name, "blk.%d.", &layer_idx) != 1) {
            // not a layer tensor, skip
            continue;
        }

        if(layer_idx < 0 || (uint32_t)layer_idx >= model->config.num_layers) {
            // invalid layer index, skip
            continue;
        }

        struct Qwen3Layer* layer = &model->layers[layer_idx];

        // map tensor to layer field based on suffix
        if(strstr(name, "attn_q.weight")) {
            layer->attn_q = tensor;
        } else if(strstr(name, "attn_k.weight")) {
            layer->attn_k = tensor;
        } else if(strstr(name, "attn_v.weight")) {
            layer->attn_v = tensor;
        } else if(strstr(name, "attn_output.weight")) {
            layer->attn_output = tensor;
        } else if(strstr(name, "attn_norm.weight")) {
            layer->attn_norm = tensor;
        } else if(strstr(name, "ffn_norm.weight")) {
            layer->ffn_norm = tensor;
        } else if(strstr(name, "ffn_up.weight")) {
            layer->ffn_up = tensor;
        } else if(strstr(name, "ffn_gate.weight")) {
            layer->ffn_gate = tensor;
        } else if(strstr(name, "ffn_down.weight")) {
            layer->ffn_down = tensor;
        } else if(strstr(name, "attn_q_norm.weight")) {
            layer->attn_q_norm = tensor;
        } else if(strstr(name, "attn_k_norm.weight")) {
            layer->attn_k_norm = tensor;
        }
    }

    if(!found_output_norm || !found_token_embedding) {
        fprintf(stderr, "Missing output norm or token embedding tensor\n");
        goto fail_mapped;
    }

    for(uint32_t i = 0; i < model->config.num_layers; i++) {
        struct Qwen3Layer* layer = &model->layers[i];

        if(!layer->attn_q.data || !layer->attn_k.data || !layer->attn_v.data ||
           !layer->attn_output.data || !layer->attn_norm.data || !layer->ffn_norm.data ||
           !layer->ffn_up.data || !layer->ffn_gate.data || !layer->ffn_down.data ||
           !layer->attn_q_norm.data || !layer->attn_k_norm.data) {
            fprintf(stderr, "Missing tensor in layer %u\n", i);
            goto fail_mapped;
        }
    }

    return true;

fail_mapped:
    free(model->layers);
    model->layers = NULL;
    munmap(model->mmap_base, model->mmap_size);
    model->mmap_base = NULL;
    model->mmap_size = 0;
    if(file) fclose(file);
    return false;
}

// ============================================================
// TOKENIZER
// ============================================================

// In C, char occupies exactly one byte. But one byte is not necessarily one complete
// Unicode character.

// our vocabulary contains the tokenizer's serialized token representations, not
// necessarily the original raw text bytes.

struct Chunk {
    size_t start;
    size_t len;
};

struct ChunkList {
    struct Chunk* items;
    size_t count;
};

// splits text using Qwen2 pre-tokenization rules.
// start and len are UTF-8 byte offsets.
// we use qwen2 pretokenization scheme because that's what qwen3 uses (verified from the
// fileformat)
// used PCRE2, a C-compatible regex engine that supports Unicode properties

static const char* QWEN2_PATTERN =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
    "|\\p{N}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+"
    "|\\s+(?!\\S)"
    "|\\s+";

/*
 * Splits UTF-8 text into chunks using Qwen2 pre-tokenization rules.
 *
 * The function does not allocate separate strings for each chunk.
 * Instead, it returns byte offsets and lengths into the original text.
 *
 * Example:
 *
 *   Input:
 *       "Hello world! I'm learning C in 2026."
 *
 *   Output chunks:
 *       ["Hello", " world", "!", " I", "'m",
 *        " learning", " C", " in", " ", "2", "0", "2", "6", "."]
 *
 *   Each chunk is represented by:
 *
 *       struct Chunk {
 *           size_t start; // byte offset into original text
 *           size_t len;   // number of bytes in chunk
 *       };
 *
 *   Example:
 *       "Hello"  -> { start: 0,  len: 5 }
 *       " world" -> { start: 5,  len: 6 }
 *       "!"      -> { start: 11, len: 1 }
 *
 *   Usage:
 *       struct ChunkList chunks =
 *           qwen2_pre_tokenize(text, strlen(text));
 *
 *       for (size_t i = 0; i < chunks.count; i++) {
 *           struct Chunk c = chunks.items[i];
 *           printf("%.*s\n", (int)c.len, text + c.start);
 *       }
 *
 *       free(chunks.items);
 *
 *   The returned chunks are processed independently by byte-level BPE.
 */

struct ChunkList qwen2_pre_tokenize(const char* text, size_t text_len) {
    struct ChunkList result = {0};

    if(!text || text_len == 0) return result;

    int error;
    PCRE2_SIZE error_offset;

    pcre2_code* regex = pcre2_compile((PCRE2_SPTR)QWEN2_PATTERN, PCRE2_ZERO_TERMINATED,
                                      PCRE2_UTF | PCRE2_UCP, &error, &error_offset, NULL);

    if(!regex) return result;

    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(regex, NULL);

    if(!match_data) {
        pcre2_code_free(regex);
        return result;
    }

    size_t capacity = 16;
    result.items = malloc(capacity * sizeof(struct Chunk));

    if(!result.items) goto cleanup;

    PCRE2_SIZE offset = 0;

    while(offset < text_len) {
        int rc =
            pcre2_match(regex, (PCRE2_SPTR)text, text_len, offset, 0, match_data, NULL);

        if(rc < 0) goto fail;

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);

        size_t start = ovector[0];
        size_t end = ovector[1];

        // The pattern must consume input continuously.
        if(start != offset || end <= start) goto fail;

        if(result.count == capacity) {
            if(capacity > SIZE_MAX / (2 * sizeof(struct Chunk))) goto fail;

            capacity *= 2;

            struct Chunk* items = realloc(result.items, capacity * sizeof(struct Chunk));

            if(!items) goto fail;

            result.items = items;
        }

        result.items[result.count++] = (struct Chunk){.start = start, .len = end - start};

        offset = end;
    }

    goto cleanup;

fail:
    free(result.items);
    result = (struct ChunkList){0};

cleanup:
    pcre2_match_data_free(match_data);
    pcre2_code_free(regex);

    return result;
}

struct BPESymbol {
    size_t start;
    size_t len;
};

struct BPESymbolList {
    struct BPESymbol* items;
    size_t count;
};

struct TokenIDs {
    uint32_t* items;
    size_t count;
};

/*

mapped = "Ġworld"

Symbol 0: "Ġ"   start=0, len=2
Symbol 1: "w"   start=2, len=1
Symbol 2: "o"   start=3, len=1
Symbol 3: "r"   start=4, len=1
Symbol 4: "l"   start=5, len=1
Symbol 5: "d"   start=6, len=1

*/

struct BPESymbolList bpe_init_symbols(const char* mapped, size_t mapped_len) {
    struct BPESymbolList result = {0};

    // Upper bound: one symbol per byte.
    result.items = malloc(mapped_len * sizeof(struct BPESymbol));

    if(!result.items) return result;

    size_t i = 0;

    while(i < mapped_len) {
        unsigned char b = (unsigned char)mapped[i];

        size_t len;

        if(b < 0x80)
            len = 1;
        else if((b & 0xE0) == 0xC0)
            len = 2;
        else if((b & 0xF0) == 0xE0)
            len = 3;
        else if((b & 0xF8) == 0xF0)
            len = 4;
        else
            goto fail;

        if(len > mapped_len - i) goto fail;

        for(size_t j = 1; j < len; j++) {
            if(((unsigned char)mapped[i + j] & 0xC0) != 0x80) goto fail;
        }

        result.items[result.count++] = (struct BPESymbol){.start = i, .len = len};

        i += len;
    }

    return result;

fail:
    free(result.items);
    return (struct BPESymbolList){0};
}

static bool bpe_pair_matches(const char* mapped, struct BPESymbol left,
                             struct BPESymbol right, const char* rule) {
    // A merge rule is "left right".
    const char* separator = strchr(rule, ' ');
    if(!separator) return false;

    size_t left_len = (size_t)(separator - rule);
    size_t right_len = strlen(separator + 1);

    return left.len == left_len && right.len == right_len &&
           memcmp(mapped + left.start, rule, left_len) == 0 &&
           memcmp(mapped + right.start, separator + 1, right_len) == 0;
}

bool bpe_apply_merges(struct Qwen3Tokenizer* tokenizer, struct BPESymbolList* symbols,
                      const char* mapped) {
    if(!tokenizer || !symbols || !mapped) return false;

    while(symbols->count > 1) {
        size_t best_rank = SIZE_MAX;
        size_t best_index = SIZE_MAX;

        // Examine every adjacent pair.
        for(size_t i = 0; i + 1 < symbols->count; i++) {
            struct BPESymbol left = symbols->items[i];
            struct BPESymbol right = symbols->items[i + 1];

            // Find the merge rule for this pair.
            for(size_t rank = 0; rank < tokenizer->merge_count; rank++) {
                if(rank >= best_rank) break;

                if(bpe_pair_matches(mapped, left, right, tokenizer->merges[rank])) {
                    best_rank = rank;
                    best_index = i;
                    break;
                }
            }
        }

        // No pair can be merged anymore.
        if(best_index == SIZE_MAX) break;

        // Merge symbols[best_index] with the next symbol.
        symbols->items[best_index].len += symbols->items[best_index + 1].len;

        // Remove the second symbol by shifting the array left.
        for(size_t j = best_index + 1; j + 1 < symbols->count; j++) {
            symbols->items[j] = symbols->items[j + 1];
        }

        symbols->count--;
    }

    return true;
}

bool bpe_to_token_ids(struct Qwen3Tokenizer* tokenizer, struct BPESymbolList* symbols,
                      const char* mapped, struct TokenIDs* tokenIds) {
    if(!tokenizer || !symbols || !mapped || !tokenIds) return false;

    for(size_t i = 0; i < symbols->count; i++) {
        struct BPESymbol symbol = symbols->items[i];

        bool found = false;

        for(size_t id = 0; id < tokenizer->vocab_size; id++) {
            const char* token = tokenizer->vocab[id];

            if(strlen(token) == symbol.len &&
               memcmp(token, mapped + symbol.start, symbol.len) == 0) {
                // Assumes tokenIds->items has sufficient capacity.
                tokenIds->items[tokenIds->count++] = (uint32_t)id;

                found = true;
                break;
            }
        }

        if(!found) return false;
    }

    return true;
}

/*


*/

void bytes_to_unicode(uint32_t table[256]) {
    // Step 1: Keep printable bytes unchanged.
    for(int b = 33; b <= 126; b++) table[b] = b;

    for(int b = 161; b <= 172; b++) table[b] = b;

    for(int b = 174; b <= 255; b++) table[b] = b;

    // Step 2: Map all remaining bytes to code points >= 256.
    uint32_t next = 256;

    for(int b = 0; b < 256; b++) {
        bool printable =
            (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);

        if(!printable) {
            table[b] = next++;
        }
    }
}

// icl, chatgpt gave me this code lmao
static int encode_utf8(uint32_t cp, char* out) {
    if(cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }

    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}

bool qwen3_tokenize(struct Qwen3Tokenizer* tokenizer, struct ChunkList* chunks,
                    struct TokenIDs* tokenIds, const char* text) {
    if(!chunks || !tokenIds || !text) return false;

    uint32_t table[256];
    bytes_to_unicode(table);

    // 1. Enable the system's default locale (required for proper UTF-8 interpretation)
    setlocale(LC_ALL, "");
    struct Chunk c;

    for(size_t i = 0; i < chunks->count; i++) {
        c = chunks->items[i];

        const unsigned char* bytes = (const unsigned char*)text + c.start;

        char* mapped = malloc(c.len * 2 + 1);
        if(!mapped) return false;

        size_t mapped_len = 0;

        for(size_t j = 0; j < c.len; j++) {
            uint8_t b = bytes[j];

            // Original byte -> Unicode code point
            uint32_t cp = table[b];

            // Unicode code point -> UTF-8 bytes
            mapped_len += encode_utf8(cp, mapped + mapped_len);
        }

        mapped[mapped_len] = '\0';

        struct BPESymbolList symbols = bpe_init_symbols(mapped, mapped_len);

        if(!symbols.items && mapped_len != 0) {
            free(mapped);
            return false;
        }

        if(!bpe_apply_merges(tokenizer, &symbols, mapped)) {
            free(symbols.items);
            free(mapped);
            return false;
        }

        if(!bpe_to_token_ids(tokenizer, &symbols, mapped, tokenIds)) {
            free(symbols.items);
            free(mapped);
            return false;
        }

        free(symbols.items);
        free(mapped);
    }

    return true;
}

// ============================================================
// KERNELS
// ============================================================

inline bool dot(float * a, float * b, float * c, size_t n){
    float sum = 0.0f;

    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }

    *c = sum;
    return true;
}

inline bool mulvec(float * a, float * b, float * c, size_t n){
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
    return true;
}

// work for an F32 embedding, but not for the Q8_0 model.
bool embed_lookup_scalar_FP32 (
    const uint32_t* token_ids, 
    const void* embedding, 
    float*output, 
    size_t n_tokens, 
    size_t hidden_size){

    for (size_t i = 0 ; i < n_tokens; i++) {
        uint32_t token_id = token_ids[i];
        float * row_start = (float *)embedding + (token_id * hidden_size);
        memcpy(output + i * hidden_size, row_start, hidden_size * sizeof(float));
    }

    return true;
}

bool embed_lookup_scalar_Q8_0 (
    const uint32_t* token_ids, 
    const void* embedding, 
    float*output, 
    size_t n_tokens, 
    size_t hidden_size){

    if (hidden_size % 32 != 0)
        return false;

    // n blocks for q8_0 because there are 32 weights in a block
    size_t n_blocks = hidden_size / 32;

    for (size_t i = 0; i < n_tokens; i++) {
        uint32_t token_id = token_ids[i];
        const block_q8_0* blocks = (const block_q8_0 *)embedding + token_id * n_blocks;

        for (size_t block_idx = 0; block_idx < n_blocks; block_idx++) {
            const block_q8_0* block = &blocks[block_idx];
            float scale = (__fp16)block->scale;

            size_t base_idx = i * hidden_size + block_idx * 32;
            size_t limit = (block_idx == n_blocks - 1) ? (hidden_size - block_idx * 32) : 32;

            for (size_t j = 0; j < limit; j++) {
                output[base_idx + j] = scale * (float)block->qs[j];
            }
        }
    }

    return true;
}



bool rmsnorm_scalar(
    const float *input,
    const float *gamma,
    float *output,
    size_t n_tokens,
    size_t hidden_size,
    float eps
) {
    if (!input || !gamma || !output || hidden_size == 0)
        return false;

    for (size_t i = 0; i < n_tokens; i++) {
        const float *row = input + i * hidden_size;
        float *out = output + i * hidden_size;

        float sum_sq = 0.0f;

        for (size_t j = 0; j < hidden_size; j++) {
            sum_sq += row[j] * row[j];
        }

        float inv_rms = 1.0f / sqrtf(sum_sq / hidden_size + eps);

        for (size_t j = 0; j < hidden_size; j++) {
            out[j] = row[j] * inv_rms * gamma[j];
        }
    }
    return true;
}

// ============================================================
// FORWARD
// ============================================================

bool embedding_lookup(const struct Tensor* embedding, const uint32_t* token_ids,
                      size_t n_tokens, size_t hidden_size, float* output){

    if (!token_ids || !embedding || !output || n_tokens == 0 || hidden_size == 0) {
        return false;
    }

        switch (embedding->type) {
        case 0:
            return embed_lookup_scalar_FP32(token_ids, embedding->data, output, n_tokens, hidden_size);
        case 8:
            return embed_lookup_scalar_Q8_0(token_ids, embedding->data, output, n_tokens, hidden_size);
        default:
            return false;
        }
}

bool rms_norm(const struct Tensor* gamma, const float* input, float* output,
              size_t n_tokens, size_t hidden_size, float eps) {

    // input  is [n_tokens, hidden_size]
    // gamaa is            [hidden_size]
    // output is [n_tokens, hidden_size]

    if (!gamma || !gamma->data || !input || !output)
        return false;

    return rmsnorm_scalar(
        input, gamma->data, output,
        n_tokens, hidden_size, eps
    );
}

/*
qwen3_forward()
    |
    ├── embedding_lookup()
    |
    ├── for each transformer layer
    |      |
    |      ├── rms_norm()
    |      |      └── rmsnorm_avx2()
    |      |
    |      ├── matmul()
    |      |      └── matmul_avx2()
    |      |
    |      ├── attention()
    |      |
    |      └── ...
    |
    ├── rms_norm()
    |
    └── lm_head()
*/

bool qwen3_forward(struct Qwen3Model* model, const uint32_t* token_ids, size_t n_tokens,
                   float* logits) {

    if(!model || !token_ids || !logits || n_tokens == 0) return false;

    size_t hidden_size = model->config.hidden_size;

    if(hidden_size == 0 || n_tokens > SIZE_MAX / hidden_size ||
       n_tokens * hidden_size > SIZE_MAX / sizeof(float))
        return false;

    size_t num_values = n_tokens * hidden_size;

    float* hidden = malloc(num_values * sizeof(float));
    float* normed = malloc(num_values * sizeof(float));

    if(!hidden || !normed) {
        free(hidden);
        free(normed);
        return false;
    }

    // 1. Embedding lookup: token IDs -> F32 hidden states.
    if(!embedding_lookup(&model->token_embedding, token_ids, n_tokens, hidden_size,
                         hidden))
        goto cleanup;

    float rms_norm_eps = model->config.rms_norm_eps;

    // 2. all attention layers ran
    for(size_t i = 0; i < model->config.num_layers; i++) {

        struct Qwen3Layer* layer = &model->layers[i];

        // attn_rms_norm (qwen architecture employs pre-norm)
        rms_norm(&layer->attn_norm, hidden, normed, n_tokens, hidden_size, rms_norm_eps);

        // attention(...)
        // mlp(...)
        // residual connections
    }

    // Final RMSNorm
    // LM head

    free(hidden);
    free(normed);
    return true;

cleanup:
    free(hidden);
    free(normed);
    return false;
}

// ============================================================
// PROGRAM CONTROLS
// ============================================================

void destroy_model(struct Qwen3Model* model) {
    if(!model) return;

    free(model->layers);
    model->layers = NULL;

    if(model->mmap_base) {
        munmap(model->mmap_base, model->mmap_size);
        model->mmap_base = NULL;
        model->mmap_size = 0;
    }
}

int main(int argc, char** argv) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    FILE* file = fopen(argv[1], "rb");

    if(!file) {
        perror("fopen");
        return 1;
    }

    struct GGUFHeader header;

    if(!read_gguf_header(file, &header)) {
        fprintf(stderr, "Invalid GGUF header\n");
        fclose(file);
        return 1;
    }

    struct Qwen3Config config;
    struct Qwen3Tokenizer tokenizer;

    if(!read_gguf_metadata(file, &header, &config, &tokenizer)) {
        fprintf(stderr, "Failed to parse GGUF metadata\n");
        fclose(file);
        return 1;
    }

    struct GGUFTensorInfo* infos = NULL;

    if(!read_gguf_tensor_infos(file, &header, &infos)) {
        fprintf(stderr, "Failed to parse tensor descriptors\n");
        fclose(file);
        return 1;
    }

    // INFO: GGUF normally aligns the beginning of the tensor-data section to a
    // 32-byte boundary (unless metadata specifies another alignment).

    // assume that the current offset just after reading the tensors desriptions
    // is not aligned the gguf's 32 byte alignement

    long position = ftell(file);

    if(position < 0) {
        perror("ftell");
        // handle cleanup
        return 1;
    }

    uint64_t alignment = 32;  // GGUF default
    uint64_t offset = (uint64_t)position;

    uint64_t tensor_data_start = offset + (alignment - (offset % alignment)) % alignment;

    struct Qwen3Model* model = malloc(sizeof(struct Qwen3Model));

    if(!model) {
        // TODO: lol handle allocation failure
        return 1;
    }

    model->header = header;
    model->config = config;

    if(!qwen3_map_file(argv[1], infos, tensor_data_start, model)) {
        fprintf(stderr, "Failed to mmap the model file\n");
        free(infos);
        free(model);
        fclose(file);
        return 1;
    }

    printf("Loaded Qwen3 GGUF successfully: %u layers, %llu tensors, %u hidden size\n",
           config.num_layers, (unsigned long long)header.tensor_count,
           config.hidden_size);
    printf(" Tokenizer vocab size: %llu\n", (unsigned long long)tokenizer.vocab_size);
    printf(" Tokenizer merge count: %llu\n", (unsigned long long)tokenizer.merge_count);

    // make use of qwen3_tokenize here for an example string
    const char* input_text = "Hello world! I'm learning C in 2026.";
    size_t text_len = strlen(input_text);
    struct ChunkList chunks = qwen2_pre_tokenize(input_text, strlen(input_text));

    struct TokenIDs token_ids = {0};
    token_ids.items = malloc(text_len * sizeof(uint32_t));

    if(!token_ids.items) {
        fprintf(stderr, "Failed to allocate token IDs\n");
        free(chunks.items);
        return 1;
    }

    if(!qwen3_tokenize(&tokenizer, &chunks, &token_ids, input_text)) {
        fprintf(stderr, "Tokenization failed\n");
        free(token_ids.items);
        free(chunks.items);
        return 1;
    }

    printf("Token IDs: ");
    for(size_t i = 0; i < token_ids.count; i++) {
        printf("%u ", token_ids.items[i]);
    }
    printf("\n");

    // NOTE: free pointers
    free(token_ids.items);
    free(chunks.items);

    for(uint64_t i = 0; i < header.tensor_count; i++) free(infos[i].name);

    free(infos);
    fclose(file);
    destroy_model(model);
    free(model);
    return 0;
}
