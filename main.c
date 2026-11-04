#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Tensor {
    const void *data;
    int64_t shape[4];
    int ndim;
    int type;
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

    void *mmap_base;
    size_t mmap_size;

    struct GGUFHeader *header;
    struct Qwen3Config config;
    struct Qwen3Layer *layers;
};

// ============================================================
// GGUF METADATA PARSER
// ============================================================

enum {
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

// Read exactly n bytes from the file.
static bool gguf_read(FILE *file, void *dst, size_t n) {
    return fread(dst, 1, n, file) == n;
}

static bool gguf_read_u32(FILE *file, uint32_t *value) {
    unsigned char b[4];

    if(!gguf_read(file, b, sizeof(b)))
        return false;

    *value = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);

    return true;
}

static bool gguf_read_u64(FILE *file, uint64_t *value) {
    unsigned char b[8];

    if(!gguf_read(file, b, sizeof(b)))
        return false;

    *value = 0;

    for(int i = 0; i < 8; i++)
        *value |= (uint64_t)b[i] << (8 * i);

    return true;
}

static bool gguf_read_f32(FILE *file, float *value) {
    uint32_t bits;

    if(!gguf_read_u32(file, &bits))
        return false;

    _Static_assert(sizeof(float) == 4, "GGUF requires 32-bit floats");

    memcpy(value, &bits, sizeof(bits));
    return true;
}

// Read a length-prefixed GGUF string.
// Caller must free(*out).
static bool gguf_read_string(FILE *file, char **out) {
    uint64_t len;

    *out = NULL;

    if(!gguf_read_u64(file, &len))
        return false;

    if(len > SIZE_MAX - 1)
        return false;

    // Prevent absurd allocations from malformed files.
    // Increase this limit if a legitimate metadata string needs it.
    const uint64_t MAX_STRING = 64ULL * 1024 * 1024;

    if(len > MAX_STRING)
        return false;

    char *str = malloc((size_t)len + 1);

    if(!str)
        return false;

    if(!gguf_read(file, str, (size_t)len)) {
        free(str);
        return false;
    }

    str[len] = '\0';

    *out = str;
    return true;
}

// Skip bytes without requiring a seekable stream.
static bool gguf_skip_bytes(FILE *file, uint64_t count) {
    unsigned char buffer[4096];

    while(count > 0) {
        size_t n = count < sizeof(buffer) ? (size_t)count : sizeof(buffer);

        if(!gguf_read(file, buffer, n))
            return false;

        count -= n;
    }

    return true;
}

// Skip a GGUF value, including arrays.
// GGUF v3 arrays have a single element type and cannot
// contain nested arrays.
static bool gguf_skip_value(FILE *file, uint32_t type) {
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
        if(!gguf_read_u64(file, &len))
            return false;

        return gguf_skip_bytes(file, len);

    case GGUF_TYPE_ARRAY: {
        uint32_t element_type;
        uint64_t count;

        if(!gguf_read_u32(file, &element_type))
            return false;

        if(!gguf_read_u64(file, &count))
            return false;

        // Nested arrays are not valid in GGUF v3.
        if(element_type > GGUF_TYPE_FLOAT64 || element_type == GGUF_TYPE_ARRAY)
            return false;

        // Avoid enormous loops on malformed input.
        if(count > 1000000000ULL)
            return false;

        for(uint64_t i = 0; i < count; i++) {
            if(!gguf_skip_value(file, element_type))
                return false;
        }

        return true;
    }

    default:
        return false;
    }
}

// read a UINT32 metadata value, or reject an unexpected type.
static bool gguf_config_u32(FILE *file, uint32_t type, uint32_t *out) {
    if(type != GGUF_TYPE_UINT32)
        return false;

    return gguf_read_u32(file, out);
}

// read a FLOAT32 metadata value.
static bool gguf_config_f32(FILE *file, uint32_t type, float *out) {
    if(type != GGUF_TYPE_FLOAT32)
        return false;

    return gguf_read_f32(file, out);
}

bool read_gguf_header(FILE *model, struct GGUFHeader *header) {
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
bool read_gguf_metadata(FILE *file, const struct GGUFHeader *header, struct Qwen3Config *config) {
    if(!file || !header || !config)
        return false;

    memset(config, 0, sizeof(*config));

    for(uint64_t i = 0; i < header->metadata_kv_count; i++) {
        char *key = NULL;
        uint32_t type;
        bool ok = false;

        if(!gguf_read_string(file, &key))
            return false;

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
            // We'll parse tokenizer vocabulary later.
            ok = gguf_skip_value(file, type);
        } else {
            // Unknown or currently unused metadata.
            ok = gguf_skip_value(file, type);
        }

        free(key);

        if(!ok)
            return false;
    }

    // Minimal sanity checks.
    if(config->num_layers == 0 || config->hidden_size == 0 || config->num_heads == 0 || config->num_kv_heads == 0 ||
       config->num_heads % config->num_kv_heads != 0) {
        return false;
    }

    return true;
}

int main(int argc, char **argv) {

    if(argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "rb");

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

    if(!read_gguf_metadata(file, &header, &config)) {
        fprintf(stderr, "Failed to parse GGUF metadata\n");
        fclose(file);
        return 1;
    }

    printf("GGUF version: %u\n", header.version);
    printf("Tensor count: %llu\n", (unsigned long long)header.tensor_count);
    printf("Layers: %u\n", config.num_layers);
    printf("Hidden size: %u\n", config.hidden_size);
    printf("Attention heads: %u\n", config.num_heads);
    printf("KV heads: %u\n", config.num_kv_heads);
    printf("Head dimension: %u\n", config.head_dim);
    printf("Context length: %u\n", config.context_length);

    fclose(file);
    return 0;
}
