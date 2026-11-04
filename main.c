#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

struct GGUFTensorInfo {
    char *name;
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
    const void *data;
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

    void *mmap_base;
    size_t mmap_size;

    struct GGUFHeader header;
    struct Qwen3Config config;
    struct Qwen3Layer *layers;

    struct Tensor output_norm;
    struct Tensor token_embedding;
    struct Tensor output;
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

// read exactly n bytes from the file into dst.
static bool gguf_read(FILE *file, void *dst, size_t n) {
    return fread(dst, 1, n, file) == n;
}

static bool gguf_read_u32(FILE *file, uint32_t *value) {
    unsigned char b[4];

    if(!gguf_read(file, b, sizeof(b)))
        return false;

    *value = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
             ((uint32_t)b[3] << 24);

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
// important reading:
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md?utm_source=chatgpt.com#file-structure
bool read_gguf_metadata(FILE *file, const struct GGUFHeader *header,
                        struct Qwen3Config *config) {
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
    if(config->num_layers == 0 || config->hidden_size == 0 || config->num_heads == 0 ||
       config->num_kv_heads == 0 || config->num_heads % config->num_kv_heads != 0) {
        return false;
    }

    return true;
}

static bool read_gguf_tensor_info(FILE *file, struct GGUFTensorInfo *info) {
    memset(info, 0, sizeof(*info));

    if(!gguf_read_string(file, &info->name))
        return false;

    if(!gguf_read_u32(file, &info->ndim))
        goto fail;

    if(info->ndim == 0 || info->ndim > 4)
        goto fail;

    for(uint32_t i = 0; i < info->ndim; i++) {
        if(!gguf_read_u64(file, &info->shape[i]))
            goto fail;

        if(info->shape[i] == 0)
            goto fail;
    }

    if(!gguf_read_u32(file, &info->type))
        goto fail;

    if(!gguf_read_u64(file, &info->offset))
        goto fail;

    return true;

fail:
    free(info->name);
    info->name = NULL;
    return false;
}

bool read_gguf_tensor_infos(FILE *file, const struct GGUFHeader *header,
                            struct GGUFTensorInfo **out_infos) {

    if(!file || !header || !out_infos)
        return false;

    *out_infos = NULL;

    if(header->tensor_count == 0 ||
       header->tensor_count > SIZE_MAX / sizeof(struct GGUFTensorInfo))
        return false;

    struct GGUFTensorInfo *infos =
        calloc((size_t)header->tensor_count, sizeof(struct GGUFTensorInfo));

    if(!infos)
        return false;

    for(uint64_t i = 0; i < header->tensor_count; i++) {
        if(!read_gguf_tensor_info(file, &infos[i])) {
            for(uint64_t j = 0; j < i; j++)
                free(infos[j].name);

            free(infos);
            return false;
        }
    }

    *out_infos = infos;
    return true;
}

bool qwen3_map_file(const char *path, struct GGUFTensorInfo *infos,
                    uint64_t tensor_data_start, struct Qwen3Model *model) {

    FILE *file = fopen(path, "rb");
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
    if(file_size == -1) {
        fclose(file);
        return false;
    }

    if(fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return false;
    }

    void *base = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(base == MAP_FAILED) {
        fclose(file);
        return false;
    }

    model->mmap_base = base;
    model->mmap_size = file_size;

    // start mapping tensors;

    // INFO: output_norm.weight is the first tensor in the tensor info array
    struct Tensor output_norm = {.ndim = infos[0].ndim,
                                 .type = infos[0].type,
                                 .data = (const uint8_t *)model->mmap_base +
                                         tensor_data_start + infos[0].offset};
    memcpy(output_norm.shape, infos[0].shape, sizeof(infos[0].shape));

    // INFO: token_embed.weight is the second tensor in the tensor info array
    struct Tensor token_embedding = {.ndim = infos[1].ndim,
                                     .type = infos[1].type,
                                     .data = (const uint8_t *)model->mmap_base +
                                             tensor_data_start + infos[1].offset};
    memcpy(token_embedding.shape, infos[1].shape, sizeof(infos[1].shape));

    // INFO: embed the output norm and token_embed in the model

    model->output_norm = output_norm;
    model->token_embedding = token_embedding;

    // now we want to start mapping the remaining tensors and layers
    // we will iterate over the tensor infos starting from index 2
    // parse tensor names to find layer index and component
    // assign tensors to the correct layer fields

    // allocate layers array
    model->layers = calloc(model->config.num_layers, sizeof(struct Qwen3Layer));
    if(!model->layers) {
        fclose(file);
        return false;
    }

    for(uint64_t i = 2; i < model->header.tensor_count; i++) {
        char *name = infos[i].name;
        uint64_t offset = infos[i].offset;

        struct Tensor tensor = {.data = (const uint8_t *)model->mmap_base +
                                        tensor_data_start + offset,
                                .ndim = infos[i].ndim,
                                .type = infos[i].type};

        memcpy(tensor.shape, infos[i].shape, sizeof(infos[i].shape));

        // simple parsing: find layer index from name
        int layer_idx = -1;
        if(sscanf(name, "transformer.layers.%d.", &layer_idx) != 1) {
            // not a layer tensor, skip
            continue;
        }

        if(layer_idx < 0 || (uint32_t)layer_idx >= model->config.num_layers) {
            // invalid layer index, skip
            continue;
        }

        struct Qwen3Layer *layer = &model->layers[layer_idx];

        // map tensor to layer field based on suffix
        if(strstr(name, "attention.wq.weight")) {
            layer->attn_q = tensor;
        } else if(strstr(name, "attention.wk.weight")) {
            layer->attn_k = tensor;
        } else if(strstr(name, "attention.wv.weight")) {
            layer->attn_v = tensor;
        } else if(strstr(name, "attention.wo.weight")) {
            layer->attn_output = tensor;
        } else if(strstr(name, "attention_norm.weight")) {
            layer->attn_norm = tensor;
        } else if(strstr(name, "ffn_norm.weight")) {
            layer->ffn_norm = tensor;
        } else if(strstr(name, "ffn.up_proj.weight")) {
            layer->ffn_up = tensor;
        } else if(strstr(name, "ffn.gate_proj.weight")) {
            layer->ffn_gate = tensor;
        } else if(strstr(name, "ffn.down_proj.weight")) {
            layer->ffn_down = tensor;
        } else if(strstr(name, "attention.wq_norm.weight")) {
            layer->attn_q_norm = tensor;
        } else if(strstr(name, "attention.wk_norm.weight")) {
            layer->attn_k_norm = tensor;
        }
    }

    return true;

    // this honestly should do the work, pretty short and cool, obv better ways
    // to optimize this but i'm a strict deadline for this engine
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

    struct GGUFTensorInfo *infos = NULL;

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

    uint64_t alignment = 32; // GGUF default
    uint64_t offset = (uint64_t)position;

    uint64_t tensor_data_start = offset + (alignment - (offset % alignment)) % alignment;

    struct Qwen3Model *model = malloc(sizeof(struct Qwen3Model));

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

    // we could theoretically free the info here, the main problem is that i
    // print everything out right after
    // maybe i should print out the actual layers and their tensors instead ?
    // idk i'm lazy asfuck rn

    // NOTE: print model data basically

    printf("GGUF version: %u\n", header.version);
    printf("Tensor count: %llu\n", (unsigned long long)header.tensor_count);
    printf("Layers: %u\n", config.num_layers);
    printf("Hidden size: %u\n", config.hidden_size);
    printf("Attention heads: %u\n", config.num_heads);
    printf("KV heads: %u\n", config.num_kv_heads);
    printf("Head dimension: %u\n", config.head_dim);
    printf("Context length: %u\n", config.context_length);

    for(uint64_t i = 0; i < header.tensor_count; i++) {
        printf("%s | ndim=%u | type=%u | offset=%llu\n", infos[i].name, infos[i].ndim,
               infos[i].type, (unsigned long long)infos[i].offset);
    }

    // print some model layer and tensor stuff to verify it worked
    for(uint32_t i = 0; i < config.num_layers; i++) {
        struct Qwen3Layer *layer = &model->layers[i];
        printf("layer %u:\n", i);
        printf("  attn_q shape: %ld %ld\n", layer->attn_q.shape[0],
               layer->attn_q.shape[1]);
        printf("  attn_k shape: %ld %ld\n", layer->attn_k.shape[0],
               layer->attn_k.shape[1]);
        printf("  attn_v shape: %ld %ld\n", layer->attn_v.shape[0],
               layer->attn_v.shape[1]);
        printf("  attn_output shape: %ld %ld\n", layer->attn_output.shape[0],
               layer->attn_output.shape[1]);
        printf("  attn_norm shape: %ld\n", layer->attn_norm.shape[0]);
        printf("  ffn_norm shape: %ld\n", layer->ffn_norm.shape[0]);
        printf("  ffn_up shape: %ld %ld\n", layer->ffn_up.shape[0],
               layer->ffn_up.shape[1]);
        printf("  ffn_gate shape: %ld %ld\n", layer->ffn_gate.shape[0],
               layer->ffn_gate.shape[1]);
        printf("  ffn_down shape: %ld %ld\n", layer->ffn_down.shape[0],
               layer->ffn_down.shape[1]);
        printf("  attn_q_norm shape: %ld\n", layer->attn_q_norm.shape[0]);
        printf("  attn_k_norm shape: %ld\n", layer->attn_k_norm.shape[0]);
    }

    // NOTE: free pointers

    for(uint64_t i = 0; i < header.tensor_count; i++)
        free(infos[i].name);

    free(infos);
    fclose(file);
    return 0;
}
