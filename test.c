#define QWEN3_TEST
#include "main.c"

static void free_tensor_infos(struct GGUFTensorInfo *infos, uint64_t count) {
    if(!infos)
        return;

    for(uint64_t i = 0; i < count; i++)
        free(infos[i].name);

    free(infos);
}

static bool layer_is_complete(const struct Qwen3Layer *layer) {
    return layer->attn_q.data && layer->attn_k.data && layer->attn_v.data &&
           layer->attn_output.data && layer->attn_norm.data && layer->ffn_norm.data &&
           layer->ffn_up.data && layer->ffn_gate.data && layer->ffn_down.data &&
           layer->attn_q_norm.data && layer->attn_k_norm.data;
}

int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "rb");
    if(!file) {
        perror("fopen");
        return 1;
    }

    struct GGUFHeader header = {0};
    struct Qwen3Config config = {0};
    struct GGUFTensorInfo *infos = NULL;
    struct Qwen3Model model = {0};
    int result = 1;

    if(!read_gguf_header(file, &header) || !read_gguf_metadata(file, &header, &config) ||
       !read_gguf_tensor_infos(file, &header, &infos)) {
        fprintf(stderr, "loader test: failed to parse GGUF\n");
        goto cleanup;
    }

    long position = ftell(file);
    if(position < 0) {
        perror("ftell");
        goto cleanup;
    }

    const uint64_t alignment = 32;
    const uint64_t offset = (uint64_t)position;
    const uint64_t tensor_data_start =
        offset + (alignment - (offset % alignment)) % alignment;

    model.header = header;
    model.config = config;

    uint64_t original_offset = infos[2].offset;
    infos[2].offset = UINT64_MAX;
    if(qwen3_map_file(argv[1], infos, tensor_data_start, &model)) {
        fprintf(stderr, "loader test: invalid tensor offset was accepted\n");
        goto cleanup;
    }
    infos[2].offset = original_offset;

    if(!qwen3_map_file(argv[1], infos, tensor_data_start, &model)) {
        fprintf(stderr, "loader test: qwen3_map_file failed\n");
        goto cleanup;
    }

    if(header.version != 3 || header.tensor_count != 310 || config.num_layers != 28 ||
       config.hidden_size != 1024 || !model.output_norm.data ||
       !model.token_embedding.data) {
        fprintf(stderr, "loader test: header/config/model assertions failed\n");
        goto cleanup;
    }

    for(uint32_t i = 0; i < config.num_layers; i++) {
        if(!layer_is_complete(&model.layers[i])) {
            fprintf(stderr, "loader test: layer %u is incomplete\n", i);
            goto cleanup;
        }
    }

    if(model.layers[0].attn_q.shape[0] != 1024 ||
       model.layers[0].attn_q.shape[1] != 2048 ||
       model.layers[0].ffn_down.shape[0] != 3072 ||
       model.layers[0].ffn_down.shape[1] != 1024) {
        fprintf(stderr, "loader test: tensor shape assertions failed\n");
        goto cleanup;
    }

    printf("loader tests passed: %u layers, %llu tensors\n", config.num_layers,
           (unsigned long long)header.tensor_count);
    result = 0;

cleanup:
    destroy_model(&model);
    free_tensor_infos(infos, header.tensor_count);
    fclose(file);
    return result;
}
