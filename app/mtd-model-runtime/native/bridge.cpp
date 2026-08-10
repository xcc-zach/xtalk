#include "audio_encoder.hpp"
#include "audio_io.hpp"
#include "audio_span.hpp"
#include "backend.hpp"
#include "generate.hpp"
#include "model_loader.hpp"
#include "qwen3_decoder.hpp"
#include "tokenizer.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

struct xtalk_mtd_context {
    std::unique_ptr<mt::ModelLoader> loader;
    std::string last_error;
};

struct xtalk_mtd_cancel_token {
    std::atomic_bool cancelled{false};
};

namespace {

char* duplicate_string(const std::string& value) {
    auto* result = static_cast<char*>(std::malloc(value.size() + 1));
    if (result == nullptr) return nullptr;
    std::memcpy(result, value.data(), value.size());
    result[value.size()] = '\0';
    return result;
}

std::string trim(const std::string& value) {
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && static_cast<unsigned char>(value[begin]) <= ' ') ++begin;
    while (end > begin && static_cast<unsigned char>(value[end - 1]) <= ' ') --end;
    return value.substr(begin, end - begin);
}

int argmax_first(const std::vector<float>& values) {
    int best = 0;
    for (int index = 1; index < static_cast<int>(values.size()); ++index) {
        if (values[index] > values[best]) best = index;
    }
    return best;
}

std::vector<int32_t> generate(
    mt::Qwen3Decoder& decoder,
    mt::ModelLoader& model,
    const std::vector<float>& fused,
    int sequence_length,
    int max_new,
    int eos,
    const xtalk_mtd_cancel_token* cancel) {
    std::vector<int32_t> ids;
    const int hidden = decoder.hidden();
    if (hidden <= 0 || sequence_length <= 0 || max_new <= 0) return ids;
    if (cancel != nullptr && cancel->cancelled.load(std::memory_order_acquire)) return ids;

    std::vector<float> hidden_states;
    if (!decoder.prefill(fused, sequence_length, &hidden_states)) return ids;
    if (cancel != nullptr && cancel->cancelled.load(std::memory_order_acquire)) return ids;
    if (static_cast<int>(hidden_states.size()) < hidden * sequence_length) return ids;

    std::vector<float> last(hidden_states.end() - hidden, hidden_states.end());
    std::vector<float> logits = decoder.logits_from_hidden(last);
    if (logits.empty()) return ids;
    ids.reserve(static_cast<size_t>(max_new));
    for (;;) {
        if (cancel != nullptr && cancel->cancelled.load(std::memory_order_acquire)) {
            ids.clear();
            return ids;
        }
        const int token = argmax_first(logits);
        ids.push_back(token);
        if (token == eos || static_cast<int>(ids.size()) >= max_new) break;
        std::vector<float> embedding = mt::embed_token(model, token, hidden);
        if (embedding.empty()) return {};
        std::vector<float> decoded = decoder.decode_one(embedding);
        if (static_cast<int>(decoded.size()) < hidden) return {};
        logits = decoder.logits_from_hidden(decoded);
        if (logits.empty()) return {};
    }
    return ids;
}

std::string transcribe(
    mt::ModelLoader& model,
    const std::vector<float>& samples,
    const std::string& instruction,
    const std::string& decoder_prefix,
    int max_new,
    const xtalk_mtd_cancel_token* cancel) {
    if (cancel != nullptr && cancel->cancelled.load(std::memory_order_acquire)) return {};
    const mt::Config& config = model.config();
    const int hidden = config.text_hidden;
    mt::AudioEncoder encoder(model);
    int audio_tokens = 0;
    std::vector<float> audio_embeddings = encoder.encode(samples, audio_tokens, hidden);
    if (audio_embeddings.empty() || audio_tokens <= 0) return {};
    if (cancel != nullptr && cancel->cancelled.load(std::memory_order_acquire)) return {};

    mt::Tokenizer tokenizer;
    if (!tokenizer.load(model)) return {};
    const std::string& prompt = instruction.empty() ? config.default_prompt : instruction;
    std::vector<int32_t> input_ids =
        mt::build_input_ids(tokenizer, config, prompt, audio_tokens);
    if (input_ids.empty()) return {};
    if (!decoder_prefix.empty()) {
        std::vector<int32_t> prefix_ids = tokenizer.encode(decoder_prefix);
        input_ids.insert(input_ids.end(), prefix_ids.begin(), prefix_ids.end());
    }

    std::vector<float> fused = mt::fuse_embeds(
        model,
        input_ids,
        audio_embeddings,
        audio_tokens,
        hidden,
        config.audio_token_id);
    if (fused.empty()) return {};

    const int sequence_length = static_cast<int>(input_ids.size());
    mt::Qwen3Decoder decoder;
    if (!decoder.load(model, sequence_length + max_new + 16)) return {};
    std::vector<int32_t> generated = generate(
        decoder,
        model,
        fused,
        sequence_length,
        max_new,
        config.eos_token_id,
        cancel);
    if (generated.empty()) return {};
    return trim(tokenizer.decode(generated));
}

}  // namespace

extern "C" {

int xtalk_mtd_runtime_available() { return 1; }

const char* xtalk_mtd_backend_name() { return mt::backend_name(); }

int xtalk_mtd_backend_is_cpu() { return ggml_backend_is_cpu(mt::backend()) ? 1 : 0; }

xtalk_mtd_context* xtalk_mtd_load(const char* gguf_path) {
    if (gguf_path == nullptr) return nullptr;
    try {
        auto loader = std::make_unique<mt::ModelLoader>();
        if (!loader->load(gguf_path)) return nullptr;
        loader->promote_small_f16_to_f32();
        auto* context = new xtalk_mtd_context();
        context->loader = std::move(loader);
        return context;
    } catch (...) {
        return nullptr;
    }
}

void xtalk_mtd_free(xtalk_mtd_context* context) { delete context; }

xtalk_mtd_cancel_token* xtalk_mtd_cancel_token_new() {
    return new (std::nothrow) xtalk_mtd_cancel_token();
}

void xtalk_mtd_cancel_token_cancel(xtalk_mtd_cancel_token* token) {
    if (token != nullptr) token->cancelled.store(true, std::memory_order_release);
}

void xtalk_mtd_cancel_token_free(xtalk_mtd_cancel_token* token) { delete token; }

char* xtalk_mtd_transcribe_pcm(
    xtalk_mtd_context* context,
    const float* samples,
    int sample_count,
    int sample_rate,
    const char* instruction,
    const char* decoder_prefix,
    int max_new,
    xtalk_mtd_cancel_token* cancel) {
    if (context == nullptr || context->loader == nullptr) return nullptr;
    if (samples == nullptr || sample_count < 0 || sample_rate <= 0 || max_new <= 0) {
        context->last_error = "invalid transcription arguments";
        return nullptr;
    }
    if (cancel != nullptr && cancel->cancelled.load(std::memory_order_acquire)) {
        context->last_error = "request cancelled";
        return nullptr;
    }
    try {
        std::vector<float> pcm(samples, samples + sample_count);
        if (sample_rate != 16000) pcm = mt::resample_linear(pcm, sample_rate, 16000);
        std::string result = transcribe(
            *context->loader,
            pcm,
            instruction == nullptr ? "" : instruction,
            decoder_prefix == nullptr ? "" : decoder_prefix,
            max_new,
            cancel);
        if (result.empty()) {
            context->last_error = cancel != nullptr
                    && cancel->cancelled.load(std::memory_order_acquire)
                ? "request cancelled"
                : "transcription failed";
            return nullptr;
        }
        char* output = duplicate_string(result);
        if (output == nullptr) {
            context->last_error = "out of memory";
            return nullptr;
        }
        context->last_error.clear();
        return output;
    } catch (const std::exception& error) {
        context->last_error = error.what();
        return nullptr;
    } catch (...) {
        context->last_error = "unknown transcription error";
        return nullptr;
    }
}

void xtalk_mtd_free_string(char* value) { std::free(value); }

const char* xtalk_mtd_last_error(xtalk_mtd_context* context) {
    return context == nullptr ? "invalid model context" : context->last_error.c_str();
}

}  // extern "C"
