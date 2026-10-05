// Compile against the installed parakeet headers; Python never guesses ABI structs.
#include "parakeet.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static bool expired(void * data) { return Clock::now() >= *static_cast<Clock::time_point *>(data); }

extern "C" void * spark_asr_open(const char * path) {
    try {
        ggml_backend_load_all();
        auto p = parakeet_context_default_params();
        p.use_gpu = false;
        return parakeet_init_from_file_with_params(path, p);
    } catch (...) { return nullptr; }
}

extern "C" int spark_asr_decode(void * handle, const int16_t * pcm, int n,
                               int threads, char * output, int capacity) {
    if (!handle || n <= 0 || n > 16000 * 30 || capacity < 2) return -1;
    try {
        std::vector<float> samples(n);
        for (int i = 0; i < n; ++i) samples[i] = pcm[i] / 32768.0f;
        auto p = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
        p.n_threads = threads;
        p.no_context = true; // no previous utterance may leak into the next one
        auto deadline = Clock::now() + std::chrono::seconds(5);
        p.abort_callback = expired;
        p.abort_callback_user_data = &deadline;
        auto ctx = static_cast<parakeet_context *>(handle);
        int result = parakeet_full(ctx, p, samples.data(), n);
        if (result != 0 || expired(&deadline)) return -2;
        std::string text;
        for (int i = 0; i < parakeet_full_n_segments(ctx); ++i) {
            if (!text.empty()) text += ' ';
            text += parakeet_full_get_segment_text(ctx, i);
        }
        if (text.size() >= static_cast<size_t>(capacity)) return -3;
        std::memcpy(output, text.c_str(), text.size() + 1);
        return 0;
    } catch (...) { return -4; }
}
