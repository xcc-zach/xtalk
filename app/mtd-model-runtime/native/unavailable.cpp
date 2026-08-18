#include <cstdlib>

extern "C" {

int xtalk_mtd_runtime_available() { return 0; }
const char* xtalk_mtd_backend_name() { return "unavailable"; }
int xtalk_mtd_backend_is_cpu() { return 0; }
void* xtalk_mtd_load(const char*) { return nullptr; }
void xtalk_mtd_free(void*) {}
void* xtalk_mtd_cancel_token_new() { return nullptr; }
void xtalk_mtd_cancel_token_cancel(void*) {}
void xtalk_mtd_cancel_token_free(void*) {}
char* xtalk_mtd_transcribe_pcm(
    void*, const float*, int, int, const char*, const char*, int, void*) {
    return nullptr;
}
void xtalk_mtd_free_string(char* value) { std::free(value); }
const char* xtalk_mtd_last_error(void*) {
    return "moss-transcribe.cpp was not included in this build";
}

}
