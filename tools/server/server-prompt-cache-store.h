#pragma once

#include <cstddef>
#include <cstdint>

void server_prompt_cache_sha256_raw(const uint8_t * data, size_t len, uint8_t * out);
