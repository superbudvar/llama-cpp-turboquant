get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

# File-presence sentinel: when LLAMA_TEST_FIXTURES is unset or the directory
# is missing, exit with non-zero so the ctest test is marked FAILED (intentional,
# labelled). Downstream tests with FIXTURES_REQUIRED test-download-model skip
# cleanly due to the fixture failure (ctest only skips fixture dependents when
# the setup test fails, not when it is skipped). With a populated fixture
# directory the test runs as before; no runtime network probe is performed.
if(NOT DEFINED ENV{LLAMA_TEST_FIXTURES})
    message(STATUS "test-download-model SKIPPED: fixture not present at $ENV{LLAMA_TEST_FIXTURES}")
    message(FATAL_ERROR "skipping per LLAMA_TEST_FIXTURES sentinel")
endif()
set(FIXTURES_DIR "$ENV{LLAMA_TEST_FIXTURES}")
if(NOT IS_DIRECTORY "${FIXTURES_DIR}")
    message(STATUS "test-download-model SKIPPED: fixture not present at ${FIXTURES_DIR}")
    message(FATAL_ERROR "skipping per LLAMA_TEST_FIXTURES sentinel")
endif()

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ggml-org/models...")
endif()

file(DOWNLOAD
    "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true"
    "${DEST}"
    TLS_VERIFY ON
    EXPECTED_HASH ${HASH}
    STATUS status
)

list(GET status 0 code)

if(NOT code EQUAL 0)
    list(GET status 1 msg)
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
endif()
