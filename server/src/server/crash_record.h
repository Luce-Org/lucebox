// The crash record: what a fix needs when the server dies on a fatal signal.
//
// With LUCE_CRASH_DIR set, a SIGSEGV, SIGBUS, SIGILL, SIGFPE or SIGABRT (an
// assert, an uncaught exception) writes crash-<pid>-<unix time>.txt there: the
// signal, the faulting address, the stack's return addresses and the process's
// memory map, so the addresses resolve offline against the binary and its
// libraries. Then the process dies as it would have. The record holds no prompt
// or output text. Linux only; a no-op elsewhere.
//
// The handler allocates nothing and formats nothing it did not prepare: it runs
// on its own stack, writes with write(2) to a file it opens with openat(2) in a
// folder opened at startup, and copies /proc/self/maps with read(2).
#pragma once

#include <string>

namespace luce::common {

// Opens `dir` and installs the handlers; false, with nothing installed, when
// `dir` is empty or cannot be opened. `about` is a line of the record's header
// (the binary, its build), formatted now.
bool install_crash_record(const char * dir, const std::string & about);

// Writes a record to `fd` as the handler does: header, signal, address, frames,
// then the memory map. Exposed so a test reads the format without dying.
void write_crash_record(int fd, int sig, const void * address, void * const * frames, int n_frames);

}  // namespace luce::common
