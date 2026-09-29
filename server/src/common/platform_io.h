// Portable host I/O and host-shared words: the POSIX / Windows differences
// live in platform_io.cpp only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace luce::common {

// A read-only file with positional reads that several threads may issue at
// once. With `direct` the page cache is bypassed (O_DIRECT); reads must then
// cover whole kDirectAlign blocks into kDirectAlign-aligned memory, and open
// fails where direct reads are unsupported (Windows) so the caller can fall
// back to another path.
class ReadOnlyFile {
public:
    static constexpr size_t kDirectAlign = 4096;

    ReadOnlyFile() = default;
    ReadOnlyFile(ReadOnlyFile && o) noexcept { *this = static_cast<ReadOnlyFile &&>(o); }
    ReadOnlyFile & operator=(ReadOnlyFile && o) noexcept;
    ReadOnlyFile(const ReadOnlyFile &) = delete;
    ReadOnlyFile & operator=(const ReadOnlyFile &) = delete;
    ~ReadOnlyFile() { close(); }

    bool open(const std::string & path, bool direct = false, std::string * err = nullptr);
    void close();
    bool is_open() const { return handle_ >= 0; }
    bool direct() const { return direct_; }
    uint64_t size() const { return size_; }

    // Reads exactly `size` bytes at `offset`; false on an error or end of
    // file (errno set).
    bool read_at(uint64_t offset, void * buf, size_t size) const;
    // Reads up to `size` bytes at `offset`, stopping early only at the end
    // of the file; the byte count, or -1 on an error (errno set).
    int64_t read_upto(uint64_t offset, void * buf, size_t size) const;
    // Hint that [offset, offset + size) will be read soon; no-op where the
    // platform has no such hint.
    void advise_willneed(uint64_t offset, size_t size) const;

private:
    intptr_t handle_ = -1;   // POSIX descriptor or Windows HANDLE
    uint64_t size_ = 0;
    bool direct_ = false;
};

// Memory the host can still hand out without swapping (Linux MemAvailable,
// Windows available physical memory); 0 when unknown.
uint64_t host_available_bytes();

// Hint that [offset, offset + size) of a read-only mapping of `map_size`
// bytes at `map` will be read soon; no-op where unsupported.
void advise_mapped_willneed(const void * map, size_t map_size, uint64_t offset, size_t size);

// Bytes of [offset, offset + size) of a read-only mapping at `map` that are in
// memory now (whole pages counted); 0 where the platform cannot tell.
size_t mapped_resident_bytes(const void * map, uint64_t offset, size_t size);

// Acquire / release on a 32-bit word shared with a device or another thread
// through host-mapped memory, and the spin-wait pause.
uint32_t host_word_load_acquire(const uint32_t * p);
void host_word_store_release(uint32_t * p, uint32_t v);
void cpu_relax();

}  // namespace luce::common
