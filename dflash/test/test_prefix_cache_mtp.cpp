#include "internal.h"
#include "qwen36/qwen36_mtp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

using dflash27b::PrefixSnapshot;
using dflash27b::mtp::Qwen36MtpModule;
using dflash27b::mtp::Qwen36MtpWeights;

void require(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", msg);
        std::abort();
    }
}

Qwen36MtpWeights tiny_mtp_weights() {
    Qwen36MtpWeights w;
    w.n_embd = 4;
    w.n_vocab = 16;
    w.n_heads = 1;
    w.n_head_kv = 1;
    w.n_key_length = 2;
    w.n_value_length = 3;
    w.heads.resize(1);
    return w;
}

std::vector<uint8_t> patterned_bytes(size_t n, uint8_t seed) {
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint8_t>(seed + (i * 17u));
    }
    return out;
}

struct FakeQwen35Backend {
    Qwen36MtpModule mtp;
    PrefixSnapshot snap;
    int cold_warm_calls = 0;

    FakeQwen35Backend() {
        mtp.attach_weights_for_test(tiny_mtp_weights());
    }

    bool cold_warm_from_bytes(const std::vector<uint8_t> & bytes,
                              int pos,
                              int32_t prefill_next) {
        cold_warm_calls++;
        snap.prefill_next_tok = prefill_next;
        return mtp.restore_head_kv(std::vector<std::vector<uint8_t>>{bytes},
                                   std::vector<int>{pos});
    }

    bool save_snapshot(int32_t prefill_next) {
        snap.prefill_next_tok = prefill_next;
        return mtp.snapshot_head_kv(snap.mtp_head_kv, snap.mtp_head_pos);
    }

    enum class RestoreStatus {
        Restored,
        Mismatch,
        RestoreFailed,
    };

    RestoreStatus restore_or_report_mismatch(int32_t post_prefill_argmax) {
        if (snap.prefill_next_tok != post_prefill_argmax) {
            return RestoreStatus::Mismatch;
        }
        if (!mtp.restore_head_kv(snap.mtp_head_kv, snap.mtp_head_pos)) {
            return RestoreStatus::RestoreFailed;
        }
        return RestoreStatus::Restored;
    }

    // R1: caller checks captured-vs-current shape contract before restore.
    // Returns Mismatch when any of γ / n_head_kv / n_ctx diverge.
    RestoreStatus restore_with_shape_check(int32_t post_prefill_argmax,
                                           int cur_gamma,
                                           int cur_n_head_kv,
                                           int cur_n_ctx) {
        if (snap.prefill_next_tok != post_prefill_argmax) {
            return RestoreStatus::Mismatch;
        }
        if (snap.mtp_gamma_at_capture != cur_gamma ||
            snap.mtp_n_head_kv        != cur_n_head_kv ||
            snap.mtp_n_ctx            != cur_n_ctx) {
            return RestoreStatus::Mismatch;
        }
        if (!mtp.restore_head_kv(snap.mtp_head_kv, snap.mtp_head_pos)) {
            return RestoreStatus::RestoreFailed;
        }
        return RestoreStatus::Restored;
    }

    void capture_shape_contract(int gamma, int n_head_kv, int n_ctx) {
        snap.mtp_gamma_at_capture = gamma;
        snap.mtp_n_head_kv        = n_head_kv;
        snap.mtp_n_ctx            = n_ctx;
    }

    std::vector<uint8_t> head_kv_bytes() {
        std::vector<std::vector<uint8_t>> out;
        std::vector<int> pos;
        require(mtp.snapshot_head_kv(out, pos), "snapshot_head_kv failed");
        require(out.size() == 1, "expected one MTP head snapshot");
        return out[0];
    }
};

void t1_snapshot_restore_head_kv_bit_equal() {
    FakeQwen35Backend backend;
    const size_t bytes_per_head =
        (64u * 1u * 2u + 64u * 1u * 3u) * sizeof(float);
    const auto cold = patterned_bytes(bytes_per_head, 0x31);

    require(backend.cold_warm_from_bytes(cold, 5, 1234), "cold warm inject failed");
    const auto cold_capture = backend.head_kv_bytes();
    require(cold_capture == cold, "cold capture mismatch");
    require(backend.save_snapshot(1234), "snapshot save failed");

    const auto wiped = patterned_bytes(bytes_per_head, 0x00);
    require(backend.cold_warm_from_bytes(wiped, 0, 1234), "wipe inject failed");
    require(backend.head_kv_bytes() != cold_capture, "wipe did not change KV bytes");

    require(backend.restore_or_report_mismatch(1234)
            == FakeQwen35Backend::RestoreStatus::Restored,
            "restore did not report success");
    require(backend.head_kv_bytes() == cold_capture, "restored KV is not bit-equal");
    std::puts("T1 snapshot_restore_head_kv_bit_equal PASS");
}

void t2_prefill_next_mismatch_falls_back_to_cold_warm() {
    FakeQwen35Backend backend;
    const size_t bytes_per_head =
        (64u * 1u * 2u + 64u * 1u * 3u) * sizeof(float);
    const auto warm_a = patterned_bytes(bytes_per_head, 0x42);
    const auto warm_b = patterned_bytes(bytes_per_head, 0x99);

    require(backend.cold_warm_from_bytes(warm_a, 7, 111), "warm A inject failed");
    require(backend.save_snapshot(111), "snapshot save failed");

    const int before = backend.cold_warm_calls;
    auto status = backend.restore_or_report_mismatch(222);
    require(status == FakeQwen35Backend::RestoreStatus::Mismatch,
            "mismatch was not reported");
    require(backend.cold_warm_calls == before, "mismatch restored or warmed eagerly");

    require(backend.cold_warm_from_bytes(warm_b, 7, 222), "fallback warm failed");
    require(backend.cold_warm_calls == before + 1, "fallback warm call not counted");
    require(backend.head_kv_bytes() == warm_b, "fallback warm bytes not installed");
    std::puts("T2 prefill_next_mismatch_falls_back_to_cold_warm PASS");
}

void t3_prefill_next_round_trip_exact_int32() {
    FakeQwen35Backend backend;
    const size_t bytes_per_head =
        (64u * 1u * 2u + 64u * 1u * 3u) * sizeof(float);
    const int32_t tok = std::numeric_limits<int32_t>::max();

    require(backend.cold_warm_from_bytes(patterned_bytes(bytes_per_head, 0x55),
                                         11,
                                         tok),
            "round-trip inject failed");
    require(backend.save_snapshot(tok), "round-trip snapshot save failed");

    PrefixSnapshot round_trip = backend.snap;
    require(round_trip.prefill_next_tok == tok, "prefill_next_tok narrowed");
    require(round_trip.mtp_head_pos == std::vector<int>{11}, "head position changed");
    std::puts("T3 prefill_next_round_trip_exact_int32 PASS");
}

void t4_shape_contract_mismatch_rejects_restore() {
    FakeQwen35Backend backend;
    const size_t bytes_per_head =
        (64u * 1u * 2u + 64u * 1u * 3u) * sizeof(float);
    const auto warm = patterned_bytes(bytes_per_head, 0x7c);

    require(backend.cold_warm_from_bytes(warm, 9, 555), "warm inject failed");
    require(backend.save_snapshot(555), "snapshot save failed");
    // Capture a "production" shape contract on this snapshot.
    backend.capture_shape_contract(/*gamma=*/3, /*n_head_kv=*/1, /*n_ctx=*/24576);

    // Same token, same shape -> Restored.
    require(backend.restore_with_shape_check(555, 3, 1, 24576)
            == FakeQwen35Backend::RestoreStatus::Restored,
            "matching shape failed to restore");

    // γ change at restore time -> Mismatch (silent corruption blocked).
    require(backend.restore_with_shape_check(555, 4, 1, 24576)
            == FakeQwen35Backend::RestoreStatus::Mismatch,
            "γ mismatch was not rejected");
    // n_head_kv change -> Mismatch.
    require(backend.restore_with_shape_check(555, 3, 2, 24576)
            == FakeQwen35Backend::RestoreStatus::Mismatch,
            "n_head_kv mismatch was not rejected");
    // n_ctx change -> Mismatch.
    require(backend.restore_with_shape_check(555, 3, 1, 32768)
            == FakeQwen35Backend::RestoreStatus::Mismatch,
            "n_ctx mismatch was not rejected");

    std::puts("T4 shape_contract_mismatch_rejects_restore PASS");
}

}  // namespace

int main() {
    t1_snapshot_restore_head_kv_bit_equal();
    t2_prefill_next_mismatch_falls_back_to_cold_warm();
    t3_prefill_next_round_trip_exact_int32();
    t4_shape_contract_mismatch_rejects_restore();
    std::puts("ALL PASS");
    return 0;
}
