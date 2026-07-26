// SA4 (SO-ARM101 validation plan): quantified A/B evidence for the stream
// layer's reason to exist — a low-rate intent stream (VLA / teleoperation
// class, ~33 frames/s) delivered to a 1 kHz command cycle.
//
//   A (baseline): zero-order hold — the setpoint jumps to the newest frame
//     at every frame boundary. This is what naive integration does with a
//     low-rate policy output.
//   B: the same frames pushed through JointStreamGroup in upsample mode
//     (KB-035): per-joint OTG filtering makes the envelope part of the
//     solution, not an afterthought.
//
// Metrics are third differences of the emitted 1 kHz position stream
// (discrete jerk, per-cycle units). The B path must respect the configured
// jerk limit BY CONSTRUCTION; the A/B separation ratio and the tracking
// error are pinned as recorded-baseline ratchets. The numbers printed here
// are the SA4 evidence feeding the plan's demo material.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "stream/joint_group.h"

namespace
{

using namespace plcopen::core;

constexpr std::size_t JointCount = 2;
constexpr std::int64_t TotalCycles = 3000;
constexpr std::int64_t FrameInterval = 30; // ~33 frames/s at 1 kHz

int fail(const char *name)
{
    std::printf("FAIL %s\n", name);
    return 1;
}

// Smooth reference intent (per-cycle time base): two incommensurate tones,
// phase-shifted per joint. Amplitudes sized so the filter limits below are
// comfortable — the test isolates delivery smoothness, not saturation.
double reference(std::size_t joint, std::int64_t cycle)
{
    const double t = static_cast<double>(cycle);
    const double phase = 0.7 * static_cast<double>(joint);
    return 0.35 * std::sin(2.0 * 3.14159265358979323846 * t / 2000.0 + phase) +
           0.12 * std::sin(2.0 * 3.14159265358979323846 * t / 770.0);
}

double reference_velocity(std::size_t joint, std::int64_t cycle)
{
    return reference(joint, cycle + 1) - reference(joint, cycle);
}

struct JerkStats
{
    double max_abs = 0.0;
    double rms = 0.0;
};

JerkStats jerk_of(const double *positions, std::int64_t count)
{
    JerkStats stats{};
    double sum_sq = 0.0;
    std::int64_t samples = 0;
    for(std::int64_t i = 3; i < count; ++i) {
        const double jerk = positions[i] - 3.0 * positions[i - 1] +
                            3.0 * positions[i - 2] - positions[i - 3];
        const double magnitude = std::fabs(jerk);
        if(magnitude > stats.max_abs) stats.max_abs = magnitude;
        sum_sq += jerk * jerk;
        ++samples;
    }
    stats.rms = samples > 0 ? std::sqrt(sum_sq / static_cast<double>(samples))
                            : 0.0;
    return stats;
}

int check_upsample_ab_metrics()
{
    stream::JointStreamGroupConfig config{};
    config.mode = stream::JointFrameMode::upsample;
    config.joint_count = JointCount;
    config.gain_ramp_cycles = 4;
    for(std::size_t joint = 0; joint < JointCount; ++joint) {
        stream::JointStreamConfig &member = config.joints[joint];
        member.filter.limits = {0.5, 0.05, 0.05, 0.01};
        member.filter.timeout_cycles = 2 * FrameInterval;
        member.filter.extrapolation_cycles = FrameInterval;
        member.max_abs_tau_ff = 1.0;
        member.min_kp = 0.0;
        member.max_kp = 100.0;
        member.min_kd = 0.0;
        member.max_kd = 20.0;
        member.safe_kp = 2.0;
        member.safe_kd = 1.0;
    }

    stream::JointStreamGroup group;
    if(group.configure_frame(config) != rt::ErrorCode::ok) {
        return fail("configure_frame");
    }
    for(std::size_t joint = 0; joint < JointCount; ++joint) {
        if(group.reset(joint, {reference(joint, 0), 0.0, 0.0}) !=
           rt::ErrorCode::ok) {
            return fail("reset");
        }
    }

    static double baseline[JointCount][TotalCycles];
    static double filtered[JointCount][TotalCycles];

    double hold[JointCount];
    for(std::size_t joint = 0; joint < JointCount; ++joint) {
        hold[joint] = reference(joint, 0);
    }

    for(std::int64_t cycle = 0; cycle < TotalCycles; ++cycle) {
        if(cycle % FrameInterval == 0) {
            stream::JointCommandFrame frame{};
            frame.joint_count = JointCount;
            frame.timestamp_cycles = cycle + 1;
            for(std::size_t joint = 0; joint < JointCount; ++joint) {
                frame.joints[joint].q_des = reference(joint, cycle);
                frame.joints[joint].dq_des = reference_velocity(joint, cycle);
                frame.joints[joint].kp = 10.0;
                frame.joints[joint].kd = 1.0;
                hold[joint] = reference(joint, cycle);
            }
            if(group.push_frame(frame) != rt::ErrorCode::ok) {
                return fail("push_frame");
            }
        }
        group.cycle();
        const stream::JointSetpointFrame &setpoints =
            group.read_setpoint_frame();
        for(std::size_t joint = 0; joint < JointCount; ++joint) {
            baseline[joint][cycle] = hold[joint];
            filtered[joint][cycle] = setpoints.joints[joint].position;
        }
    }

    int failures = 0;
    for(std::size_t joint = 0; joint < JointCount; ++joint) {
        const JerkStats a = jerk_of(baseline[joint], TotalCycles);
        const JerkStats b = jerk_of(filtered[joint], TotalCycles);

        double worst_tracking = 0.0;
        for(std::int64_t cycle = 500; cycle < TotalCycles; ++cycle) {
            const double error =
                std::fabs(filtered[joint][cycle] - reference(joint, cycle));
            if(error > worst_tracking) worst_tracking = error;
        }

        std::printf(
            "STREAM_AB joint=%zu zoh_max_jerk=%.3e zoh_rms_jerk=%.3e "
            "filtered_max_jerk=%.3e filtered_rms_jerk=%.3e "
            "separation=%.1fx tracking_error=%.4f\n",
            joint, a.max_abs, a.rms, b.max_abs, b.rms,
            b.max_abs > 0.0 ? a.max_abs / b.max_abs : 0.0, worst_tracking);

        // Construction promise: the filtered stream respects the configured
        // per-cycle jerk limit (0.01) with quantization headroom.
        if(b.max_abs > 0.011) {
            failures += fail("filtered jerk exceeds configured limit");
        }
        // Recorded-baseline ratchets (measured 268x/303x separation and
        // 0.007/0.015 tracking; generous headroom so only real regressions
        // trip): the ZOH step train carries frame-sized third differences
        // while the filtered stream stays two orders of magnitude below.
        if(a.max_abs / b.max_abs < 100.0) {
            failures += fail("A/B separation collapsed");
        }
        if(worst_tracking > 0.03) {
            failures += fail("filtered tracking error above ratchet");
        }
    }
    return failures;
}

} // namespace

int main()
{
    int failures = 0;
    failures += check_upsample_ab_metrics();
    if(failures == 0) {
        std::printf("PASS stream AB metrics\n");
    }
    return failures > 0 ? 1 : 0;
}
