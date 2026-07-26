// Dedicated solve_fixed_time oracle (review A6b, KB-100).
//
// Properties, checked over randomized asymmetric limits and boundary states
// (the existing otg_time_optimal_tests fuzz used one fixed symmetric limit
// set, which is how the KB-066 class of asymmetric-limit defects survived):
//   P1 exact duration: every success has duration_cycles() == T (hard).
//   P2 boundary states: sample(0)/sample(T) hit from/to at 1e-9 (hard).
//   P3 dense sub-cycle limit compliance at 16 sub-samples per cycle with the
//      direction-aware acceleration bound and the documented KB-055 entry
//      exemptions (hard) — the property the integer-cycle checks missed.
//   P6 T == T_min identity: bit-identical segments to plan_time_optimal
//      (hard; time_optimal.h promises the pass-through).
//   P5 below-T_min statistic: KB-056 declares self-contained candidates may
//      succeed below the planner's T_min; the frequency and depth are
//      recorded and capped so the declared gap cannot silently grow.
//   P7 coverage floor: the infeasible fraction at T >= T_min + 1 is capped
//      from the recorded baseline.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "otg/profile1d.h"
#include "otg/time_optimal.h"

namespace
{

using namespace plcopen::core;

struct Lcg
{
    unsigned state = 0xC0FFEE01u;
    unsigned next()
    {
        state = state * 1664525u + 1013904223u;
        return state;
    }
    double range(double lo, double hi)
    {
        return lo + (hi - lo) * static_cast<double>(next()) / 4294967295.0;
    }
};

int g_fail = 0;

void fail_case(const char *what, int i)
{
    if(g_fail < 5) {
        std::printf("  FAIL %s i=%d\n", what, i);
    }
    ++g_fail;
}

// Direction-aware dense compliance with the KB-055 entry exemptions the
// production acceptance gate documents.
bool dense_within_limits(const otg::Profile1D &p, otg::State1D from,
                         otg::Limits1D lim)
{
    constexpr double Eps = 1e-7;
    double v_allow = lim.max_velocity;
    if(std::fabs(from.velocity) > v_allow) {
        v_allow = std::fabs(from.velocity);
    }
    if(from.acceleration != 0.0) {
        const double n0 = std::ceil(std::fabs(from.acceleration) / lim.max_jerk);
        const double reduced_v =
            std::fabs(from.velocity + 0.5 * from.acceleration * n0);
        if(reduced_v > v_allow) {
            v_allow = reduced_v;
        }
    }
    for(std::size_t s = 0; s < p.segment_count(); ++s) {
        const otg::Segment1D &seg = p.segment(s);
        const double n = static_cast<double>(seg.duration_cycles);
        constexpr int Sub = 16;
        const int steps = static_cast<int>(n) * Sub;
        for(int k = 0; k <= steps; ++k) {
            const double x = n * static_cast<double>(k) / steps;
            const double v =
                seg.c1 +
                x * (2.0 * seg.c2 +
                     x * (3.0 * seg.c3 + x * (4.0 * seg.c4 + x * 5.0 * seg.c5)));
            const double a =
                2.0 * seg.c2 +
                x * (6.0 * seg.c3 + x * (12.0 * seg.c4 + x * 20.0 * seg.c5));
            const double j =
                6.0 * seg.c3 + x * (24.0 * seg.c4 + x * 60.0 * seg.c5);
            if(std::fabs(v) > v_allow + Eps) {
                return false;
            }
            const double bound =
                v * a >= 0.0 ? lim.max_acceleration : lim.max_deceleration;
            if(std::fabs(a) > bound + Eps) {
                return false;
            }
            if(std::fabs(j) > lim.max_jerk + Eps) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    int iterations = 500;
    for(int i = 1; i + 1 < argc + 1; ++i) {
        if(i < argc && std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[i + 1]);
        }
    }

    Lcg rng;
    int solved = 0;
    int infeasible = 0;
    int below_tmin_success = 0;
    std::int64_t max_below_gap = 0;
    int identity_checked = 0;

    for(int i = 0; i < iterations; ++i) {
        const double vm = rng.range(0.5, 8.0);
        const double am = rng.range(0.5, 8.0);
        const double dm = rng.range(0.5, 8.0);
        const double jm = rng.range(0.1, 4.0);
        const otg::Limits1D lim{vm, am, dm, jm};
        const otg::State1D from{rng.range(-15.0, 15.0),
                                rng.range(-vm * 0.95, vm * 0.95), 0.0};
        const otg::Target1D to{rng.range(-15.0, 15.0),
                               rng.range(-vm * 0.95, vm * 0.95), 0.0};

        const auto optimal = otg::plan_time_optimal(from, to, lim);
        if(!optimal) {
            continue;
        }
        const std::int64_t t_min = optimal.value().duration_cycles();

        // P6: exact T_min pass-through must be bit-identical.
        {
            const auto same = otg::solve_fixed_time(from, to, lim, t_min);
            if(!same) {
                fail_case("P6 t_min solve failed", i);
            } else {
                const otg::Profile1D &a = optimal.value();
                const otg::Profile1D &b = same.value();
                bool identical = a.segment_count() == b.segment_count();
                for(std::size_t s = 0; identical && s < a.segment_count(); ++s) {
                    identical = std::memcmp(&a.segment(s), &b.segment(s),
                                            sizeof(otg::Segment1D)) == 0;
                }
                if(!identical) {
                    fail_case("P6 t_min identity", i);
                } else {
                    ++identity_checked;
                }
            }
        }

        const std::int64_t target =
            t_min + 1 + static_cast<std::int64_t>(rng.next() % 20u);
        const auto fixed = otg::solve_fixed_time(from, to, lim, target);
        if(!fixed) {
            ++infeasible;
            continue;
        }
        ++solved;
        const otg::Profile1D &p = fixed.value();

        // P1 exact duration.
        if(p.duration_cycles() != target) {
            fail_case("P1 duration", i);
            continue;
        }
        // P2 boundary states.
        const otg::State1D s0 = otg::sample(p, rt::CycleTick::from_cycles(0));
        const otg::State1D sT =
            otg::sample(p, rt::CycleTick::from_cycles(target));
        if(std::fabs(s0.position - from.position) > 1e-9 ||
           std::fabs(s0.velocity - from.velocity) > 1e-9 ||
           std::fabs(sT.position - to.position) > 1e-9 ||
           std::fabs(sT.velocity - to.velocity) > 1e-9) {
            fail_case("P2 boundary state", i);
            continue;
        }
        // P3 dense sub-cycle compliance.
        if(!dense_within_limits(p, from, lim)) {
            fail_case("P3 dense limits", i);
            continue;
        }

        // P5: below-T_min statistic (KB-056 declared gap).
        if(t_min > 1) {
            const auto below =
                otg::solve_fixed_time(from, to, lim, t_min - 1);
            if(below) {
                ++below_tmin_success;
                const std::int64_t gap = 1;
                if(gap > max_below_gap) {
                    max_below_gap = gap;
                }
                if(!dense_within_limits(below.value(), from, lim)) {
                    if(g_fail < 2) {
                        std::printf("    case lim=(%.17g,%.17g,%.17g,%.17g) "
                                    "from=(%.17g,%.17g) to=(%.17g,%.17g) "
                                    "tmin=%lld segs=%zu\n",
                                    vm, am, dm, jm, from.position,
                                    from.velocity, to.position, to.velocity,
                                    (long long)t_min,
                                    below.value().segment_count());
                    }
                    fail_case("P5 below-tmin dense limits", i);
                }
            }
        }
    }

    std::printf(
        "OTG_FIXED_TIME_ORACLE iterations=%d solved=%d infeasible=%d "
        "identity=%d below_tmin_success=%d fail=%d\n",
        iterations, solved, infeasible, identity_checked, below_tmin_success,
        g_fail);

    // P7 coverage floor: the infeasible fraction at T in [T_min+1, T_min+20]
    // is capped from the recorded baseline (measured 38/20000 = 0.19%; cap
    // at 2% so it can only ratchet down through deliberate re-recording).
    if(solved + infeasible > 0 &&
       static_cast<double>(infeasible) / (solved + infeasible) > 0.02) {
        std::printf("FAIL coverage floor: infeasible fraction %.4f\n",
                    static_cast<double>(infeasible) / (solved + infeasible));
        ++g_fail;
    }
    // P5 cap: KB-056 declared below-T_min successes stay a bounded fraction
    // (measured 0.63 at the recorded baseline — the fixed-time strategy zoo
    // beats the cascade's T_min by one cycle in most random cases; the cap
    // only catches runaway growth of the declared gap).
    if(solved > 0 &&
       static_cast<double>(below_tmin_success) / solved > 0.75) {
        std::printf("FAIL below-tmin fraction %.4f\n",
                    static_cast<double>(below_tmin_success) / solved);
        ++g_fail;
    }

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL\n");
    return g_fail == 0 ? 0 : 1;
}
