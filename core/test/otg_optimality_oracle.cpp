// Y0 OTG optimality oracle — PMP switch-structure exhaustive + shooting.
// Test-time only. Zero runtime cost, zero external dependency.
// Design: doc/design/core/otg-oracle-design.md

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "otg/profile1d.h"
#include "otg/time_optimal.h"

namespace
{

using namespace plcopen::core;

// ================================================================
// Phase types (Pontryagin bang-singular-bang)
// ================================================================

enum class Phase : int
{
    PJ,
    NJ,
    AcP,
    AcN,
    VcP,
    VcN,
};

bool is_bang(Phase p) { return p == Phase::PJ || p == Phase::NJ; }
bool is_a_sat(Phase p) { return p == Phase::AcP || p == Phase::AcN; }
bool is_v_sat(Phase p) { return p == Phase::VcP || p == Phase::VcN; }

const char *phase_tag(Phase p)
{
    switch(p) {
    case Phase::PJ:  return "+";
    case Phase::NJ:  return "-";
    case Phase::AcP: return "A+";
    case Phase::AcN: return "A-";
    case Phase::VcP: return "V+";
    case Phase::VcN: return "V-";
    }
    return "?";
}

// ================================================================
// Transition graph
//   PJ  → {NJ, AcP, VcN}
//   NJ  → {PJ, AcN, VcP}
//   AcP → {NJ}   AcN → {PJ}
//   VcP → {NJ}   VcN → {PJ}
// ================================================================

static constexpr int MaxSucc = 3;
struct Successors
{
    Phase list[MaxSucc]{};
    int count = 0;
};

Successors successors_of(Phase p)
{
    Successors s{};
    switch(p) {
    case Phase::PJ:
        s.list[0] = Phase::NJ;
        s.list[1] = Phase::AcP;
        s.list[2] = Phase::VcN;
        s.count = 3;
        break;
    case Phase::NJ:
        s.list[0] = Phase::PJ;
        s.list[1] = Phase::AcN;
        s.list[2] = Phase::VcP;
        s.count = 3;
        break;
    case Phase::AcP:
    case Phase::VcP:
        s.list[0] = Phase::NJ;
        s.count = 1;
        break;
    case Phase::AcN:
    case Phase::VcN:
        s.list[0] = Phase::PJ;
        s.count = 1;
        break;
    }
    return s;
}

// ================================================================
// Structure table — grammar-based enumeration
// ================================================================

static constexpr int MaxPhases = 7;
static constexpr int TableCap = 700;

struct Structure
{
    Phase phases[MaxPhases]{};
    int len = 0;
    int bangs = 0;
    int a_sats = 0;
    int v_sats = 0;

    bool is_square() const { return bangs == 3 + v_sats; }
    bool is_bang_start() const { return len > 0 && is_bang(phases[0]); }
};

struct StructureTable
{
    Structure entries[TableCap]{};
    int count = 0;

    bool push(const Structure &s)
    {
        if(count >= TableCap) return false;
        entries[count++] = s;
        return true;
    }
};

void generate(StructureTable &tab, Structure &cur)
{
    if(cur.len > 0) tab.push(cur);
    if(cur.len >= MaxPhases) return;

    Successors next = successors_of(cur.phases[cur.len - 1]);
    for(int i = 0; i < next.count; ++i) {
        Phase ph = next.list[i];
        cur.phases[cur.len] = ph;
        ++cur.len;
        cur.bangs += is_bang(ph) ? 1 : 0;
        cur.a_sats += is_a_sat(ph) ? 1 : 0;
        cur.v_sats += is_v_sat(ph) ? 1 : 0;

        generate(tab, cur);

        --cur.len;
        cur.bangs -= is_bang(ph) ? 1 : 0;
        cur.a_sats -= is_a_sat(ph) ? 1 : 0;
        cur.v_sats -= is_v_sat(ph) ? 1 : 0;
    }
}

StructureTable build_table()
{
    StructureTable tab;
    Phase starts[] = {Phase::PJ, Phase::NJ, Phase::AcP, Phase::AcN,
                      Phase::VcP, Phase::VcN};
    for(int si = 0; si < 6; ++si) {
        Structure s{};
        s.phases[0] = starts[si];
        s.len = 1;
        s.bangs = is_bang(starts[si]) ? 1 : 0;
        s.a_sats = is_a_sat(starts[si]) ? 1 : 0;
        s.v_sats = is_v_sat(starts[si]) ? 1 : 0;
        generate(tab, s);
    }
    return tab;
}

// ================================================================
// Continuous-time state propagation
// ================================================================

struct CState
{
    double p = 0.0;
    double v = 0.0;
    double a = 0.0;
};

CState advance(CState s, double jerk, double dt)
{
    return {s.p + s.v * dt + 0.5 * s.a * dt * dt + jerk * dt * dt * dt / 6.0,
            s.v + s.a * dt + 0.5 * jerk * dt * dt,
            s.a + jerk * dt};
}

double jerk_of(Phase ph, const otg::Limits1D &lim)
{
    if(ph == Phase::PJ) return lim.max_jerk;
    if(ph == Phase::NJ) return -lim.max_jerk;
    return 0.0;
}

double a_sat_target(Phase ph, const otg::Limits1D &lim)
{
    return ph == Phase::AcP ? lim.max_acceleration : -lim.max_deceleration;
}

double v_sat_target(Phase ph, const otg::Limits1D &lim)
{
    return ph == Phase::VcP ? lim.max_velocity : -lim.max_velocity;
}

// ================================================================
// Residual for Newton solver (square systems: k = 3 + a_sats + 2*v_sats)
//   [0..2]  boundary: p_f−p_t, v_f−v_t, a_f−a_t
//   [3..]   saturation entry conditions in phase order
// ================================================================

void eval_residual(const Structure &str, const double *t,
                   CState from, CState target, const otg::Limits1D &lim,
                   double *r)
{
    CState s = from;
    int ri = 3;
    for(int i = 0; i < str.len; ++i) {
        Phase ph = str.phases[i];
        if(is_a_sat(ph))
            r[ri++] = s.a - a_sat_target(ph, lim);
        else if(is_v_sat(ph)) {
            r[ri++] = s.a;
            r[ri++] = s.v - v_sat_target(ph, lim);
        }
        s = advance(s, jerk_of(ph, lim), t[i]);
    }
    r[0] = s.p - target.p;
    r[1] = s.v - target.v;
    r[2] = s.a - target.a;
}

double vec_norm(const double *v, int n)
{
    double s = 0.0;
    for(int i = 0; i < n; ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

// ================================================================
// Gaussian elimination with partial pivoting (n ≤ 7)
// ================================================================

bool gauss(double A[][MaxPhases], double *b, int n)
{
    for(int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::fabs(A[col][col]);
        for(int row = col + 1; row < n; ++row) {
            double v = std::fabs(A[row][col]);
            if(v > best) {
                best = v;
                piv = row;
            }
        }
        if(best < 1e-15) return false;
        if(piv != col) {
            for(int j = 0; j < n; ++j) {
                double tmp = A[col][j];
                A[col][j] = A[piv][j];
                A[piv][j] = tmp;
            }
            double tmp = b[col];
            b[col] = b[piv];
            b[piv] = tmp;
        }
        for(int row = col + 1; row < n; ++row) {
            double f = A[row][col] / A[col][col];
            for(int j = col; j < n; ++j) A[row][j] -= f * A[col][j];
            b[row] -= f * b[col];
        }
    }
    for(int i = n - 1; i >= 0; --i) {
        for(int j = i + 1; j < n; ++j) b[i] -= A[i][j] * b[j];
        if(std::fabs(A[i][i]) < 1e-15) return false;
        b[i] /= A[i][i];
    }
    return true;
}

// ================================================================
// Newton shooting solver
// ================================================================

static constexpr double FdEps = 1e-8;
static constexpr double Tol = 1e-10;
static constexpr int MaxIter = 60;

struct ShootResult
{
    double dur[MaxPhases]{};
    double total = 1e30;
    double res_norm = 1e30;
    bool ok = false;
};

ShootResult shoot(const Structure &str, CState from, CState target,
                  const otg::Limits1D &lim, const double *t0)
{
    const int n = str.len;
    ShootResult res;
    double t[MaxPhases];
    for(int i = 0; i < n; ++i) t[i] = t0[i] > 1e-14 ? t0[i] : 0.01;

    double r[MaxPhases];
    eval_residual(str, t, from, target, lim, r);
    double rn = vec_norm(r, n);

    for(int iter = 0; iter < MaxIter && rn > Tol; ++iter) {
        double J[MaxPhases][MaxPhases];
        for(int j = 0; j < n; ++j) {
            double h = std::fmax(std::fabs(t[j]) * FdEps, FdEps);
            double sv = t[j];
            t[j] = sv + h;
            double rp[MaxPhases];
            eval_residual(str, t, from, target, lim, rp);
            t[j] = sv;
            for(int i = 0; i < n; ++i) J[i][j] = (rp[i] - r[i]) / h;
        }

        double neg_r[MaxPhases];
        for(int i = 0; i < n; ++i) neg_r[i] = -r[i];
        double Jc[MaxPhases][MaxPhases];
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j)
                Jc[i][j] = J[i][j];
        if(!gauss(Jc, neg_r, n)) break;

        double alpha = 1.0;
        bool stepped = false;
        for(int ls = 0; ls < 12; ++ls) {
            double tt[MaxPhases];
            bool neg = false;
            for(int i = 0; i < n; ++i) {
                tt[i] = t[i] + alpha * neg_r[i];
                if(tt[i] < -1e-12) neg = true;
            }
            if(neg) {
                alpha *= 0.5;
                continue;
            }
            double rr[MaxPhases];
            eval_residual(str, tt, from, target, lim, rr);
            double nn = vec_norm(rr, n);
            if(nn < rn) {
                for(int i = 0; i < n; ++i) {
                    t[i] = tt[i];
                    r[i] = rr[i];
                }
                rn = nn;
                stepped = true;
                break;
            }
            alpha *= 0.5;
        }
        if(!stepped) break;
    }

    res.res_norm = rn;
    res.total = 0.0;
    bool good = rn < Tol;
    for(int i = 0; i < n; ++i) {
        if(t[i] < -1e-10) good = false;
        res.dur[i] = t[i] < 0.0 ? 0.0 : t[i];
        res.total += res.dur[i];
    }
    res.ok = good;
    return res;
}

// ================================================================
// Trajectory limit verification
// ================================================================

bool limits_ok(const Structure &str, const double *dur, CState from,
               const otg::Limits1D &lim)
{
    CState s = from;
    for(int i = 0; i < str.len; ++i) {
        double j = jerk_of(str.phases[i], lim);
        int samp = static_cast<int>(std::ceil(dur[i])) + 1;
        if(samp < 8) samp = 8;
        if(samp > 200) samp = 200;
        for(int k = 0; k <= samp; ++k) {
            double frac = dur[i] * static_cast<double>(k) /
                          static_cast<double>(samp);
            CState sp = advance(s, j, frac);
            if(std::fabs(sp.v) > lim.max_velocity * 1.002) return false;
            if(sp.a > lim.max_acceleration * 1.002) return false;
            if(sp.a < -lim.max_deceleration * 1.002) return false;
        }
        s = advance(s, j, dur[i]);
    }
    return true;
}

// ================================================================
// Oracle: minimum T* over all bang-start square structures
// ================================================================

struct OracleResult
{
    double time = 1e30;
    int idx = -1;
    double dur[MaxPhases]{};
    bool found = false;
};

OracleResult oracle_solve(otg::State1D from, otg::Target1D to,
                          otg::Limits1D lim, const StructureTable &tab)
{
    CState cf{from.position, from.velocity, from.acceleration};
    CState ct{to.position, to.velocity, to.acceleration};

    if(std::fabs(cf.p - ct.p) < 1e-15 && std::fabs(cf.v - ct.v) < 1e-15 &&
       std::fabs(cf.a - ct.a) < 1e-15) {
        OracleResult r;
        r.time = 0.0;
        r.found = true;
        return r;
    }

    double dist = std::fabs(to.position - from.position);
    double tc = 0.1;
    if(lim.max_velocity > 0.0)
        tc = std::fmax(tc, dist / lim.max_velocity);
    if(lim.max_acceleration > 0.0)
        tc = std::fmax(tc, std::sqrt(2.0 * dist / lim.max_acceleration));
    if(lim.max_jerk > 0.0)
        tc = std::fmax(tc, std::cbrt(6.0 * dist / lim.max_jerk));
    tc = std::fmax(tc, (std::fabs(from.velocity) + std::fabs(to.velocity)) /
                           std::fmax(lim.max_acceleration, 0.01));
    tc = std::fmax(tc, std::fabs(from.acceleration) /
                           std::fmax(lim.max_jerk, 0.01));

    static const double scales[] = {1.0, 0.3, 3.0, 0.1, 10.0, 0.03, 30.0,
                                    0.005};
    static constexpr int NS = 8;

    OracleResult best;
    for(int si = 0; si < tab.count; ++si) {
        const Structure &str = tab.entries[si];
        if(!str.is_bang_start() || !str.is_square()) continue;

        for(int sc = 0; sc < NS; ++sc) {
            double t0[MaxPhases];
            double base = tc * scales[sc] / str.len;
            for(int i = 0; i < str.len; ++i)
                t0[i] = is_bang(str.phases[i]) ? base * 1.2 : base * 0.6;

            ShootResult sr = shoot(str, cf, ct, lim, t0);
            if(!sr.ok || sr.total <= 0.0 || sr.total >= best.time) continue;

            if(limits_ok(str, sr.dur, cf, lim)) {
                best.time = sr.total;
                best.idx = si;
                for(int i = 0; i < str.len; ++i) best.dur[i] = sr.dur[i];
                best.found = true;
            }
        }
    }
    return best;
}

// ================================================================
// Tests
// ================================================================

struct Lcg
{
    unsigned int state = 0xDEAD0001u;
    unsigned int next_u32()
    {
        state = state * 1664525u + 1013904223u;
        return state;
    }
    double range(double lo, double hi)
    {
        return lo +
               (hi - lo) * (static_cast<double>(next_u32()) / 4294967295.0);
    }
};

int check_structure_table()
{
    std::printf("--- structure_table ---\n");
    StructureTable tab = build_table();
    std::printf("  total: %d\n", tab.count);

    int sq = 0, bsq = 0;
    int by_len[MaxPhases + 1] = {};
    for(int i = 0; i < tab.count; ++i) {
        by_len[tab.entries[i].len]++;
        if(tab.entries[i].is_square()) ++sq;
        if(tab.entries[i].is_bang_start() && tab.entries[i].is_square()) ++bsq;
    }
    std::printf("  square: %d  bang_start_square: %d\n", sq, bsq);
    for(int l = 1; l <= MaxPhases; ++l)
        if(by_len[l] > 0)
            std::printf("  len %d: %d\n", l, by_len[l]);

    int bad = 0;
    for(int i = 0; i < tab.count; ++i) {
        const Structure &s = tab.entries[i];
        for(int j = 1; j < s.len; ++j) {
            Successors next = successors_of(s.phases[j - 1]);
            bool ok = false;
            for(int k = 0; k < next.count; ++k)
                if(next.list[k] == s.phases[j]) ok = true;
            if(!ok) ++bad;
        }
    }
    if(bad > 0 || tab.count == 0) {
        std::printf("FAIL structure_table bad=%d count=%d\n", bad, tab.count);
        return 1;
    }
    std::printf("  OK\n");
    return 0;
}

int check_fixed_cases()
{
    std::printf("--- fixed_oracle_cases ---\n");
    StructureTable tab = build_table();
    int fail = 0;

    struct Case
    {
        const char *name;
        otg::State1D from;
        otg::Target1D to;
        otg::Limits1D lim;
    };

    Case cases[] = {
        {"rest_short", {0, 0, 0}, {1, 0, 0}, {10, 10, 10, 1}},
        {"rest_vlim", {0, 0, 0}, {100, 0, 0}, {2, 4, 4, 1}},
        {"same_dir", {0, 1, 0}, {10, 0, 0}, {4, 2, 2, 1}},
        {"reverse", {0, 2, 0}, {-5, 0, 0}, {4, 2, 2, 1}},
        {"asymmetric", {0, 0, 0}, {10, 0, 0}, {4, 3, 1, 1}},
        {"nonzero_a0", {0, 0, 0.5}, {5, 0, 0}, {4, 2, 2, 1}},
        {"short_dist", {0, 0, 0}, {0.01, 0, 0}, {4, 2, 2, 1}},
        {"neg_target_v", {0, 0, 0}, {0, -1, 0}, {4, 2, 2, 1}},
        {"backward", {5, 0, 0}, {0, 0, 0}, {4, 2, 2, 1}},
        {"high_v0", {0, 3.5, 0}, {20, 0, 0}, {4, 2, 2, 1}},
    };
    int n_cases = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

    for(int ci = 0; ci < n_cases; ++ci) {
        const Case &c = cases[ci];
        OracleResult res = oracle_solve(c.from, c.to, c.lim, tab);
        if(!res.found) {
            std::printf("  FAIL %s no_solution\n", c.name);
            ++fail;
            continue;
        }

        const Structure &str = tab.entries[res.idx];
        CState s{c.from.position, c.from.velocity, c.from.acceleration};
        for(int i = 0; i < str.len; ++i)
            s = advance(s, jerk_of(str.phases[i], c.lim), res.dur[i]);

        double ep = std::fabs(s.p - c.to.position);
        double ev = std::fabs(s.v - c.to.velocity);
        double ea = std::fabs(s.a - c.to.acceleration);
        if(ep > 1e-7 || ev > 1e-7 || ea > 1e-7) {
            std::printf("  FAIL %s backsub p=%.2e v=%.2e a=%.2e\n", c.name,
                        ep, ev, ea);
            ++fail;
            continue;
        }

        std::printf("  OK %s T*=%.4f [", c.name, res.time);
        for(int i = 0; i < str.len; ++i) {
            if(i > 0) std::printf(",");
            std::printf("%s", phase_tag(str.phases[i]));
        }
        std::printf("]\n");
    }
    return fail;
}

struct DomainStats
{
    int attempted = 0;
    int planner_fail = 0;
    int compared = 0;
    int oracle_miss = 0;
    int fail = 0;
    int convention_excluded = 0;
    int relaxed_compared = 0;
    int ec_max = 0;
    int ec_min = 0;
    double ec_sum = 0.0;
    bool hard_gate = true;
    // Suboptimality upper bound (KB-100): excess above this cap fails the
    // domain even where the lower-bound gate is soft. 0 disables.
    int ec_cap = 0;

    void record(otg::State1D from, otg::Target1D to, otg::Limits1D lim,
                const StructureTable &tab, int case_id)
    {
        ++attempted;
        rt::Result<otg::Profile1D> planned =
            otg::plan_time_optimal(from, to, lim);
        if(!planned) {
            ++planner_fail;
            return;
        }

        // Direction-aware vs signed-alphabet comparability (KB-100): the
        // production checker bounds acceleration relative to the motion
        // direction (PLCopen Acc/Dec semantics), while this oracle's
        // saturation arcs are the signed pair {+a_max, -d_max}. The two
        // agree exactly while v >= 0. Direction-aware physics is mirror
        // invariant, so a trajectory confined to v <= 0 is solved on the
        // mirrored problem (where the signed alphabet is direction-correct);
        // mixed-sign trajectories under asymmetric bounds have no valid T*
        // in this alphabet and are counted out instead of asserted.
        const bool asymmetric = lim.max_acceleration != lim.max_deceleration;
        // Mirror the production gate's KB-055 velocity exemptions into the
        // reference limits so the comparison is like for like: an entry
        // velocity above the limit decays into the envelope, and the
        // quantized zeroing ramp legally reaches v0 + a0*n0/2.
        otg::Limits1D ref = lim;
        if(std::fabs(from.velocity) > ref.max_velocity) {
            ref.max_velocity = std::fabs(from.velocity);
        }
        if(from.acceleration != 0.0) {
            const double n0 =
                std::ceil(std::fabs(from.acceleration) / lim.max_jerk);
            const double reduced_v =
                std::fabs(from.velocity + 0.5 * from.acceleration * n0);
            if(reduced_v > ref.max_velocity) {
                ref.max_velocity = reduced_v;
            }
        }
        otg::State1D oracle_from = from;
        otg::Target1D oracle_to = to;
        if(asymmetric) {
            bool has_pos = from.velocity > 1e-12 || to.velocity > 1e-12;
            bool has_neg = from.velocity < -1e-12 || to.velocity < -1e-12;
            const std::int64_t total = planned.value().duration_cycles();
            for(std::int64_t cyc = 1; cyc <= total && !(has_pos && has_neg);
                ++cyc) {
                const double v =
                    otg::sample(planned.value(),
                                plcopen::core::rt::CycleTick::from_cycles(cyc))
                        .velocity;
                if(v > 1e-12) {
                    has_pos = true;
                } else if(v < -1e-12) {
                    has_neg = true;
                }
            }
            if(has_pos && has_neg) {
                // Mixed-sign trajectory: no exact T* in the signed alphabet.
                // Solve with the symmetric relaxed bound max(a_max, d_max) —
                // a superset of the direction-aware admissible controls in
                // every regime, so T*_relaxed <= T*_aware and the negative
                // gate stays a valid violation detector (tight optimality
                // statistics come from the single-sign tiers only).
                otg::Limits1D relaxed = ref;
                relaxed.max_acceleration =
                    std::fmax(lim.max_acceleration, lim.max_deceleration);
                relaxed.max_deceleration = relaxed.max_acceleration;
                OracleResult relaxed_orc =
                    oracle_solve(from, to, relaxed, tab);
                if(!relaxed_orc.found) {
                    ++oracle_miss;
                    return;
                }
                std::int64_t relaxed_c =
                    static_cast<std::int64_t>(std::ceil(relaxed_orc.time));
                if(relaxed_c < 1) relaxed_c = 1;
                const std::int64_t plan_relaxed =
                    planned.value().duration_cycles();
                if(plan_relaxed < relaxed_c && hard_gate) {
                    if(fail < 3)
                        std::printf(
                            "  FAIL-RELAXED i=%d plan=%lld oracle=%lld "
                            "T*=%.6f from=(%.17g,%.17g,%.17g) "
                            "to=(%.17g,%.17g,%.17g) "
                            "lim=(%.17g,%.17g,%.17g,%.17g)\n",
                            case_id, (long long)plan_relaxed,
                            (long long)relaxed_c, relaxed_orc.time,
                            from.position, from.velocity, from.acceleration,
                            to.position, to.velocity, to.acceleration,
                            lim.max_velocity, lim.max_acceleration,
                            lim.max_deceleration, lim.max_jerk);
                    ++fail;
                    return;
                }
                ++relaxed_compared;
                return;
            }
            if(has_neg) {
                oracle_from = {-from.position, -from.velocity,
                               -from.acceleration};
                oracle_to = {-to.position, -to.velocity, -to.acceleration};
            }
        }

        OracleResult orc = oracle_solve(oracle_from, oracle_to, ref, tab);
        if(!orc.found) {
            ++oracle_miss;
            return;
        }

        std::int64_t plan_c = planned.value().duration_cycles();
        std::int64_t oracle_c =
            static_cast<std::int64_t>(std::ceil(orc.time));
        if(oracle_c < 1) oracle_c = 1;

        std::int64_t excess = plan_c - oracle_c;
        if(hard_gate && excess < 0) {
            if(fail < 3)
                std::printf(
                    "  FAIL i=%d excess=%lld plan=%lld oracle=%lld "
                    "T*=%.6f from=(%.17g,%.17g,%.17g) to=(%.17g,%.17g,%.17g) "
                    "lim=(%.17g,%.17g,%.17g,%.17g)\n",
                    case_id, (long long)excess, (long long)plan_c,
                    (long long)oracle_c, orc.time,
                    from.position, from.velocity, from.acceleration,
                    to.position, to.velocity, to.acceleration,
                    lim.max_velocity, lim.max_acceleration,
                    lim.max_deceleration, lim.max_jerk);
            ++fail;
            return;
        }
        if(ec_cap > 0 && excess > ec_cap) {
            if(fail < 3)
                std::printf(
                    "  FAIL-CAP i=%d excess=%lld cap=%d plan=%lld "
                    "oracle=%lld\n",
                    case_id, (long long)excess, ec_cap, (long long)plan_c,
                    (long long)oracle_c);
            ++fail;
            return;
        }

        int ec = static_cast<int>(excess);
        if(compared == 0 || ec > ec_max) ec_max = ec;
        if(compared == 0 || ec < ec_min) ec_min = ec;
        ec_sum += ec;
        ++compared;
    }

    void report(const char *domain) const
    {
        std::printf("  [%s] attempted=%d planner_fail=%d oracle_miss=%d "
                    "negative_fail=%d relaxed_compared=%d "
                    "convention_excluded=%d compared=%d",
                    domain, attempted, planner_fail, oracle_miss, fail,
                    relaxed_compared, convention_excluded, compared);
        if(compared > 0) {
            std::printf(" max=%d avg=%.2f", ec_max, ec_sum / compared);
            if(ec_min < 0) std::printf(" min=%d", ec_min);
        }
        std::printf("\n");
        std::printf(
            "OTG_ORACLE_METRICS domain=%s attempted=%d planner_fail=%d "
            "oracle_miss=%d negative_fail=%d relaxed_compared=%d "
            "convention_excluded=%d "
            "compared=%d excess_min=%d "
            "excess_max=%d excess_sum=%.0f hard_gate=%d ec_cap=%d\n",
            domain, attempted, planner_fail, oracle_miss, fail,
            relaxed_compared, convention_excluded, compared,
            ec_min, ec_max, ec_sum, hard_gate ? 1 : 0, ec_cap);
    }
};

int check_excess_cycles(int iterations)
{
    std::printf("--- excess_cycles (n=%d per domain) ---\n", iterations);
    StructureTable tab = build_table();
    Lcg rng;
    int total_fail = 0;

    // Domain 1: regular — random states within limits
    {
        DomainStats ds;
        ds.ec_cap = 8; // KB-100 suboptimality upper bound (measured max x ~2)
        for(int i = 0; i < iterations; ++i) {
            double vm = rng.range(0.5, 8.0);
            double am = rng.range(0.5, 8.0);
            double dm = rng.range(0.5, 8.0);
            double jm = rng.range(0.1, 4.0);
            otg::Limits1D lim{vm, am, dm, jm};
            double p0 = rng.range(-15.0, 15.0);
            double v0 = rng.range(-vm * 0.95, vm * 0.95);
            double pt = rng.range(-15.0, 15.0);
            double vt = rng.range(-vm * 0.95, vm * 0.95);
            ds.record({p0, v0, 0.0}, {pt, vt, 0.0}, lim, tab, i);
        }
        ds.report("regular");
        total_fail += ds.fail;
    }

    // Domain 2: pin-boundary — velocities near ±v_max
    {
        DomainStats ds;
        ds.ec_cap = 6; // KB-100 suboptimality upper bound (measured max x ~2)
        for(int i = 0; i < iterations; ++i) {
            double vm = rng.range(1.0, 6.0);
            double am = rng.range(0.5, 6.0);
            double dm = rng.range(0.5, 6.0);
            double jm = rng.range(0.2, 3.0);
            otg::Limits1D lim{vm, am, dm, jm};
            double p0 = rng.range(-10.0, 10.0);
            double sign0 = rng.range(0.0, 1.0) > 0.5 ? 1.0 : -1.0;
            double v0 = sign0 * vm * rng.range(0.9, 1.0);
            double pt = rng.range(-10.0, 10.0);
            double signt = rng.range(0.0, 1.0) > 0.5 ? 1.0 : -1.0;
            double vt = signt * vm * rng.range(0.9, 1.0);
            ds.record({p0, v0, 0.0}, {pt, vt, 0.0}, lim, tab, i);
        }
        ds.report("pin-boundary");
        total_fail += ds.fail;
    }

    // Domain 3: bump — short distance, similar start/end velocities
    {
        DomainStats ds;
        ds.ec_cap = 6; // KB-100 suboptimality upper bound (measured max x ~2)
        for(int i = 0; i < iterations; ++i) {
            double vm = rng.range(1.0, 6.0);
            double am = rng.range(0.5, 6.0);
            double dm = rng.range(0.5, 6.0);
            double jm = rng.range(0.2, 3.0);
            otg::Limits1D lim{vm, am, dm, jm};
            double p0 = rng.range(-5.0, 5.0);
            double v0 = rng.range(-vm * 0.8, vm * 0.8);
            double pt = p0 + rng.range(-0.5, 0.5);
            double vt = v0 + rng.range(-0.3, 0.3);
            if(std::fabs(vt) > vm) vt = (vt > 0 ? 1 : -1) * vm * 0.95;
            ds.record({p0, v0, 0.0}, {pt, vt, 0.0}, lim, tab, i);
        }
        ds.report("bump");
        total_fail += ds.fail;
    }

    // Domain 4: nonzero-a0 — high initial acceleration
    // No hard gate: plan_time_optimal uses adjusted-jerk zeroing (j'=-a0/n0),
    // which is not a PMP {±j,0} phase, so the PMP-based T* is not a strict
    // lower bound on plan's integer-cycle count.
    {
        DomainStats ds;
        ds.ec_cap = 8; // KB-100 suboptimality upper bound (measured max x ~2)
        ds.hard_gate = false;
        for(int i = 0; i < iterations; ++i) {
            double vm = rng.range(1.0, 6.0);
            double am = rng.range(1.0, 6.0);
            double dm = rng.range(1.0, 6.0);
            double jm = rng.range(0.2, 3.0);
            otg::Limits1D lim{vm, am, dm, jm};
            double p0 = rng.range(-10.0, 10.0);
            double v0 = rng.range(-vm * 0.8, vm * 0.8);
            double a0_sign = rng.range(0.0, 1.0) > 0.5 ? 1.0 : -1.0;
            double a0_mag = rng.range(0.7, 1.0) *
                            (a0_sign > 0 ? am : dm);
            double a0 = a0_sign * a0_mag;
            double pt = rng.range(-10.0, 10.0);
            double vt = rng.range(-vm * 0.8, vm * 0.8);
            ds.record({p0, v0, a0}, {pt, vt, 0.0}, lim, tab, i);
        }
        ds.report("high-a0");
        total_fail += ds.fail;
    }

    // Domain 5: nonzero target acceleration
    // No hard gate: targeting ramp uses adjusted jerk (j'=at/n), not PMP.
    {
        DomainStats ds;
        ds.ec_cap = 10; // KB-100 suboptimality upper bound (measured max x ~2)
        ds.hard_gate = false;
        for(int i = 0; i < iterations; ++i) {
            double vm = rng.range(1.0, 6.0);
            double am = rng.range(1.0, 6.0);
            double dm = rng.range(1.0, 6.0);
            double jm = rng.range(0.2, 3.0);
            otg::Limits1D lim{vm, am, dm, jm};
            double p0 = rng.range(-10.0, 10.0);
            double v0 = rng.range(-vm * 0.8, vm * 0.8);
            double pt = rng.range(-10.0, 10.0);
            double vt = rng.range(-vm * 0.8, vm * 0.8);
            double at_sign = rng.range(0.0, 1.0) > 0.5 ? 1.0 : -1.0;
            double at_bound = at_sign > 0 ? am : dm;
            double at = at_sign * rng.range(0.3, 0.95) * at_bound;
            ds.record({p0, v0, 0.0}, {pt, vt, at}, lim, tab, i);
        }
        ds.report("nonzero-at");
        total_fail += ds.fail;
    }

    // Domain 6: both a0 and at nonzero
    {
        DomainStats ds;
        ds.ec_cap = 10; // KB-100 suboptimality upper bound (measured max x ~2)
        ds.hard_gate = false;
        for(int i = 0; i < iterations; ++i) {
            double vm = rng.range(1.0, 6.0);
            double am = rng.range(1.0, 6.0);
            double dm = rng.range(1.0, 6.0);
            double jm = rng.range(0.2, 3.0);
            otg::Limits1D lim{vm, am, dm, jm};
            double p0 = rng.range(-10.0, 10.0);
            double v0 = rng.range(-vm * 0.8, vm * 0.8);
            double a0_sign = rng.range(0.0, 1.0) > 0.5 ? 1.0 : -1.0;
            double a0 = a0_sign * rng.range(0.3, 0.95) *
                        (a0_sign > 0 ? am : dm);
            double pt = rng.range(-10.0, 10.0);
            double vt = rng.range(-vm * 0.8, vm * 0.8);
            double at_sign = rng.range(0.0, 1.0) > 0.5 ? 1.0 : -1.0;
            double at = at_sign * rng.range(0.3, 0.95) *
                        (at_sign > 0 ? am : dm);
            ds.record({p0, v0, a0}, {pt, vt, at}, lim, tab, i);
        }
        ds.report("a0-and-at");
        total_fail += ds.fail;
    }

    if(total_fail > 0) {
        std::printf("FAIL excess_negative=%d\n", total_fail);
        return 1;
    }
    std::printf("  OK\n");
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    int iterations = 500;
    for(int i = 1; i < argc; ++i)
        if(std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
            iterations = std::atoi(argv[++i]);

    int f = 0;
    f += check_structure_table();
    std::printf("\n");
    f += check_fixed_cases();
    std::printf("\n");
    f += check_excess_cycles(iterations);

    std::printf("\n%s\n", f ? "FAIL" : "PASS");
    return f ? 1 : 0;
}
