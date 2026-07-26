#pragma once

#include <cmath>
#include <cstdint>

#include "rt/cycle.h"
#include "rt/error.h"
#include "rt/static_vector.h"

namespace plcopen::core::otg
{

struct State1D
{
    double position = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
};

struct Target1D
{
    double position = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
};

struct Limits1D
{
    double max_velocity = 0.0;
    double max_acceleration = 0.0;
    double max_deceleration = 0.0;
    double max_jerk = 0.0;
};

struct Segment1D
{
    std::int64_t duration_cycles = 0;
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
    double c4 = 0.0;
    double c5 = 0.0;
    State1D start{};
    State1D finish{};
};

class Profile1D
{
public:
    // Sized for the time-optimal planner's worst case: a zero-crossing entry
    // ramp (up to 6 jerk phases), cruise, exit ramp (3 phases), and the
    // integer-quantization correction segment.
    static constexpr std::size_t MaxSegments = 16;

    rt::ErrorCode add_segment(const Segment1D &segment)
    {
        const rt::ErrorCode pushed = segments_.push_back(segment);
        if(pushed == rt::ErrorCode::ok) {
            duration_cycles_ += segment.duration_cycles;
        }
        return pushed;
    }

    std::size_t segment_count() const
    {
        return segments_.size();
    }

    std::int64_t duration_cycles() const
    {
        return duration_cycles_;
    }

    const Segment1D &segment(std::size_t index) const
    {
        return segments_[index];
    }

    rt::ErrorCode translate(double delta)
    {
        if(!std::isfinite(delta)) return rt::ErrorCode::invalid_argument;
        for(std::size_t i = 0; i < segments_.size(); ++i) {
            segments_[i].c0 += delta;
            segments_[i].start.position += delta;
            segments_[i].finish.position += delta;
        }
        return rt::ErrorCode::ok;
    }

private:
    rt::StaticVector<Segment1D, MaxSegments> segments_{};
    std::int64_t duration_cycles_ = 0;
};

inline bool is_finite(State1D state)
{
    return std::isfinite(state.position) && std::isfinite(state.velocity) &&
           std::isfinite(state.acceleration);
}

inline bool is_finite(Target1D target)
{
    return std::isfinite(target.position) && std::isfinite(target.velocity) &&
           std::isfinite(target.acceleration);
}

inline bool is_finite(Limits1D limits)
{
    return std::isfinite(limits.max_velocity) && std::isfinite(limits.max_acceleration) &&
           std::isfinite(limits.max_deceleration) && std::isfinite(limits.max_jerk);
}

inline State1D sample_segment(const Segment1D &segment, std::int64_t cycle)
{
    if(cycle <= 0) {
        return segment.start;
    }
    if(cycle >= segment.duration_cycles) {
        return segment.finish;
    }

    const double x = static_cast<double>(cycle);
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x3 * x;
    const double x5 = x4 * x;

    State1D state{};
    state.position =
        segment.c0 + segment.c1 * x + segment.c2 * x2 + segment.c3 * x3 +
        segment.c4 * x4 + segment.c5 * x5;
    state.velocity =
        segment.c1 + 2.0 * segment.c2 * x + 3.0 * segment.c3 * x2 +
        4.0 * segment.c4 * x3 + 5.0 * segment.c5 * x4;
    state.acceleration =
        2.0 * segment.c2 + 6.0 * segment.c3 * x + 12.0 * segment.c4 * x2 +
        20.0 * segment.c5 * x3;
    return state;
}

inline State1D sample(const Profile1D &profile, rt::CycleTick tick)
{
    std::int64_t remaining = tick.cycles();
    for(std::size_t i = 0; i < profile.segment_count(); ++i) {
        const Segment1D &current = profile.segment(i);
        if(remaining <= current.duration_cycles) {
            return sample_segment(current, remaining);
        }
        remaining -= current.duration_cycles;
    }

    if(profile.segment_count() == 0) {
        return State1D{};
    }
    const Segment1D &last = profile.segment(profile.segment_count() - 1);
    return last.finish;
}

inline double jerk_at(const Segment1D &segment, std::int64_t cycle)
{
    const double x = static_cast<double>(cycle);
    return 6.0 * segment.c3 + 24.0 * segment.c4 * x + 60.0 * segment.c5 * x * x;
}

inline Segment1D make_quintic_segment(State1D from, Target1D to, std::int64_t cycles)
{
    const double t = static_cast<double>(cycles);
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double t4 = t3 * t;
    const double t5 = t4 * t;
    const double dp = to.position - from.position;

    Segment1D segment{};
    segment.duration_cycles = cycles;
    segment.start = from;
    segment.finish = {to.position, to.velocity, to.acceleration};
    segment.c0 = from.position;
    segment.c1 = from.velocity;
    segment.c2 = 0.5 * from.acceleration;
    segment.c3 =
        (20.0 * dp - (8.0 * to.velocity + 12.0 * from.velocity) * t -
         (3.0 * from.acceleration - to.acceleration) * t2) /
        (2.0 * t3);
    segment.c4 =
        (-30.0 * dp + (14.0 * to.velocity + 16.0 * from.velocity) * t +
         (3.0 * from.acceleration - 2.0 * to.acceleration) * t2) /
        (2.0 * t4);
    segment.c5 =
        (12.0 * dp - (6.0 * to.velocity + 6.0 * from.velocity) * t -
         (from.acceleration - to.acceleration) * t2) /
        (2.0 * t5);
    return segment;
}

namespace limit_detail
{

// Real roots of a*x^2 + b*x + c = 0 appended to roots[count]; bounded and
// deterministic (no iteration). The degeneracy test is relative: a leading
// coefficient far below the other terms would make the closed form blow up
// numerically, so such cases fall through to the lower-degree solver.
inline void quadratic_roots(double a, double b, double c, double *roots, int &count)
{
    if(std::fabs(a) <= 1e-14 * (std::fabs(b) + std::fabs(c))) {
        if(std::fabs(b) > 1e-300) {
            roots[count++] = -c / b;
        }
        return;
    }
    const double disc = b * b - 4.0 * a * c;
    if(disc < 0.0) {
        return;
    }
    const double s = std::sqrt(disc);
    // Numerically stable pairing: avoid cancellation in -b ± s.
    const double q = b >= 0.0 ? -0.5 * (b + s) : -0.5 * (b - s);
    roots[count++] = q / a;
    if(q != 0.0) {
        roots[count++] = c / q;
    } else {
        roots[count++] = 0.0;
    }
}

// Two bounded Newton iterations per root: the closed forms lose accuracy
// when the leading coefficient is small relative to the rest, and a slightly
// misplaced extremum candidate would make the limit check miss a peak.
inline void polish_cubic_roots(double a, double b, double c, double d,
                               double *roots, int base, int count)
{
    for(int i = base; i < count; ++i) {
        double x = roots[i];
        for(int it = 0; it < 2; ++it) {
            const double f = ((a * x + b) * x + c) * x + d;
            const double df = (3.0 * a * x + 2.0 * b) * x + c;
            if(std::fabs(df) <= 1e-300) {
                break;
            }
            x -= f / df;
        }
        if(std::isfinite(x)) {
            roots[i] = x;
        }
    }
}

// Real roots of a*x^3 + b*x^2 + c*x + d = 0 (Cardano / trigonometric form),
// appended to roots[count]. Bounded and deterministic.
inline void cubic_roots(double a, double b, double c, double d, double *roots, int &count)
{
    if(std::fabs(a) <=
       1e-14 * (std::fabs(b) + std::fabs(c) + std::fabs(d))) {
        quadratic_roots(b, c, d, roots, count);
        return;
    }
    const int base = count;
    const double bn = b / a;
    const double cn = c / a;
    const double dn = d / a;
    // Depressed cubic t^3 + p t + q with x = t - bn/3.
    const double shift = bn / 3.0;
    const double p = cn - bn * bn / 3.0;
    const double q = 2.0 * bn * bn * bn / 27.0 - bn * cn / 3.0 + dn;
    const double disc = 0.25 * q * q + p * p * p / 27.0;
    if(disc > 0.0) {
        const double s = std::sqrt(disc);
        const double u = std::cbrt(-0.5 * q + s);
        const double v = std::cbrt(-0.5 * q - s);
        roots[count++] = u + v - shift;
        polish_cubic_roots(a, b, c, d, roots, base, count);
        return;
    }
    if(p >= 0.0) {
        // disc <= 0 with p >= 0 only at the triple-root degeneracy.
        roots[count++] = -shift;
        return;
    }
    const double m = 2.0 * std::sqrt(-p / 3.0);
    double arg = 3.0 * q / (p * m);
    if(arg > 1.0) {
        arg = 1.0;
    }
    if(arg < -1.0) {
        arg = -1.0;
    }
    const double theta = std::acos(arg) / 3.0;
    constexpr double TwoPiOverThree = 2.0943951023931953;
    roots[count++] = m * std::cos(theta) - shift;
    roots[count++] = m * std::cos(theta - TwoPiOverThree) - shift;
    roots[count++] = m * std::cos(theta + TwoPiOverThree) - shift;
    polish_cubic_roots(a, b, c, d, roots, base, count);
}

} // namespace limit_detail

// Exact limit compliance over the whole segment domain: velocity, acceleration
// and jerk are checked at both endpoints and at every interior stationary
// point (velocity extrema = real roots of the acceleration cubic,
// acceleration extrema = real roots of the jerk quadratic, jerk extremum =
// root of the linear snap). This replaces the earlier 96-point integer-cycle
// sampler, which missed sub-cycle extrema; it is also cheaper (a bounded
// handful of closed-form evaluations instead of 97 samples).
inline bool within_limits(const Segment1D &segment, Limits1D limits)
{
    constexpr double Epsilon = 1e-9;
    const double n = static_cast<double>(segment.duration_cycles);
    if(segment.duration_cycles <= 0) {
        return true;
    }

    const auto velocity_at = [&](double x) {
        return segment.c1 +
               x * (2.0 * segment.c2 +
                    x * (3.0 * segment.c3 +
                         x * (4.0 * segment.c4 + x * 5.0 * segment.c5)));
    };
    const auto acceleration_at = [&](double x) {
        return 2.0 * segment.c2 +
               x * (6.0 * segment.c3 +
                    x * (12.0 * segment.c4 + x * 20.0 * segment.c5));
    };
    const auto jerk_value_at = [&](double x) {
        return 6.0 * segment.c3 + x * (24.0 * segment.c4 + x * 60.0 * segment.c5);
    };

    // PLCopen asymmetric bounds are direction-of-motion relative: while |v|
    // grows (v·a >= 0, accelerating) the magnitude bound is max_acceleration,
    // while |v| shrinks (v·a < 0, decelerating) it is max_deceleration. The
    // earlier signed check (a in [-d_max, +a_max]) was only correct for
    // forward motion and mis-bounded reversed segments.
    const auto accel_ok = [&](double a, double v) {
        const double bound = v * a >= 0.0 ? limits.max_acceleration
                                          : limits.max_deceleration;
        return std::fabs(a) <= bound + Epsilon;
    };

    const auto check_state = [&](double x) {
        const double v = velocity_at(x);
        if(std::fabs(v) > limits.max_velocity + Epsilon) {
            return false;
        }
        if(!accel_ok(acceleration_at(x), v)) {
            return false;
        }
        return std::fabs(jerk_value_at(x)) <= limits.max_jerk + Epsilon;
    };

    if(!check_state(0.0) || !check_state(n)) {
        return false;
    }

    // Interior jerk extremum: snap root 24 c4 + 120 c5 x = 0.
    if(segment.c5 != 0.0) {
        const double x_snap = -segment.c4 / (5.0 * segment.c5);
        if(x_snap > 0.0 && x_snap < n &&
           std::fabs(jerk_value_at(x_snap)) > limits.max_jerk + Epsilon) {
            return false;
        }
    }

    // Interior acceleration extrema: jerk roots (quadratic).
    {
        double roots[2];
        int count = 0;
        limit_detail::quadratic_roots(60.0 * segment.c5, 24.0 * segment.c4,
                                      6.0 * segment.c3, roots, count);
        for(int i = 0; i < count; ++i) {
            const double x = roots[i];
            if(x > 0.0 && x < n &&
               !accel_ok(acceleration_at(x), velocity_at(x))) {
                return false;
            }
        }
    }

    // Interior velocity extrema: acceleration roots (cubic). The same points
    // partition the segment into velocity-monotone intervals, which the
    // crossing check below relies on.
    double v_extrema[3];
    int v_extrema_count = 0;
    {
        limit_detail::cubic_roots(20.0 * segment.c5, 12.0 * segment.c4,
                                  6.0 * segment.c3, 2.0 * segment.c2,
                                  v_extrema, v_extrema_count);
        for(int i = 0; i < v_extrema_count; ++i) {
            const double x = v_extrema[i];
            if(x > 0.0 && x < n &&
               std::fabs(velocity_at(x)) > limits.max_velocity + Epsilon) {
                return false;
            }
        }
    }

    // Velocity sign changes flip the direction-aware acceleration bound:
    // deceleration (d_max) applies up to the crossing and acceleration
    // (a_max) from it, so by continuity the crossing point itself must
    // satisfy the tighter of the two. Between consecutive velocity extrema
    // the velocity is monotone, so each interval holds at most one crossing,
    // located by bounded bisection.
    {
        double marks[5];
        int mark_count = 0;
        marks[mark_count++] = 0.0;
        for(int i = 0; i < v_extrema_count; ++i) {
            if(v_extrema[i] > 0.0 && v_extrema[i] < n) {
                marks[mark_count++] = v_extrema[i];
            }
        }
        marks[mark_count++] = n;
        // insertion sort (<= 5 entries)
        for(int i = 1; i < mark_count; ++i) {
            const double key = marks[i];
            int k = i - 1;
            while(k >= 0 && marks[k] > key) {
                marks[k + 1] = marks[k];
                --k;
            }
            marks[k + 1] = key;
        }
        const double tight =
            limits.max_acceleration < limits.max_deceleration
                ? limits.max_acceleration
                : limits.max_deceleration;
        for(int i = 0; i + 1 < mark_count; ++i) {
            double lo = marks[i];
            double hi = marks[i + 1];
            double v_lo = velocity_at(lo);
            double v_hi = velocity_at(hi);
            if(v_lo == 0.0) {
                if(std::fabs(acceleration_at(lo)) > tight + Epsilon) {
                    return false;
                }
                continue;
            }
            if(v_lo * v_hi >= 0.0) {
                continue;
            }
            for(int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                const double v_mid = velocity_at(mid);
                if(v_lo * v_mid <= 0.0) {
                    hi = mid;
                } else {
                    lo = mid;
                    v_lo = v_mid;
                }
            }
            const double x_cross = 0.5 * (lo + hi);
            if(std::fabs(acceleration_at(x_cross)) > tight + Epsilon) {
                return false;
            }
        }
    }

    return true;
}

inline bool state_within_limits(State1D state, Limits1D limits)
{
    // Direction-of-motion-relative bound, matching within_limits: |v| growing
    // (v·a >= 0) is acceleration, |v| shrinking is deceleration.
    const double bound = state.velocity * state.acceleration >= 0.0
                             ? limits.max_acceleration
                             : limits.max_deceleration;
    return std::fabs(state.velocity) <= limits.max_velocity &&
           std::fabs(state.acceleration) <= bound;
}

inline rt::Result<Profile1D> plan(State1D from, Target1D to, Limits1D limits)
{
    if(!is_finite(from) || !is_finite(to) || !is_finite(limits) || limits.max_velocity <= 0.0 ||
       limits.max_acceleration <= 0.0 || limits.max_deceleration <= 0.0 ||
       limits.max_jerk <= 0.0) {
        return rt::Result<Profile1D>::failure(rt::ErrorCode::invalid_argument);
    }

    const State1D finish{to.position, to.velocity, to.acceleration};
    if(!state_within_limits(from, limits) || !state_within_limits(finish, limits)) {
        return rt::Result<Profile1D>::failure(rt::ErrorCode::infeasible);
    }

    const double distance = std::fabs(to.position - from.position);
    const double velocity_floor = limits.max_velocity * 0.25;
    double guess = 1.0;
    if(distance > 0.0) {
        guess = distance / velocity_floor;
        const double by_accel =
            std::sqrt((2.0 * distance) / limits.max_acceleration);
        const double by_decel =
            std::sqrt((2.0 * distance) / limits.max_deceleration);
        const double by_jerk = std::cbrt((6.0 * distance) / limits.max_jerk);
        if(by_accel > guess) {
            guess = by_accel;
        }
        if(by_decel > guess) {
            guess = by_decel;
        }
        if(by_jerk > guess) {
            guess = by_jerk;
        }
    }

    const double start_stop =
        (std::fabs(from.velocity) + std::fabs(to.velocity)) / limits.max_acceleration + 1.0;
    if(start_stop > guess) {
        guess = start_stop;
    }

    std::int64_t cycles = static_cast<std::int64_t>(std::ceil(guess));
    if(cycles < 1) {
        cycles = 1;
    }

    for(int attempt = 0; attempt < 80; ++attempt) {
        const Segment1D segment = make_quintic_segment(from, to, cycles);
        if(within_limits(segment, limits)) {
            Profile1D profile{};
            const rt::ErrorCode added = profile.add_segment(segment);
            if(added != rt::ErrorCode::ok) {
                return rt::Result<Profile1D>::failure(added);
            }
            return rt::Result<Profile1D>::success(profile);
        }
        cycles = cycles + cycles / 4 + 1;
    }

    return rt::Result<Profile1D>::failure(rt::ErrorCode::infeasible);
}

} // namespace plcopen::core::otg
