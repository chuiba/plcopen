#pragma once

// B2 kinematics plugin contract v1 (approved matrix:
// doc/compliance/kinematics-plugin-semantics.md). Header-file ABI: users
// link their mechanism at compile time; a cross-DSO stable ABI is Phase C
// scope. Implementations must honor the RT-safe contract — no heap
// allocation, no exceptions, no blocking, numeric iteration bounded (<= 3
// Newton steps with a warm seed), out-of-budget reports `infeasible`
// instead of waiting for convergence. The verification harness
// (core/kin/verify.h) asserts the contract; conformance is not on trust.
//
// v1 constraints (decision #9, tightened for the first slice): the joint
// count equals the Cartesian coordinate count (2 or 3) and equals the group
// axis count; the 6R batch lifts this together with orientation support.

#include <cstddef>

#include "geom/geometry.h"
#include "rt/error.h"

namespace plcopen::core::kin
{

// How `singularity_margin` is to be read (decision #3). `angular` means the
// value is a real distance measure to the nearest singular configuration, in
// radians, that L5 may gate a configured threshold against. `none` means the
// mechanism exposes no such measure and the returned value is an inert
// sentinel — a positive gate threshold over a `none` plugin would never bite,
// so L5 rejects that configuration instead of silently not gating.
enum class MarginSemantics
{
    angular,
    none,
};

class Kinematics
{
public:
    virtual ~Kinematics() = default;

    virtual std::size_t joint_count() const = 0;
    // 2 or 3 translational Cartesian coordinates (orientation is v2 scope).
    virtual std::size_t cartesian_count() const = 0;

    virtual rt::ErrorCode forward(const double *joints,
                                  std::size_t joint_count,
                                  geom::Vec3 &cartesian) const = 0;

    // Seed-branch semantics (decision #4): the returned solution stays on
    // the seed's configuration branch; when no same-branch solution exists
    // the call reports `infeasible` — an implicit branch flip is a machine
    // hazard, never a convenience.
    virtual rt::ErrorCode inverse(geom::Vec3 cartesian,
                                  const double *seed_joints,
                                  std::size_t joint_count,
                                  double *joints_out) const = 0;

    // Distance measure to the nearest singular configuration (decision #3):
    // strictly positive away from singularities, approaching zero at them.
    // Read it through `margin_semantics()` — mechanisms that declare `none`
    // return an inert sentinel rather than a measure.
    virtual double singularity_margin(const double *joints,
                                      std::size_t joint_count) const = 0;

    // Capability query for the value above. Defaulted (not pure) so existing
    // plugins stay source-compatible; only sentinel mechanisms override.
    virtual MarginSemantics margin_semantics() const
    {
        return MarginSemantics::angular;
    }
};

} // namespace plcopen::core::kin
