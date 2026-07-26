// SA2 host byte pump for the Feetech STS bus (SO-ARM101 validation plan).
//
// The approved adapter matrix (feetech-sts-adapter-semantics.md 2.1) keeps
// the kernel protocol layer IO-free and assigns serial transport to the
// host: this demo is that host. Per cycle it runs exactly one
// SYNC_WRITE + SYNC_READ exchange (matrix 2.8), feeds replies back into
// FeetechBus, and reports the communication health counters.
//
// Two transports:
//   --dry-run          in-process FeetechSim register image (CI path)
//   --port <device>    POSIX serial (real bus; Linux host; requires the
//                      matrix 2.6 unit scales on the command line — the
//                      4.8 gate is still open, so no built-in constants)
//
// Motion is HOLD by default (latch the first feedback positions). The only
// scripted motion is a small sine on one joint and it requires
// --confirm-motion. Mechanical safety remains the operator's
// responsibility; the kernel soft-limit machinery is NOT in this loop —
// this is a bus-transport validation harness, not a motion controller.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "adapters/feetech.h"

#ifndef _WIN32
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace
{

using namespace plcopen::core;

struct Options
{
    std::uint8_t ids[adapters::FeetechBus::StorageCapacity] = {};
    std::size_t id_count = 0;
    bool dry_run = false;
    const char *port = nullptr;
    int baud = 1000000;
    int hz = 50;
    long cycles = 200;
    bool motion_sine = false;
    bool confirm_motion = false;
    std::size_t sine_joint = 0;
    double amplitude_rad = 0.15;
    double period_s = 4.0;
    double velocity_units_per_rad_s = 1.0;
    double acceleration_units_per_rad_s2 = 1.0;
};

bool parse_ids(const char *text, Options &options)
{
    std::size_t count = 0;
    const char *cursor = text;
    while(*cursor != '\0') {
        char *end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if(end == cursor || value <= 0 || value >= 0xFE ||
           count >= adapters::FeetechBus::StorageCapacity) {
            return false;
        }
        options.ids[count++] = static_cast<std::uint8_t>(value);
        cursor = *end == ',' ? end + 1 : end;
    }
    options.id_count = count;
    return count > 0;
}

class Transport
{
public:
    virtual ~Transport() = default;
    virtual bool exchange(const adapters::FeetechPacket &sync_write,
                          const adapters::FeetechPacket &sync_read,
                          adapters::FeetechBus &bus) = 0;
};

class SimTransport final : public Transport
{
public:
    bool bind(const std::uint8_t *ids, std::size_t count,
              const adapters::FeetechConfig &config)
    {
        return sim_.configure(config) == rt::ErrorCode::ok &&
               sim_.bind(ids, count) == rt::ErrorCode::ok;
    }

    bool exchange(const adapters::FeetechPacket &sync_write,
                  const adapters::FeetechPacket &sync_read,
                  adapters::FeetechBus &bus) override
    {
        adapters::FeetechPacket responses[adapters::FeetechBus::StorageCapacity];
        if(sim_.exchange(sync_write, sync_read, responses,
                         adapters::FeetechBus::StorageCapacity) !=
           rt::ErrorCode::ok) {
            return false;
        }
        for(std::size_t i = 0; i < count_; ++i) {
            for(std::size_t b = 0; b < responses[i].size; ++b) {
                bus.push(responses[i].bytes[b]);
            }
        }
        return true;
    }

    void set_count(std::size_t count) { count_ = count; }

private:
    adapters::FeetechSim sim_;
    std::size_t count_ = 0;
};

#ifndef _WIN32
class SerialTransport final : public Transport
{
public:
    bool open_port(const char *device, int baud, int reply_budget_us)
    {
        reply_budget_us_ = reply_budget_us;
        fd_ = ::open(device, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if(fd_ < 0) {
            std::printf("RUNNER serial open failed: %s\n", device);
            return false;
        }
        termios tio{};
        if(tcgetattr(fd_, &tio) != 0) return false;
        cfmakeraw(&tio);
        tio.c_cc[VMIN] = 0;
        tio.c_cc[VTIME] = 0;
        speed_t speed = B1000000;
        if(baud == 500000) speed = B500000;
        if(baud == 115200) speed = B115200;
        cfsetispeed(&tio, speed);
        cfsetospeed(&tio, speed);
        return tcsetattr(fd_, TCSANOW, &tio) == 0;
    }

    ~SerialTransport() override
    {
        if(fd_ >= 0) ::close(fd_);
    }

    bool exchange(const adapters::FeetechPacket &sync_write,
                  const adapters::FeetechPacket &sync_read,
                  adapters::FeetechBus &bus) override
    {
        if(::write(fd_, sync_write.bytes, sync_write.size) < 0) return false;
        if(::write(fd_, sync_read.bytes, sync_read.size) < 0) return false;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::microseconds(reply_budget_us_);
        std::uint8_t chunk[256];
        while(std::chrono::steady_clock::now() < deadline) {
            const ssize_t got = ::read(fd_, chunk, sizeof(chunk));
            if(got > 0) {
                for(ssize_t i = 0; i < got; ++i) bus.push(chunk[i]);
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
        return true;
    }

    bool write_raw(const adapters::FeetechPacket &packet)
    {
        return ::write(fd_, packet.bytes, packet.size) ==
               static_cast<ssize_t>(packet.size);
    }

private:
    int fd_ = -1;
    int reply_budget_us_ = 4000;
};
#endif

} // namespace

int main(int argc, char **argv)
{
    Options options{};
    const std::uint8_t default_ids[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(options.ids, default_ids, sizeof(default_ids));
    options.id_count = 6;

    for(int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const auto next = [&]() -> const char * {
            return i + 1 < argc ? argv[++i] : nullptr;
        };
        if(std::strcmp(arg, "--ids") == 0) {
            const char *value = next();
            if(value == nullptr || !parse_ids(value, options)) {
                std::printf("RUNNER invalid --ids\n");
                return 1;
            }
        } else if(std::strcmp(arg, "--dry-run") == 0) {
            options.dry_run = true;
        } else if(std::strcmp(arg, "--port") == 0) {
            options.port = next();
        } else if(std::strcmp(arg, "--baud") == 0) {
            const char *value = next();
            options.baud = value ? std::atoi(value) : options.baud;
        } else if(std::strcmp(arg, "--hz") == 0) {
            const char *value = next();
            options.hz = value ? std::atoi(value) : options.hz;
        } else if(std::strcmp(arg, "--cycles") == 0) {
            const char *value = next();
            options.cycles = value ? std::atol(value) : options.cycles;
        } else if(std::strcmp(arg, "--motion") == 0) {
            const char *value = next();
            options.motion_sine = value && std::strcmp(value, "sine") == 0;
        } else if(std::strcmp(arg, "--confirm-motion") == 0) {
            options.confirm_motion = true;
        } else if(std::strcmp(arg, "--joint") == 0) {
            const char *value = next();
            options.sine_joint = value ? static_cast<std::size_t>(std::atoi(value)) : 0;
        } else if(std::strcmp(arg, "--amplitude-rad") == 0) {
            const char *value = next();
            options.amplitude_rad = value ? std::atof(value) : options.amplitude_rad;
        } else if(std::strcmp(arg, "--period-s") == 0) {
            const char *value = next();
            options.period_s = value ? std::atof(value) : options.period_s;
        } else if(std::strcmp(arg, "--velocity-units-per-rad-s") == 0) {
            const char *value = next();
            options.velocity_units_per_rad_s = value ? std::atof(value) : 0.0;
        } else if(std::strcmp(arg, "--acceleration-units-per-rad-s2") == 0) {
            const char *value = next();
            options.acceleration_units_per_rad_s2 = value ? std::atof(value) : 0.0;
        }
    }

    if(options.hz <= 0 || options.hz > 100) {
        // matrix 2.8: 50Hz default tier, 100Hz upper tier pending S5-HW.
        std::printf("RUNNER --hz must be within 1..100 (matrix 2.8)\n");
        return 1;
    }
    if(options.motion_sine && !options.confirm_motion) {
        std::printf("RUNNER sine motion moves the arm: clear the workspace "
                    "and re-run with --confirm-motion\n");
        return 1;
    }
    if(!options.dry_run && options.port == nullptr) {
        std::printf("RUNNER either --dry-run or --port is required\n");
        return 1;
    }

    adapters::FeetechConfig config{};
    config.velocity_units_per_radian_second = options.velocity_units_per_rad_s;
    config.acceleration_units_per_radian_second2 =
        options.acceleration_units_per_rad_s2;

    adapters::FeetechBus bus;
    if(bus.configure(config) != rt::ErrorCode::ok ||
       bus.bind(options.ids, options.id_count) != rt::ErrorCode::ok) {
        std::printf("RUNNER bus bind failed\n");
        return 1;
    }

    SimTransport sim_transport;
#ifndef _WIN32
    SerialTransport serial_transport;
#endif
    Transport *transport = nullptr;
    if(options.dry_run) {
        if(!sim_transport.bind(options.ids, options.id_count, config)) {
            std::printf("RUNNER sim bind failed\n");
            return 1;
        }
        sim_transport.set_count(options.id_count);
        transport = &sim_transport;
    } else {
#ifdef _WIN32
        std::printf("RUNNER real serial is POSIX-only in SA2; use --dry-run\n");
        return 1;
#else
        const int cycle_us = 1000000 / options.hz;
        if(!serial_transport.open_port(options.port, options.baud,
                                       cycle_us * 2 / 5)) {
            return 1;
        }
        transport = &serial_transport;
        adapters::FeetechPacket initialize{};
        for(std::size_t i = 0; i < options.id_count; ++i) {
            if(bus.build_initialize(i, initialize) != rt::ErrorCode::ok ||
               !serial_transport.write_raw(initialize)) {
                std::printf("RUNNER initialize (Operating_Mode=0) failed\n");
                return 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
#endif
    }

    // Cycle 0 latch: hold whatever the first feedback reports (real bus) or
    // mid position (sim starts at goal 2048 => 0 rad).
    adapters::ServoSetpoints hold[adapters::FeetechBus::StorageCapacity] = {};
    bool hold_captured = options.dry_run;

    long responses_total = 0;
    long stale_events = 0;
    const auto cycle_duration =
        std::chrono::microseconds(1000000 / options.hz);
    auto next_tick = std::chrono::steady_clock::now();

    for(long cycle = 0; cycle < options.cycles; ++cycle) {
        for(std::size_t i = 0; i < options.id_count; ++i) {
            adapters::ServoSetpoints setpoints = hold[i];
            if(options.motion_sine && hold_captured &&
               i == options.sine_joint) {
                const double phase = 2.0 * adapters::FeetechConfig::Pi *
                                     static_cast<double>(cycle) /
                                     (options.period_s * options.hz);
                setpoints.position += options.amplitude_rad * std::sin(phase);
                setpoints.velocity = options.amplitude_rad * 2.0 *
                                     adapters::FeetechConfig::Pi / options.period_s;
            }
            bus.latch(i, setpoints);
        }

        adapters::FeetechPacket sync_write{};
        adapters::FeetechPacket sync_read{};
        if(bus.build_cycle(sync_write, sync_read) != rt::ErrorCode::ok) {
            std::printf("RUNNER build_cycle failed\n");
            return 1;
        }
        bus.begin_feedback_cycle();
        if(!transport->exchange(sync_write, sync_read, bus)) {
            std::printf("RUNNER transport exchange failed\n");
            return 1;
        }
        bus.end_feedback_cycle();

        bool all_ready = true;
        for(std::size_t i = 0; i < options.id_count; ++i) {
            adapters::ServoFeedback feedback{};
            bus.read_feedback(i, feedback);
            if(feedback.info.communication_ready) {
                ++responses_total;
            } else {
                all_ready = false;
            }
            if(bus.stale(i)) ++stale_events;
            if(!hold_captured && feedback.info.communication_ready) {
                hold[i].position = feedback.position;
            }
        }
        if(!hold_captured && all_ready) {
            hold_captured = true;
        }

        if(!options.dry_run) {
            next_tick += cycle_duration;
            std::this_thread::sleep_until(next_tick);
        }
    }

    const long expected =
        options.cycles * static_cast<long>(options.id_count);
    const bool healthy = responses_total == expected && stale_events == 0;
    std::printf("RUNNER cycles=%ld servos=%zu hz=%d responses=%ld "
                "expected=%ld stale_events=%ld wire_bytes_per_cycle=%zu "
                "bus_utilization=%.1f%%\n",
                options.cycles, options.id_count, options.hz,
                responses_total, expected, stale_events,
                adapters::FeetechBus::cycle_wire_bytes(options.id_count),
                adapters::FeetechBus::utilization_percent(
                    options.id_count,
                    static_cast<std::size_t>(options.hz)));
    std::printf("RUNNER %s\n", healthy ? "PASS" : "FAIL");
    return healthy ? 0 : 1;
}
