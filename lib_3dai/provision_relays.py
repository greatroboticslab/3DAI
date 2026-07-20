"""
provision_relays.py
===================
One-shot provisioning for the scanner rig's ESP32 GPIO/relay board.

The ESP32 firmware (ESP32_Laser_Control) hard-codes no pins: every channel is
assigned at runtime with a serial ``CONFIG`` command. This script holds the pin
map for THIS rig in one place and pushes it to the board, so the mapping lives
in version control instead of someone's memory.

SAFETY (see AGENTS.md)
----------------------
- Importing this module does nothing to hardware. No serial port is opened and
  no command is sent at import time.
- The default action is a DRY RUN: it only prints the CONFIG commands it would
  send. You must pass ``--commit`` to actually open the port and configure the
  board.
- Every channel is provisioned with SAFE OFF, so on every ESP32 boot the outputs
  are driven de-energized before the board reports READY. This script refuses to
  provision a channel as SAFE ON: near the 5.5 W laser, a relay must never come
  up energized.
- This script never issues SET / never energizes an output. It only writes the
  channel configuration (pin, polarity, safe state). Turning a relay ON is a
  separate, deliberate act done elsewhere.

Usage
-----
    # Show what would be sent (no hardware touched):
    python lib_3dai/provision_relays.py

    # Actually configure the board on a specific port:
    python lib_3dai/provision_relays.py --commit --port COM3

    # Auto-detect the ESP32 and configure it:
    python lib_3dai/provision_relays.py --commit

    # Read back the board's current channel table (opens the port, no writes):
    python lib_3dai/provision_relays.py --status --port COM3
"""

from dataclasses import dataclass


# ── ESP32 pin-safety constants ─────────────────────────────────────────────
# GPIO6-11 are bonded to the on-board SPI flash; driving them hangs/bricks the
# boot. GPIO34-39 are input-only silicon and cannot be outputs.
_SPI_FLASH_PINS = range(6, 12)      # 6,7,8,9,10,11
_INPUT_ONLY_PINS = range(34, 40)    # 34..39


@dataclass(frozen=True)
class RelayChannel:
    """One provisioned output channel on the rig."""
    ch: int              # 1-based channel number (matches firmware)
    pin: int             # ESP32 GPIO number
    label: str           # what this channel drives, for humans
    active_high: bool = True   # relay energizes on HIGH
    safe_on: bool = False      # boot/idle state; MUST stay False for this rig


# ── THIS RIG'S PIN MAP ─────────────────────────────────────────────────────
# Known-safe, output-capable GPIOs confirmed for the scanner ESP32.
#
# Wiring: GPIO18/19/25/26 are the four signal lines; the board's other two wired
# pins are 5V and GND (power for the relay module, not controllable GPIO, so not
# listed here).
#
# Polarity: this board was bench-verified ACTIVE-HIGH on 2026-07-20 -- with
# active_high=True and safe_on=False the firmware drives the pin LOW on boot,
# which DE-energizes the relay (confirmed: CH1/GPIO18 relay dropped out with
# POL HIGH SAFE OFF, and energized with POL LOW). Do NOT set these active_high
# =False: on this board that drives the pin HIGH for "off" and energizes the
# relay on boot -- the exact failure mode to avoid near the laser.
#
# p6 / p7 were intentionally dropped: on this board they are not usable GPIO
# (GPIO6/7 are SPI-flash pins). Re-add them here once the real GPIO numbers
# behind those silkscreen labels are confirmed, e.g.:
#     RelayChannel(ch=5, pin=<real gpio>, label="...", active_high=True),
RIG_CHANNELS: list[RelayChannel] = [
    RelayChannel(ch=1, pin=18, label="relay on GPIO18 (active-HIGH)", active_high=True),
    RelayChannel(ch=2, pin=19, label="relay on GPIO19 (active-HIGH)", active_high=True),
    RelayChannel(ch=3, pin=25, label="relay on GPIO25 (active-HIGH)", active_high=True),
    RelayChannel(ch=4, pin=26, label="relay on GPIO26 (active-HIGH)", active_high=True),
]


def validate_channels(channels: list[RelayChannel]) -> None:
    """Raise ValueError if any channel is unsafe or the map is inconsistent.

    Pure, no I/O. Safe to call from tests or before ever touching a port.
    """
    seen_ch: dict[int, RelayChannel] = {}
    seen_pin: dict[int, RelayChannel] = {}
    for c in channels:
        if c.pin in _SPI_FLASH_PINS:
            raise ValueError(
                f"CH {c.ch}: GPIO{c.pin} is an SPI-flash pin (6-11); it would "
                f"hang the ESP32 on boot. Refusing to provision."
            )
        if c.pin in _INPUT_ONLY_PINS:
            raise ValueError(
                f"CH {c.ch}: GPIO{c.pin} is input-only (34-39) and cannot drive "
                f"an output."
            )
        if not (0 <= c.pin <= 39):
            raise ValueError(f"CH {c.ch}: GPIO{c.pin} out of range 0-39.")
        if c.safe_on:
            raise ValueError(
                f"CH {c.ch}: safe_on=True is not allowed on this rig; outputs "
                f"must boot de-energized (SAFE OFF)."
            )
        if c.ch in seen_ch:
            raise ValueError(f"Duplicate channel number {c.ch}.")
        if c.pin in seen_pin:
            raise ValueError(
                f"GPIO{c.pin} assigned to both CH {seen_pin[c.pin].ch} and "
                f"CH {c.ch}."
            )
        seen_ch[c.ch] = c
        seen_pin[c.pin] = c


def config_command(c: RelayChannel) -> str:
    """Return the exact firmware CONFIG line for a channel (no I/O)."""
    pol = "HIGH" if c.active_high else "LOW"
    safe = "ON" if c.safe_on else "OFF"
    return f"CONFIG {c.ch} PIN {c.pin} POL {pol} SAFE {safe}"


def print_plan(channels: list[RelayChannel]) -> None:
    print("Planned ESP32 relay provisioning (SAFE=OFF on every channel):\n")
    for c in channels:
        print(f"  {config_command(c):<45}  # {c.label}")
    print(
        "\nDRY RUN: nothing was sent. Re-run with --commit to configure the "
        "board.\nThis only writes channel config; it does not turn any output ON."
    )


def commit(channels: list[RelayChannel], port: str | None) -> int:
    """Open the serial port and send CONFIG for each channel. Returns exit code.

    Imported lazily so that importing this module never pulls in pyserial or
    touches a port.
    """
    from relay_controller import RelayController  # local import, on purpose

    if port is None:
        print("No --port given; scanning for an ESP32...")
        port = RelayController.find_esp32()
        if port is None:
            print("ERROR: no ESP32 found on any serial port.")
            return 2
        print(f"Found ESP32 on {port}.")

    rc = RelayController(port)
    if not rc.connect():
        print(f"ERROR: could not connect / no PONG on {port}.")
        return 2

    try:
        ok = True
        for c in channels:
            sent = rc.configure_channel(
                c.ch, pin=c.pin, active_high=c.active_high, safe_on=c.safe_on
            )
            print(f"  {config_command(c):<45}  -> {'OK' if sent else 'FAILED'}")
            ok = ok and sent
        # Belt-and-suspenders: drive everything to its safe (OFF) state now.
        rc.safe_all()
        print("\nAll channels driven to SAFE (OFF).")
        rc.print_status()
        return 0 if ok else 1
    finally:
        rc.disconnect()


def show_status(port: str | None) -> int:
    """Read back the board's channel table without writing anything."""
    from relay_controller import RelayController  # local import, on purpose

    if port is None:
        port = RelayController.find_esp32()
        if port is None:
            print("ERROR: no ESP32 found on any serial port.")
            return 2
    rc = RelayController(port)
    if not rc.connect():
        print(f"ERROR: could not connect / no PONG on {port}.")
        return 2
    try:
        rc.print_status()
        return 0
    finally:
        rc.disconnect()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", action="store_true",
        help="Actually open the serial port and send CONFIG (default: dry run).",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Read back the board's current channel table (no writes).",
    )
    parser.add_argument(
        "--port", default=None,
        help="Serial port (e.g. COM3). If omitted, the ESP32 is auto-detected.",
    )
    args = parser.parse_args(argv)

    # Validate the pin map before anything else, dry run included.
    validate_channels(RIG_CHANNELS)

    if args.status:
        return show_status(args.port)
    if args.commit:
        return commit(RIG_CHANNELS, args.port)

    print_plan(RIG_CHANNELS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
