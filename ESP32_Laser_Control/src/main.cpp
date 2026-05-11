/**
 * ESP32 Generic GPIO Output Controller
 * ======================================
 * Firmware for the NodeMCU-32S (or any ESP32 Arduino target).
 *
 * All channel configuration is done at runtime via the serial command
 * protocol — no pin assignments are hard-coded.  Configuration is saved
 * to NVS flash so it survives power cycles.  On every boot, each
 * configured output is immediately driven to its safe state before the
 * device signals READY, so relays are never left in an unknown condition.
 *
 * SERIAL INTERFACE
 * ─────────────────
 * Baud : 115200  8N1
 * Line endings accepted: LF, CR, or CR+LF
 *
 * COMMANDS
 * ─────────
 *   CONFIG <ch> PIN <n> POL <HIGH|LOW> SAFE <ON|OFF>
 *       Configure (or re-configure) a channel.
 *       ch   : 1-based channel number (1 – MAX_CHANNELS)
 *       n    : GPIO pin number
 *       POL  : HIGH → relay energises on HIGH; LOW → energises on LOW
 *       SAFE : output state applied on boot and when SAFE command issued
 *
 *   SET <ch> <ON|OFF>        Drive a channel ON or OFF
 *   GET <ch>                 Report a single channel's configuration + state
 *   STATUS                   Report all configured channels
 *   SAFE                     Drive all channels to their safe states now
 *   REMOVE <ch>              Delete a channel (drives safe before removing)
 *   FACTORY                  Erase all channel config, drive all safe
 *   PING                     Connectivity check → PONG
 *   HELP                     Print command reference
 *
 * RESPONSES
 * ──────────
 *   OK                          Success
 *   OK FACTORY RESET            Factory command succeeded
 *   PONG                        Response to PING
 *   CH <n> PIN <p> POL <pol> SAFE <s> STATE <st>   Channel status line
 *   CH <n> UNCONFIGURED
 *   NO CHANNELS CONFIGURED
 *   ERR <REASON>[: detail]      Error
 */

#include <Arduino.h>
#include <Preferences.h>

// ── Configuration constants ────────────────────────────────────────────────

static constexpr uint8_t  MAX_CHANNELS   = 16;
static constexpr uint32_t BAUD_RATE      = 115200;
static constexpr size_t   CMD_BUF_SIZE   = 160;

// GPIO 34-39 on the ESP32 are input-only silicon; refuse to configure them.
static constexpr uint8_t INPUT_ONLY_MIN  = 34;
static constexpr uint8_t INPUT_ONLY_MAX  = 39;

// NVS namespace
static constexpr char NVS_NS[] = "gpio_ctrl";

// ── Data model ─────────────────────────────────────────────────────────────

struct Channel {
    bool    configured;
    uint8_t pin;
    bool    activeHigh;    // true  → output HIGH energises relay
    bool    safeState;     // true  → ON,  false → OFF
    bool    currentState;  // last state set via SET (or safe on init)
};

static Channel    channels[MAX_CHANNELS];
static Preferences prefs;
static String     inputBuffer;

// ── Pin helpers ────────────────────────────────────────────────────────────

static bool isInputOnly(uint8_t pin) {
    return pin >= INPUT_ONLY_MIN && pin <= INPUT_ONLY_MAX;
}

/** Returns true if another channel (not excludeCh) already owns this pin. */
static bool pinInUse(uint8_t pin, int excludeCh) {
    for (int i = 0; i < MAX_CHANNELS; i++) {
        if (i == excludeCh) continue;
        if (channels[i].configured && channels[i].pin == pin) return true;
    }
    return false;
}

/** Drive the physical pin to match ch.currentState, respecting polarity. */
static void writePin(int ch) {
    const Channel& c = channels[ch];
    bool level = c.activeHigh ? c.currentState : !c.currentState;
    digitalWrite(c.pin, level ? HIGH : LOW);
}

/** Set logical state for a channel and immediately update the pin. */
static void applyState(int ch, bool state) {
    channels[ch].currentState = state;
    writePin(ch);
}

/** Drive every configured channel to its saved safe state. */
static void applySafeStates() {
    for (int i = 0; i < MAX_CHANNELS; i++) {
        if (channels[i].configured) {
            applyState(i, channels[i].safeState);
        }
    }
}

// ── NVS persistence ────────────────────────────────────────────────────────
//
// Key scheme (all ≤ 15 chars, NVS limit):
//   "cNc"  bool   – configured flag  (N = 0-15)
//   "cNp"  uint8  – GPIO pin
//   "cNa"  bool   – activeHigh
//   "cNs"  bool   – safeState

static void saveChannel(int ch) {
    char key[8];
    prefs.begin(NVS_NS, false);

    snprintf(key, sizeof(key), "c%dc", ch);
    if (channels[ch].configured) {
        prefs.putBool(key, true);
        snprintf(key, sizeof(key), "c%dp", ch); prefs.putUChar(key, channels[ch].pin);
        snprintf(key, sizeof(key), "c%da", ch); prefs.putBool(key,  channels[ch].activeHigh);
        snprintf(key, sizeof(key), "c%ds", ch); prefs.putBool(key,  channels[ch].safeState);
    } else {
        prefs.putBool(key, false);
    }

    prefs.end();
}

static void loadChannels() {
    char key[8];
    prefs.begin(NVS_NS, true);  // read-only open

    for (int i = 0; i < MAX_CHANNELS; i++) {
        snprintf(key, sizeof(key), "c%dc", i);
        channels[i].configured = prefs.getBool(key, false);
        if (channels[i].configured) {
            snprintf(key, sizeof(key), "c%dp", i);
            channels[i].pin = prefs.getUChar(key, 0);
            snprintf(key, sizeof(key), "c%da", i);
            channels[i].activeHigh = prefs.getBool(key, true);
            snprintf(key, sizeof(key), "c%ds", i);
            channels[i].safeState  = prefs.getBool(key, false);
            channels[i].currentState = channels[i].safeState;
            pinMode(channels[i].pin, OUTPUT);
        }
    }

    prefs.end();
}

static void eraseAllChannels() {
    // First, drive everything safe and release pins.
    for (int i = 0; i < MAX_CHANNELS; i++) {
        if (channels[i].configured) {
            applyState(i, channels[i].safeState);
            pinMode(channels[i].pin, INPUT);
            channels[i].configured = false;
        }
    }
    // Then wipe NVS namespace.
    prefs.begin(NVS_NS, false);
    prefs.clear();
    prefs.end();
}

// ── Serial output helpers ──────────────────────────────────────────────────

static void printChannelStatus(int ch) {
    const Channel& c = channels[ch];
    if (!c.configured) {
        Serial.printf("CH %d UNCONFIGURED\r\n", ch + 1);
    } else {
        Serial.printf("CH %d PIN %d POL %s SAFE %s STATE %s\r\n",
            ch + 1,
            c.pin,
            c.activeHigh   ? "HIGH" : "LOW",
            c.safeState    ? "ON"   : "OFF",
            c.currentState ? "ON"   : "OFF");
    }
}

static void printHelp() {
    Serial.print(
        "COMMANDS\r\n"
        "  CONFIG <ch> PIN <n> POL <HIGH|LOW> SAFE <ON|OFF>  Configure a channel\r\n"
        "  SET <ch> <ON|OFF>                                  Drive output state\r\n"
        "  GET <ch>                                           Single channel status\r\n"
        "  STATUS                                             All configured channels\r\n"
        "  SAFE                                               Drive all to safe state\r\n"
        "  REMOVE <ch>                                        Remove channel config\r\n"
        "  FACTORY                                            Erase all config\r\n"
        "  PING                                               Connectivity check\r\n"
        "  HELP                                               This message\r\n"
        "  ch = 1-based channel number (1-");
    Serial.print(MAX_CHANNELS);
    Serial.print(")\r\n");
}

// ── Command processor ──────────────────────────────────────────────────────

static void processCommand(const String& raw) {
    String cmd = raw;
    cmd.trim();
    if (cmd.length() == 0) return;

    // Uppercase copy for keyword parsing.
    String u = cmd;
    u.toUpperCase();

    // ── PING ────────────────────────────────────────────────────────────────
    if (u == "PING") {
        Serial.print("PONG\r\n");
        return;
    }

    // ── HELP ────────────────────────────────────────────────────────────────
    if (u == "HELP") {
        printHelp();
        return;
    }

    // ── STATUS ──────────────────────────────────────────────────────────────
    if (u == "STATUS") {
        bool any = false;
        for (int i = 0; i < MAX_CHANNELS; i++) {
            if (channels[i].configured) {
                printChannelStatus(i);
                any = true;
            }
        }
        if (!any) Serial.print("NO CHANNELS CONFIGURED\r\n");
        return;
    }

    // ── SAFE ────────────────────────────────────────────────────────────────
    if (u == "SAFE") {
        applySafeStates();
        Serial.print("OK\r\n");
        return;
    }

    // ── FACTORY ─────────────────────────────────────────────────────────────
    if (u == "FACTORY") {
        eraseAllChannels();
        Serial.print("OK FACTORY RESET\r\n");
        return;
    }

    // ── CONFIG <ch> PIN <n> POL <HIGH|LOW> SAFE <ON|OFF> ───────────────────
    if (u.startsWith("CONFIG ")) {
        int  ch = 0, pin = 0;
        char pol[8] = {}, safe[8] = {};
        int  n = sscanf(u.c_str(), "CONFIG %d PIN %d POL %7s SAFE %7s",
                        &ch, &pin, pol, safe);
        if (n != 4) {
            Serial.print("ERR SYNTAX: CONFIG <ch> PIN <n> POL <HIGH|LOW> SAFE <ON|OFF>\r\n");
            return;
        }

        ch--;  // convert to 0-based
        if (ch < 0 || ch >= MAX_CHANNELS) { Serial.print("ERR INVALID_CHANNEL\r\n");   return; }
        if (pin < 0 || pin > 39)          { Serial.print("ERR PIN_OUT_OF_RANGE\r\n");  return; }
        if (isInputOnly((uint8_t)pin))     { Serial.print("ERR PIN_INPUT_ONLY\r\n");   return; }
        if (pinInUse((uint8_t)pin, ch))   { Serial.print("ERR PIN_IN_USE\r\n");        return; }

        if (strcmp(pol,  "HIGH") != 0 && strcmp(pol,  "LOW") != 0) {
            Serial.print("ERR INVALID_POL: use HIGH or LOW\r\n"); return;
        }
        if (strcmp(safe, "ON") != 0 && strcmp(safe, "OFF") != 0) {
            Serial.print("ERR INVALID_SAFE: use ON or OFF\r\n"); return;
        }

        // Release old pin if the pin number is changing.
        if (channels[ch].configured && channels[ch].pin != (uint8_t)pin) {
            pinMode(channels[ch].pin, INPUT);
        }

        channels[ch].configured  = true;
        channels[ch].pin         = (uint8_t)pin;
        channels[ch].activeHigh  = (strcmp(pol, "HIGH") == 0);
        channels[ch].safeState   = (strcmp(safe, "ON")  == 0);
        channels[ch].currentState = channels[ch].safeState;

        pinMode(pin, OUTPUT);
        applyState(ch, channels[ch].safeState);
        saveChannel(ch);
        Serial.print("OK\r\n");
        return;
    }

    // ── SET <ch> <ON|OFF> ───────────────────────────────────────────────────
    if (u.startsWith("SET ")) {
        int  ch = 0;
        char state[8] = {};
        if (sscanf(u.c_str(), "SET %d %7s", &ch, state) != 2) {
            Serial.print("ERR SYNTAX: SET <ch> <ON|OFF>\r\n");
            return;
        }
        ch--;
        if (ch < 0 || ch >= MAX_CHANNELS || !channels[ch].configured) {
            Serial.print("ERR INVALID_CHANNEL\r\n"); return;
        }
        if (strcmp(state, "ON") != 0 && strcmp(state, "OFF") != 0) {
            Serial.print("ERR INVALID_STATE: use ON or OFF\r\n"); return;
        }
        applyState(ch, strcmp(state, "ON") == 0);
        Serial.print("OK\r\n");
        return;
    }

    // ── GET <ch> ────────────────────────────────────────────────────────────
    if (u.startsWith("GET ")) {
        int ch = 0;
        if (sscanf(u.c_str(), "GET %d", &ch) != 1) {
            Serial.print("ERR SYNTAX: GET <ch>\r\n"); return;
        }
        ch--;
        if (ch < 0 || ch >= MAX_CHANNELS) {
            Serial.print("ERR INVALID_CHANNEL\r\n"); return;
        }
        printChannelStatus(ch);
        return;
    }

    // ── REMOVE <ch> ─────────────────────────────────────────────────────────
    if (u.startsWith("REMOVE ")) {
        int ch = 0;
        if (sscanf(u.c_str(), "REMOVE %d", &ch) != 1) {
            Serial.print("ERR SYNTAX: REMOVE <ch>\r\n"); return;
        }
        ch--;
        if (ch < 0 || ch >= MAX_CHANNELS) {
            Serial.print("ERR INVALID_CHANNEL\r\n"); return;
        }
        if (channels[ch].configured) {
            applyState(ch, channels[ch].safeState);
            pinMode(channels[ch].pin, INPUT);
            channels[ch].configured = false;
            saveChannel(ch);
        }
        Serial.print("OK\r\n");
        return;
    }

    Serial.print("ERR UNKNOWN_COMMAND - type HELP\r\n");
}

// ── Arduino entry points ───────────────────────────────────────────────────

void setup() {
    Serial.begin(BAUD_RATE);
    delay(500);  // give the host a moment to open the port

    inputBuffer.reserve(CMD_BUF_SIZE);

    // Load channel config from flash and immediately apply safe states.
    // This happens BEFORE signalling READY so outputs are never in an
    // unknown state when the host sees the device come online.
    loadChannels();
    applySafeStates();

    Serial.print("READY\r\n");
}

void loop() {
    while (Serial.available()) {
        char c = static_cast<char>(Serial.read());

        if (c == '\n' || c == '\r') {
            if (inputBuffer.length() > 0) {
                processCommand(inputBuffer);
                inputBuffer = "";
            }
        } else if (inputBuffer.length() < CMD_BUF_SIZE - 1) {
            inputBuffer += c;
        }
        // Silently discard characters when buffer is full.
    }
}