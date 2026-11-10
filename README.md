# 3DAI scanner

A bench scanner that collects data for recognizing what an object is made of.
For every object it records:

- **3D shape** from a projector that shines striped "fringe" patterns on it,
- **how it scatters light** from four lasers in different colors,
- **color, depth and infrared pictures** from a Kinect camera.

This page walks you through turning everything on, opening the data
collection GUI and collecting data, one step at a time. Read it top to bottom
the first time.

> **Use the scanner PC in the lab.** This doesn't work on a laptop or a fresh
> download. The scanner PC has three things that aren't in this GitHub repo:
> the Kinect software (`C:\KinectEnv`), the 3D calibration and fringe scripts
> (the `data\` folder) and the scanner's Python setup
> (`scanner_system\.venv`). On that PC the project is in
> `C:\Users\Robotics_Lab\3DAI\3DAI`.

---

## Quick version (once you've done it before)

1. Turn on the projector, the Kinect and the laser board ([Step 1](#step-1-turn-on-the-hardware)).
2. Room lights off. Nothing on the white table.
3. Double-click **`scan.bat`** (Desktop shortcut **SCAN objects**) and answer its questions.
4. When you're done, double-click **`push_data.bat`** (Desktop shortcut **PUSH data to GitHub**).

---

## What's on the bench

| Thing | What it is | How it connects to the PC |
|---|---|---|
| **Projector** | TI DLP LightCrafter 4500. Shines the fringe stripes for 3D. | HDMI cable. Windows sees it as a **second monitor**. |
| **Kinect** | Xbox One Kinect (Kinect v2). The camera for everything. | Its own power box, plus a USB 3 cable (the **blue** USB ports). |
| **Laser board** | A small ESP32 circuit board that switches the lasers on and off through relays. | USB cable. Windows sees it as **COM3**. |
| **4 lasers** | CH1 and CH2 red (635 nm), CH3 near-infrared (940 nm, **invisible to your eyes**), CH4 green (530 nm). | Wired to the laser board's relays. |
| **Big blue laser** | 455 nm, 5.5 W. **Parked. Never use it, never touch it.** | Not used by any of this. |

---

## Step 1: Turn on the hardware

Do these in order. Each one ends with a way to check that it worked.

### 1a. The projector (fringe)

1. Plug in the projector's power and switch it on. It should light up and
   shine an image onto the table.
2. Make sure its HDMI cable goes into the PC.
3. On the PC, press **Windows key + P** and pick **Extend**. Not Duplicate,
   not "PC screen only".
4. Open **Settings > System > Display**. You'll see two numbered boxes. One is
   the normal monitor and one is the projector (click **Identify** to see which
   is which).
5. Drag the projector's box so it sits **to the right of** the main monitor,
   with their top edges level. Click **Apply** if Windows asks.
6. The main monitor must stay the main display, at 1920 pixels wide. The
   scanner opens its stripe window 1920 pixels to the right of the main
   screen's left edge, which only lands on the projector if the projector is
   placed like this.

**Check it works:** open a Command Prompt in the project folder (see
[How to open a Command Prompt here](#how-to-open-a-command-prompt-in-the-project-folder))
and run:

```
C:\KinectEnv\Scripts\python.exe scanner_system\project_target.py 20
```

A dark grid with a bullseye should appear **on the table** for 20 seconds. If
it shows up on your monitor instead, go back to steps 3 to 5.

### 1b. The Kinect

1. Plug the Kinect's power box into the wall. The light on the power box
   should come on.
2. Plug its USB cable into a **blue** USB 3 port on the PC. A black USB 2 port
   won't work.
3. Wait about 10 seconds.

**Check it works:** in the GUI's **Hardware** page ([Step 2](#step-2-open-the-data-collection-gui)),
the Kinect line should be green. Click **Test Kinect capture**, and you should
see a live picture of the table. Or from a Command Prompt in the project
folder:

```
C:\KinectEnv\Scripts\python.exe scanner_system\kinect_grab_once.py kinect_test.png
```

then open `kinect_test.png`.

### 1c. The lasers (laser board)

The lasers are only switched on by the software, for a second or two at a
time. You never switch them on by hand.

1. Make sure the laser board's USB cable goes into the PC. It must be a
   **data** cable: a charge-only cable lights the board up but Windows won't see
   it.
2. Close anything else that could be using COM3 (the Arduino IDE, PlatformIO's
   serial monitor, a second copy of the GUI or of `scan.bat`). Only one program
   can talk to the board at a time.
3. Put on laser goggles before firing anything. Never look into a beam.

**Check it works (the GUI way):**

1. Open the GUI ([Step 2](#step-2-open-the-data-collection-gui)) and go to **Hardware**.
2. Under **ESP32 (laser + relay controller)** it should say a board is on
   `COM3`. Click **Probe ESP32**. It should say **Connected** and list four relay
   channels (CH1 to CH4).
3. Under **Blink test**, pick a channel, leave it at 1 second and click
   **Fire CH1** (or whichever). The laser should light for a second and then go
   off by itself. Tick **Project aiming target while firing** to see the dot
   against the projector's bullseye.
4. CH3 is near-infrared and you **can't see it**. Point a phone camera at the
   table while you fire it: the phone shows the dot.

**Check it works (the command way):** this fires each laser once and reports
which ones lit:

```
scanner_system\.venv\Scripts\python.exe -m scanner_system.selftest
```

`[OK]` for each channel is good. `NO SPOT` means that laser's wire or connector
is loose: CH4's wire has come loose twice before.

### 1d. Check everything at once

`scan.bat` checks everything before it scans. It also starts the database
(Docker) if it isn't running, so you never have to touch Docker yourself. You
want every line to say `[OK]`:

```
Checking the scanner...
  [OK] Database:    Connected (... samples).
  [OK] Kinect:      Connected: ...
  [OK] Laser board: Connected on COM3. 4 relay channel(s) ...
  [OK] Projector:   Connected as a display: ...
```

If it says `STOP`, it tells you which thing to fix. Fix it and double-click
`scan.bat` again.

---

## Step 2: Open the data collection GUI

The GUI runs in your web browser. It's where you see each piece of hardware,
fire the lasers to test them, scan one object at a time and look through
everything that's been collected.

1. Open **Docker Desktop** from the Start menu and wait until it says the
   engine is **running**. That's the database. Running `scan.bat` once also
   starts it.
2. Double-click **`scanner_system\launch_demo.bat`**.
3. Your browser opens **http://127.0.0.1:8502**. Keep the black window open:
   closing it closes the GUI.

The pages are in the left sidebar:

| Page | What it's for |
|---|---|
| **Capture** | Scan one object: type its label and material, pick the lasers, click **Run capture**. |
| **Samples** | Every object scanned so far, with its pictures, laser numbers and 3D height map. |
| **Dataset export** | Export the labeled data for training. |
| **Hardware** | Is each device connected? Test the Kinect, fire the lasers (Step 1). |

> **Only see Samples and Dataset export?** You opened the read-only public view
> on port **8501**, which starts by itself every time the PC logs in (for
> Dr. Zhang, see [ONLINE.md](ONLINE.md)). Use **http://127.0.0.1:8502** from
> `launch_demo.bat` instead.

If you'd rather type it, from a Command Prompt in the project folder:

```
scanner_system\.venv\Scripts\python.exe -m streamlit run scanner_system\gui.py --server.port 8502
```

---

## Step 3: Collect data

There are two ways. Use `scan.bat` for real collection sessions.

### Way A: `scan.bat` (recommended)

Full walkthrough with every question it asks: **[HOW_TO_SCAN.md](HOW_TO_SCAN.md)**.
In short:

1. Projector on, Kinect on, laser board plugged in. **Room lights off.**
   Nothing on the white table.
2. Double-click **`scan.bat`**. It checks the hardware (Step 1d), then does
   the laser self-test (press Enter to fire).
3. It asks for an object's name, material and a few details. Put the object
   in the middle of the white table where the lasers point, step back and press
   Enter. A scan takes about a minute: the projector flashes stripes and the
   lasers blink.
4. It asks you to turn the object about a quarter turn for pose 2, then press
   Enter.
5. After each pose it runs a quality check. If something was wrong (a laser
   didn't fire, the lights were on), press Enter to redo it.
6. Next object. **To stop, leave the name empty and press Enter.**

Everything you enter goes into `collection.xlsx` and the database.

### Way B: the GUI's Capture page

Good for scanning a single object or trying something out. Open the GUI
(Step 2), go to **Capture**, choose **Create new**, type the label, material
class and subclass, keep all four laser channels ticked and click
**Run capture**. Each instrument then shows green (ok), red (failed) or white
(skipped). Note that the label boxes **don't clear** between scans, so retype
them for every new object.

---

## Step 4: Send the data to Dr. Zhang

Double-click **`push_data.bat`** at the end of every session. It copies the
pictures and spreadsheets into `dataset\` and pushes them to GitHub. `Pushed.`
means you're done. If the push fails, check the internet connection and run
it again.

---

## When something doesn't work

| What you see | What to do |
|---|---|
| `[XX] Database` or the GUI says *Cannot reach the scanner MongoDB* | Open **Docker Desktop**, wait until it says running, try again. |
| `[XX] Kinect: not detected on USB` | Power box plugged in and its light on? USB in a **blue** port? Unplug both, plug back in, wait 10 s. |
| `Kinect detected but not ready` | Same as above: usually the power box isn't plugged in. |
| `[XX] Laser board: No ESP32-like serial port found` | USB cable unplugged, or it's a charge-only cable. Try another cable. |
| `Found COM3 but no response to PING` | Something else has COM3 open (close the Arduino IDE, any other GUI or `scan.bat` window). Otherwise unplug the board's USB, wait 5 s, plug it back in. |
| Board connects but shows **no relay channels** | The board lost its settings. From the project folder run `scanner_system\.venv\Scripts\python.exe lib_3dai\provision_relays.py --commit --port COM3`, then probe it again. |
| One laser shows `NO SPOT` | That laser's wire or connector is loose. Fix it at the bench. |
| Nothing lights at all and the board still doesn't answer after a replug | Tell William. The deeper repair (erasing the board's stored settings) is in `3DAI Scanner Handoff - 2026-08-26 - Adain.md`, Part 1. |
| `[??] Projector: not detected as a display` | If the projector is on and shows an image, ignore it. Otherwise check its power and HDMI cable. |
| The stripes show up on your **monitor**, not the table | Redo [Step 1a](#1a-the-projector-fringe), steps 3 to 5 (Extend, projector to the right). |
| `scan.bat` window closes at once or says a file is locked | Close `collection.xlsx` in Excel or LibreOffice and run `scan.bat` again. |
| A scan says `partial` or `failed` | Run `scan.bat` again: it redoes only the unfinished rows. If one object fails twice, skip it with `s` and tell William. |

## Laser safety

- Goggles on whenever lasers can fire (the self-test, the Blink test, any scan).
- Never look into a beam, and keep shiny things off the table unless they're
  what you're scanning.
- CH3 is invisible. Assume it's on whenever a scan is running.
- **The big blue 455 nm laser is parked. Never use it.**
- Don't move the Kinect, the projector or the lasers. The 3D calibration and
  the laser measurements depend on them staying exactly where they are.

## Where the data goes

- `collection.xlsx`: the list of objects (tab `samples`) and the laser numbers
  (tab `laser_features`).
- `scanner_data\scans\`: every full-size picture, one folder per scan.
- `dataset\`: the previews that get pushed to GitHub, sorted by material
  ([dataset/README.md](dataset/README.md) says what each file is).
- The database itself runs in Docker as `scanner-mongo`.

## How to open a Command Prompt in the project folder

1. Open File Explorer and go to `C:\Users\Robotics_Lab\3DAI\3DAI`.
2. Click the address bar at the top, type `cmd` and press Enter.

A black window opens already in the right folder. Paste a command and press
Enter.

## Other docs

- [HOW_TO_SCAN.md](HOW_TO_SCAN.md): the full `scan.bat` walkthrough.
- [ONLINE.md](ONLINE.md): the read-only view on the internet (`go_online.bat`).
- [SIDE_CAMERA.md](SIDE_CAMERA.md): adding a second camera for laser photos.
- `3DAI Scanner Handoff - 2026-08-26 - Adain.md` and
  `3DAI Scanner Handoff - 2026-09-15.md`: what was built and fixed, and what's
  still open.

---

## For developers: the 3DAI API and `capture_tool.py` (data hub)

This part is separate from the scanning above. It runs a web API with a
Postgres database for the data hub. **`capture_tool.py` doesn't turn on the
lasers or the projector**: by default it writes placeholder images, and with
`--real` it grabs Kinect frames only. To collect scanner data, use Steps 1 to
4 above.

Requirements: Docker.

1. Clone the repository
   ```bash
   git clone https://github.com/greatroboticslab/3DAI
   cd 3DAI
   ```
2. Set up a Python virtual environment

   Windows:
   ```shell
   python -m venv venv
   venv\Scripts\activate
   ```
   Linux / macOS:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Create the environment file
   ```bash
   cp enviroment.example .env
   ```
5. Start the database and API
   ```bash
   docker compose up -d
   ```
6. Start the capture tool
   ```bash
   python3 capture_tool.py
   ```
   On the scanner PC, with the Kinect on, for real Kinect frames:
   `C:\KinectEnv\Scripts\python.exe capture_tool.py --real --steps 4`
7. Check the API is running: [http://localhost:8000/docs](http://localhost:8000/docs)
