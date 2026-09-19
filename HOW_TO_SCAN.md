# How to scan objects (read this, nothing else)

Everything happens by double-clicking two files in this folder. There are
shortcuts to both on the Desktop.

## Before you start (2 minutes)

1. Turn the projector on. Turn the room lights off.
2. Make sure the Kinect's power brick is plugged in (its light is on).
3. Nothing must be on the white table.

## Scanning: double-click `scan.bat`

A black window opens and checks the machine. If every line says `[OK]`, keep
going. If it says `STOP`, it tells you what to plug in. Fix it and
double-click `scan.bat` again.

Next it does a **laser self-test**: with the projector on and the room dark,
press Enter and it fires each laser once to confirm they all light. If one
says `NO SPOT`, a wire is loose; fix it before scanning. (Press `s` to skip.)

After every pose it runs a **quality check** and prints the result. If it
finds a problem (a laser that did not fire, the room lights on, the camera
exposure still moving, an object not on the table), it asks whether to redo
that pose. Press Enter to redo it right away, `k` to keep it anyway, or `s`
to skip. Good scans just say `QC: all checks passed` and move on.

Then it asks you questions, one object at a time:

```
Object name:            wooden block 03
Material:               wood
More detail (optional): oak
Surface:  1 matte  2 glossy  3 textured  4 mixed   [1]
Transparency:  1 opaque  2 translucent  3 transparent   [1]
Height in mm (only if you measured it with calipers, else Enter):
Notes (optional):
>>> Place 'wooden block 03' on the stage, then press Enter
```

Put the object in the middle of the white table where the lasers point,
step back, press Enter. It scans (about a minute, the lasers blink, the
projector flashes stripes). Then it says:

```
>>> Rotate 'wooden block 03' to pose 2/2, then press Enter
```

Turn the object roughly a quarter turn, press Enter. That object is done.
It asks for the next object. **When you are finished, leave the object name
empty and press Enter.**

Rules that matter for the science:

- Name objects `<what it is> <number>`: `cardboard box 01`, `steel ruler 02`.
- Material is the one word the model learns: wood, metal, plastic, cardboard,
  paper, fabric, glass, ceramic, foam, rubber, stone, leather. Pick the one
  that is true for the surface the lasers hit.
- If an object is flat on top and you measured its thickness with the
  calipers, type the height. Those objects also calibrate the 3D scanner.
- Do not move the Kinect, the projector or the lasers. Ever.
- The big blue laser is parked. Do not touch it.

## Sending data to Dr. Zhang: double-click `push_data.bat`

Do this at the end of every session. It copies the pictures and spreadsheets
into the `dataset` folder and pushes them to GitHub. If it says
`Pushed.`, you are done.

## If something goes wrong

- The window closes immediately or says a file is locked: close
  `collection.xlsx` in LibreOffice/Excel, then run `scan.bat` again.
- A scan says `partial` or `failed`: run `scan.bat` again; it re-scans only
  the rows that did not complete. If the same object fails twice, the
  message next to it says which instrument failed (kinect / projector /
  laser). Skip the object with `s` and tell William.
- You typed a wrong name or material: open `collection.xlsx`, fix the cell,
  save, close the file. The database is updated on the next push.

## Where the data lives

- `collection.xlsx` - the list of objects (tab `samples`) and the measured laser
  numbers (tab `laser_features`).
- `scanner_data\scans\` - every picture, sorted by scan.
- `dataset\` - what gets pushed to GitHub.
- The database itself runs in Docker (`scanner-mongo`); `scan.bat` starts it.
