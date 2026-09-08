# Blender Live Link

RTXPT can optionally run a small local TCP server that lets the **RTXPT Live Link**
Blender add-on (see the companion [`rtxpt-exporter`](https://github.com/NVIDIA-RTX/rtxpt-exporter)
repository, or wherever your fork of it lives) drive the free-fly camera live while you
navigate the 3D viewport in Blender, and trigger a full scene reload after you re-export
geometry - without having to close and relaunch `Rtxpt.exe` every time.

This is intentionally a small, dependency-free feature: no new third-party library was
added to either the Blender add-on or the RTXPT executable to support it.

## Enabling it

Live Link is off by default. Start RTXPT with:

```
Rtxpt.exe --scene YourProject.scene.json --liveLink
```

Optionally choose a different port (default `42042`):

```
Rtxpt.exe --scene YourProject.scene.json --liveLink --liveLinkPort 42999
```

When enabled, RTXPT opens a listening socket on `127.0.0.1:<port>` (loopback only, by
design - Blender and RTXPT are expected to run on the same machine) and logs:

```
LiveLink: listening on 127.0.0.1:42042 for the Blender Live Link add-on
```

From Blender, install and enable the **RTXPT Live Link** add-on, set the host/port to
match (defaults are `127.0.0.1` / `42042`), and click **Connect**. Once connected,
moving the camera in the Blender viewport (with Live Link's viewport-camera sync
enabled) updates RTXPT's camera in real time, and the **Sync Full Scene** button
re-exports the scene via the RTXPT Exporter add-on and asks RTXPT to hot-reload it.

## Protocol

The protocol is plain ASCII text, one command per line, terminated with `\n`. Fields
within a line are separated by single spaces. This was chosen over JSON so that neither
side needs a JSON library - the Blender side already ships with Python's standard
library, and the RTXPT side ships with nothing beyond the C++ standard library and
Winsock2.

Coordinates are always in RTXPT/glTF space: right-handed, **Y-up**. Blender is Z-up, so
the add-on is responsible for converting `(x, y, z) -> (x, z, -y)` before sending
anything - RTXPT does not perform any axis conversion on received commands.

### Client (Blender) → Server (RTXPT)

| Command | Format | Description |
|---|---|---|
| `HELLO` | `HELLO <info...>` | Handshake. `info` is a free-form, human-readable string (client name/version) logged by RTXPT for diagnostics. |
| `CAM` | `CAM px py pz dx dy dz ux uy uz fovY zNear` | Sets the free-fly camera position `(px,py,pz)`, view direction `(dx,dy,dz)` (need not be normalized), up vector `(ux,uy,uz)`, vertical FOV in **radians**, and near clip plane. Send `fovY <= 0` or `zNear <= 0` to leave that particular value unchanged. Meant to be sent continuously while the Blender viewport camera moves (20-60 Hz is reasonable). |
| `RELOAD` | `RELOAD <path>` | Asks RTXPT to (re)load `<path>`, a `.scene.json` file relative to the `Assets` folder - exactly the same string you'd pass to `--scene`. Triggers a full scene reload (same code path as switching scenes from the UI), so expect a brief hitch. |
| `PING` | `PING` | Keepalive; server replies `PONG`. |

### Server (RTXPT) → Client (Blender)

| Reply | Meaning |
|---|---|
| `HELLO_OK 1 RTXPT` | Handshake acknowledged; `1` is the protocol version implemented by this build. |
| `OK` | The last command (currently only `RELOAD`) was accepted and queued. |
| `ERR <message>` | The last line could not be parsed or was rejected. |
| `PONG` | Reply to `PING`. |

`CAM` commands are not acknowledged individually - at typical viewport update rates an
ack round-trip per message isn't useful, and the client already knows whether the TCP
connection is alive.

### Example session

```
> HELLO Blender 4.2 / RTXPT Live Link 1.0.0
< HELLO_OK 1 RTXPT
> CAM -20.0 1.8 12.0 0.94 -0.10 -0.30 0.0 1.0 0.0 1.0472 0.001
> CAM -19.8 1.8 12.1 0.95 -0.09 -0.29 0.0 1.0 0.0 1.0472 0.001
> RELOAD TestProject.scene.json
< OK
```

## What Live Link does *not* do (yet)

- **Per-object live transform sync.** Moving individual objects in Blender is not
  streamed live; use **Sync Full Scene** (a glTF + `.scene.json` re-export followed by
  a `RELOAD`) to push geometry, material or transform edits into a running RTXPT.
- **Multi-client / remote use.** The server accepts one client at a time and only binds
  to `127.0.0.1`. Driving RTXPT from a different machine is not supported out of the
  box (you could tunnel the TCP port over SSH, or edit `LiveLinkServer.cpp` to bind to
  `INADDR_ANY` if you understand the security implications of doing so).
- **Non-Windows platforms.** `--liveLink` currently logs a warning and does nothing
  outside of Windows, matching the rest of RTXPT's current platform support.

## Implementation

- `Rtxpt/LiveLink/LiveLinkServer.h` / `.cpp` - the TCP server and protocol parser. Runs
  entirely on its own background thread, independent of the render loop.
- `Rtxpt/AdvancedSample.cpp` - owns the `LiveLinkServer` instance (started in
  `AdvancedPathTracer`'s constructor when `--liveLink` is passed) and drains queued
  commands once per frame from its `Animate()` override (`LiveLinkServer::PopCommands()`),
  applying them via `Sample::LiveLinkApplyCamera()` (which wraps
  `FirstPersonCamera::LookTo`) or `Sample::SetCurrentScene(path, true)`.
- `Rtxpt/Sample.h` - adds the small protected `LiveLinkApplyCamera()` helper used above;
  no changes to `Sample.cpp` were needed.
- `Rtxpt/SampleCommon/CommandLine.h` / `.cpp` - adds the `--liveLink` / `--liveLinkPort`
  command line options.
