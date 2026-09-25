"""Govee smart-light control — local LAN first, cloud fallback.

LAN (no key, sub-second): UDP JSON to the light's :4003, replies to our
:4001. Requires "LAN Control" enabled per device in the Govee Home app
and the device on the same (non-isolated) WiFi.

Cloud (api.govee.com, needs `govee.api_key` from the Govee Home app:
Me -> Apply for API Key) covers WiFi devices regardless of the LAN
toggle, at ~1s latency, and is used automatically when LAN discovery
finds nothing or a LAN command goes unanswered.

Never raises; every call returns a human-speakable result string.
"""
import json
import socket
import sys
import threading
import time
import urllib.request

_LISTEN_PORT = 4001   # devices reply here
_DEV_PORT = 4003      # devices listen here
_UDP_TIMEOUT = 1.2
_CLOUD = "https://api.govee.com/v1"

# cmds.COLORS names -> RGB (Govee uses 0-255)
_RGB = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "white": (255, 255, 255), "yellow": (255, 220, 60), "orange": (255, 130, 0),
    "purple": (160, 32, 240), "pink": (255, 80, 180), "cyan": (0, 220, 220),
    "magenta": (255, 0, 255), "darkgreen": (0, 128, 0),
    "lightblue": (120, 180, 255), "black": (0, 0, 0),
    "warmwhite": (255, 180, 100), "gold": (255, 200, 40),
    "teal": (0, 160, 160), "lime": (160, 255, 60),
}


def _log(msg):
    print(f"[govee] {msg}", file=sys.stderr)


def color_rgb(name):
    """Named color -> (r, g, b) or None."""
    return _RGB.get((name or "").lower().replace(" ", ""))


class GoveeLights:
    """Thread-safe facade over the LAN and cloud clients."""

    def __init__(self, cfg):
        g = cfg.get("govee", {}) or {}
        self.enabled = g.get("enabled", True)
        self.api_key = g.get("api_key") or None
        self.state_path = None
        sd = cfg.get("state_dir")
        if sd:
            import pathlib
            self.state_path = pathlib.Path(sd) / "govee.json"
        self._lock = threading.Lock()
        self._sock = None
        self._devices = []   # [{device, model, ip, name, supportCmds}]
        self._last_scan = 0.0
        self._load_cache()

    # ------------------------------------------------------------- cache
    def _load_cache(self):
        if not (self.state_path and self.state_path.exists()):
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._devices = data.get("devices", [])
        except Exception as e:
            _log(f"cache load failed: {e}")

    def _save_cache(self):
        if not self.state_path:
            return
        try:
            self.state_path.write_text(
                json.dumps({"devices": self._devices}, indent=1), encoding="utf-8")
        except Exception as e:
            _log(f"cache save failed: {e}")

    # --------------------------------------------------------------- lan
    def _udp(self):
        if self._sock is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.bind(("0.0.0.0", _LISTEN_PORT))
            self._sock = s
        return self._sock

    def _send_recv(self, payload, ip, wait_s=_UDP_TIMEOUT):
        """One UDP round trip; returns the parsed reply dict or None."""
        s = self._udp()
        s.settimeout(wait_s)
        try:
            s.sendto(json.dumps(payload).encode(), (ip, _DEV_PORT))
        except OSError as e:
            _log(f"udp send to {ip} failed: {e}")
            return None
        end = time.time() + wait_s
        while time.time() < end:
            try:
                data, addr = s.recvfrom(4096)
            except socket.timeout:
                return None
            except OSError:
                return None
            try:
                reply = json.loads(data)
            except ValueError:
                continue
            if addr[0] == ip or ip.endswith("255"):
                return reply
        return None

    def scan(self, force=False):
        """Discover LAN devices (broadcast). Cached for 5 minutes."""
        with self._lock:
            if not force and self._devices and time.time() - self._last_scan < 300:
                return self._devices
            s = self._udp()
            s.settimeout(1.0)
            msg = json.dumps({"msg": {"cmd": "scan",
                                      "data": {"account_msg": "spark"}}}).encode()
            for dst in ("255.255.255.255", "192.168.50.255"):
                try:
                    s.sendto(msg, (dst, _DEV_PORT))
                except OSError:
                    pass
            end = time.time() + 3
            found = {}
            while time.time() < end:
                try:
                    data, addr = s.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    d = json.loads(data).get("msg", {}).get("data", {})
                except ValueError:
                    continue
                dev = d.get("device")
                if dev:
                    found[dev] = {
                        "device": dev,
                        "model": d.get("model", ""),
                        "ip": addr[0],
                        "name": f"{d.get('model', 'light')} {dev[-4:]}",
                        "supportCmds": sorted(d.get("supportCmds", [])
                                              or ["turn", "brightness", "colorwc"]),
                    }
            if found:
                self._devices = list(found.values())
                self._last_scan = time.time()
                self._save_cache()
                _log(f"scan: {len(self._devices)} device(s) on LAN")
            return self._devices

    def _lan_cmd(self, dev, command, data):
        payload = {"msg": {"cmd": "dev", "device": dev["device"],
                           "cmd": {"command": command, "data": data}}}
        reply = self._send_recv(payload, dev.get("ip", ""))
        if reply and reply.get("msg", {}).get("code") == 300:
            return "ok"
        return None

    # ------------------------------------------------------------- cloud
    def _cloud(self, method, path, body=None):
        req = urllib.request.Request(
            _CLOUD + path, data=json.dumps(body).encode() if body else None,
            headers={"Govee-API-Key": self.api_key,
                     "Content-Type": "application/json"},
            method=method)
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read())

    def _cloud_devices(self):
        try:
            data = self._cloud("GET", "/devices").get("data", [])
        except Exception as e:
            _log(f"cloud devices failed: {e}")
            return []
        out = []
        for d in data.get("devices", []) if isinstance(data, dict) else data:
            out.append({
                "device": d.get("device", ""),
                "model": d.get("model", ""),
                "ip": None,
                "name": d.get("deviceName") or f"{d.get('model', 'light')}",
                "supportCmds": sorted(d.get("supportCmds", [])
                                      or ["turn", "brightness", "colorwc"]),
                "cloud": True,
            })
        return out

    # ---------------------------------------------------- command translation
    # Canonical ops ("turn"/"brightness"/"color"/"temp"), two dialects: the
    # LAN protocol packs color+temperature into one colorwc object; the
    # CLOUD API uses separate color / colorTem commands with flat values.
    @staticmethod
    def _lan_payload(op, value):
        if op == "turn":
            return "turn", ("on" if value else "off")
        if op == "brightness":
            return "brightness", max(1, min(100, int(value)))
        if op == "color":
            return "colorwc", {"color": {"r": value[0], "g": value[1], "b": value[2]},
                               "colorTemInKelvin": 0}
        return "colorwc", {"color": {"r": 0, "g": 0, "b": 0},
                           "colorTemInKelvin": max(2000, min(9000, int(value)))}

    @staticmethod
    def _cloud_payload(op, value):
        if op == "turn":
            return "turn", ("on" if value else "off")
        if op == "brightness":
            return "brightness", max(1, min(100, int(value)))
        if op == "color":
            return "color", {"r": value[0], "g": value[1], "b": value[2]}
        return "colorTem", max(2000, min(9000, int(value)))

    def _cmd(self, dev, op, value):
        """Run one canonical op on one device over its transport."""
        if dev.get("cloud") or not dev.get("ip"):
            if not self.api_key:
                return None
            name, val = self._cloud_payload(op, value)
            try:
                self._cloud("PUT", "/devices/control",
                            {"device": dev["device"], "model": dev["model"],
                             "cmd": {"name": name, "value": val}})
                return "ok"
            except Exception as e:
                _log(f"cloud control failed: {e}")
                return None
        name, val = self._lan_payload(op, value)
        payload = {"msg": {"cmd": "dev", "device": dev["device"],
                           "cmd": {"command": name, "data": val}}}
        reply = self._send_recv(payload, dev.get("ip", ""))
        if reply and reply.get("msg", {}).get("code") == 300:
            return "ok"
        return None

    # -------------------------------------------------------------- api
    def devices(self):
        """Usable device list: LAN cache, else cloud, else empty."""
        if not self.enabled:
            return []
        devs = self.scan()
        if devs:
            return devs
        if self.api_key:
            if not self._devices or time.time() - self._last_scan >= 300:
                cloud = self._cloud_devices()
                if cloud:
                    self._devices = cloud
                    self._last_scan = time.time()
                    self._save_cache()
                    _log(f"cloud: {len(cloud)} device(s)")
            return self._devices
        return []

    def _targets(self, label):
        devs = self.devices()
        if not devs:
            return []
        label = (label or "").lower()
        if label in ("", "all", "my", "room", "the"):
            return devs
        return [d for d in devs if label in d["name"].lower()]

    def _apply(self, command, data, label="all"):
        with self._lock:
            devs = self._targets(label)
            if not devs:
                return ("none", "I can't find your Govee lights right now.")
            for dev in devs:
                if self._cmd(dev, command, data) != "ok":
                    return ("error", "I couldn't reach your lights just then.")
            return ("ok", "")

    # speech-facing -------------------------------------------------------
    def turn(self, on, label="all"):
        state, why = self._apply("turn", bool(on), label)
        if state == "none" or state == "error":
            return why
        return "Lights on!" if on else "Lights off."

    def brightness(self, pct, label="all"):
        pct = max(1, min(100, int(pct)))
        state, why = self._apply("brightness", pct, label)
        if state == "none" or state == "error":
            return why
        return f"Lights at {pct} percent."

    def color(self, name, label="all"):
        rgb = color_rgb(name) if isinstance(name, str) else name
        if not rgb:
            return "Hmm, I don't know that color."
        state, why = self._apply("color", rgb, label)
        if state == "none" or state == "error":
            return why
        n = (name if isinstance(name, str) else "that color").replace("_", " ").lower()
        return f"Lights {n}."

    def color_temp(self, kelvin, label="all"):
        kelvin = max(2000, min(9000, int(kelvin)))
        state, why = self._apply("temp", kelvin, label)
        if state == "none" or state == "error":
            return why
        return "Warmer light." if kelvin < 4000 else "Cooler light."

    def status(self, label="all"):
        devs = self.devices()
        if not devs:
            return "I can't find your Govee lights right now."
        n = len(devs)
        return f"I see {n} Govee light{'s' if n != 1 else ''} ready to go."
