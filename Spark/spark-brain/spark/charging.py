"""Read Doly's INA219 without altering the SDK's configuration/calibration.

Hardware map and current polarity: SDK/docs/AI_Integration_Developer_FAQ.md.
INA219 shunt register 0x01 is signed, big endian, 10 microvolts per count.
Positive averaged shunt voltage means current flowing into Doly's battery.
"""
from collections import deque
import threading
import time


class ChargingMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._samples = deque(maxlen=8)
        self._last_read = 0.0
        self._bus = None
        self._pending = None
        self.charging = None
        self.average = None
        self.voltage = None
        self.error = None

    def _read(self):
        from smbus2 import SMBus, i2c_msg
        if self._bus is None:
            self._bus = SMBus(3)

        def register(reg):
            pointer = i2c_msg.write(0x41, [reg])
            result = i2c_msg.read(0x41, 2)
            # Combined transaction: no other SDK reader can change the
            # register pointer between selecting it and reading its value.
            self._bus.i2c_rdwr(pointer, result)
            return int.from_bytes(bytes(result), "big")

        config = register(0)
        shunt = register(1)
        voltage = (register(2) >> 3) * 0.004
        if config & 7 not in (5, 7) or not 2.0 < voltage < 5.0:
            raise ValueError("INA219 has no valid continuous shunt reading")
        return shunt if shunt < 32768 else shunt - 65536, voltage

    def _read_bounded(self):
        # At most one I2C transaction worker. A stalled adapter must not
        # block the voice loop, hold checks, shutdown, or spawn more jobs.
        if self._pending is None:
            result = []

            def read():
                try:
                    result.append((self._read(), None))
                except Exception as exc:
                    result.append((None, exc))

            worker = threading.Thread(target=read, daemon=True)
            self._pending = worker, result, time.monotonic()
            worker.start()
        worker, result, started = self._pending
        worker.join(timeout=0.1)
        if worker.is_alive():
            raise TimeoutError("INA219 read did not finish within 100ms")
        self._pending = None
        if time.monotonic() - started > 2:
            raise TimeoutError("INA219 returned a stale reading")
        value, error = result[0]
        if error is not None:
            raise error
        return value

    def sample(self):
        with self._lock:
            now = time.monotonic()
            if now - self._last_read < 0.25:
                return self.charging
            if now - self._last_read > 2.0:
                self._samples.clear()  # old readings cannot authorize motion
                self.charging = None
            self._last_read = now
            try:
                shunt, self.voltage = self._read_bounded()
                self._samples.append(shunt)
                self.average = sum(self._samples) / len(self._samples)
                self.error = None
                if len(self._samples) >= 5:
                    if shunt > 0:
                        # Hold on the first possible contact; old discharge
                        # samples cannot authorize motion during charge onset.
                        self.charging = True if self.average > 5 else None
                    elif self.average < -5 and all(x < -5 for x in list(self._samples)[-5:]):
                        self.charging = False
                    else:
                        self.charging = None  # mixed/tapering evidence holds motion
                return self.charging
            except Exception as exc:
                self.error = str(exc)
                self.charging = self.average = self.voltage = None
                self._samples.clear()
                if self._bus is not None and self._pending is None:
                    self._bus.close()
                    self._bus = None
                return None

    def close(self):
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._pending is not None and self._pending[0].is_alive():
                return False
            if self._bus is not None:
                self._bus.close()
                self._bus = None
            return True
        finally:
            self._lock.release()

    def healthy(self):
        """Fresh electrical readings, even during charge/discharge transition.

        This is NOT proof of charging or permission for ordinary movement.
        A bounded departure can cross zero current without treating it as
        a failed sensor; stale readings and I2C faults still stop it.
        """
        with self._lock:
            return (self.error is None and self.voltage is not None
                    and 2.0 < self.voltage < 5.0 and len(self._samples) >= 5
                    and time.monotonic() - self._last_read <= .6)
