import time
from dataclasses import dataclass

import serial


HAND_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_swing"]
TOUCH_FINGER_NAMES = ["little", "ring", "middle", "index", "thumb"]

REG = {
    "mode": 1100,
    "angleSet": 1040,
    "forceSet": 1046,
    "speedSet": 1052,
    "angleAct": 1064,
    "forceAct": 1070,
    "errCode": 1082,
    "statusCode": 1088,
    "temp": 1094,
    "touchData": 3000,
}

DEFAULT_OPEN = {
    "little": 1720,
    "ring": 1720,
    "middle": 1720,
    "index": 1720,
    "thumb_bend": 1450,
    "thumb_swing": 1700,
}

DEFAULT_CLOSED = {
    "little": 900,
    "ring": 900,
    "middle": 900,
    "index": 900,
    "thumb_bend": 1130,
    "thumb_swing": 1700,
}

HAND_LIMITS = {
    "little": (850, 1800),
    "ring": (850, 1800),
    "middle": (850, 1800),
    "index": (850, 1800),
    "thumb_bend": (1050, 1500),
    "thumb_swing": (450, 1800),
}


@dataclass
class RH56F2HandConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    hand_id: int = 1
    speed: int = 800
    force: int = 1500
    mode: int = 0
    startup_settle_s: float = 0.5
    read_retries: int = 3
    read_retry_delay_s: float = 0.05


def _checksum(frame: list[int]) -> int:
    return sum(frame[2:]) & 0xFF


def _pack_six(values: list[int]) -> list[int]:
    packed: list[int] = []
    for value in values:
        value = int(value)
        packed.append(value & 0xFF)
        packed.append((value >> 8) & 0xFF)
    return packed


def _unpack_six(raw: list[int]) -> list[int]:
    if len(raw) < 12:
        return []
    values: list[int] = []
    for i in range(6):
        value = raw[2 * i] | (raw[2 * i + 1] << 8)
        if value > 32767:
            value -= 65536
        values.append(value)
    return values


def _u16_le(raw: list[int], idx: int) -> int:
    return int(raw[idx]) | (int(raw[idx + 1]) << 8)


class RH56F2Hand:
    """Small RS485 driver for Inspire RH56F2 dexterous hand.

    The public position unit is the hand register angle unit from the vendor SDK.
    Positive movement opens/extends fingers on the unit we tested.
    """

    def __init__(self, config: RH56F2HandConfig):
        self.config = config
        self.ser: serial.Serial | None = None
        self.last_write_ack: int | None = None

    @property
    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def connect(self) -> None:
        self.ser = serial.Serial(self.config.port, self.config.baudrate, timeout=0.2, write_timeout=0.2)
        self.configure()
        time.sleep(self.config.startup_settle_s)

    def configure(self) -> None:
        self.write_positions("mode", {name: self.config.mode for name in HAND_NAMES})
        self.write_positions("speedSet", {name: self.config.speed for name in HAND_NAMES})
        self.write_positions("forceSet", {name: self.config.force for name in HAND_NAMES})

    def disconnect(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def _write_register(self, address: int, values: list[int]) -> bytes:
        if self.ser is None:
            raise RuntimeError("RH56F2 hand is not connected")
        frame = [0xEB, 0x90, self.config.hand_id, len(values) + 3, 0x12]
        frame.extend([address & 0xFF, (address >> 8) & 0xFF])
        frame.extend(values)
        frame.append(_checksum(frame))
        self.ser.write(bytes(frame))
        time.sleep(0.02)
        response = self.ser.read_all()
        self.last_write_ack = response[7] if len(response) > 7 else None
        return response

    def _read_register(self, address: int, length: int) -> list[int]:
        if self.ser is None:
            raise RuntimeError("RH56F2 hand is not connected")
        self.ser.read_all()
        frame = [0xEB, 0x90, self.config.hand_id, 0x04, 0x11]
        frame.extend([address & 0xFF, (address >> 8) & 0xFF, length])
        frame.append(_checksum(frame))
        self.ser.write(bytes(frame))

        deadline = time.time() + 1.0
        recv = bytearray()
        while time.time() < deadline:
            chunk = self.ser.read_all()
            if chunk:
                recv.extend(chunk)
                if len(recv) >= length + 8:
                    break
            time.sleep(0.02)
        if len(recv) < length + 8:
            return []
        return list(recv[7 : 7 + length])

    def read_positions(self, key: str = "angleAct") -> dict[str, float]:
        last_values: list[int] = []
        last_error: str | None = None
        for _ in range(max(1, int(self.config.read_retries))):
            try:
                values = _unpack_six(self._read_register(REG[key], 12))
                if values:
                    return {name: float(value) for name, value in zip(HAND_NAMES, values, strict=True)}
                last_error = f"No RH56F2 response while reading {key}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(self.config.read_retry_delay_s)
        raise RuntimeError(last_error or f"No RH56F2 response while reading {key}")

    def read_touch_data(self) -> dict[str, dict[str, dict[str, float]] | dict[str, float]]:
        """Read RH56F2 fingertip tactile data from the vendor touch register.

        The vendor SDK exposes one tactile tuple per finger, not a fingertip
        taxel matrix: normal force, tangential force, tangential direction, and
        proximity. It also exposes nine palm values.
        """
        raw = self._read_register(REG["touchData"], 0x44)
        if len(raw) < 68:
            raise RuntimeError("No RH56F2 response while reading touchData")

        fingers: dict[str, dict[str, float]] = {}
        for idx, name in enumerate(TOUCH_FINGER_NAMES):
            base = idx * 10
            proximity = raw[base + 6] | (raw[base + 7] << 8) | (raw[base + 8] << 16)
            fingers[name] = {
                "normal": float(_u16_le(raw, base)),
                "tangential": float(_u16_le(raw, base + 2)),
                "angle": float(_u16_le(raw, base + 4)),
                "proximity": float(proximity),
            }

        palm_start = len(TOUCH_FINGER_NAMES) * 10
        palm = {
            f"palm_{idx + 1}": float(_u16_le(raw, palm_start + idx * 2))
            for idx in range(9)
        }
        return {"fingers": fingers, "palm": palm}

    def write_positions(self, key: str, values_by_name: dict[str, float]) -> bool:
        current = {name: 0.0 for name in HAND_NAMES}
        if key == "angleSet":
            try:
                current = self.read_positions("angleAct")
            except RuntimeError:
                pass
        values: list[int] = []
        for name in HAND_NAMES:
            value = values_by_name.get(name, current[name])
            if key == "angleSet":
                lo, hi = HAND_LIMITS[name]
                value = min(max(float(value), lo), hi)
            values.append(int(round(value)))
        self._write_register(REG[key], _pack_six(values))
        return self.last_write_ack == 0x01

    def set_angles(self, values_by_name: dict[str, float]) -> bool:
        return self.write_positions("angleSet", values_by_name)
