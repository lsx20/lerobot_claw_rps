import time
from dataclasses import dataclass

import serial


HAND_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_swing"]

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


class RH56F2Hand:
    """Small RS485 driver for Inspire RH56F2 dexterous hand.

    The public position unit is the hand register angle unit from the vendor SDK.
    Positive movement opens/extends fingers on the unit we tested.
    """

    def __init__(self, config: RH56F2HandConfig):
        self.config = config
        self.ser: serial.Serial | None = None

    @property
    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def connect(self) -> None:
        self.ser = serial.Serial(self.config.port, self.config.baudrate, timeout=0.2, write_timeout=0.2)
        self.configure()
        time.sleep(self.config.startup_settle_s)

    def configure(self) -> None:
        self.write_positions("mode", {name: 0 for name in HAND_NAMES})
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
        return self.ser.read_all()

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

    def write_positions(self, key: str, values_by_name: dict[str, float]) -> None:
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

    def set_angles(self, values_by_name: dict[str, float]) -> None:
        self.write_positions("angleSet", values_by_name)
