import serial
import numpy as np
from collections import deque


# CONFIGURATION
PORT = "/dev/cu.usbmodem5C941308011"
BAUD = 921600

SAMPLE_RATE = 16000
FRAME_COUNT = 256

MIC_DISTANCE = 0.50       # meters
SPEED_OF_SOUND = 343.0    # m/s

HEADER = b"\xAA\x55\xAA\x55"

# 256 frames
# × 2 channels
# × 2 bytes
PAYLOAD_SIZE = FRAME_COUNT * 2 * 2

# Start with 500 for a clap / sharp loud sound.
RMS_THRESHOLD = 700.0
WINDOW_SAMPLES = 2048
PRE_TRIGGER_SAMPLES = 512


# Maximum physically possible sample delay
MAX_LAG = int(
    np.ceil(
        MIC_DISTANCE
        / SPEED_OF_SOUND
        * SAMPLE_RATE
    )
)

print(
    f"Maximum physical lag: "
    f"{MAX_LAG} samples"
)

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

ser.reset_input_buffer()

print("Connected to ESP32")
print("Listening for loud sound...")
print(f"Trigger threshold: RMS > {RMS_THRESHOLD}")

def read_exactly(size):

    data = bytearray()

    while len(data) < size:

        chunk = ser.read(
            size - len(data)
        )

        if not chunk:
            return None

        data.extend(chunk)

    return bytes(data)


def find_header():

    state = 0

    while True:

        byte = ser.read(1)

        if not byte:
            continue

        value = byte[0]

        if value == HEADER[state]:

            state += 1

            if state == len(HEADER):
                return

        else:

            if value == HEADER[0]:
                state = 1

            else:
                state = 0


def read_packet():

    find_header()

    payload = read_exactly(
        PAYLOAD_SIZE
    )

    if payload is None:
        return None

    samples = np.frombuffer(
        payload,
        dtype="<i2"
    )

    # L0 R0 L1 R1 ...
    left = samples[0::2].astype(
        np.float64
    )

    right = samples[1::2].astype(
        np.float64
    )

    return left, right


# ============================================================
# RMS
# ============================================================

def calculate_rms(samples):

    centered = (
        samples
        - np.mean(samples)
    )

    return np.sqrt(
        np.mean(centered ** 2)
    )


# ============================================================
# TDOA
# ============================================================

def estimate_delay(left, right):

    # Remove DC offset
    left = (
        left
        - np.mean(left)
    )

    right = (
        right
        - np.mean(right)
    )

    # Cross-correlation
    correlation = np.correlate(
        left,
        right,
        mode="full"
    )

    # Corresponding lag values
    lags = np.arange(
        -len(right) + 1,
        len(left)
    )

    # Only search physically possible range
    valid = (
        (lags >= -MAX_LAG)
        &
        (lags <= MAX_LAG)
    )

    correlation_valid = (
        correlation[valid]
    )

    lags_valid = (
        lags[valid]
    )

    # Strongest match
    best_index = np.argmax(
        correlation_valid
    )

    best_lag = int(
        lags_valid[best_index]
    )

    delay_seconds = (
        best_lag
        / SAMPLE_RATE
    )

    delay_us = (
        delay_seconds
        * 1_000_000
    )

    return best_lag, delay_us


# Keeps the most recent samples before the loud sound happens.

pre_left = deque(
    maxlen=PRE_TRIGGER_SAMPLES
)

pre_right = deque(
    maxlen=PRE_TRIGGER_SAMPLES
)

try:

    while True:

        result = read_packet()

        if result is None:
            continue

        left, right = result

        rms_left = calculate_rms(left)
        rms_right = calculate_rms(right)

        print(
            f"\r"
            f"L RMS={rms_left:7.1f}   "
            f"R RMS={rms_right:7.1f}",
            end="",
            flush=True
        )


        # ------------------------------------
        # Wait until pre-trigger buffer
        # has enough history
        # ------------------------------------

        buffer_ready = (
            len(pre_left)
            >= PRE_TRIGGER_SAMPLES
        )


        # ------------------------------------
        # Detect loud sound
        # ------------------------------------

        triggered = (
            buffer_ready
            and
            max(
                rms_left,
                rms_right
            )
            >= RMS_THRESHOLD
        )


        if triggered:

            print("\n\nLOUD SOUND DETECTED!")

            print(
                f"L RMS = {rms_left:.1f}"
            )

            print(
                f"R RMS = {rms_right:.1f}"
            )


            # --------------------------------
            # Start event with audio from
            # BEFORE the trigger
            # --------------------------------

            event_left = list(
                pre_left
            )

            event_right = list(
                pre_right
            )


            # Add the packet that caused
            # the trigger
            event_left.extend(left)

            event_right.extend(right)


            # --------------------------------
            # Continue collecting audio
            # AFTER the trigger
            # --------------------------------

            while (
                len(event_left)
                < WINDOW_SAMPLES
            ):

                result = read_packet()

                if result is None:
                    continue

                new_left, new_right = result

                event_left.extend(
                    new_left
                )

                event_right.extend(
                    new_right
                )


            # --------------------------------
            # Use exactly 2048 samples
            # --------------------------------

            event_left = np.array(
                event_left[
                    :WINDOW_SAMPLES
                ],
                dtype=np.float64
            )

            event_right = np.array(
                event_right[
                    :WINDOW_SAMPLES
                ],
                dtype=np.float64
            )


            # --------------------------------
            # Calculate TDOA
            # --------------------------------

            lag, delay_us = (
                estimate_delay(
                    event_left,
                    event_right
                )
            )


            print("\nTDOA RESULT")
            print("-------------------")

            print(
                f"Lag: {lag:+d} samples"
            )

            print(
                f"TDOA: {delay_us:+.1f} us"
            )


            # Optional:
            # convert delay to path-length
            # difference

            distance_difference = (
                SPEED_OF_SOUND
                * delay_us
                / 1_000_000
            )

            print(
                "Path difference: "
                f"{distance_difference:+.3f} m"
            )


            print("\nFinished.")

            # Exit main loop
            break


        # ------------------------------------
        # No trigger:
        # save current samples as history
        # ------------------------------------

        pre_left.extend(left)
        pre_right.extend(right)


except KeyboardInterrupt:

    print("\nStopped by user")


finally:

    ser.close()

    print("Serial port closed")