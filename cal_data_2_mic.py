import serial
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

PORT = "/dev/cu.usbmodem5C941308011"
BAUD = 921600

SAMPLE_RATE = 16000

# ESP32 sends 256 stereo frames per packet
FRAME_COUNT = 256

# We combine multiple packets before estimating TDOA
WINDOW_SAMPLES = 2048

# Approximate microphone spacing
MIC_DISTANCE = 0.50  # meters

SPEED_OF_SOUND = 343.0  # m/s

HEADER = b"\xAA\x55\xAA\x55"

# 256 frames
# × 2 channels
# × 2 bytes per int16 sample
PAYLOAD_SIZE = FRAME_COUNT * 2 * 2


# Maximum physically possible lag
MAX_LAG = int(
    np.ceil(
        MIC_DISTANCE
        / SPEED_OF_SOUND
        * SAMPLE_RATE
    )
)

print("Maximum lag:", MAX_LAG, "samples")


# ============================================================
# SERIAL SETUP
# ============================================================

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

ser.reset_input_buffer()

print("Connected to ESP32")
print("Press Ctrl+C to stop")


# ============================================================
# SERIAL HELPERS
# ============================================================

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

    left = samples[0::2].astype(
        np.float64
    )

    right = samples[1::2].astype(
        np.float64
    )

    return left, right


# ============================================================
# TDOA ESTIMATION
# ============================================================

def estimate_delay(left, right):

    # Remove DC offset
    left = left - np.mean(left)
    right = right - np.mean(right)

    # Cross correlation
    correlation = np.correlate(
        left,
        right,
        mode="full"
    )

    # Lag values corresponding to correlation
    lags = np.arange(
        -len(right) + 1,
        len(left)
    )

    # Restrict search to physically possible lag
    valid = (
        (lags >= -MAX_LAG)
        &
        (lags <= MAX_LAG)
    )

    correlation_valid = correlation[valid]
    lags_valid = lags[valid]

    # Find strongest correlation
    best_index = np.argmax(
        correlation_valid
    )

    best_lag = int(
        lags_valid[best_index]
    )

    delay_seconds = (
        best_lag / SAMPLE_RATE
    )

    delay_us = (
        delay_seconds * 1_000_000
    )

    return (
        best_lag,
        delay_us
    )


# ============================================================
# MAIN LOOP
# ============================================================

left_buffer = []
right_buffer = []

try:

    while True:

        result = read_packet()

        if result is None:
            continue

        left, right = result

        left_buffer.extend(left)
        right_buffer.extend(right)

        # Wait until we have enough samples
        if len(left_buffer) < WINDOW_SAMPLES:
            continue


        # Take exactly WINDOW_SAMPLES
        left_window = np.array(
            left_buffer[:WINDOW_SAMPLES],
            dtype=np.float64
        )

        right_window = np.array(
            right_buffer[:WINDOW_SAMPLES],
            dtype=np.float64
        )


        # Remove samples we just used
        left_buffer = left_buffer[
            WINDOW_SAMPLES:
        ]

        right_buffer = right_buffer[
            WINDOW_SAMPLES:
        ]


        # Calculate RMS so we know whether
        # there is meaningful sound
        left_centered = (
            left_window
            - np.mean(left_window)
        )

        right_centered = (
            right_window
            - np.mean(right_window)
        )

        rms_left = np.sqrt(
            np.mean(
                left_centered ** 2
            )
        )

        rms_right = np.sqrt(
            np.mean(
                right_centered ** 2
            )
        )


        # Estimate TDOA
        lag, delay_us = estimate_delay(
            left_window,
            right_window
        )


        print(
            f"L RMS={rms_left:7.1f}  "
            f"R RMS={rms_right:7.1f}  "
            f"Lag={lag:+3d} samples  "
            f"TDOA={delay_us:+8.1f} us"
        )


except KeyboardInterrupt:

    print("\nStopping...")


finally:

    ser.close()

    print("Serial port closed")