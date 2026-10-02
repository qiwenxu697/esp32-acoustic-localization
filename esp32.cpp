import serial
import numpy as np

PORT = "/dev/cu.usbmodem5C941308011"
BAUD = 921600

FRAME_COUNT = 256

HEADER = b"\xAA\x55\xAA\x55"

# 256 frames
# × 2 microphones
# × 2 bytes
PAYLOAD_SIZE = FRAME_COUNT * 2 * 2


ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

ser.reset_input_buffer()

print("Connected")


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

        b = ser.read(1)

        if not b:
            continue

        if b[0] == HEADER[state]:

            state += 1

            if state == len(HEADER):
                return

        else:

            if b[0] == HEADER[0]:
                state = 1
            else:
                state = 0


count = 0

try:

    while True:

        find_header()

        payload = read_exactly(
            PAYLOAD_SIZE
        )

        if payload is None:
            continue


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


        # Remove DC offset
        left -= np.mean(left)
        right -= np.mean(right)


        rms_left = np.sqrt(
            np.mean(left ** 2)
        )

        rms_right = np.sqrt(
            np.mean(right ** 2)
        )


        count += 1


        # IMPORTANT:
        # still read every packet,
        # but only print occasionally

        if count % 20 == 0:

            print(
                f"L RMS={rms_left:7.1f}   "
                f"R RMS={rms_right:7.1f}"
            )


except KeyboardInterrupt:

    print("\nStopping")


finally:

    ser.close()