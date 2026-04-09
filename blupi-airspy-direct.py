
import ctypes
import os
import sys
import time
import numpy as np # Enige Dep!
from collections import deque
from math import sqrt


freqmin     = 390_000_000
freqmax     = 395_000_000
sensitivity = 50
sysdamping  = 10
freqdamping = 100
totalbins   = 960 * 3       # 2880 bins over 5 MHz → ~1.7 kHz/bin

AIRSPY_DLL  = r"C:\Program Files\PothosSDR\bin\airspy.dll"
SAMPLE_RATE = 6_000_000     # 6 MSPS — Airspy Mini max, dekt ruim 5 MHz band
GAIN        = 14            # Linearity gain 0–21 (verhoog bij zwak signaal)

# FFT_SIZE zodat bins_in_band ≈ totalbins
# bins_in_band = FFT_SIZE * 5MHz / 6MHz → FFT_SIZE = 2880 * 6/5 = 3456 → 4096
FFT_SIZE    = 4096
FREQ_CENTER = (freqmin + freqmax) // 2   # 382.5 MHz

AIRSPY_SUCCESS           = 0
AIRSPY_SAMPLE_FLOAT32_IQ = 0

class airspy_transfer(ctypes.Structure):
    _fields_ = [
        ("device",          ctypes.c_void_p),
        ("ctx",             ctypes.c_void_p),
        ("samples",         ctypes.c_void_p),
        ("sample_count",    ctypes.c_int),
        ("dropped_samples", ctypes.c_uint64),
        ("sample_type",     ctypes.c_int),
    ]

CALLBACK_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(airspy_transfer))


def load_dll():
    if not os.path.isfile(AIRSPY_DLL):
        print(f"[FOUT] airspy.dll niet gevonden: {AIRSPY_DLL}")
        sys.exit(1)
    try:
        return ctypes.CDLL(AIRSPY_DLL)
    except OSError as e:
        print(f"[FOUT] Kan airspy.dll niet laden: {e}")
        sys.exit(1)


def average(p):
    return sum(p) / float(len(p))

def variance(p):
    avg = average(p)
    return [(x - avg) ** 2 for x in p]

def std_dev(p):
    return sqrt(average(variance(p)))

def alert(freq_hz, power_db):
    freq = round(freq_hz / 1_000_000, 4)
    print(f"At {time.strftime('%H:%M:%S')}, a {round(power_db, 1)} dB/Hz signal was detected at {freq} MHz.")

class AirspyScanner:
    def __init__(self):
        self.lib       = load_dll()
        self.device    = ctypes.c_void_p()
        self._iq_buf   = np.zeros(FFT_SIZE, dtype=np.complex64)
        self._buf_fill = 0

        # Frequentie-as voor alle FFT bins
        freqs_full = FREQ_CENTER + np.fft.fftshift(
            np.fft.fftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
        )
        self._freq_mask = (freqs_full >= freqmin) & (freqs_full <= freqmax)
        self._freq_axis = freqs_full[self._freq_mask]

        self.rolling     = []
        self.rolling_avg = deque([])
        self.sweep       = deque([])
        self.i           = 0
        self.stddev      = 100.0

        self._window       = np.hanning(FFT_SIZE).astype(np.float32)
        self._callback_ref = CALLBACK_TYPE(self._on_samples)
        self._last_status  = time.time()

    def open(self):
        ret = self.lib.airspy_open(ctypes.byref(self.device))
        if ret != AIRSPY_SUCCESS:
            print(f"[FOUT] airspy_open mislukt (code {ret})")
            print("       Airspy Mini aangesloten? WinUSB driver via Zadig OK?")
            sys.exit(1)

        self.lib.airspy_set_sample_type(self.device, AIRSPY_SAMPLE_FLOAT32_IQ)
        self.lib.airspy_set_samplerate(self.device, ctypes.c_uint32(SAMPLE_RATE))
        self.lib.airspy_set_freq(self.device, ctypes.c_uint32(FREQ_CENTER))
        self.lib.airspy_set_linearity_gain(self.device, ctypes.c_uint8(GAIN))

        print(f"[OK]  Airspy Mini geopend")
        print(f"[OK]  Center : {FREQ_CENTER/1e6:.1f} MHz")
        print(f"[OK]  Band   : {freqmin/1e6:.0f}–{freqmax/1e6:.0f} MHz")
        print(f"[OK]  Gain   : {GAIN}")

    def _on_samples(self, transfer_ptr):
        t     = transfer_ptr.contents
        count = t.sample_count
        raw   = ctypes.cast(t.samples, ctypes.POINTER(ctypes.c_float))
        flat  = np.ctypeslib.as_array(raw, shape=(count * 2,)).copy()
        iq    = flat[0::2] + 1j * flat[1::2]

        pos = 0
        while pos < len(iq):
            space = FFT_SIZE - self._buf_fill
            chunk = iq[pos: pos + space]
            self._iq_buf[self._buf_fill: self._buf_fill + len(chunk)] = chunk
            self._buf_fill += len(chunk)
            pos += len(chunk)
            if self._buf_fill == FFT_SIZE:
                self._process_fft()
                self._buf_fill = 0

        return 0

    def _process_fft(self):
        fft_out    = np.fft.fftshift(np.fft.fft(self._iq_buf * self._window))
        power_full = 10 * np.log10(np.abs(fft_out) ** 2 + 1e-12)
        power      = power_full[self._freq_mask]

        for b in range(len(power)):
            freq_hz  = float(self._freq_axis[b])
            power_db = float(power[b])

            if len(self.rolling) < totalbins:
                self.rolling.append(deque([]))

            self.rolling[self.i].append(power_db)
            self.sweep.append(power_db)

            if len(self.rolling[self.i]) >= freqdamping:
                self.rolling[self.i].popleft()
                alarmthresh = (average(self.rolling[self.i]) +
                               self.stddev / sensitivity * 25000)
                if power_db > alarmthresh:
                    alert(freq_hz, power_db)

            if len(self.sweep) > totalbins:
                self.sweep.popleft()

            if self.i < totalbins - 1:
                self.i += 1
            else:
                self.i = 0
                self.rolling_avg.append(average(self.sweep))
                if len(self.rolling_avg) > sysdamping:
                    self.rolling_avg.popleft()
                    self.stddev = std_dev(self.rolling_avg)

        if time.time() - self._last_status >= 10:
            self._last_status = time.time()
            print(f"[{time.strftime('%H:%M:%S')}]  "
                  f"{freqmin/1e6:.0f}–{freqmax/1e6:.0f} MHz  |  "
                  f"gem: {float(np.mean(power)):.1f} dB  |  "
                  f"stddev: {self.stddev:.4f}")

    def run(self):
        self.open()
        ret = self.lib.airspy_start_rx(self.device, self._callback_ref, None)
        if ret != AIRSPY_SUCCESS:
            print(f"[FOUT] airspy_start_rx mislukt (code {ret})")
            sys.exit(1)
        print(f"\n[OK]  Scan gestart — Ctrl+C om te stoppen\n")
        try:
            while self.lib.airspy_is_streaming(self.device):
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nBuh-bye")
        finally:
            self.lib.airspy_stop_rx(self.device)
            self.lib.airspy_close(self.device)


if __name__ == "__main__":
    scanner = AirspyScanner()
    scanner.run()