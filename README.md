# xddsp_accumphase
```python
"""
xddsp_accumphase.py

Floating-point phase accumulator in XDDSP style.

This module implements a through-zero, wrapped phase accumulator with an internal
phase range of [-radius, radius). It is designed to match the behavior of the
original Gamma-style floating accumulator:

    phase in [-radius, radius)
    phase_unit in [0, 1)  →  phase = (2*radius)*phase_unit - radius
    inc = freq_hz * (2 * radius) / sr_hz

Per-sample optional frequency modulation is provided via a time-varying
freq_offset_hz signal.

All DSP is written in a functional way:
- State is a tuple of scalars.
- tick() returns (y, new_state)
- process() returns (y, new_state) for a whole block.
- All jitted DSP code is @njit(cache=True, fastmath=True) and uses only
  NumPy scalars/arrays and tuples (no dicts, no classes, no Python objects).
"""

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Internal helpers (Numba-jitted)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _accumphase_recache(freq_hz, sr_hz, radius):
    """
    Compute per-sample increment (inc) for a given freq_hz.

    Gamma does:
        mInc = freq * ups() * (2*radius)
    where ups() ~ 1/sr.
    That reduces to: inc = freq_hz * (2*radius) / sr_hz
    """
    return (float(freq_hz) * (2.0 * float(radius))) / float(sr_hz)


@njit(cache=True, fastmath=True)
def _accumphase_wrap(phase, radius):
    """
    Wrap phase to [-radius, radius).

    Equivalent to:
        span = 2*radius
        p0   = (phase + radius) % span
        p    = p0 - radius
    """
    span = 2.0 * radius
    p0 = (phase + radius) % span
    return p0 - radius


@njit(cache=True, fastmath=True)
def _accumphase_tick_jit(freq_offset_hz, state):
    """
    Jitted inner tick for one sample.

    Parameters
    ----------
    freq_offset_hz : float
        Per-sample frequency modulation in Hz.
    state : tuple
        (phase, freq_hz, sr_hz, radius, inc)

    Returns
    -------
    phase_before : float
        Wrapped phase in [-radius, radius) BEFORE increment.
    new_state : tuple
        Updated state.
    """
    phase, freq_hz, sr_hz, radius, inc = state

    # Wrap current phase; that's the "output" phase
    phase_wrapped = _accumphase_wrap(phase, radius)

    # Dynamic increment for this sample (no branching on array values)
    inc_extra = _accumphase_recache(freq_offset_hz, sr_hz, radius)
    inc_now = inc + inc_extra

    # Phase after stepping (unwrapped, will be wrapped next call)
    new_phase_unwrapped = phase_wrapped + inc_now

    new_state = (new_phase_unwrapped, freq_hz, sr_hz, radius, inc)
    return phase_wrapped, new_state


@njit(cache=True, fastmath=True)
def _accumphase_process_jit(freq_offset_hz, state, phase_out):
    """
    Jitted block processor.

    Parameters
    ----------
    freq_offset_hz : ndarray (N,)
        Time-varying frequency modulation (Hz) per sample.
    state : tuple
        (phase, freq_hz, sr_hz, radius, inc)
    phase_out : ndarray (N,)
        Preallocated output buffer for the phases.

    Returns
    -------
    new_state : tuple
        Updated state after N samples.
    """
    phase, freq_hz, sr_hz, radius, inc = state
    n = freq_offset_hz.shape[0]

    for i in range(n):
        # Wrap current phase and store as output
        span = 2.0 * radius
        p0 = (phase + radius) % span
        phase_wrapped = p0 - radius
        phase_out[i] = phase_wrapped

        # Per-sample FM increment (no branching)
        inc_extra = (freq_offset_hz[i] * (2.0 * radius)) / sr_hz
        inc_now = inc + inc_extra

        # Advance phase
        phase = phase_wrapped + inc_now

    new_state = (phase, freq_hz, sr_hz, radius, inc)
    return new_state


# ---------------------------------------------------------------------------
# Public API: init / update_state / tick / process
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def accumphase_init(freq_hz, phase_unit, sr_hz, radius=np.pi):
    """
    Initialize floating-phase accumulator state.

    Parameters
    ----------
    freq_hz : float
        Oscillator frequency in Hz.
    phase_unit : float
        Starting phase as unit fraction in [0,1),
        where 0   -> -radius
              0.5 -> 0
              1   -> radius (wraps to -radius).
    sr_hz : float
        Sample rate in Hz.
    radius : float
        Internal +/- phase range. For a normal sine, radius = pi.

    Returns
    -------
    state : tuple
        (phase, freq_hz, sr_hz, radius, inc)
    """
    phase = (2.0 * radius) * float(phase_unit) - radius
    inc = _accumphase_recache(freq_hz, sr_hz, radius)
    return (phase, float(freq_hz), float(sr_hz), float(radius), inc)


@njit(cache=True, fastmath=True)
def accumphase_update_state(state, freq_hz):
    """
    Return new state with updated base frequency (and increment).

    Parameters
    ----------
    state : tuple
        (phase, freq_hz, sr_hz, radius, inc)
    freq_hz : float
        New base frequency in Hz.

    Returns
    -------
    new_state : tuple
        (phase, new_freq_hz, sr_hz, radius, new_inc)
    """
    phase, _, sr_hz, radius, _ = state
    new_inc = _accumphase_recache(freq_hz, sr_hz, radius)
    return (phase, float(freq_hz), sr_hz, radius, new_inc)


@njit(cache=True, fastmath=True)
def accumphase_tick(freq_offset_hz, state):
    """
    Advance the accumulator by one sample.

    Parameters
    ----------
    freq_offset_hz : float
        Per-sample frequency modulation in Hz.
        (0.0 => no modulation).
    state : tuple
        (phase, freq_hz, sr_hz, radius, inc)

    Returns
    -------
    phase_before : float
        Wrapped phase in [-radius, radius) BEFORE increment.
    new_state : tuple
        Updated state after one sample.
    """
    return _accumphase_tick_jit(freq_offset_hz, state)


def accumphase_process(freq_offset_hz, state):
    """
    Block processing wrapper.

    Parameters
    ----------
    freq_offset_hz : ndarray (N,)
        Time-varying frequency modulation (Hz) for each sample.
        Use np.zeros(N) for no modulation.
    state : tuple
        (phase, freq_hz, sr_hz, radius, inc)

    Returns
    -------
    phase_out : ndarray (N,)
        Wrapped phase per sample in [-radius, radius).
    new_state : tuple
        Updated state after processing the whole block.
    """
    freq_offset_hz = np.asarray(freq_offset_hz, dtype=np.float64)
    n = freq_offset_hz.shape[0]
    phase_out = np.empty(n, dtype=np.float64)

    new_state = _accumphase_process_jit(freq_offset_hz, state, phase_out)
    return phase_out, new_state


# ---------------------------------------------------------------------------
# Smoke test / example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
    except ImportError:
        sd = None

    # Parameters
    sr = 48000.0
    dur_sec = 1.0
    n_samples = int(sr * dur_sec)

    base_freq = 440.0
    phase_unit_start = 0.0
    radius = np.pi

    # Initialize state
    state = accumphase_init(base_freq, phase_unit_start, sr, radius)

    # No FM for this example
    freq_offset = np.zeros(n_samples, dtype=np.float64)

    # Process one block of phases
    phases, state_out = accumphase_process(freq_offset, state)

    # Turn phases into a sine waveform for listening/plotting
    y = np.sin(phases).astype(np.float32)

    # Plot a short segment of the waveform
    t = np.arange(200) / sr
    plt.figure(figsize=(8, 4))
    plt.plot(t, y[:200])
    plt.title("accumphase sine @ 440 Hz (first 200 samples)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Listen example (if sounddevice is available)
    if sd is not None:
        print("Playing 1 second of 440 Hz sine from accumphase accumulator...")
        sd.play(y, int(sr))
        sd.wait()
    else:
        print("sounddevice not installed; skipping audio playback.")

```
