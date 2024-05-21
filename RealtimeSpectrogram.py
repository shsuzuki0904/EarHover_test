from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import rtmixer
import math
import sounddevice as sd

FRAMES_PER_BUFFER = 512
NFFT = 4096
NOVERLAP = 128

window = 100000
downsample = 10  
channels = 1

def create_specgram(frame):
    global plotdata
    
    spec, freqs, t = plt.mlab.specgram(plotdata[:,-1], Fs=samplerate)
    xmin, xmax = np.min(t) - pad_xextent, np.max(t) + pad_xextent
    extent = xmin, xmax, freqs[0], freqs[-1]
    arr = np.flipud(10. * np.log10(spec))

    return arr, extent

def update_plot(frame):

    global plotdata

    while ringBuffer.read_available >= FRAMES_PER_BUFFER:
        read, buf1, buf2 = ringBuffer.get_read_buffers(FRAMES_PER_BUFFER)
        assert read == FRAMES_PER_BUFFER
        buffer = np.frombuffer(buf1, dtype='float32')
        buffer.shape = -1, channels
        buffer = buffer[::downsample]

        assert buffer.base.base == buf1
        shift = len(buffer)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = buffer
        ringBuffer.advance_read_index(FRAMES_PER_BUFFER)

    arr, _  = create_specgram(frame)
    image.set_array(arr)
    return image,

device_info = sd.query_devices(device=None, kind='input')
samplerate = device_info['default_samplerate']

pad_xextent = (NFFT - NOVERLAP) / samplerate / 2
length = int(window * samplerate / (1000 * downsample))
plotdata = np.zeros((length, channels))

stream = rtmixer.Recorder(device=None, channels=channels, blocksize=FRAMES_PER_BUFFER,
                          latency='low', samplerate=samplerate)

ringbufferSize = 2**int(math.log2(3 * samplerate))

ringBuffer = rtmixer.RingBuffer(channels * stream.samplesize, ringbufferSize)

fig, ax = plt.subplots(figsize=(10, 5))
arr, extent = create_specgram(0)
image = plt.imshow(arr, animated=True, extent=extent, aspect='auto')
fig.colorbar(image)

ani = FuncAnimation(fig, update_plot, interval=1, blit=True, cache_frame_data=False)           

with stream:
    ringBuffer = rtmixer.RingBuffer(channels * stream.samplesize, ringbufferSize)
    action = stream.record_ringbuffer(ringBuffer)
    plt.show()
