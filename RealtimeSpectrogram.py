import sys
import numpy as np
import pyqtgraph as pg
import pyaudio
import librosa
from pyqtgraph.Qt import QtWidgets, QtCore

class RealTimeMelSpectrogram:
    def __init__(self, 
                 sr=44100, 
                 n_mels=128, 
                 n_fft=2048, 
                 hop_length=512
        ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sr, input=True, frames_per_buffer=self.hop_length, stream_callback=self.callback)

        # Initialize PyQtGraph
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Mel Spectrogram")
        self.plot = self.win.addPlot()
        self.image = pg.ImageItem(axisOrder='row-major')
        self.plot.addItem(self.image)
        self.plot.setLabel('left', 'Frequency', units='Hz')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setXRange(0, 3)  # Time axis from 0 to 3 seconds
        self.plot.setYRange(0, 100)  # Frequency axis from 0 to 100 Hz

        # Create a colormap
        self.cmap = pg.colormap.get('viridis', source='matplotlib')
        self.image.setLookupTable(self.cmap.getLookupTable())

        # Prepare the data buffer
        self.spec_data = np.zeros((int(3 * self.sr / self.hop_length), self.n_mels))

        # Start the update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        S = librosa.feature.melspectrogram(y=audio_data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, fmax=100)
        S_db = librosa.power_to_db(S, ref=np.max)
        self.spec_data = np.roll(self.spec_data, -1, axis=0)
        self.spec_data[-1, :] = S_db[:, 0]
        return (in_data, pyaudio.paContinue)

    def update(self):
        self.image.setImage(self.spec_data.T, autoLevels=False, levels=(self.spec_data.min(), self.spec_data.max()))
        QtWidgets.QApplication.processEvents()

    def start(self):
        self.stream.start_stream()
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    spectrogram = RealTimeMelSpectrogram()
    spectrogram.start()
