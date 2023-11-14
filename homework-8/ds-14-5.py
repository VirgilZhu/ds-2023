import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

audio_path = r".\test.wav"
y, sr = librosa.load(audio_path, sr=None, mono=True)

fft_result = np.fft.fft(y)
magnitude = np.abs(fft_result)
frequency = np.fft.fftfreq(len(magnitude), 1/sr)

plt.figure(figsize=(12, 4))

plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')

plt.subplot(2, 1, 2)
plt.plot(frequency[:int(len(frequency)/2)], magnitude[:int(len(frequency)/2)])
plt.title('Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
