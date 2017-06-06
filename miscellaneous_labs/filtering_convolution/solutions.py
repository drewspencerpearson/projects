# Name this file solutions.py.
"""Volume II Lab 10: Fourier II (Filtering and Convolution)
Drew Pearson
Math 320
Novemeber 12
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile as wf
import scipy as sp

# Problem 1: Implement this function.
def clean_signal(outfile='prob1.wav'):
    """Clean the 'Noisysignal2.wav' file. Plot the resulting sound
    wave and write the resulting sound to the specified outfile.
    """
    rate, data = wf.read('Noisysignal2.wav')
    fsig = sp.fft(data, axis = 0)
    plt.plot(fsig)
    plt.show()
    for j in xrange(14000,50000):
        fsig[j] = 0
        fsig[-j] = 0

    newsig = sp.ifft(fsig)
    newsig = np.real(newsig)
    newsig = sp.int16(newsig/sp.absolute(newsig).max()*32767)

    wf.write(outfile, rate, newsig)
    print "F.D.R. is saying only thing we have to fear is fear itself."

# Problem 2 is not required. Use balloon.wav for problem 3.

# Problem 3: Implement this function.
def convolve(source='chopin.wav', pulse='balloon.wav', outfile='prob3.wav'):
    """Convolve the specified source file with the specified pulse file, then
    write the resulting sound wave to the specified outfile.
    """
    rate_piano, data_piano = wf.read(source)
    rate_impulse, data_impulse = wf.read(pulse)
    data_impulse = data_impulse[:,0]
    data_piano = data_piano[:,0]
    new_data_piano = np.hstack((data_piano, np.zeros(len(data_piano))))
    wf.write('teststack', rate_piano, new_data_piano)

    dst = np.hstack((np.zeros(len(new_data_piano)-len(data_impulse))))

    half1 = data_impulse[:len(data_impulse)/2:1]
    half2 = data_impulse[len(data_impulse)/2::1]
    new_impulse = np.hstack((np.hstack((half1, dst)), half2))
    convolution = sp.ifft(sp.fft(new_data_piano)*sp.fft(new_impulse))
    convolution = np.real(convolution)
    convolution = sp.int16(convolution/sp.absolute(convolution).max()*32767)

    wf.write(outfile, rate_piano, convolution)





# Problem 4: Implement this function.
def white_noise(outfile='prob4.wav'):
    """Generate some white noise, write it to the specified outfile,
    and plot the spectrum (DFT) of the signal.
    """
    samplerate = 22050
    noise = sp.int16(sp.random.randint(-32767, 32767, samplerate * 10))
    wf.write(outfile, samplerate, noise)
    plt.plot(abs(noise))
    plt.show()


if __name__ == '__main__':
    #clean_signal()
    #convolve('chopin_full.wav')
    white_noise()
    #white_noise()