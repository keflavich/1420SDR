# James Cheshire
# Last modified 5/11/17
# frequency dropoffs: (approx.) index 416 for redshift, 1631 for blueshift

import pylab as plt
from rtlsdr import RtlSdr
import datetime, csv, sys, numpy as np, getopt
from astropy import constants, units as u

hi_restfreq = 1420.405751786*u.MHz

def main(argv):

    if(not len(argv)):
        sys.exit("Usage:\n1420_psd.py -i <integration time(s)>")
    try:
        opts, args = getopt.getopt(argv, "hi:", ["integrate=", "background"])
    except getopt.GetoptError:
        print('Usage:\n1420_psd.py -i <integration time (s)> (--background)')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage:\n1420_psd.py -i <integration time (s)>')
            sys.exit()
        elif opt in ("-i", "--integrate"):
            if arg.isdigit():
                int_time = int(arg)
                if(int_time <= 0):
                    sys.exit("Integration time must be a positive integer number of seconds")
            else:
                sys.exit("Error: argument must be an integer number of seconds")
        if opt == '--background':
            filesuffix = '_background'
        else:
            filesuffix = ""



    # initialize SDR
    sdr = RtlSdr()

    freqcorr = 55

    sdr.sample_rate = 2.4e6
    sdr.center_freq = hi_restfreq.to(u.MHz).value
    sdr.gain = 50
    sdr.set_freq_correction(freqcorr)

    numsamples = 2**11
    passes = int(int_time * sdr.rs / numsamples)


    # collect data

    power = []
    frequency = []

    print('Warning: expect execution to take 4-5x your integration time')
    print('Collecting Data...')
    for ii in range(passes):
        samples = sdr.read_samples(numsamples)

        ps = np.abs(np.fft.fft(samples))**2

        frq = np.fft.fftfreq(samples.size)
        idx = np.argsort(frq)
        if ii == 0:
            frequency = sdr.fc + sdr.rs * frq[idx]

        ps[0] = np.mean(ps)

        n = len(samples)
        power.append(ps[idx]/n)

    print('Averaging samples...')
    avgpower = []
    for ii in range(numsamples):
        avg_i = 0
        for jj in range(passes):
            avg_i += power[jj][ii]
        avgpower.append(avg_i/passes)

    rvel = (constants.c*((hi_restfreq - frequency)/hi_restfreq)).to(u.km/u.s).value

    print('Writing output files...')
    # write plot

    fig, ax = plt.subplots()
    ax.plot(rvel, 10*np.log10(avgpower), 'b-')
    ax.tick_params(labelsize=6)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.set_xlabel('Relative Velocity (km/s)')
    ax.set_ylabel('Measured Power (dB)')
    fig.savefig('psd_' + str(datetime.datetime.now()) + '.pdf')

    # write csv

    filename = 'psd_' + str(datetime.datetime.now()) + "_t=" + str(int_time) + filesuffix + '.csv'

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for ii in range(numsamples):
            writer.writerow([frequency[ii], 10*np.log10(avgpower[ii]),
                             rvel[ii]])

    plt.plot(rvel, 10*np.log10(avgpower))
    plt.xlabel('Relative Velocity (km/s)')
    plt.ylabel('Measured Power (dB)')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
