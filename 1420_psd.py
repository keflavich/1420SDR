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
        opts, args = getopt.getopt(argv, "hi:", ["integrate=", "background", "doplot", "verbose"])
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

    opts = dict(opts)
    if '--background' in opts:
        filesuffix = '_background'
    else:
        filesuffix = ""

    if "--doplot" in opts:
        doplot = True
    else:
        doplot = False

    if "--verbose" in opts:
        verbose = True
    else:
        verbose = False

    if verbose:
        print(opts)




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

    if verbose:
        print('Averaging samples...')
    avgpower = []
    for ii in range(numsamples):
        avg_i = 0
        for jj in range(passes):
            avg_i += power[jj][ii]
        avgpower.append(avg_i/passes)

    rvel = (constants.c*((hi_restfreq - u.Quantity(frequency, u.MHz))/hi_restfreq)).to(u.km/u.s).value

    now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))

    if verbose:
        print('Writing output files...')
    # write plot

    if doplot:
        fig, ax = plt.subplots()
        ax.plot(rvel, 10*np.log10(avgpower), 'b-')
        ax.tick_params(labelsize=6)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.set_xlabel('Relative Velocity (km/s)')
        ax.set_ylabel('Measured Power (dB)')
        try:
            fig.savefig('psd_' + now + '.pdf')
        except:
            print("FAILED savefig")

    # write csv

    filename = 'psd_' + now + "_t=" + str(int_time) + filesuffix + '.csv'
    filename = f"psd_{now}_tint{int_time}s{filesuffix}.csv"
    if verbose:
        print(filename)
    assert "'" not in filename
    assert '"' not in filename

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for ii in range(numsamples):
            writer.writerow([frequency[ii], 10*np.log10(avgpower[ii]),
                             rvel[ii]])


if __name__ == "__main__":
    main(sys.argv[1:])
