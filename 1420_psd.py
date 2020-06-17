# James Cheshire
# Modified by Adam Ginsburg

import pylab as plt
from rtlsdr import RtlSdr
import datetime, csv, sys, numpy as np, getopt
from astropy import constants, units as u
from astropy.table import Table
import tqdm

hi_restfreq = 1420.405751786*u.MHz

def main(argv):

    if(not len(argv)):
        sys.exit("Usage:\n1420_psd.py -i <integration time(s)>")
    try:
        opts, args = getopt.getopt(argv, "hi:", ["integrate=", "background",
                                                 "do_fsw", "doplot", "verbose",
                                                 "device_index=",
                                                 "progressbar", "freqcorr=",
                                                 "obs_lat",
                                                 "obs_lon",
                                                 "altitude",
                                                 "azimuth",
                                                ])
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

    if "--progressbar" in opts:
        progressbar = tqdm.tqdm
    else:
        progressbar = lambda x: x

    if "--do_fsw" in opts:
        do_fsw = True
        try:
            velthrow = int(opts['--do_fsw'])*u.km/u.s
        except:
            velthrow = 50*u.km/u.s
    else:
        do_fsw = False

    if '--device_index' in opts:
        device_index = int(opts['--device_index'])
        if verbose:
            print(f"Using device index {device_index}")
    else:
        device_index = 0

    if '--freqcorr' in opts:
        freqcorr = int(opts['--freqcorr'])
    else:
        freqcorr = None

    if verbose:
        print(opts)


    import time
    t0 = time.time()



    # initialize SDR
    sdr = RtlSdr(device_index=device_index)


    if do_fsw:
        freqthrow = ((velthrow/constants.c) * hi_restfreq).to(u.Hz).value

    sdr.sample_rate = 2.4e6
    # center frequency in Hertz
    sdr.center_freq = hi_restfreq.to(u.Hz).value
    # max gain is ~50?
    sdr.gain = 50

    if freqcorr is not None:
        sdr.set_freq_correction(freqcorr)

    numsamples = 2048
    passes = int(int_time * sdr.rs / numsamples)

    if do_fsw:
        nfsw = 4
        #fsw_time = int_time / nfsw

    #chanwidth = sdr.rs / numsamples
    #chanwidth_kms = (chanwidth / sdr.center_freq) * constants.c.to(u.km/u.s).value


    if verbose:
        print(f"Center freq: {sdr.fc}  readsize: {sdr.fc}  numsamples: {numsamples} passes: {passes}")

    # collect data

    if do_fsw:
        power = {1: [], -1: []}
        frequency = {}
        sign = 1
    else:
        power = []

    if verbose:
        print('Warning: expect execution to take 4-5x your integration time')
        print('Collecting Data...')

    if do_fsw:
        for fsw_id in progressbar(range(nfsw)):

            sdr.center_freq = hi_restfreq.to(u.Hz).value + freqthrow * sign
            sign = sign * -1

            frq = np.fft.fftfreq(numsamples)
            idx = np.argsort(frq)
            frequency[sign] = sdr.fc + sdr.rs * frq[idx]

            for ii in range(passes//nfsw):
                samples = sdr.read_samples(numsamples)

                ps = np.abs(np.fft.fft(samples))**2

                ps[0] = np.mean(ps)

                n = len(samples)
                power[sign].append(ps[idx]/n)
    else:
        frq = np.fft.fftfreq(numsamples)
        idx = np.argsort(frq)
        frequency = sdr.fc + sdr.rs * frq[idx]
        for ii in progressbar(range(passes)):
            samples = sdr.read_samples(numsamples)

            ps = np.abs(np.fft.fft(samples))**2

            ps[0] = np.mean(ps)

            n = len(samples)
            power.append(ps[idx]/n)

    if verbose:
        print(f"sampling time = {time.time()-t0} for tint={int_time}")

    if do_fsw:
        avgpower = {key: np.array(pow).mean(axis=0) for key, pow in power.items()}

        # fsw = low-frequency minus high frequency
        fsw = avgpower[-1] - avgpower[1]

        # radio velocity = (nu_0 - nu) / nu_0
        rvel1 = (constants.c*((hi_restfreq - u.Quantity(frequency[-1], u.Hz))/hi_restfreq)).to(u.km/u.s).value
        rvel2 = (constants.c*((hi_restfreq - u.Quantity(frequency[1], u.Hz))/hi_restfreq)).to(u.km/u.s).value
    else:
        avgpower = np.array(power).mean(axis=0)
        rvel = (constants.c*((hi_restfreq - u.Quantity(frequency, u.Hz))/hi_restfreq)).to(u.km/u.s).value


    now = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))


    if doplot:
        fig, ax = plt.subplots(figsize=(14,6))
        if do_fsw:
            ax.plot(rvel, 10*np.log10(fsw), )
        else:
            ax.plot(rvel, 10*np.log10(avgpower), )
        ax.tick_params(labelsize=6)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.set_xlabel('Relative Velocity (km/s)')
        ax.set_ylabel('Measured Power (dB)')
        try:
            fig.savefig('psd_' + now + '.pdf')
        except:
            print("FAILED savefig")

    # write csv

    if do_fsw:
        filename = f"psd_{now}_tint{int_time}s_sdr{device_index}_fsw{filesuffix}.fits"

        dat = {'freq1': frequency[-1],
               'freq2': frequency[1],
               'power1': avgpower[-1],
               'power2': avgpower[1],
               'fsw_rvel1': rvel1,
               'fsw_rvel2': rvel2,
               'fsw_pow': fsw}
    else:
        filename = f"psd_{now}_tint{int_time}s_sdr{device_index}{filesuffix}.fits"

        dat = {'freq': frequency,
               'rvel': frequency,
               'power': avgpower
               }
    tbl = Table(dat)

    if verbose:
        print(filename)

    for key in ('obs_lat', 'obs_lon', 'altitude', 'azimuth'):
        if f'--{key}' in opts:
            tbl.meta[f'--{key}'] = opts[f'--{key}']
        
    tbl.meta['date-obs'] = now
    tbl.write(filename)

    if verbose:
        print(f"total time = {time.time()-t0} for tint={int_time}")


if __name__ == "__main__":
    main(sys.argv[1:])
