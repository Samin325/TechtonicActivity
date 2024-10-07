import os
from obspy import Stream, read
from filter.filters import filter_bp

def filter_mseed_data(t,v):
    """filters the miniSEED velocity using band-pass filter

    Args:
        t (ndarray): array of relative time (s)
        v (ndarray): array of unfiltered velocity

    Returns:
        ndarray : filtered velocity
    """
    F_s = 1 / (t[1] - t[0])
    f_low = 0.3
    f_high = 1.2
    return filter_bp(v, F_s, f_low, f_high)

def main():
    # loop though all mseed files in a directory
    mseed_data_directory = "./data/earth/training/data/mseed_filtered/"
    
    # replaces filtered mseed data with
    for file in os.listdir(mseed_data_directory):
        if file.endswith(".mseed"):
            st = read(mseed_data_directory + file)
            tr = st[0]
            t = tr.times()
            v = tr.data
            v_filt = filter_mseed_data(t,v)
            
            # replace velocity with filtered velocity
            tr.data = v_filt
            st_filt = Stream(tr)
            st_filt.write(mseed_data_directory + file.replace(".mseed", "_filtered.mseed"), format="MSEED")
            
if __name__ == "__main__":
    main()
