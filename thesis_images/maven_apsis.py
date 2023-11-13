from h5py import File
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from funkyfresh import FunkyFresh
import matplotlib.ticker as ticker
from lmfit.models import PolynomialModel


if __name__ == '__main__':
    file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
    apoapse = file['apoapse']
    print(apoapse.keys())

    orbits = file['apoapse/orbit'][:]
    lt = file['apoapse/subspacecraft_local_time'][:]
    subsolar_lat = file['apoapse/subsolar_latitude'][:]
    subsolar_lon = file['apoapse/subsolar_longitude'][:]
    subsc_lat = file['apoapse/subspacecraft_latitude'][:]
    subsc_lon = file['apoapse/subspacecraft_longitude'][:]
    sc_alt = file['apoapse/spacecraft_altitude'][:]
    angle = file['apoapse/subsolar_subspacecraft_angle'][:]
    my = apoapse['mars_year'][:]
    ls = apoapse['solar_longitude'][:]

    special_ls = ls + (my-my[0]) * 360

    fit = np.poly1d(np.polyfit(orbits, special_ls, 47))
    fit_inverse = np.poly1d(np.polyfit(special_ls, orbits, 47))

    ffs = FunkyFresh()   # funky fresh style
    ffs.set_named_style('AGU')
    fig, ax = plt.subplots(3, 1, sharex='all', layout='constrained', gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(ffs.figure_widths['text'], 4.5))

    twilight = matplotlib.colormaps['twilight_shifted']
    noon = twilight(0.5)
    colors = twilight(lt/24)

    ax[0].plot(orbits, subsolar_lat, color=noon, linestyle='-')   # this is the subsolar line
    for orb in range(17501):
        ax[0].plot(orbits[orb: orb+2], subsc_lat[orb: orb+2], color=colors[orb])   # This hack makes this phd quality, not undergrad quality
    orb = 1700
    arrowprops = dict(arrowstyle="-|>, head_length=0.25,head_width=0.125",
                      shrinkA=1, shrinkB=1,
                      patchA=None, patchB=None, relpos=(0, 0.5),
                      connectionstyle="arc3,rad=0.2")
    ax[0].annotate('Sub-spacecraft latitude', xy=(orbits[orb], subsc_lat[orb]), ha='left', va='bottom', xytext=(15, 2), textcoords='offset points', arrowprops=arrowprops, fontsize=6)
    orb = 4200
    arrowprops['relpos'] = (0.5, 1)
    arrowprops['connectionstyle'] = "arc3,rad=0"
    ax[0].annotate('Sub-solar latitude', xy=(orbits[orb], subsolar_lat[orb]), ha='center', va='top', xytext=(0, -30), textcoords='offset points', arrowprops=arrowprops, fontsize=6)
    #image = ax[0].scatter(orbits, subsc_lat, s=1, c=colors)

    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax[0].set_ylabel('Latitude [degrees]')
    ax[0].set_ylim(-90, 90)

    ax[0].set_title('Apoapsis Geometry')

    # create normalization instance
    norm = matplotlib.colors.Normalize(vmin=0, vmax=24)
    # create a scalarmappable from the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=twilight, norm=norm)
    cbar = fig.colorbar(sm, ax=ax[0], orientation='vertical', cmap='twilight_shifted', pad=0, label='Local Time [hours]')
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(6))
    cbar.ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))

    ax[1].plot(orbits, angle)
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax[1].set_ylim(0, 90)
    ax[1].set_ylabel('Solar Zenith Angle \n [degrees]')
    #ax[1].set_title('Subsolar-subspacecraft angle')

    ax[2].plot(orbits, sc_alt)
    ax[2].set_ylim(4000, 7000)
    #ax[2].set_title('Spacecraft altitude')
    ax[2].set_ylabel('Altitude [km]')

    ax[2].set_xlabel('Orbit Number')
    ax[2].xaxis.set_major_locator(ticker.MultipleLocator(2500))
    ax[2].xaxis.set_minor_locator(ticker.MultipleLocator(500))
    ax[2].yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax[2].yaxis.set_minor_locator(ticker.MultipleLocator(250))
    ax[2].set_xlim(0, 17500)

    second_ax = ax[2].secondary_xaxis(location=-0.6, functions=(fit, fit_inverse))
    second_ax.set_xlabel('Solar Longitude [degrees]')

    def major_formatter(x, pos):
        return int(x%360)

    second_ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
    second_ax.xaxis.set_major_formatter(ticker.FuncFormatter(major_formatter))
    second_ax.xaxis.set_minor_locator(ticker.MultipleLocator(30))
    fig.canvas.draw()
    labels = [item.get_text() for item in second_ax.get_xticklabels()]

    print(labels)

    mys = [33, 34, 35, 36, 37]
    counter = 0
    new_labels = []
    for label in labels:
        if label == '0':
            new_labels.append(f'MY {mys[counter]}')
            counter += 1
        else:
            new_labels.append(label)

    second_ax.set_xticklabels(new_labels)
    ticks = second_ax.get_xticklabels()
    for i in range(2, 20, 4):
        ticks[i].set_rotation(90)
        ticks[i].set_rotation_mode('anchor')
        ticks[i].set(ha='right', va='center')

    plt.savefig('/home/kyle/thesis/maven_apsis.pdf')
