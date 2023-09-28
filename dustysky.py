# -*- coding: utf-8 -*-

# future import statements
from __future__ import print_function
from __future__ import division
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.interpolate as spi
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time
import matplotlib.colors as colors

# version information
__project__ = "EXPLORE"
__author__  = "ACRI-ST"
__modifiers__ = '$Author: N. Cox $'
__date__ = '$Date: 2021-10-12 $'
__version__ = '$Rev: 1.0 $'
__license__ = '$Apache 2.0 $'

plt.close('all')

#%% Load extinction cubes

def load_cube(hdf5file):
    """Load hdf5, calculate axes values corresponding to data.

    (original authors: N. Leclerc, G. Plum, S. Ferron)

    Args:
        hdf5file (str): full path for HDF5 file.



    Returns:
        dict: headers contains in HDF5 file.
        :func:`np.array`: 3D array which contains the extinction value.
        tuple: (x, y, z) where x,y,z contains array of axes
            corresponding to cube values.
        array: value min for x, y, z axes.
        array: value max for x, y, z axes.
        float: value of gridstep size
        float: value of half-width of the cube
        float: points (neeed??)
        float: value of scale (half-width*gridstep)
        step, hw, points, s

    """
    # read hdf5 file
    with h5py.File(hdf5file, 'r') as hf:
        cube = hf['explore/cube_datas'][:]
        dc = hf['explore/cube_datas']
        #cube = hf['stilism/cube_datas'][:]
        #dc = hf['stilism/cube_datas']
        
        headers = {k: v for k, v in dc.attrs.items()}

    sun_position = headers["sun_position"]
    gridstep_values = headers["gridstep_values"]
    new_sun_position = np.append(sun_position[1:],sun_position[0])

    # Calculate axes for cube value, with sun at position (0, 0, 0)
    min_axes = -1 * new_sun_position * gridstep_values
    max_axes = np.abs(min_axes)
    axes = (
        np.linspace(min_axes[0], max_axes[0], cube.shape[0]),
        np.linspace(min_axes[1], max_axes[1], cube.shape[1]),
        np.linspace(min_axes[2], max_axes[2], cube.shape[2])
    )

    step = np.array(headers["gridstep_values"])
    hw = (np.copy(cube.shape) - 1) / 2.
    points = (
        np.arange(0, cube.shape[0]),
        np.arange(0, cube.shape[1]),
        np.arange(0, cube.shape[2])
    )
    s = hw * step

    return (headers, cube,
        axes, min_axes, max_axes,
        step, hw, points, s)

#%% Compute reddening from extinction cubes

def reddening(sc, cube, axes, max_axes, step_pc=5):
    """Calculate Extinction versus distance from Sun.

    Args:
        sc: SkyCoord object

    Kwargs:
        step_pc (int): Incremental distance in parsec

    Returns:
        array: Parsec values.
        array: Extinction A(5500) value obtained with integral of linear extrapolation.

    """

    sc1=SkyCoord(sc, distance = 1 * u.pc)

    coords_xyz = sc1.transform_to('galactic').represent_as('cartesian').get_xyz().value

    # Find the number of parsec I can calculate before go out the cube
    # (exclude divide by 0)
    not0 = np.where(coords_xyz != 0)

    max_pc = np.amin(
        np.abs( np.take(max_axes, not0) / np.take(coords_xyz, not0) ) )

    # Calculate all coordinates to interpolate (use step_pc)

    distances = np.arange(0, max_pc, step_pc)

    sc2 = SkyCoord(
        sc,
        distance=distances)

    sc2 = sc2.transform_to('galactic').represent_as('cartesian')
    coords_xyz = np.array([coord.get_xyz().value for coord in sc2])

    # linear interpolation with coordinates
    interpolation = spi.interpn(
        axes,
        cube,
        coords_xyz,
        method='linear'
    )

    xvalues = np.arange(0, len(interpolation) * step_pc, step_pc)
    yvalues_cumul = np.nancumsum(interpolation) * step_pc
    yvalues = interpolation

    
    return (
        xvalues,
        np.around(yvalues_cumul, decimals=5),
        np.around(yvalues, decimals=5)
        )

#%% Milky Way plot

hdf5file50 = os.path.join('', "explore_cube_density_values_050pc_v2.h5")
hdf5file25 = os.path.join('', "explore_cube_density_values_025pc_v2.h5")
hdf5file10 = os.path.join('', "explore_cube_density_values_010pc_v2.h5")

headers50, cube50, axes50, min_axes50, max_axes50, step50, hw50, points50, s50 = load_cube(hdf5file50)
headers25, cube25, axes25, min_axes25, max_axes25, step25, hw25, points25, s25 = load_cube(hdf5file25)
headers10, cube10, axes10, min_axes10, max_axes10, step10, hw10, points10, s10 = load_cube(hdf5file10)

step_pc10 = 10
step_pc25 = 50
step_pc50 = 100


#%% 2D Animation 


def create_skymap_data(mode=None, outputfile=None):

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    import scipy.interpolate as spi
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import matplotlib.animation as animation
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import time
    import matplotlib.colors as colors

    #l = np.linspace(-np.pi, np.pi, 361)
    #b = np.linspace(-np.pi/2, np.pi/2, 181)

    l = np.linspace(-np.pi, np.pi, 721)
    b = np.linspace(-np.pi/2, np.pi/2, 361)

    Ext = np.zeros([len(b),len(l)])

    d = range(151)

    x_all = []
    Ext_all = []
    coord = []

    for k in range(len(d)):
        print(k)
        for i in range(len(b)):
            for j in range(len(l)):
                gal = SkyCoord(l[j]*u.rad, b[i]*u.rad, frame = 'galactic')
                xvalues, yvalues_cumul, yvalues = reddening(gal, cube=cube10, axes=axes10, max_axes=max_axes10, step_pc=step_pc10)
                
                if len(xvalues < 1418):
                        yvalues = np.pad(yvalues, (0, 1417-len(yvalues)), 'constant')                 
                        yvalues_cumul = np.pad(yvalues_cumul, (0, 1417-len(yvalues_cumul)), 'maximum')                    

                if (mode == 'diff'):
                    Ext[i][j] = yvalues[k]
                if (mode == 'cumul'):
                    Ext[i][j] = yvalues_cumul[k]

        Ext_all.append(Ext)
        Ext = np.zeros([len(b),len(l)])

    d = range(31, 61)

    for k in range(len(d)):
        print(k)
        for i in range(len(b)):
            for j in range(len(l)):
                gal = SkyCoord(l[j]*u.rad, b[i]*u.rad, frame = 'galactic')
                xvalues, yvalues_cumul, yvalues = reddening(gal, cube=cube25, axes=axes25, max_axes=max_axes25, step_pc=step_pc25)
                # print(len(xvalues))
                if len(xvalues < 87):
                        yvalues = np.pad(yvalues, (0, 86-len(yvalues)), 'constant')                 
                        yvalues = np.pad(yvalues_cumul, (0, 86-len(yvalues_cumul)), 'maximum')                    

                if (mode == 'diff'):
                    Ext[i][j] = yvalues[d[k]]
                if (mode == 'cumul'):
                    Ext[i][j] = yvalues_cumul[d[k]]

        Ext_all.append(Ext)
        Ext = np.zeros([len(b),len(l)])    

    d = range(31, 51)

    for k in range(len(d)):
        print(k)
        for i in range(len(b)):
            for j in range(len(l)):
                gal = SkyCoord(l[j]*u.rad, b[i]*u.rad, frame = 'galactic')
                xvalues, yvalues_cumul, yvalues = reddening(gal, cube=cube50, axes=axes50, max_axes=max_axes50, step_pc=step_pc50)
                # print(len(xvalues))
                if len(xvalues < 72):
                    yvalues = np.pad(yvalues, (0, 71-len(yvalues)), 'constant') 
                    yvalues_cumul = np.pad(yvalues_cumul, (0, 71-len(yvalues_cumul)), 'maximum')                    

                if (mode == 'diff'):
                    Ext[i][j] = yvalues[d[k]]
                if (mode == 'cumul'):
                    Ext[i][j] = yvalues_cumul[d[k]]

        Ext_all.append(Ext)
        Ext = np.zeros([len(b),len(l)])   

    np.savez(outputfile, y = Ext_all, l = l, b = b)

    return



def create_figure(mode=None, input_npz=None, outgif=None, plot_title=None):

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    import scipy.interpolate as spi
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import matplotlib.animation as animation
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import time
    import matplotlib.colors as colors

    ### load data + create (l,b) mesh
    data = np.load(input_npz)

    l = data['l']
    b = data['b']
    Ext_all = data['y']

    X,Y = np.meshgrid(-l, b)

    ### setup plot

    plt.rcParams['animation.ffmpeg_path'] ='D:\\EXPLORE\\dustysky-master\\ffmpeg'
    plt.close('all')
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="aitoff")

    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    ### add first layer
    cont = ax.pcolormesh(X, Y, Ext_all[0], vmin = 0, shading = 'auto', cmap = 'copper')
    ax2 = fig.add_subplot(111)

    ### define logos and annotations
    logo = mpimg.imread('logo_g_tomo_white.png')
    imagebox = OffsetImage(logo, zoom =0.09)
    ab = AnnotationBbox(imagebox, (0.0,0.86), frameon=False)
    ax2.add_artist(ab)

    EXPLORE_logo = mpimg.imread('EXPLORE_logo_white.png')
    imagebox = OffsetImage(EXPLORE_logo, zoom =0.035)
    ab2 = AnnotationBbox(imagebox, (0.94,0.86), frameon=False)
    ax2.add_artist(ab2)

    plt.axis('off')

    txt="This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101004214"
    plt.figtext(0.51, 0.2, txt, wrap=True, horizontalalignment='center', fontsize=9, fontname = 'Arial')

    plt.suptitle(plot_title, x = 0.511, y=0.765, fontname = 'Arial', fontsize = 10)

    def animate(i):
        #convert pc to lightyear???
        z = Ext_all[i]
        cont = ax.pcolormesh(X, Y, z, shading = 'auto', cmap = 'copper')
        if i <= 150:
            plt.title('Distance = %.1lf pc' %(i*10), x = 0.5, y=0.85, fontname = 'Arial', fontsize = 14)
        if i > 150 and i <= 180:
            plt.title('Distance = %.1lf pc' %(50*i-6000), x = 0.5, y=0.85, fontname = 'Arial', fontsize = 14)
        if i > 180:
            plt.title('Distance = %.1lf pc' %((i-180)*100+3000), x = 0.5, y=0.85, fontname = 'Arial', fontsize = 14)
        plt.show()
        return cont

    ### start animation
    
    anim = animation.FuncAnimation(fig, animate, len(Ext_all))
    anim.save(outgif, writer = 'ffmpeg')
    #FFwriter = animation.FFMpegWriter()
    #anim.save('animation.mp4', writer = FFwriter)


t = time.time()

create_skymap_data(mode='diff', outputfile='diff_extinction_0_to_5000pc')
create_skymap_data(mode='cumul', outputfile='cumul_extinction_0_to_5000pc')

create_figure(mode='diff', input_npz='diff_extinction_0_to_5000pc.npz', outgif='anim_diff.gif', plot_title='Differential Extinction')
create_figure(mode='cumul', input_npz='cumul_extinction_0_to_5000pc.npz' outgif='anim_cumul.gif', plot_title="Integrated Extinction")

elapsed = time.time() - t
print("execution time", elapsed)