#2020_05_28

#Python 3.8

# Leonardo Fortaleza, upon initial work by Ségolène Brivet

# 3D plot of tumor and antennas location
# find distance between tumor location and projection on the line between
# each antenna pairs.

# Standard library imports
import itertools
import json
import operator
import os.path
import sys

# Third-party library imports
import jsonpickle
import matplotlib.pyplot as plt
#from matplotlib import interactive
#from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d import proj3d
#import mplcursors
import natsort
import numpy as np


#interactive(True)


def reverse_coord_list(coord_list):
    """Convert coordinates from [z,x,y] to [x,y,z].

    Parameters
    ----------
    coord_list : list
        list with coordinates in form [z,x,y]

    Returns
    -------
    new_coord_list : list
        list with coordinates in form [x,y,z]
    """
    return [[x,y,z] for [z,x,y] in coord_list]

def pairs_matrix(Settings_xyz, antenna_pairs):
    """Create list of antenna pairs with mean point coordinates.

    Each element of list consists of [Tx, Rx, x, y, z]

    Parameters
    ----------
    Settings_xyz : class
        settings class with antenna locations.
    antenna_pairs : list
        list of antenna pairs in form, each element in form [Tx, Rx]

    Returns
    -------
    pairs_positions: list
        list of antenna pairs with mean point coordinates, in form [Tx, Rx, x, y, z]
    """

    pairs_positions = []

    for (Tx,Rx) in antenna_pairs:
        Tx_p = Settings_xyz.antenna_locations[Tx - 1]
        Rx_p = Settings_xyz.antenna_locations[Rx - 1]
        x, y, z = (Rx_p + Tx_p)/2.0
        pairs_positions.append([Tx, Rx, x, y ,z])

    return pairs_positions

def displacement_cart(positions_list, reference_point):
    displacements = [[Tx, Rx, x - reference_point[0], y - reference_point[1], z - reference_point[2]] for (Tx, Rx, x, y, z) in positions_list]
    return displacements

def pairs_dict(Settings_xyz, antenna_pairs):
    """Create list of antenna pairs with mean point coordinates.

    Each element of list consists of [Tx, Rx, x, y, z]

    Parameters
    ----------
    Settings_xyz : class
        settings class with antenna locations.
    antenna_pairs : list
        list of antenna pairs in form, each element in form [Tx, Rx]

    Returns
    -------
    pairs_dict: dict
        dictionat of antenna pairs with mean point coordinates
    """

    pairs_dict = {"Tx": [], "Rx": [], "mean_positions": [], "antenna_locations": Settings_xyz.antenna_locations}

    for (Tx,Rx) in antenna_pairs:

        pairs_dict["Tx"].append(Tx)
        pairs_dict["Rx"].append(Rx)

        Tx_p = pairs_dict["antenna_locations"][Tx - 1]
        Rx_p = pairs_dict["antenna_locations"][Rx - 1]
        x, y, z = (Rx_p + Tx_p)/2.0
        pairs_dict["mean_positions"].append([x, y ,z])

    return pairs_dict

def calculate_displacement_cart(pairs_dict, reference_point):

    displacements = pairs_dict["mean_positions"] - reference_point
    return displacements

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def cart2sph(x,y,z):
    # For el in [-90,90] degrees
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arctan2(y,x)
    az = np.pi/2.0 - np.arctan2(np.sqrt(x**2 + y**2), z)
    return r, el, az

def sph2cart_list(positions):
    #az = positions[0]
    #el = positions[1]
    #r = positions[2]

    r,el,az = zip(*positions)
    r = np.array(r)
    el = np.array(el)
    az = np.array(az)

    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def cart2sph_list(positions):
    # For el in [-90,90] degrees
    #x = positions[0]
    #y = positions[1]
    #z = positions[2]

    x,y,z = zip(*positions)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arctan2(y,x)
    az = np.pi/2.0 - np.arctan2(np.sqrt(x**2 + y**2), z)
    return r, el, az

def sph2polar_projection(az, el, r, preserve="latitude"):

    if (preserve=="area"):
        rho = np.sqrt(2*(1 + np.sin(az)))
        ## rho <- spherical.to.polar.area(r[,"phi"])
    if (preserve=="angle"):
        ## rho = alpha*sqrt(2*(1+sin(phi))/(1-sin(phi)))
        rho = np.sqrt(2*(1+np.sin(az))/(1-np.sin(az)))
    if (preserve=="latitude"):
        rho = np.pi/2 + az
    x = rho*np.cos(el)
    y = rho*np.sin(el)
    return x, y

def azimuthal_equidistant_projection(az, el, r):
    rho = r*(np.pi/2 - az)
    x = rho*np.sin(el)
    y = -rho*np.cos(el)

    return x, y

def wrap_angles(angles):
    return ((-angles + np.pi) % (2.0 * np.pi) - np.pi) * -1.0

def spherical2lat_long(az,el):
    latitude = el
    longitude = wrap_angles(az - np.pi/2.0)

    return longitude, latitude

def AreBadValues(A):
    # Only returns true for bad array elements.
    # Bad array elements include Inf (infinity), NA (missing), and NaN (not a number).

    # Adapted from https://www.mathworks.com/matlabcentral/fileexchange/28848-spherical-to-azimuthal-equidistant

    areBadVals = not np.isfinite(A)
    # Inf, NA, and NaN are non-finite values.
    return areBadVals


def Spherical2AzimuthalEquidistant(latitudes, longitudes, centerLatitude, centerLongitude, centerX, centerY, radius = 1):
    # Projects latitudes and longitudes onto an azimuthal equidistant image.
    # Converts the given latitudes and longitudes into x and y values suitable for
    # plotting on an azimuthal equidistant projection centered on the given latitude
    # and longitude. centerX is the x-coordinate at which to plot the center point,
    # and centerY is the y-coordinate at which to plot the center point. radius is
    # the radius of the projection.

    # Adapted from https://www.mathworks.com/matlabcentral/fileexchange/28848-spherical-to-azimuthal-equidistant

    # Input Angles need to be in radians

    cosLats = np.cos(latitudes)
    sinLats = np.sin(latitudes)
    cosCLat = np.cos(centerLatitude)
    sinCLat = np.sin(centerLatitude)
    dLongs = longitudes - centerLongitude
    Js = cosLats*np.cos(dLongs)
    cosCs = (sinCLat * sinLats) + (cosCLat * Js)
    Cs = np.arccos(cosCs)
    # C = the distance of the projection from the center of the plot on a scale
    # from 0 to pi. Mathematically, C = realsqrt((x .^ 2) + (y .^ 2)).

    sinCs = np.sqrt(1 - (cosCs**2))
    Ks = np.true_divide(Cs,sinCs)
    # The cases in which sin(C) = 0 will be handled later in the function.
    # The values calculated so far are simply intermediary expressions in the
    # computation.
    unscaledXs = Ks * (cosLats * np.sin(dLongs))
    unscaledYs = Ks * ((cosCLat * sinLats) - (sinCLat * Js))
    # Produces values scaled from -pi to +pi, with x = 0, y = 0 in the middle of
    # the projection.
    specialCases = np.argwhere(AreBadValues(Ks))
    # Collect indices of cases for which division by 0 occurred.
    centers = specialCases[Cs[specialCases] < (np.pi/2.0)]
    # Collect indices of special cases for which the point is closer to the
    # center of the projection than the rim.
    unscaledXs[centers] = 0
    unscaledYs[centers] = 0
    # These indices correspond to the special case in which the point to plot is
    # at the center of the projection, x = 0, y = 0.
    antipodes = np.setdiff1d(specialCases, centers)
    # Collect the indices of special cases for which the point is closer to the
    # rim of the projection than the center. By definition, these are all the
    # remaining special cases.
    unscaledXs[antipodes] = np.pi
    unscaledYs[antipodes] = 0
    # These indices correspond to the case in which the point to plot the
    # antipode of the point at the center of the projection. The azimuthal
    # equidistant projection does not project the antipode to any single point,
    # but rather the circle of radius pi that comprises the outer rim of the
    # projection. The right-most point on this circle is arbitrarily declared
    # to be the anitpode. Note that this is not necessarily consistent with the
    # arbitrary convention in the function GeodesicMidpoints for selecting a
    # geodesic midpoint between a point and its antipode.
    scalingFactor = radius / np.pi
    # Determine the factor by which to stretch the projection, which has a
    # radius of pi by default.
    Xs = (scalingFactor * unscaledXs) + centerX
    Ys = (scalingFactor * unscaledYs) + centerY
    # Scales the projection as the user requested.

    return Xs, Ys


class Settings_xyz:
    def __init__(self, narrowband = True, reference_rotation = 15):
        # Set narrowband to True to switch positions of antennas 4 and 13

        # We consider phantoms 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14  (not 4)

        # Number of sensors
        self.nSensor = 16
        # Boundaries of the grid (x,y,z)
        self.grid_lower = [-7e-2, -7e-2, 0]
        self.grid_upper = [7e-2, 7e-2, 7e-2]
        # Width of each voxel
        self.voxel_width = 2e-3
        # Dimensions of the breast (the radius of the region for imaging)
        self.breast_dimensions = [0.07,0.07,0.07]
        # Time step and number of samples (before and after downsampling and windowing)
        self.dt = 6.25e-12
        self.nSample_original = 4096
        self.nSample = 1000
        # Indicator of any channels or antennas to exclude
        self.excluded_antennas = []
        self.excluded_channels = []
        # If settings.align = 1, then align the data by shifting the baseline signals in time
        self.align = 1
        # maxLag indicates the maximum delay in samples that is allowed
        self.maxLag = 10

        # If there is a tumour size to test for include non-zero radius for delay calculations [OPTIONAL]
        self.tumour_radius = 10

        # Box_dimensions should be expressed in m (e.g. a 1 cm box is [0.01,0.01,0.01])
        self.box_dimensions = [0.03, 0.03, 0.03]

        # Coordinate system: looking at the phantom from the bottom up (looking at the nipple), the first four antennas are lying
        # on the positive part of axis y, axis x goes through the nipple and axis z goes through antennas P5, P6, P12, P11
        # The counting goes clockwise, so PHI goes backwards from 0 to -270 degrees.
        # PHI = (0:-pi/2:-(2*pi-pi/2))'; % azimuth angles (0, 90, 180, 270)
        # THETA = ((1:4)*pi/2/5)';   % elevation angles (18, 36, 54, 72)
        # SENSOR_RADIUS = 0.0730;
        # self.antenna_locations = SENSOR_RADIUS*[cos(antenna_THETA).*cos(antenna_PHI) cos(antenna_THETA).*sin(antenna_PHI) sin(antenna_THETA)];

        self.reference_rotation = reference_rotation
        ant_r = 70

        if narrowband:
            ant_theta = np.deg2rad([15, 15, 15, 45, 15, 15, 15, 15, 45, 45, 45, 45, 15, 45, 45, 45])
            ant_phi = np.deg2rad([-15, 15, 75, 165, 165, -165, -105, -75, -15, 15, 75, 105, 105, -165, -105, -75]) + np.deg2rad(self.reference_rotation)
        else:
            ant_theta = np.deg2rad([15, 15, 15, 15, 15, 15, 15, 15, 45, 45, 45, 45, 45, 45, 45, 45])
            ant_phi = np.deg2rad([-15, 15, 75, 105, 165, -165, -105, -75, -15, 15, 75, 105, 165, -165, -105, -75]) + np.deg2rad(self.reference_rotation)

        antenna_locations = []
        for i in range(len(ant_theta)):
            [x, y, z] = sph2cart(ant_phi[i], ant_theta[i], ant_r)
            antenna_locations.append([x, y, z])

        self.antenna_locations = [np.multiply(d,[1e-3]*3) for d in antenna_locations]

        self.index = None
        self.permittivity = None
        self.permeability = 1
        self.tumour_location = None

        self.antenna_pairs = itertools.permutations(range(1,15), 2)

    def setPhantomNb(self, phantom_nb, angle = 0):
        # Set the permittivity and permeability to estimated average values
        # perm_old = [8.3, 7.5, 8.3, 6, 6.3, 8.3, 16, 17, 17]
        # Average permittivity with centre at 3 GHz
        perm_new = [1, 8, 8, 11, 17.92, 21.44, 12.93, 15.14, 10.03, 11.85, 14.38, 16.8, 10.24, 9.36]

        # We consider phantoms 0 (air), 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14  (not 4)
        if phantom_nb == 4:
            print("No phantom nb 4, please use any of these numbers:")
            print("0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14  (not 4)")
            sys.exit()
        self.index = phantom_nb if phantom_nb <= 3 else phantom_nb - 1

        self.permittivity = perm_new[self.index]
        self.permeability = 1

        tum_r = [0, 43, 46, 42, 46, 50, 39, 34, 46, 40, 39, 50, 45, 42]
        tum_theta = np.deg2rad([0, 50, 40, 45, 41, 45, 40, 40, 50, 30, 40, 45, 48, 45])
        # Angles are equivalent to phantom angle/rotation 45 degrees
        tum_phi = np.deg2rad([0, 255, 150, 335, 125, 135, 180, 210, 80, 120, 150, 85, 125, 270]) - np.deg2rad(angle - 45) + np.deg2rad(self.reference_rotation)

        [x, y, z] = sph2cart(tum_phi[self.index], tum_theta[self.index], tum_r[self.index])
        tumor_location = [x, y, z]
        self.tumour_location = np.multiply(tumor_location, [1e-3]*3)

    def selectAntennaPairs(self, antenna_pairs = "all"):
        if isinstance(antenna_pairs, str):
            if antenna_pairs.lower() == "all":
                self.antenna_pairs = list(itertools.permutations(np.arange(1,17), 2))
            else:
                print("Invalid antenna pairs list. Please provide list of pairs or the string 'all'.")
                sys.exit()
        else:
            self.antenna_pairs = antenna_pairs

    def calc_antenna_pairs_positions(self, antenna_pairs = "all", force_pairs = False):
        if (not hasattr(self, 'antenna_pairs')) or force_pairs:
            self.selectAntennaPairs(antenna_pairs)

        pairs_mean_positions = []
        for (Tx,Rx) in self.antenna_pairs:
            Tx_p = self.antenna_locations[Tx - 1]
            Rx_p = self.antenna_locations[Rx - 1]
            pairs_mean_positions.append((Rx_p + Tx_p)/2.0)
        self.pairs_mean_positions = pairs_mean_positions

    def calc_antenna_pairs_displacements(self, reference_point = [0,0,0], antenna_pairs = "all", force_pairs = False):
        if (not hasattr(self, 'antenna_pairs')) or force_pairs:
            self.selectAntennaPairs(antenna_pairs)

        pairs_displacements = []
        for (Tx,Rx) in self.antenna_pairs:
            Tx_p = self.antenna_locations[Tx - 1]
            Rx_p = self.antenna_locations[Rx - 1]
            pairs_displacements.append([Tx_p, Rx_p])
        pairs_displacements = np.array(pairs_displacements) - reference_point
        self.pairs_displacements = pairs_displacements

    def calc_antenna_pairs_between_distances(self, antenna_pairs = "all", out = False, force_pairs = False):
        if (not hasattr(self, 'antenna_pairs')) or force_pairs:
            self.selectAntennaPairs(antenna_pairs)

        pairs_displacements = []
        for (Tx,Rx) in self.antenna_pairs:
            Tx_p = self.antenna_locations[Tx - 1]
            Rx_p = self.antenna_locations[Rx - 1]
            pairs_displacements.append([Tx_p, Rx_p])
        pairs_displacements = np.array(pairs_displacements)
        pairs_distances = np.linalg.norm(pairs_displacements[:,0] - pairs_displacements[:,1], axis=1)

        if out == False:
            self.pairs_distances = pairs_distances
        else:
            return pairs_distances

    def calc_antenna_distances(self, point):
        point_distance = np.linalg.norm((self.antenna_locations - point), axis=1)
        distances = np.zeros((16, 16))
        for tx, rx in self.antenna_pairs:
            if tx < rx:
                distances[tx - 1][rx - 1] = point_distance[tx - 1] + point_distance[rx - 1]
                distances[rx - 1][tx - 1] = distances[tx - 1][rx - 1]
            elif tx == rx:
                distances[tx - 1][rx - 1] = 2*point_distance[tx - 1]
        return distances

    def calc_distances_to_tumour(self):
        distances_to_tumour = self.calc_antenna_distances(self.tumour_location)
        self.distances_to_tumour = distances_to_tumour
        #return distances_to_tumour


    def to_json(self, filename):
        """Save Settings object to JSON file.

        Uses jsonpickle to encode.

        Parameters
        ----------
        filename : str
            file name and path
        """
        frozen = jsonpickle.encode(self)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'w') as fp:
            json.dump(frozen, fp, sort_keys=True, indent=4)

    def from_json(self, filename):
        """Load Settings object from JSON file.

        Uses jsonpickle to decode.

        Parameters
        ----------
        filename : str
            file name and path
        """
        with open(filename, 'w') as fp:
            frozen = json.load(fp)
        thawed = jsonpickle.decode(frozen)
        self = thawed


class Settings:
    def __init__(self, narrowband = True):
        # Set narrowband to True to switch positions of antennas 4 and 13

        # We consider phantoms 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14  (not 4)

        # Number of sensors
        self.nSensor = 16
        # Boundaries of the grid (z,x,y)
        self.grid_lower = [0, -7e-2, -7e-2]
        self.grid_upper = [7e-2, 7e-2, 7e-2]
        # Width of each voxel
        self.voxel_width = 2e-3
        # Dimensions of the breast (the radius of the region for imaging)
        self.breast_dimensions = [0.07,0.07,0.07]
        # Time step and number of samples (before and after downsampling and windowing)
        self.dt = 6.25e-12
        self.nSample_original = 4096
        self.nSample = 1000
        # Indicator of any channels or antennas to exclude
        self.excluded_antennas = []
        self.excluded_channels = []
        # If settings.align = 1, then align the data by shifting the baseline signals in time
        self.align = 1
        # maxLag indicates the maximum delay in samples that is allowed
        self.maxLag = 10

        # If there is a tumour size to test for include non-zero radius for delay calculations [OPTIONAL]
        self.tumour_radius = 10

        # Box_dimensions should be expressed in m (e.g. a 1 cm box is [0.01,0.01,0.01])
        self.box_dimensions = [0.03, 0.03, 0.03]

        # Coordinate system: looking at the phantom from the bottom up (looking at the nipple), the first four antennas are lying
        # on the positive part of axis y, axis x goes through the nipple and axis z goes through antennas P5, P6, P12, P11
        # The counting goes clockwise, so PHI goes backwards from 0 to -270 degrees.
        # PHI = (0:-pi/2:-(2*pi-pi/2))'; % azimuth angles (0, 90, 180, 270)
        # THETA = ((1:4)*pi/2/5)';   % elevation angles (18, 36, 54, 72)
        # SENSOR_RADIUS = 0.0730;
        # self.antenna_locations = SENSOR_RADIUS*[cos(antenna_THETA).*cos(antenna_PHI) cos(antenna_THETA).*sin(antenna_PHI) sin(antenna_THETA)];

        ant_r = 70

        if narrowband:
            ant_theta = np.deg2rad([15, 15, 15, 45, 15, 15, 15, 15, 45, 45, 45, 45, 15, 45, 45, 45])
            ant_phi = np.deg2rad([-15, 15, 75, 165, 165, -165, -105, -75, -15, 15, 75, 105, 105, -165, -105, -75])
        else:
            ant_theta = np.deg2rad([15, 15, 15, 15, 15, 15, 15, 15, 45, 45, 45, 45, 45, 45, 45, 45])
            ant_phi = np.deg2rad([-15, 15, 75, 105, 165, -165, -105, -75, -15, 15, 75, 105, 165, -165, -105, -75])

        antenna_locations = []
        for i in range(len(ant_theta)):
            [x, y, z] = sph2cart(ant_phi[i], ant_theta[i], ant_r)
            antenna_locations.append([z, x, y])

        self.antenna_locations = [np.multiply(d,[1e-3]*3) for d in antenna_locations]

        self.index = None
        self.permittivity = None
        self.permeability = 1
        self.tumour_location = None

    def setPhantomNb(self, phantom_nb, angle = 0):
        # Set the permittivity and permeability to estimated average values
        # perm_old = [8.3, 7.5, 8.3, 6, 6.3, 8.3, 16, 17, 17]
        perm_new = [8, 8, 11, 17.92, 21.44, 12.93, 15.14, 10.03, 11.85, 14.38, 16.8, 10.24, 9.36]

        # We consider phantoms 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14  (not 4)
        if phantom_nb == 4:
            print("No phantom nb 4, please use any of these numbers:")
            print("1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14  (not 4)")
            sys.exit()
        self.index = phantom_nb-1 if phantom_nb <= 3 else phantom_nb - 2

        self.permittivity = perm_new[self.index]
        self.permeability = 1

        tum_r = [43, 46, 42, 46, 50, 39, 34, 46, 40, 39, 50, 45, 42]
        tum_theta = np.deg2rad([50, 40, 45, 41, 45, 40, 40, 50, 30, 40, 45, 48, 45])
        # Angles are equivalent to phantom angle/rotation 45 degrees
        tum_phi = np.deg2rad([255, 150, 335, 125, 135, 180, 210, 80, 120, 150, 85, 125, 270] - angle - 45)
        index = phantom_nb - 1 if phantom_nb <= 3 else phantom_nb - 2

        [x, y, z] = sph2cart(tum_phi[index], tum_theta[index], tum_r[index])
        tumor_location = [z, x, y]
        self.tumour_location = np.multiply(tumor_location, [1e-3]*3)

"""
    def getTumorLoc(self, phantom_nb):
        self.setPhantomNb(phantom_nb)
        tum_loc = self.tumour_location

        ax = plt.gca()
        ax.scatter(tum_loc[2], tum_loc[1], tum_loc[0], c='r', marker='o', s=120, edgecolors='k', alpha=0.7)
        ax.text(tum_loc[2], tum_loc[1], tum_loc[0] + 0.003, "Tumor",
                horizontalalignment='left',
                verticalalignment='center')
        plt.show(block=True)

        return tum_loc

def plot_antennas(obj):
    fig = plt.figure(figsize=(17, 9))
    x = [obj.antenna_locations[i][0] for i in range(len(obj.antenna_locations))]
    y = [obj.antenna_locations[i][1] for i in range(len(obj.antenna_locations))]
    z = [obj.antenna_locations[i][2] for i in range(len(obj.antenna_locations))]
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z, y, x, c='b', marker='o', s=80)

    for i, ant in enumerate(obj.antenna_locations):
        posx, posy, posz = ant
        ax.text(posz + 0.001, posy + 0.001, posx + 0.005, "%d" % (i + 1),
                horizontalalignment='center',
                verticalalignment='center')
    ax.set_xlim(-0.07, 0.07)
    #ax.set_ylim(-0.07, 0.07)
    ax.set_ylim(0.07, -0.07)
    ax.set_zlim(0, 0.06)
    tum_loc = obj.tumour_location
    ax.scatter(tum_loc[2], tum_loc[1], tum_loc[0], c='r', marker='o', s=120, edgecolors='k', alpha=0.7)

    #for i1, ant1 in enumerate(obj.antenna_locations):
    #    for i2, ant2 in enumerate(obj.antenna_locations):
    #        if i2 > i1:
    #            ax.plot3D([ant1[2], ant2[2]], [ant1[1], ant2[1]], [ant1[0], ant2[0]],
    #                     label="%d - %d" % (i1 + 1, i2 + 1))

    onHover = mplcursors.cursor(hover=True)

    @onHover.connect("add")
    def _(sel):
        sel.annotation.set_text(sel.artist.get_label())
        sel.annotation.get_bbox_patch().set(fc="white")

    onClick = mplcursors.cursor(multiple=True)

    @onClick.connect("add")
    def _(sel):
        sel.extras.append(onClick.add_highlight(sel.artist))
        sel.annotation.set_text(sel.artist.get_label())
        sel.annotation.get_bbox_patch().set(fc="white")

    plt.show()
"""

def spherical_distances(position, center = np.array([0,0,0])):

    r, el, az = cart2sph_list(position - center)

    return r, el, az

def distances_for_sorting(Settings_xyz, center = [0,0,0], decimals = 4):

    antenna_pairs = Settings_xyz.antenna_pairs

    #distances_matrix = Settings_xyz.distances_to_tumour()
    distances_matrix = Settings_xyz.calc_antenna_pairs_displacements(reference_point = center)

    distances = [distances_matrix[x-1][y-1] for x,y in antenna_pairs]

    positions = Settings_xyz.pairs_mean_positions

    r, el, az = spherical_distances(positions, center = center)

    sort_key = np.around(list(zip(distances, r, el, az)), decimals = decimals)

    return sort_key

def sort_antenna_pairs_by_distances_to_point(Settings_xyz, reference_point = [0,0,0], sort_type = "distance", decimals = 4):

    antenna_pairs = Settings_xyz.antenna_pairs

    distances_matrix = Settings_xyz.distances_to_tumour

    distances = [distances_matrix[x-1][y-1] for x,y in antenna_pairs]

    positions = Settings_xyz.pairs_mean_positions

    r, el, az = spherical_distances(positions, center = reference_point)

    if sort_type.lower() == "mean-point":
        sort_key = np.around(list(zip(r, el, az)), decimals = decimals)
    else:
        sort_key = np.around(list(zip(distances, r, el, az)), decimals = decimals)

    idx = natsort.index_natsorted(sort_key)

    sorted_pairs = np.array(Settings_xyz.antenna_pairs)[idx,:]

    sorted_values = np.array(sort_key)[idx,:]

    return sorted_pairs, sorted_values

def sort_antenna_pairs_by_distances_to_tumour(Settings_xyz, sort_type = "distance", decimals = 4):

    antenna_pairs = Settings_xyz.antenna_pairs

    distances_matrix = Settings_xyz.distances_to_tumour

    distances = [distances_matrix[x-1][y-1] for x,y in antenna_pairs]

    positions = Settings_xyz.pairs_mean_positions

    r, el, az = spherical_distances(positions, center = Settings_xyz.tumour_location)

    if sort_type.lower() == "mean-point":
        sort_key = np.around(list(zip(r, el, az)), decimals = decimals)
    else:
        sort_key = np.around(list(zip(distances, r, el, az)), decimals = decimals)

    idx = natsort.index_natsorted(sort_key)

    sorted_pairs = np.array(Settings_xyz.antenna_pairs)[idx,:]

    sorted_values = np.array(sort_key)[idx,:]

    return sorted_pairs, sorted_values

def sort_antenna_pairs_by_direct_distances(Settings_xyz, decimals = 4):

    antenna_pairs = Settings_xyz.antenna_pairs

    Settings_xyz.calc_antenna_pairs_between_distances()

    distances = Settings_xyz.pairs_distances

    #distances = [distances[x-1][y-1] for x,y in antenna_pairs]

    positions = Settings_xyz.pairs_mean_positions

    r, el, az = spherical_distances(positions, center = Settings_xyz.tumour_location)

    #sort_key = list(zip(np.around(distances, decimals = decimals), antenna_pairs))
    sort_key = np.around(list(zip(distances, r, el, az)), decimals = decimals)

    idx = natsort.index_natsorted(sort_key)

    sorted_pairs = np.array(Settings_xyz.antenna_pairs)[idx,:]

    sorted_values = np.array(sort_key)[idx,:]

    return sorted_pairs, sorted_values

def extract_reciprocals(pairs):
    """ Splits list of pairs into filtered list without reciprocals, list of reciprocal pairs [(b,a) only] and list of pairs that had reciprocals [(a,b) only].

    Parameters
    ----------
    pairs: list of tuples
        input list of pairs

    Returns
    ----------
    pairs_filt: list of tuples
        list filtered to remove reciprocals only include pairs in the form (a,b)
    recs: list of tuples
        list only including reciprocal pairs (b,a)
    pairs_twice: list of tuples
        list including pairs (a,b) only when the reciprocal (b,a) was present in original list
    """
    recs = []
    pairs_filt = pairs
    pairs_twice = []

    for p in pairs:
        r = (p[1],p[0])
        if (p in pairs_filt) and (r in pairs):
            recs.append(r)
            pairs_filt.remove(r)
            pairs_twice.append(p)

    return pairs_filt, recs, pairs_twice

def sort_pairs(phantom, angle = 0, selected_pairs = "all", reference_point = "tumor", sort_type = "distance", decimals = 4, out_distances=False):
    """Sort antenna pairs by distance to reference point for specific phantom and angle.

    If a column name is not found (KeyError), displays a message and continues with other columns.

    Parameters
    ----------
    phantom : int
        phantom number
    angle : int, optional
        position angle in degrees, by default 0
    selected_pairs: list of tuples or str, optional
        list of pairs to sort or "all" for all 240 possible pairs, by default "all"
    reference_point: str or array-like, default "tumor"
        input "tumor", "tumour" or "plug" to calculate based on position of plug inside phantom, otherwise receives a 3-D coordinate in form [x,y,z]
    sort_type : str, default "distance"
       available selections:
            "distance": calculate minimum travel distance between each antenna in a pair and the reference point
            "mean-point": calculate mean-point between antennas in a pair and its distance to the reference point
            "between_antennas": direct distance between antennas, disregarding the reference point


    decimals : int, default 4
        number of decimals for rounding
    out_distances: bool, optional
        set to True to provide optional return list with distances, by default False

    Returns
    ----------
    sorted_pairs: list of str
        list with DataFrames split by phantom, angle with the "pair" column sorted by distances
    distances: list of float, optional
        list with distances for each pair in sorted_pairs, only provided if out_distances=True
    """
    obj = Settings_xyz()

    obj.setPhantomNb(phantom, angle)

    # check if selected_pairs is list of tuples or "all", otherwise assumes list of strings and converts to list of tuples

    if (selected_pairs == "all") or (all(isinstance(item, tuple) for item in selected_pairs)):
        pass
    else:
        selected_pairs = [tuple(int(x.lstrip('(').lstrip('[').lstrip('{').rstrip(')').rstrip(']').rstrip('}')) for x in el.split(',')) for el in selected_pairs]

    obj.selectAntennaPairs(antenna_pairs = selected_pairs)

    obj.calc_antenna_pairs_positions(antenna_pairs = selected_pairs)

    obj.calc_distances_to_tumour()

    if sort_type.lower() == "between_antennas":
        sorted_pairs, distances = sort_antenna_pairs_by_direct_distances(obj, decimals = decimals)
        if out_distances:
            return sorted_pairs, distances[:,0]
        else:
            return sorted_pairs

    if reference_point.lower() == "tumor" or "tumour" or "plug":
        sorted_pairs, distances = sort_antenna_pairs_by_distances_to_tumour(obj, sort_type = sort_type, decimals = decimals)
    else:
        sorted_pairs, distances = sort_antenna_pairs_by_distances_to_point(obj, reference_point = reference_point, sort_type = sort_type, decimals = decimals)

    if out_distances:
        distances = [d for d in distances[:,0]]
        return sorted_pairs, distances
    else:
        return sorted_pairs

if __name__ == "__main__":
    #obj = Settings()
    #plot_antennas(obj)

    # Specify tumor number (=phantom number) here
    # We consider phantoms nb 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14  (not 4)
    #obj.getTumorLoc(1)

    #obj.setPhantomNb(1)

    #antenna_locations = reverse_coord_list(obj.antenna_locations)
    #tumour_locations = reverse_coord_list(obj.tumour_locations)


    # 36 pairs - Configuration A

    #pairs = [(1,2), (2,3), (3,5), (5,6), (7,8), (9,10), (10,11), (11,12), (14,15), (15,16),
    #			(1,6), (2,5), (4,10), (4,12), (4,14), (9,14), (9,16)
    #            ]
    #pairs_rev = [tuple(reversed(t)) for t in pairs]
    #extra_pairs = [(3, 13), (5,13)]

    #pairs.extend(pairs_rev)
    #pairs.extend(extra_pairs)

    # 56 pairs - Configuration B

    pairs = [(1,6), (1,7), (1,8), (1,9), (1,14), (1,15), (1,16), (6,7), (6,8), (6,9), (6,14), (6,15), (6,16), (7,8), (7,9), (7,14), (7,15), (7,16),
            (8,9), (8,14), (8,15), (8,16), (9,14), (9,15), (9,16), (14,15), (14,16), (15,16)
                ]
    pairs_rev = [tuple(reversed(t)) for t in pairs]
    pairs.extend(pairs_rev)

    #pairs = 'all'

    obj = Settings_xyz()

    obj.setPhantomNb(13, angle = 0)

    print("Tumor Location:\n")
    print(obj.tumour_location)

    obj.selectAntennaPairs(antenna_pairs = pairs)

    obj.calc_antenna_pairs_positions(antenna_pairs = pairs)

    obj.calc_distances_to_tumour()

    #print obj.antenna_pairs
    #print obj.distances_to_tumour

    #sorted_pairs, sorted_values = sort_antenna_pairs_by_distances_to_tumour(obj, sort_type="distance")
    #sorted_pairs, sorted_values = sort_antenna_pairs_by_distances_to_tumour(obj, sort_type="mean-point")
    sorted_pairs, sorted_values = sort_antenna_pairs_by_direct_distances(obj, decimals = 4)

    print("Sorted Pairs: ","(",sorted_pairs.shape[0],")")
    print(sorted_pairs)
    print("\n", "Distance Values:","(",sorted_values.shape[0],")")
    print(sorted_values)

    #plot_antennas(obj)

    print("End")
