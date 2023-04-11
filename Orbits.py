#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:29:02 2020

@author: ZhexingLi
"""

'''
Note: Orbit plot employs the convention where the longitude of ascending node is measured
from the celestial north in the direction of increasing position angle, i.e. counterclockwise
on the sky with North in the up direction and East in the left direction. Ascending node in 
this convention is defined as when the orbiting body crosses the plane of the sky moving AWAY 
from the observer (Earth). Orbit is face on if inclination (i) is 0 degree, and edge on if i 
is 90 degrees. If inclination if between 0 and 90 degrees, the object orbits in couterclockwise 
direction (prograde), if inclination is between 90 and 180 degrees, the objects orbits in 
clockwise direction (retrograde).
'''

###########################################################################################
# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
from astropy import constants as const
from matplotlib.patches import Circle,Wedge
import matplotlib.colors as colors
import pandas as pd

############################################################################################
############################################################################################
# Modify parameters, switches, options etc. in this section

# input orbital parameters
a = [1.34]   # semi-major axis (in AU)
e = [0.]   # eccentricity
w = [np.pi/2]   # argument of periastron of PLANETS (in radians)
Omega = [3*np.pi/2]   # longitude of ascending node (in radians)
radius = [0.09]     # radius of the planet (in jupiter radius)
albedo = [0.3]          # geometric albedo
inclination = [1.57]     # inclination, in radians
distance = 20.34             # distance to the system, in pc
threshold = 4e-11          # flux ratio threshold, color code only if above this
# cgi: 0.14,1.4; starshade: 0.1,5 (72 mas for 425-552 nm, 104 mas for 615-800 for SRP, 70 ms for HabEx SS)
FOV = [0.070,5]            # FOV of the instrument, [IWA,OWA], in arcsec, if don't need it, leave empty, if owa not provided, default 5
letter = ['b']   # label for planets
style = ['dotted','dashed']       # linestyles for different orbits.
system = 'test'
view = ['topdown','sky']        # viewing orientation, 'topdown' or 'sky' or both

# If have occulter transmittance file, turn it on, file format: two columns, distance (mas) and tau
# values, first row will be skipped for header info
transmittance = 'Yes'        # if there's a file for tau, put 'Yes', if no then 'No'
tau_file = 'HabExSS_tau.txt'      # file name only, file should be in the path specified below

# Habitable zone boundaries option, if turned on (HabZone = 'Yes'), HZ boundaries will be plotted on the 
# top down view only, if provided Teff and Rs (or Ms and gs) of the star.
HabZone = 'Yes'             # 'Yes' or 'No', only need Teff and (Rs or (Ms and gs))
case = 1            # planetary mass for HZ calculation; 1: 1 earth mass, 2: 5 earth mass, 3: 0.1 earth mass
Teff = 5429             # effective temperature of the star, in Kelvins
Rs = 1.005            # radius of the star, in solar radii, if unknown, put 'NaN'
Ms = 1.0                # stellar mass in solar masses, put value for this if Rs is unknown
gs = 4.41               # stellar surface gravity in log10(cm/s^2), put value for this if Rs is unkown

# Turn 'coordinates' option on to print sky x,y coordinates during the specified time period, display 
# window of observation, add plot position with time on flux ratio plot
coordinates = 'No'             # put 'Yes' if want to utilize parameters below, otherwise 'No'
period = [8241]            # period in days
tp = [2450812.0]      # time of periastron in JD
time = np.arange(2455071,2455571,100)   # observing time stamps to get positions
timetype = 'range'              # indicate the type of time give above, 'range' or 'specific'
plotpos = 'Yes'       # if plot actual positions on top of orbits, 'Yes' or 'No'

# Turn this on or off to save plots for the functionalities that are turned on
save = 'No'

# path for input files and data products
path = '/Users/zhexingli/Desktop/UCR/01-RESEARCH/PROJECTS/MISC/HD83443/'
savepath = '/Users/zhexingli/Desktop/UCR/01-RESEARCH/PROJECTS/MISC/'
#plt.style.use('default')

############################################################################################
############################################################################################
if len(FOV) == 0:         # extra check just in case wrong input is provided when no iwa and owa are given
    transmittance = 'No'
else:
    pass

def main():
    # Main function to create all the stuff in the plot depending on the viewing angle
    #
    # Outputs:
    #   - Final plot products
    
    if len(view) == 1:
        # only need one orbit plot
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel('AU')
        ax.set_ylabel('AU')
        
        inner = 0
        outer = 0
        science = 0
        
        # get coordinates and flux values
        x_all,y_all,x_all2,y_all2,flux_all,flux_all2,orbital_phase,phi_angle = gather()
        
        if view[0] == 'topdown':
            HZ(ax)
            plot_orbit(x_all,y_all,flux_all2,fig,ax)
        elif view[0] == 'sky':
            inner,outer,science = WA(ax)
            plot_orbit(x_all2,y_all2,flux_all2,fig,ax)
        else:
            print('Please provide a valid viewing orientation. Either \'topdown\', or \'sky\'.')
            exit()
            
        # actual sky coordinates for all planets' locations
        if coordinates == 'Yes':
            coord(ax)
        else:
            pass
        
        # plot window of visibility in orbital phases
        if view[0] == 'sky':
            window(inner,outer,science,x_all2,y_all2,flux_all,flux_all2,orbital_phase)
        else:
            pass
        
        # set plot size (save space, not showing owa)
        sidex = np.max(x_all) - np.min(x_all)
        sidey = np.max(y_all) - np.min(y_all)
        xposlim = 0
        xneglim = 0
        yposlim = 0
        yneglim = 0
        if sidex >= sidey:
            xposlim = np.max(x_all)+2
            xneglim = np.min(x_all)-2
            ax.set_xlim(xneglim,xposlim)
            y_extra = (sidex - sidey)/2
            yposlim = np.max(y_all)+y_extra+1
            yneglim = np.min(y_all)-y_extra-1
            ax.set_ylim(yneglim,yposlim)
        elif sidex < sidey:
            yposlim = np.max(y_all)+2
            yneglim = np.min(y_all)-2
            ax.set_ylim(yneglim,yposlim)
            x_extra = (sidey - sidex)/2
            xposlim = np.max(x_all)+x_extra+2
            xneglim = np.min(x_all)-x_extra-2
            ax.set_xlim(xneglim,xposlim)
        else:
            pass
            
        ax.plot(0,0,'r*',label='Star')
        ax.legend(loc='best',fontsize='small')
        if view[0] == 'topdown':
            ax.set_title(str(system) + ' Top Down View' ,loc='center')
        elif view[0] == 'sky':
            ax.set_title(str(system) + ' Sky View' ,loc='center')
            
            # add arrows to indicate directionality (north up, east left)
            xlen = xposlim - xneglim                # range of x and y axes
            ylen = yposlim - yneglim
            
            arr_basex = xposlim - 0.1*xlen               # location of base of the arrows
            arr_basey = yposlim - 0.2*ylen
            arr_xdelta = xlen*0.1             # length of horizontal arrow  
            arr_ydelta = ylen*0.1            # length of vertical arrow
            size = xlen*0.02               # size of the arrow head
            ax.arrow(arr_basex,arr_basey,0,arr_ydelta,shape='full',head_width=size, \
                      head_length=size)             # vertical arrow
            ax.arrow(arr_basex,arr_basey,-arr_xdelta,0,shape='full',head_width=size, \
                      head_length=size)             # horizontal arrow
            textsize = xlen*0.3
            vtextx = arr_basex - xlen*0.02          # vertical arrow text location
            vtexty = arr_basey + arr_ydelta + size + ylen*(textsize*0.001)    
            htextx = arr_basex - arr_xdelta - size - xlen*(textsize*0.003)     # horizontal arrow text location
            htexty = arr_basey - ylen*0.02
            ax.text(vtextx, vtexty, 'N',fontsize=textsize)              # vertical arrow text
            ax.text(htextx, htexty, 'E',fontsize=textsize)              # horizontal arrow text
        else:
            pass
        
        if save == 'Yes':
            plt.savefig(savepath + 'Orbits_SS_' + str(view[0]) + '.png',dpi=1000)
        else:
            pass
    
    
    elif len(view) == 2:
        # need two separate orbit plot
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        ax1.set_xlabel('Separation (AU)')
        ax1.set_ylabel('Separation (AU)')
        
        fig2, ax2 = plt.subplots()
        ax2.set_aspect('equal')
        ax2.set_xlabel('Separation (AU)')
        ax2.set_ylabel('Separation (AU)')
        
        inner = 0
        outer = 0
        science = 0
        
        # get coordinates and flux values
        x_all,y_all,x_all2,y_all2,flux_all,flux_all2,orbital_phase,phi_angle = gather()
        
        if ('topdown' in view) and ('sky' in view):
            HZ(ax1)
            plot_orbit(x_all,y_all,flux_all2,fig1,ax1)             # topdown plot
            inner,outer,science = WA(ax2)           # plot iwa and owa
            plot_orbit(x_all2,y_all2,flux_all2,fig2,ax2)           # sky plot
        else:
            print('Please provide a valid viewing orientation. Either \'topdown\', or \'sky\'.')
            exit()
            
        # actual sky coordinates for all planets' locations
        if coordinates == 'Yes':
            coord(ax1,ax2)
        else:
            pass
        
        # plot window of visibility in orbital phases
        window(inner,outer,science,x_all2,y_all2,flux_all,flux_all2,orbital_phase)
        
        # set plot size (save space, not showing owa)
        sidex1 = np.max(x_all) - np.min(x_all)
        sidey1 = np.max(y_all) - np.min(y_all)
        sidex2 = np.max(x_all2) - np.min(x_all2)
        sidey2 = np.max(y_all2) - np.min(y_all2)
        
        side = {'a':sidex1,'b':sidey1,'c':sidex2,'d':sidey2}
        maxside = max(side,key=side.get)
        xskyposlim = 0             # maximum positive x axis value
        xskyneglim = 0              # minimum negative x axis value
        yskyposlim = 0             # maximum positive y axis value
        yskyneglim = 0             # niminimum negative y axis value
        if maxside == 'a':
            ax1.set_xlim(np.min(x_all)-2,np.max(x_all)+2)
            y1_extra = (sidex1 - sidey1)/2
            ax1.set_ylim(np.min(y_all)-y1_extra-2,np.max(y_all)+y1_extra+2)
            x2_extra = (sidex1 - sidex2)/2
            xskyposlim = np.max(x_all2)+x2_extra+2
            xskyneglim = np.min(x_all2)-x2_extra-2
            ax2.set_xlim(xskyneglim,xskyposlim)
            y2_extra = (sidex1 - sidey2)/2
            yskyposlim = np.max(y_all2)+y2_extra+2
            yskyneglim = np.min(y_all2)-y2_extra-2
            ax2.set_ylim(yskyneglim,yskyposlim)
        elif maxside == 'b':
            ax1.set_ylim(np.min(y_all)-2,np.max(y_all)+2)
            x1_extra = (sidey1 - sidex1)/2
            ax1.set_xlim(np.min(x_all)-x1_extra-2,np.max(x_all)+x1_extra+2)
            x2_extra = (sidey1 - sidex2)/2
            xskyposlim = np.max(x_all2)+x2_extra+2
            xskyneglim = np.min(x_all2)-x2_extra-2
            ax2.set_xlim(xskyneglim,xskyposlim)
            y2_extra = (sidey1 - sidey2)/2
            yskyposlim = np.max(y_all2)+y2_extra+2
            yskyneglim = np.min(y_all2)-y2_extra-2
            ax2.set_ylim(yskyneglim,yskyposlim)
        elif maxside == 'c':
            xskyposlim = np.max(x_all2)+2
            xskyneglim = np.min(x_all2)-2
            ax2.set_xlim(xskyneglim,xskyposlim)
            x1_extra = (sidex2 - sidex1)/2
            ax1.set_xlim(np.min(x_all)-x1_extra-2,np.max(x_all)+x1_extra+2)
            y1_extra = (sidex2 - sidey1)/2
            ax1.set_ylim(np.min(y_all)-y1_extra-2,np.max(y_all)+y1_extra+2)
            y2_extra = (sidex2 - sidey2)/2
            yskyposlim = np.max(y_all2)+y2_extra+2
            yskyneglim = np.min(y_all2)-y2_extra-2
            ax2.set_ylim(yskyneglim,yskyposlim)
        elif maxside == 'd':
            yskyposlim = np.max(y_all2)+2
            yskyneglim = np.min(y_all2)-2
            ax2.set_ylim(yskyneglim,yskyposlim)
            x1_extra = (sidey2 - sidex1)/2
            ax1.set_xlim(np.min(x_all)-x1_extra-2,np.max(x_all)+x1_extra+2)
            y1_extra = (sidey2 - sidey1)/2
            ax1.set_ylim(np.min(y_all)-y1_extra-2,np.max(y_all)+y1_extra+2)
            x2_extra = (sidey2 - sidex2)/2
            xskyposlim = np.max(x_all2)+x2_extra+2
            xskyneglim = np.min(x_all2)-x2_extra-2
            ax2.set_xlim(xskyneglim,xskyposlim)
            
        ax1.plot(0,0,'r*',label='Star')
        ax1.legend(loc='upper left',fontsize='small')
        ax1.set_title(str(system) + ' Top Down View',loc='center')  
        ax2.plot(0,0,'r*',label='Star')
        ax2.legend(loc='upper left',fontsize='small')
        ax2.set_title(str(system) + ' Sky View',loc='center')
        
        # add arrows to indicate directionality (north up, east left)
        xlen = xskyposlim - xskyneglim                    # range of x and y axes
        ylen = yskyposlim - yskyneglim

        arr_basex = xskyposlim - 0.1*xlen               # location of base of the arrows
        arr_basey = yskyposlim - 0.2*ylen
        arr_xdelta = xlen*0.1             # length of horizontal arrow  
        arr_ydelta = ylen*0.1            # length of vertical arrow
        size = xlen*0.025               # size of the arrow head
        ax2.arrow(arr_basex,arr_basey,0,arr_ydelta,shape='full',head_width=size, \
                  head_length=size)             # vertical arrow
        ax2.arrow(arr_basex,arr_basey,-arr_xdelta,0,shape='full',head_width=size, \
                  head_length=size)             # horizontal arrow
        textsize = xlen*0.45
        vtextx = arr_basex - xlen*0.01          # vertical arrow text location
        vtexty = arr_basey + arr_ydelta + size + ylen*(textsize*0.001)    
        htextx = arr_basex - arr_xdelta - size - xlen*(textsize*0.003)     # horizontal arrow text location
        htexty = arr_basey - ylen*0.01
        ax2.text(vtextx, vtexty, 'N',fontsize=textsize)              # vertical arrow text
        ax2.text(htextx, htexty, 'E',fontsize=textsize)              # horizontal arrow text
        
        if save == 'Yes':
            fig1.savefig(savepath + 'Orbits_SS_topdown.png',dpi=500)
            fig2.savefig(savepath + 'Orbits_SS_sky.png',dpi=500)
        else:
            pass
        
    else:
        print('Please provide a valid viewing angle(s).')


############################################################################################
def cal_pos(planet,phi=None):
    # Function to calculate the x and y positions of orbits and planets in either 
    # 'topdown' or 'sky' viewing angle.
    #
    # Inputs:
    #   - planet: a number that indicate the nth planet it's currently working on
    #   - phi: if given, is orbital phase of the planet at a specified time
    # Outputs:
    #   - x_temp: a list of x-coordinate of planet positions for top down view in AU
    #   - y_temp: a list of y-coordinate of planet positions for top down view in AU
    #   - x_temp2: a list of x-coordinate of planet actual positions on the sky in AU
    #   - y_temp2: a list of y-coordinate of planet actual positions on the sky in AU
    #   - f: a list of true anomaly values
    #   - phi: orbital phase of the planet at a specified time
    
    x_temp = []
    y_temp = []
    x_temp2 = []
    y_temp2 = []
    
    # calculate true anomaly values
    if 0 <= w[planet] <= np.pi*2:
        if phi is not None:
            phi = phi
        else:
            phi = np.arange(0,1.001,0.001)

        # Get respective steps of mean anomaly according to orbital phase
        m = 2*(np.pi)*phi/1.0

        # Get respective steps of eccentric anomaly according to orbital phase
        eps = []
        for j in range(len(phi)):
            def func(x):
                return m[j] - x + e[planet]*(np.sin(x))
            sol = root(func,1)
            eps.append(sol.x[0])
        eps = np.array(eps)
        
        # Get respective steps of true anomaly according to orbital phase
        f = []
        for j in range(len(phi)):
            v = np.arccos(((np.cos(eps[j])) - e[planet])/(1 - e[planet]*(np.cos(eps[j]))))
            if eps[j] <= np.pi:
                f.append(v)
            else:
                v = np.pi + abs(np.pi - v)
                f.append(v)
        f = np.array(f)
        
        # Calculates distance at each step
        distance = a[planet]*(1-e[planet]**2)/(1+e[planet]*np.cos(f))
        
        # Calculates x and y positions, starts at periastron
        # x and y coordinates of the top down view, zero inclination
        for k in range(len(f)):
            x = -distance[k]*(np.sin(Omega[planet])*np.cos(w[planet]+f[k]) + \
                              np.cos(Omega[planet])*np.sin(w[planet]+f[k])*np.cos(0))
            y = distance[k]*(np.cos(Omega[planet])*np.cos(w[planet]+f[k]) - \
                             np.sin(Omega[planet])*np.sin(w[planet]+f[k])*np.cos(0))
            
            x_temp.append(x)
            y_temp.append(y)
        
        # x and y coordinates on the plane of sky
        for k in range(len(f)):
            x = -distance[k]*(np.sin(Omega[planet])*np.cos(w[planet]+f[k]) + \
                              np.cos(Omega[planet])*np.sin(w[planet]+f[k])*np.cos(inclination[planet]))
            y = distance[k]*(np.cos(Omega[planet])*np.cos(w[planet]+f[k]) - \
                             np.sin(Omega[planet])*np.sin(w[planet]+f[k])*np.cos(inclination[planet]))
            
            x_temp2.append(x)
            y_temp2.append(y)
    
    else:
        print('Invalid argument of periastron input. Please input between 0 and 2pi.')
        
    x_temp = np.array(x_temp)
    y_temp = np.array(y_temp)
    
    x_temp2 = np.array(x_temp2)
    y_temp2 = np.array(y_temp2)
        
    return(x_temp,y_temp,x_temp2,y_temp2,f,phi)


############################################################################################
def flux_ratio(planet,x_cor,y_cor,x_cor2,y_cor2,anom):
    # Function to calculate the flux ratio at each point in the entire orbit for
    # all planets.
    #
    # Inputs:
    #   - planet: a number that indicate the nth planet it's currently working on
    #   - x_cor: a list of x-coordinate of planet positions in AU
    #   - y_cor: a list of y-coordinate of planet positions in AU
    #   - x_cor2: a list of x-coordinate of planet positions in the sky in AU
    #   - y_cor2: a list of y-coordinate of planet positions in the sky in AU
    #   - anom: a list of true anomaly values
    # Output:
    #   - ratio: flux ratio values for one planet at different positions without occulter transmittance profile
    #   - ratio2: flux ratio values for one planet at different positions with occulter transmittance profile
    #   - alpha: phase angle values for one planet along its orbit
    #   - mas: separation between star and planet in mas
    
    ratio = []
    alpha = []
    mas = []
    
    size = radius[planet]*const.R_jup.value
    
    for pos in range(len(x_cor)):
        p_dist = np.sqrt(x_cor[pos]**2 + y_cor[pos]**2)       # actual star-planet separation
        p_dist = p_dist*const.au.value     # convert to meters
        p_dist2 = np.sqrt(x_cor2[pos]**2 + y_cor2[pos]**2)      # star-planet separation on sky
        p_dist2 = p_dist2*const.au.value
        
        if e[planet] >= 0 and e[planet] < 1:
            phase = np.arccos(np.sin(w[planet]+anom[pos])*np.sin(inclination[planet]))    # effective phase angle
            alpha.append(phase)
        else:
            print('Wrong eccentricity value. Eccentricity needs to be betwwen 0' \
                  + ' (inclusive) and 1 (exclusive).')
            
        g = (np.sin(phase) + (np.pi - phase)*np.cos(phase))/np.pi   # phase function
        
        fr = albedo[planet]*g*(size**2/(p_dist**2))      # flux ratio at one step
        
        ratio.append(fr)
        
        p_dist2 = np.arctan(206265*p_dist2/(distance*const.pc.value))*1e3      # sky view separation converted to mas
        mas.append(p_dist2)
    
    # get file with tau values
    ratio2 = []
    if transmittance == 'Yes':
        tau_path = path + tau_file
        trans = pd.read_csv(tau_path,header=None,skiprows=1,delim_whitespace=True,
                            names=('distance','tau'))
        sep = trans['distance']
        tau = trans['tau']
        
        fit = interp1d(sep,tau,kind='cubic')           # interpolation between data points
        
        for i in range (len(mas)):
            if mas[i] > np.max(sep):            # correct flux ratio values with transmittance if within maximum distance values in the tau file
                ratio2.append(ratio[i])
            else:
                new_ratio = ratio[i]*fit(mas[i])
                ratio2.append(new_ratio)
    else:
        ratio2 = ratio
    
    ratio = np.array(ratio)
    ratio2 = np.array(ratio2)
    alpha = np.array(alpha)
    mas = np.array(mas)
    
    return ratio, ratio2, alpha, mas


############################################################################################
def plot_orbit(x_cor,y_cor,fluxes,figure,axes):
    # Function to plot out all the planetary orbits and color code orbits based on
    # flux ratio information.
    #
    # Inputs:
    #   - x_cor: a list of x-coordinate of planet positions (list within list)
    #   - y_cor: a list of y-coordinate of planet positions (list within list)
    #   - fluxes: a list of flux ratio values at different positions (list within list)
    #   - figure: a figure object created in which all plots are displayed on
    #   - axes: an axis object on which all plots are created on
    # Output:
    #   - Plot including planets' orbits and positions
        
    # Plot out the orbit for the planet first
    for i in range(len(a)):
        axes.plot(x_cor[i], y_cor[i], linestyle=style[i], color='grey', alpha=0.3,\
                  linewidth=3, label=letter[i])

    
    # Put all fluxes together, pick out those above the flux threshold, and normalize
    total = []
    for i in range(len(a)):
        total = total + list(fluxes[i])
        
    thres = []
    for fl in total:
        if fl > threshold:
            thres.append(fl)
        else:
            pass
        
    thres = np.array(thres)
    
    # Normalize colors based on flux values
    if len(thres) > 0:
        #norm = plt.Normalize(thres.min(), thres.max())
        norm = colors.LogNorm(vmin=thres.min(), vmax=thres.max())
    
        colorbar = False
        
        for i in range(len(a)):
            # Pick out those coordinates that need to be color coded        
            temp_x = []
            temp_y = []
            temp_flux = []
            
            new_x = []
            new_y = []
            new_flux = []
            print(np.min(fluxes[i]),np.max(fluxes[i]))
            for j in range(len(x_cor[i])):
                if fluxes[i][j] >= threshold:
                    if j == len(x_cor[i])-1:
                        temp_flux.append(fluxes[i][j])
                        temp_x.append(x_cor[i][j])
                        temp_y.append(y_cor[i][j])
                        new_x.append(temp_x)
                        new_y.append(temp_y)
                        new_flux.append(temp_flux)
                        temp_x = []
                        temp_y = []
                        temp_flux = []
                    else:
                        if fluxes[i][j+1] >= threshold:
                            temp_flux.append(fluxes[i][j])
                            temp_x.append(x_cor[i][j])
                            temp_y.append(y_cor[i][j])
                        else:
                            temp_flux.append(fluxes[i][j])
                            temp_x.append(x_cor[i][j])
                            temp_y.append(y_cor[i][j])
                            new_x.append(temp_x)
                            new_y.append(temp_y)
                            new_flux.append(temp_flux)
                            temp_x = []
                            temp_y = []
                            temp_flux = []
                else:
                    pass
                
            # Remove lists that contain only a single value, if there's any
            for k in range(len(new_x)):
                if len(new_x[k]) == 0 or len(new_x[k]) == 1:
                    del new_x[k]
                    del new_y[k]
                    del new_flux[k]
                else:
                    new_x[k] = np.array(new_x[k])
                    new_y[k] = np.array(new_y[k])
                    new_flux[k] = np.array(new_flux[k])
            
            if len(new_x) > 0:
                for p in range(len(new_x)):
                    # Color code the orbit based on flux ratios
                    points = np.array([new_x[p], new_y[p]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                            
                    # Create a continuous norm to map from data points to colors
                    lc = LineCollection(segments, cmap='autumn_r', norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(new_flux[p])
                    lc.set_linewidth(2)
                    line = axes.add_collection(lc)
                    
                    # Only creates colorbar once
                    if colorbar == False:
                        figure.colorbar(line, ax=axes,label='Flux Ratio')
                        #line.set_clim([1e-8,1e-10])
                        colorbar = True
                    else:
                        pass
            else:
                pass
        
            new_x.clear()
            new_y.clear()
            new_flux.clear()

    else:
        print('Fluxes for all planets at all positions are below the threshold.')


############################################################################################
def orbphi (planet):
    # Function to determine the orbital phases of all planets when given a specific
    # time range in order to calculate the corresponding x,y positions.
    #
    # Input:
    #   - planet: a number that indicate the (n+1)th planet it's currently working on
    # Output:
    #   - time: time stamps at which coordinates need to be calculated
    #   - phase: orbital phase values for all planets corresponding to the time given
    
    # orbital phases to store in
    phase = []
    
    # calculate orbital phases at each time step
    for i in range(len(time)):
        ratio = (time[i] - tp[planet])/period[planet]
        if ratio > 1:   # when t is more than one period bigger than tp
            phi = ratio - int(ratio)
            phase.append(phi)
        elif 0 <= ratio <= 1:   # when t is within one period bigger than tp
            phase.append(ratio)
        elif -1 <= ratio < 0:           # when t is within one period less than tp
            phi = 1 + ratio
            phase.append(phi)
        else:       # when t is more than one period less than tp
            phi = 1 + (ratio - int(ratio))
            phase.append(phi)
            
    phase = np.array(phase)
    
    return phase


############################################################################################
def coord (axes,axes2=None):
    # Function to calculate, plot, and store actualy sky coordinates if given specific
    # time range.
    #
    # Inputs:
    #   - axes: an axis object on which all plots are created on
    #   - axes2: a second axis object on which plots are created on
    # Outputs:
    #   - Text file that saves relevant information
    
    x2_obs = []
    y2_obs = []
    
    with open(savepath+'coordinates.txt','w') as outfile:
        outfile.write('Sky coordinates for all planets.\n\n')
    
    position = False
    
    for i in range(len(a)):
        orb_phase = orbphi(i)
        x_loc,y_loc,x_loc2,y_loc2,true_anom,phi = cal_pos(i,orb_phase)
        brightness,brightness2,phase_angle,separation = flux_ratio(i,x_loc,y_loc,x_loc2,y_loc2,true_anom)
        x2_obs.append(x_loc2)
        y2_obs.append(y_loc2)
        
        phase_angle = (phase_angle/(np.pi*2))*360
        
        if len(view) == 1:
            if view[0] == 'topdown':
                if plotpos == 'Yes':
                    if position is False:
                        axes.plot(x_loc,y_loc,'bo',markersize=3,alpha=0.5,label='Positions')
                        position = True
                    else:
                        axes.plot(x_loc,y_loc,'bo',markersize=3,alpha=0.5)
                else:
                    pass
            elif view[0] == 'sky':
                if plotpos == 'Yes':
                    if position is False:
                        axes2.plot(x_loc2,y_loc2,'bo',markersize=3,alpha=0.5,label='Positions')
                        position = True
                    else:
                        axes2.plot(x_loc2,y_loc2,'bo',markersize=3,alpha=0.5)
                else:
                    pass
                
                # only save sky x and y coordinates in a file
                with open(savepath+'coordinates.txt','a') as outfile:
                    outfile.write('**********************************************************\n')
                    outfile.write('Sky coordinates for planet ' + str(letter[i]) + '.\n')
                    
                df = pd.DataFrame()
                df['time(JD)'] = time         # time stamps in JD
                df['orbital_phase'] = orb_phase     # corresponding orbital phases
                df['phase_angle(degrees)'] = phase_angle
                df['sky_x(AU)'] = x_loc2
                df['sky_y(AU)'] = y_loc2
                df['separation(mas)'] = separation
                df['flux_ratio'] = brightness2
                df.to_csv(savepath+'coordinates.txt',sep='\t',index=False, header=True, mode='a')
                
                with open(savepath+'coordinates.txt','a') as outfile:
                    outfile.write('\n')
            else:
                pass
            
        elif len(view) == 2:
            if ('topdown' in view) and ('sky' in view):
                if plotpos == 'Yes':
                    if position is False:
                        axes.plot(x_loc,y_loc,'bo',markersize=3,alpha=0.5,label='Positions')
                        axes2.plot(x_loc2,y_loc2,'bo',markersize=3,alpha=0.5,label='Positions')
                        position = True
                    else:
                        axes.plot(x_loc,y_loc,'bo',markersize=3,alpha=0.5)
                        axes2.plot(x_loc2,y_loc2,'bo',markersize=3,alpha=0.5)
                else:
                    pass
                
                # only save sky x and y coordinates in a file
                with open(savepath+'coordinates.txt','a') as outfile:
                    outfile.write('**********************************************************\n')
                    outfile.write('Sky coordinates for planet ' + str(letter[i]) + '.\n')
                    
                df = pd.DataFrame()
                df['time(JD)'] = time         # time stamps in JD
                df['orbital_phase'] = orb_phase     # corresponding orbital phases
                df['phase_angle(degrees)'] = phase_angle
                df['sky_x(AU)'] = x_loc2
                df['sky_y(AU)'] = y_loc2
                df['separation(mas)'] = separation
                df['flux_ratio'] = brightness2
                df.to_csv(savepath+'coordinates.txt',sep='\t',index=False, header=True, mode='a')
                
                with open(savepath+'coordinates.txt','a') as outfile:
                    outfile.write('\n')
            else:
                pass
        else:
            pass


############################################################################################
def window (iwa,owa,sciwa,x_pos2,y_pos2,flux,flux2,phi):
    # Function to plot the flux ratio variation for each planet wrt orbital phase and
    # indicate the phases at which the specified time range is looking at.
    #
    # Input:
    #   - iwa: inner working angle in AU
    #   - owa: outer working angle in AU
    #   - sciwa: science inner working angle in AU
    #   - x_pos2: sky x coordinates for all planets (array, lists in array)
    #   - y_pos2: sky y coordinates for all planets (array, lists in array)
    #   - flux: flux ratios for all planets without occulter transmittance (array, lists in array)
    #   - flux2: same as flux, except with occulter transmittance
    #   - phi: orbital phase, same for all planets (array)
    #
    # Output:
    #   - One plot for each planet showing flux ratio variation and observing window
    
    for i in range(len(a)):
        winfig, winax = plt.subplots()
        x_loc2 = x_pos2[i]
        y_loc2 = y_pos2[i]
        r = np.sqrt(abs(x_loc2)**2 + abs(y_loc2)**2)

        f_ratio = flux[i]
        f_ratio2 = flux2[i]
        
        # plot out flux variations vs orbital phase
        if transmittance == 'Yes':
            winax.plot(phi,f_ratio,'gold',label=r"Flux w/o $\tau$",linestyle='dashed')
            winax.plot(phi,f_ratio2,'gold',label=r"Flux w/ $\tau$",linestyle='solid')
        else:
            winax.plot(phi,f_ratio2,'gold',label="Flux",linestyle='solid')
        thres = np.array([threshold]*len(f_ratio2))
        winax.plot(phi,thres,'r',label='Threshold',linestyle='dashed')
        
        if coordinates == 'Yes':
            orbital_phase = orbphi(i)
            if timetype == 'range':
                if (time[-1] - time[0]) >= period[i]:
                    winax.axvspan(phi[0],phi[-1],alpha=0.8,color='limegreen',lw=0,
                                  label='Observation')
                else:
                    if orbital_phase[0] < orbital_phase[-1]:
                        winax.axvspan(orbital_phase[0],orbital_phase[-1],alpha=0.8,\
                                      color='limegreen',lw=0,label='Observation')
                    elif orbital_phase[0] > orbital_phase[-1]:
                        winax.axvspan(orbital_phase[0],phi[-1],alpha=0.8,\
                                      color='limegreen',lw=0)
                        winax.axvspan(phi[0],orbital_phase[-1],alpha=0.8,\
                                      color='limegreen',lw=0,label='Observation')
                    else:
                        pass
            elif timetype == 'specific':
                label = False
                for stamp in orbital_phase:
                    if label is False:
                        winax.axvline(stamp,alpha=0.8,linewidth=1,color='limegreen',\
                                      label='Observation')
                        label = True
                    else:
                        winax.axvline(stamp,alpha=0.8,linewidth=1,color='limegreen')
            else:
                pass
        else:
            pass
        
        if iwa != 'NaN':          # if there's provided values for iwa and owa
            visIWA = []         # sky coordinates where planet is inside IWA
            visOWA = []         # sky coordinates where planet is outside OWA
            visSCIWA = []       # sky coordinates where planet is inside Science IWA
            
            # determine which orbital phase is about to enter or leave IWA or OWA
            for j in range(len(x_loc2)):
                if r[j] < iwa:           # if orbit is within iwa
                    if j == 0 or len(visIWA) == 0 or j == len(x_loc2) - 1:
                        visIWA.append(phi[j])
                    elif r[j-1] < iwa and r[j+1] >= iwa:
                        visIWA.append(phi[j])
                    elif r[j-1] >= iwa and r[j+1] < iwa:
                        visIWA.append(phi[j])
                    else:
                        pass
                elif r[j] > owa:         # if orbit is beyond owa
                    if j == 0 or len(visOWA) == 0 or j == len(x_loc2) - 1:
                        visOWA.append(phi[j])
                    elif r[j-1] > owa and r[j+1] <= owa:
                        visIWA.append(phi[j])
                    elif r[j-1] <= owa and r[j+1] > owa:
                        visIWA.append(phi[j])
                    else:
                        pass
                else:           # visible orbit
                    pass
                
            # determine which orbital phase is about to enter or leave Science IWA
            if sciwa != 'NaN':
                for s in range(len(x_loc2)):
                    if r[s] < sciwa:
                        if s == 0 or len(visSCIWA) == 0 or s == len(x_loc2) - 1:
                            visSCIWA.append(phi[s])
                        elif r[s-1] < sciwa and r[s+1] >= sciwa:
                            visSCIWA.append(phi[s])
                        elif r[s-1] >= sciwa and r[s+1] < sciwa:
                            visSCIWA.append(phi[s])
                        else:
                            pass
                    else:
                        pass
            else:
                pass
            
            # check if there're cases where both phase 0 and are recordeed, if yes, then
            # remove phase 1 for plotting purposes.
            if len(visIWA) % 2 == 1:
                visIWA = visIWA[:-1]
            else:
                pass
            
            if len(visOWA) % 2 == 1:
                visOWA == visOWA[:-1]
            else:
                pass
            
            if len(visSCIWA) > 0:
                if len(visSCIWA) % 2 == 1:
                    visSCIWA == visSCIWA[:-1]
                else:
                    pass
            else:
                pass
            
            # plot greyed out phases that are inside IWA or outside OWA
            label_iwa = False
            for k in range(0,len(visIWA),2):
                if label_iwa is False:
                    winax.axvspan(visIWA[k],visIWA[k+1],fill=False,color='grey',\
                                  hatch=r'\\',alpha=0.5,label='IWA')
                    label_iwa = True
                else:
                    winax.axvspan(visIWA[k],visIWA[k+1],fill=False,color='grey',\
                                  alpha=0.5,hatch=r'\\')
            
            for k in range(0,len(visOWA),2):
                if label_iwa is False:
                    winax.axvspan(visIWA[k],visIWA[k+1],fill=False,color='grey',\
                                  hatch=r'\\',alpha=0.5,label='IWA')
                    label_iwa = True
                else:
                    winax.axvspan(visIWA[k],visIWA[k+1],fill=False,color='grey',\
                                  alpha=0.5,hatch=r'\\')
        
            if len(visSCIWA) > 0:
                label_sciwa = False
                for k in range(0,len(visSCIWA),2):
                    if label_sciwa is False:
                        winax.axvspan(visSCIWA[k],visSCIWA[k+1],color='r',\
                                      alpha=0.3, lw=0, label='Science IWA')
                        label_sciwa = True
                    else:
                        winax.axvspan(visSCIWA[k],visSCIWA[k+1],color='r',\
                                      alpha=0.3, lw=0)
            else:
                pass
        else:
            pass
        
        winax.set_xlim(phi[0],phi[-1])
        winax.set_xlabel('Orbital phase')
        winax.set_ylabel('Flux ratio')
        winax.set_title('Planet ' + str(letter[i]),loc='center')
        winax.legend(loc='upper right',fontsize='small')
        
        if save == 'Yes':
            winfig.savefig(savepath+'visibility_planet_' + str(letter[i]) + '.png', dpi=500)
        else:
            pass


############################################################################################
def WA (axes):
    # Function that plots the inner and outer working angle as greyed out areas.
    #
    # Input:
    #   - axes: an axis object on which all plots are created on
    # Output:
    #   - inner: inner working angle radius in AU
    #   - outer: outer working angle radius in AU
    #   - science: science inner working angle radius in AU
    #   - Inner and outer working angle circles
    
    # Convert FOV numbers to be in AU    
    inner = 0
    outer = 0
    science = 0
    
    if len(FOV) > 0:      # if iwa and owa values are provided
        iwa = FOV[0]*(2*np.pi)/(360*3600)       # angle in radians
        if len(FOV) > 1:           # if owa is provided
            owa = FOV[1]*(2*np.pi)/(360*3600)
        else:         # or else owa use default value of 5 arcsec
            owa = 5*(2*np.pi)/(360*3600)
        
        inner = np.tan(iwa)*distance*const.pc.value/const.au.value     # get inner angles in au
        outer = np.tan(owa)*distance*const.pc.value/const.au.value     # get outer angles in au
        
        if transmittance == 'Yes':
            tau_path = path + tau_file
            trans = pd.read_csv(tau_path,header=None,skiprows=1,delim_whitespace=True,
                                names=('distance','tau'))
            sep = trans['distance']
            tau = trans['tau']
        
            fit = interp1d(tau,sep,kind='cubic')
            
            sci = fit(0.5)/1000          # science iwa in as
            sci_rad = sci*(2*np.pi)/(360*3600)       # science iwa in radians
            science = np.tan(sci_rad)*distance*const.pc.value/const.au.value
        else:
            science = 'NaN'
            
        if axes is None:
            pass
        else:
            # Grey out areas beyond outer working angle
            xnode = [-outer-0.5,outer+0.5,outer+0.5,-outer-0.5]
            ynode = [-outer-0.5,-outer-0.5,outer+0.5,outer+0.5]
            
            axes.fill(xnode,ynode,'grey',alpha = 0.2)
            
            # Make areas in between iwa and owa visible
            circle1 = Circle((0, 0), outer, facecolor='white',
                             edgecolor='white', linewidth=0, alpha=1.0)
            axes.add_patch(circle1)
            
            # Grey out areas inside inner working angle
            circle2 = Circle((0, 0), inner, facecolor='grey',
                             edgecolor='white', linewidth=0, alpha=0.1,label='IWA')
            axes.add_patch(circle2)
            
            if science == 'NaN':
                pass
            else:
                # Grey out areas (darker) inside science inner working angle
                circle2 = Circle((0, 0), science, facecolor='grey',
                                 edgecolor='grey', linewidth=0, alpha=0.2,label='IWA$_{0.5}$')
                axes.add_patch(circle2)
            
            # Set plot size (including showing owa)
            axes.set_xlim(-outer-0.5,outer+0.5)
            axes.set_ylim(-outer-0.5,outer+0.5)
    else:       # if iwa and owa values not provided
        inner = 'NaN'
        outer = 'NaN'
        science = 'NaN'

    return inner, outer, science


############################################################################################
def gather ():
    # Function to gather all x, y coordinates, and flux information for all planets.
    #
    # Outputs:
    #   - x: x coordinates of topdown view for all planets (array, lists in a list)
    #   - y: y coordinates of topdown view for all planets (array, lists in a list)
    #   - x2: x coordinates of sky view for all planets (array, lists in a list)
    #   - y2: y coordinates of sky view for all planets (array, lists in a list)
    #   - flux1: corresponding flux ratio values for all planets (array, list in a list)
    #   - flux2: same as flux1, but takes into account occulter transmittance
    #   - phi: orbital phase values, same for all planets (array)
    #   - phase: phase angle values for all planets (array, lists in a list)
    
    # orbit coordinates for all planets depending on viewing geometry
    x = []
    y = []
    x2 = []
    y2 = []
    
    # flux ratios of all planets
    flux = []
    flux2 = []
    phase = []
    
    # gather x and y coordinates for planets
    for i in range(len(a)):
        x_loc,y_loc,x_loc2,y_loc2,true_anom,phi = cal_pos(i)
        x.append(x_loc)
        y.append(y_loc)
        x2.append(x_loc2)
        y2.append(y_loc2)
        
        # get flux ratio values for planets
        flux_ind,flux_ind2,phase_ind,separation_ind = flux_ratio(i,x_loc,y_loc,x_loc2,y_loc2,true_anom)
        flux.append(flux_ind)
        flux2.append(flux_ind2)
        phase.append(phase_ind)
    
    x = np.array(x)
    y = np.array(y)
    x2 = np.array(x2)
    y2 = np.array(y2)
    flux = np.array(flux)
    flux2 = np.array(flux2)
    phase = np.array(phase)
    
    return x, y, x2, y2, flux, flux2, phi, phase


############################################################################################
def HZ (axes):
    # Function to calculate and plot habitable zone (conservative and optimistic) boundaries 
    # on top of the top down orbit plot.
    #
    # Input:
    #   - axes: an axis object on which all plots are created on
    # Output:
    #   - Conservative and optimistic habitable zone boundaries plotted on orbit plot
    
    if HabZone == 'Yes':
        Rstar = 0  
        if Rs == 'NaN':             # derive R from g (Smalley 2005)
            temp = -0.5*(np.log10((10**gs)*0.01) - np.log10((10**4.4374)*0.01) - np.log10(Ms))
            Rstar = const.R_sun.value*(10**temp)
        else:
            Rstar = Rs*const.R_sun.value
        
        Lstar = 4*np.pi*(Rstar**2)*const.sigma_sb.value*(Teff**4)
        
        cons_in = 0             # conservative HZ inner boundary (runaway greenhouse)
        cons_out = 0            # conservative HZ outer boundary (maximum greenhouse)
        opti_in = 0             # optimistic HZ inner boundary (recent Venus)
        opti_out = 0            # optimistic HZ outer boundary (early Mars)
        
        Ts = Teff - 5780
        
        # conservative inner boundary
        if case == 1:
            S_cons_in = 1.107 + (1.332e-4)*Ts + (1.58e-8)*(Ts**2) + (-8.308e-12)*(Ts**3) + \
                        (-1.931e-15)*(Ts**4)
        elif case == 2:
            S_cons_in = 1.188 + (1.433e-4)*Ts + (1.707e-8)*(Ts**2) + (-8.968e-12)*(Ts**3) + \
                        (-2.084e-15)*(Ts**4)
        elif case == 3:
            S_cons_in = 0.99 + (1.209e-4)*Ts + (1.404e-8)*(Ts**2) + (-7.418e-12)*(Ts**3) + \
                        (-1.713e-15)*(Ts**4)
        else:
            print('Please provide a valid case number for habitable zone calculation.')
            
        cons_in = np.sqrt((Lstar/const.L_sun.value)/S_cons_in)              # in AU
        
        # conservative outer boundary
        S_cons_out = 0.356 + (6.171e-5)*Ts + (1.698e-9)*(Ts**2) + (-3.198e-12)*(Ts**3) + \
                     (-5.575e-16)*(Ts**4)
        cons_out = np.sqrt((Lstar/const.L_sun.value)/S_cons_out)
        
        # optimistic inner boundary
        S_opti_in = 1.776 + (2.136e-4)*Ts + (2.533e-8)*(Ts**2) + (-1.332e-11)*(Ts**3) + \
                    (-3.097e-15)*(Ts**4)
        opti_in = np.sqrt((Lstar/const.L_sun.value)/S_opti_in)
        
        # optimistic outer boundary
        S_opti_out = 0.32 + (5.547e-5)*Ts + (1.526e-9)*(Ts**2) + (-2.874e-12)*(Ts**3) + \
                     (-5.011e-16)*(Ts**4)
        opti_out = np.sqrt((Lstar/const.L_sun.value)/S_opti_out)
        
        # plot the boundaries
        opti_out_circle = Wedge((0,0), opti_out, 0, 360, width=opti_out - opti_in,
                                facecolor='limegreen', edgecolor='white', linewidth=0, \
                                alpha=1.0, label='Optimistic HZ')
        axes.add_patch(opti_out_circle)
        
        cons_out_circle = Wedge((0,0), cons_out, 0, 360, width=cons_out - cons_in,\
                                facecolor='lime', edgecolor='white', linewidth=0, \
                                alpha=1.0, label='Conservative HZ')
        axes.add_patch(cons_out_circle)
    
    else:
        pass


############################################################################################    
############################################################################################
'''
Run the above script from the command line. It only calls the main () function.
'''

if __name__ == '__main__':
    main ()
    