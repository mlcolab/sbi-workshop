import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from scipy.linalg import inv
from numpy.random import multivariate_normal
from notebook.services.config import ConfigManager
from traitlets.config.manager import BaseJSONConfigManager
import matplotlib as mpl
import math
from scipy.integrate import odeint, solve_ivp
import random
import torch.nn as nn


# lookup table for plotting colors
COLORS = dict(
    ground_truth = 'r',
    data = '#0000FF',
    linreg = '#519872',
    abc = 'purple',
    sbi  = '#DE1A1A',
    )





def set_plot_attributes(ax, xlabel='',ylabel='',title='',xlim=None,ylim=None,legend=False, grid=False):
    """ Helper function to avoid wasting cell space on plotting code. Feel free to extend. """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    plt.show()




def interactive_projectile(simulator):
    """ Returns an interactive plot for the projectile simulation using a wrapper. """
    def visualize_projectile(launch_velocity, launch_angle, drag_coefficient):
        # perform simulation using arguments from interactive plot
        θ = launch_velocity, launch_angle, drag_coefficient
        simulation  = simulator(θ)
        # plotting
        fig, ax = plt.subplots()
        ax.plot(simulation['distance_travelled'], simulation['height'],label='Ground truth trajetory', color=COLORS['ground_truth'])
        ax.scatter(simulation['distance_travelled'], simulation['x'],label='x_o', marker='x', color=COLORS['data'])
        set_plot_attributes(ax, xlim=(0,350), ylim=(0,150), xlabel='distance traveled (m)', ylabel='height (m)', legend=True)
        
        return dict(θ=θ, distance_travelled=simulation['distance_travelled'], height=simulation['height'], x=simulation['x'])
    
    return interactive(visualize_projectile, launch_velocity=(50,250), launch_angle=(10,45), drag_coefficient=(0.2,0.7,0.01))




def displacement_deriv(theta, t,b,m,g,l):
    """ Differential equation describing the angular displacement over time of a damped pendulum. """
    theta1 = theta[0]
    theta2 = theta[1]
    dtheta1_dt = theta2
    dtheta2_dt = -(b/m * theta2) - (g/l * math.sin(theta1))
    dtheta_dt = [ dtheta1_dt, dtheta2_dt]
    
    return dtheta_dt


def interactive_pendulum(simulator):
    """ Returns an interactive plot for the damped pendulum simulation using a wrapper. """
    def visualize_pendulum(dampening_factor, mass, length):
        """ Wrapper to visualize pendulum simulator. """
        # perform simulation using parameters from the interactive plot
        θ = dampening_factor, mass, length 
        x_o  = simulator(θ)
        # plotting
        fig, ax = plt.subplots()
        t = np.linspace(0,20,150)
        ax.plot(t,x_o,marker='x', color=COLORS['data'], label='measurements')
        set_plot_attributes(ax, xlabel='$t$', grid=True, ylabel='Angular displacement', legend=True)  
        
        return dict(θ=θ, x_o=x_o)

    return interactive(visualize_pendulum, dampening_factor=(0,1,0.1), mass=(0.01,5,0.1), length=(1,4,0.5))
