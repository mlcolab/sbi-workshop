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
    model= '#DE1A1A',
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
    """ Returns an interactive plot for the projectile simulation using a wrapper."""
    
    def visualize_projectile(speed, angle, drag):
        area = 0.008 # this is here or it will show up in the interactive plot
        mass = 0.2 # same here
        
        # perform simulation using arguments from interactive widgets
        simulation = simulator(speed, angle, drag, area, mass)
        # plot
        fig, ax = plt.subplots()
        ax.plot(simulation['d'], simulation['h'],label='Ground truth trajectory', color=COLORS['ground_truth'])
        ax.scatter(simulation['d'], simulation['x'],label='$x_{\rm o}$', marker='x', color=COLORS['data'])
        set_plot_attributes(ax, xlim=(0,350), ylim=(0,150), xlabel='distance traveled (m)', ylabel='height (m)', legend=True)
        
        return dict(θ=(speed,angle,drag,area,mass), d=simulation['d'], h=simulation['h'], x=simulation['x'])
    
    return interactive(visualize_projectile, speed=(50,250), angle=(10,45), drag=(0.2,0.7,0.01))




def displacement_deriv(theta, t, b, m, g, l):
    """ Differential equation describing the angular displacement over time of a damped pendulum."""
    
    dtheta1_dt = theta[1]
    dtheta2_dt = -(b/m * theta[1]) - (g/l * math.sin(theta[0]))
    
    return (dtheta1_dt, dtheta2_dt)


def interactive_pendulum(simulator):
    """ Returns an interactive plot for the damped pendulum simulation using a wrapper. """
    def visualize_pendulum(damp:float, m:float, l:float, g=9.81):
        """ Wrapper to visualize pendulum simulator. """
        # perform simulation using parameters from the interactive plot
        θ = damp, m, l 
        x_o  = simulator(damp, m, l, g)
        # plotting
        fig, ax = plt.subplots()
        t = np.linspace(0,20,150)
        ax.plot(t,x_o,marker='x', color=COLORS['data'], label='measurements')
        set_plot_attributes(ax, xlabel='$t$', grid=True, ylabel='Angular displacement', legend=True)  
        
        return dict(θ=θ, x_o=x_o)

    return interactive(visualize_pendulum, damp=(0.1,1,0.1), m=(0.01,5,0.1), l=(1,4,0.5))


def plot_data_vs_model(domain,
                       data,
                       model,
                       data_label='data',
                       model_label='prediction',
                       plot_attrs=dict(legend=True, ylim=(0, 250), xlabel='distance traveled (m)', ylabel='height (m)')
                      ):
    fig, ax = plt.subplots()
    ax.scatter(domain, data, label=data_label, marker='x', color=COLORS['data'])
    ax.plot(domain, model, label = model_label, color=COLORS['sbi'])
    set_plot_attributes(ax, **plot_attrs)
    
    
    
#### ARCHIVE FROM OLD NOTEBOOK (WILL NOT WORK, FIX IMPORTS)

def sbi_pendulum_sim(θ):
    m, damp, l = θ
    return pendulum_sim(m, damp, l)

# define an initial, three-dimensional prior 
num_dim = 3
prior = utils.BoxUniform(low=torch.Tensor([0.1,0.01,1]), high=torch.Tensor([1,5,4]))

# and infer the posterior over θ using the sbi toolkit
posterior = infer(sbi_pendulum_sim, prior, method='SNPE', num_simulations=500)

def pendulum_sim(m:float, damp:float, l:float, g:float=9.81)->np.array:
    """
    Blackbox simulator that takes in a parameter vector
    and produces an observation of angular dispalement
    of a damped pendulum.
    """
    θ = [damp, m, l]
    theta_0 = [0,3]
    t = np.linspace(0,12,150)    
    # solve ODE
    solution = odeint(displacement_deriv, theta_0, t, args = (damp, m, g, l))
    angular_velocity_measurements =  solution[:,0] + np.random.randn(t.shape[0]) * 0.05
    # return observation
    return angular_velocity_measurements

pendulum = interactive_pendulum(pendulum_sim)
pendulum

x_o = pendulum.result['x_o']
samples = posterior.sample((10000,), x=x_o)
θ = samples.mean(dim=0).numpy()
log_probability = posterior.log_prob(samples, x=x_o)
_ = analysis.pairplot(samples, figsize=(6,6))

_, ax = plt.subplots()
ax.plot(np.linspace(0,20,150), sbi_pendulum_sim(θ), label='sbi reconstruction', color=COLORS['sbi'])
ax.plot(np.linspace(0,20,150), x_o, label='x_o', color=COLORS['data'], marker='x')
set_plot_attributes(ax, legend=True, xlabel='$t$', grid=True, ylabel='Angular displacement')