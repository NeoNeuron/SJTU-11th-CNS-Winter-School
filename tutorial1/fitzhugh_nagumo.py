'''
Interactive module for FitzHugh-Nagumo model.
John D. Murray (john.david.murray@gmail.com)
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib as mpl

dt = 0.1 # Integration dt

# Define dynamics of of the model

def dyn_fn(xinit, tmax, dt, args): # Integrates trajectory for FitzHugh-Nagumo
    x = np.zeros((int(tmax/dt), len(xinit)))
    x[0] = xinit
    for i in range(1,int(tmax/dt)):
        x[i] = x[i-1] + dt*eqs_fn(x[i-1],args) # Forward Euler
    return x

def eqs_fn(x,args): # Dynamical equations for FitzHugh-Nagumo, returns derivatives
    I, b0, b1, eps = args[0], args[1], args[2], args[3]
    u = x[0]
    w = x[1]
    dudt = u - 1./3*u**3 - w + I
    dwdt = eps*(b0+b1*u - w)
    z = np.array([dudt, dwdt])
    return z

def u_nullcline(u, I):
    return u - 1./3*u**3 + I

def w_nullcline(u,b0,b1):
    return b0 + b1*u


# Set plotting properties
params = {'axes.labelsize': 16,
          'text.fontsize': 16,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10}
mpl.rcParams.update(params)
mpl.rc('mathtext', fontset='stixsans',default='regular')

# Make figure
fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([.1,.1,.85,.275])
ax1 = fig.add_axes([.1,.475,.425,.425])
ax.set_ylim(-2.5,4.5)
ax1.set_xlim(-3,3)
ax1.set_ylim(-3,3)
ax.set_xlabel('Time')
ax.set_ylabel('u or w')
ax1.set_xlabel('u')
ax1.set_ylabel('w')
axcolor = 'lightgoldenrodyellow'
axsI = plt.axes([0.6, 0.7, 0.3, 0.04], axisbg=axcolor)
axsb0  = plt.axes([0.6, 0.65, 0.3, 0.04], axisbg=axcolor)
axsb1  = plt.axes([0.6, 0.6, 0.3, 0.04], axisbg=axcolor)
axseps  = plt.axes([0.6, 0.55, 0.3, 0.04], axisbg=axcolor)
axsT  = plt.axes([0.6, 0.5, 0.3, 0.04], axisbg=axcolor)
axh = plt.axes([-1,-1,1,1])

# Make sliders that control parameters
sI = Slider(axsI, r'$I$', -2, 2.0, valinit=0,color='maroon')
sb0 = Slider(axsb0, r'$b_0$', 0, 5.0, valinit=0,color='midnightblue')
sb1 = Slider(axsb1, r'$b_1$', 0.0, 10.0, valinit=1,color='midnightblue')
seps = Slider(axseps, r'$\epsilon$', 0.02, 1.2, valinit=0.1,color='k')
sT = Slider(axsT, r'$T$', 5., 200.0, valinit=200,color='k')
sxinit = Slider(axh ,'x',-5,5,valinit=0)
syinit = Slider(axh,'y',-5,5,valinit=0)

# Plot trajectories and nullclines
l, = ax.plot(0,0, lw=2, color='r',label='u')
lb, = ax.plot(0,0, lw=2, color='b',label='w')
l1 = ax.legend(loc=2,frameon=False)
l1, = ax1.plot(0,0, lw=2, color='k')
us = np.linspace(-3,3,1000)

vs_lim = (-3, 3)
ws_lim = (-3, 3)
vs = np.linspace(vs_lim[0], vs_lim[1],20)
ws = np.linspace(ws_lim[0], ws_lim[1],20)
vs_grid, ws_grid = np.meshgrid(vs, ws)

q1 = ax1.quiver(vs_grid,ws_grid,0*vs_grid,0*ws_grid,scale=15,
                scale_units='xy',angles='xy',headwidth=3,width=0.005,
                facecolor='gray')



#def ML(x,args):#VwIext, t):
#    V, w = x[0], y[1]
#    return eqs_fn(x,args)#deriv_V_t(V, w, I_ext), deriv_w_t(V, w)



l1n1, = ax1.plot(us,u_nullcline(us,sI.val),lw=2,color='r',ls='--',label='u nullcline')
l1n2, = ax1.plot(us,w_nullcline(us,sb0.val,sb1.val),lw=2,color='b',ls='--',label='w nullcline')

# Add figure text
fig.text(.5,.95,'FitzHugh-Nagumo model',ha='center',size=18)
fig.text(.625,.86,r'$\frac{du}{dt} = u - \frac{1}{3} u^3 - w + I$',size=14,color='r')
fig.text(.625,.8,r'$\frac{dw}{dt} = \epsilon ( b_0 + b_1 u - w) $',size=14,color='b')




# Update initial condition and parameters
def update(val):
    xinit = [sxinit.val,syinit.val]
    T = sT.val
    I = sI.val
    b0 = sb0.val
    b1 = sb1.val
    eps = seps.val
    args = (I, b0, b1, eps)
    m = dyn_fn(xinit,T,dt,args)
    t = np.linspace(0,T,int(T/dt),endpoint=False)
    l.set_xdata(t)
    l.set_ydata(m[:,0])
    lb.set_xdata(t)
    lb.set_ydata(m[:,1])
    ax.set_xlim(0,T)
    us = np.linspace(-3,3,500)
    l1n1.set_xdata(us)
    l1n2.set_xdata(us)
    l1n1.set_ydata(u_nullcline(us,I)) # u nullcline
    l1n2.set_ydata(w_nullcline(us,b0,b1)) # w nullcline
    l1.set_xdata(m[:,0])
    l1.set_ydata(m[:,1])

    z = eqs_fn([vs_grid,ws_grid],args)
    dx, dy = z[0],z[1]
    q1.set_UVC(dx,dy)

    plt.draw()
sI.on_changed(update)
sb0.on_changed(update)
sb1.on_changed(update)
seps.on_changed(update)
sT.on_changed(update)

# Make reset button
resetax = plt.axes([0.8, 0.45, 0.1, 0.04]) # Reset button
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sI.reset()
    sb0.reset()
    sb1.reset()
    seps.reset()
    sT.reset()
button.on_clicked(reset)

# Make mouse event to set initial conditions for trajectory
def onpick4(event):
    if event.inaxes == ax1:
        sxinit.val = event.xdata
        syinit.val = event.ydata
        update(0)

fig.canvas.mpl_connect('button_press_event',onpick4)

plt.show()

