import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

def StepEuler1(Derivatives, yv, t, dt):
    """ First-order Euler method """
    yv1 = yv + Derivatives(yv, t) * dt
    t1 = t + dt
    return yv1, t1


def StepEuler2(Derivatives, yv, t, dt):
    """ Second-order Euler method """
    d1 = Derivatives(yv, t)
    d2 = Derivatives(yv + d1 * dt, t)
    yv1 = yv + 0.5 * (d1 + d2) * dt
    t1 = t + dt
    return yv1, t1


def StepVerlet(Derivatives, yv, t, dt):
    """ Verlet symplectic algorithm """
    y, v = yv
    v, a = Derivatives(yv, t)
    
    y1 = y + v * dt + 0.5 * a * dt**2
    
    a1 = Derivatives([y1, v], t)[1]    
    
    v1 = v + 0.5 * (a + a1) * dt
    t1 = t + dt
    
    return np.array([y1, v1]), t1


def StepRungeKutta(Derivatives, yv, t, dt):
    """ Fourth-order Runge-Kutta algoritm """
    d1 = Derivatives(yv, t)
    d2 = Derivatives(yv + 0.5 * d1 * dt, t + 0.5 * dt)
    d3 = Derivatives(yv + 0.5 * d2 * dt, t + 0.5 * dt)
    d4 = Derivatives(yv + d3 * dt, t + dt)
    
    yv1 = yv + (d1 + 2 * d2 + 2 * d3 + d4) * dt / 6
    t1 = t + dt
    
    return yv1, t1


def StepEuler1AdvancedV(Derivatives, yv, t, dt):
    """ First-order Euler method with advanced velocity """
    y, v = yv
    v1 = v + Derivatives((y, v), t)[1] * dt     # v calculated first
    y1 = y + Derivatives((y, v1), t)[0] * dt    # ... and used the new value for y
    t1 = t + dt
    return np.array([y1, v1]), t1


def StepEuler1AdvancedY(Derivatives, yv, t, dt):
    """ First-order Euler method with advanced coordinate """
    y, v = yv
    y1 = y + Derivatives((y, v), t)[0] * dt     # y calculated first
    v1 = v + Derivatives((y1, v), t)[1] * dt    # ... and used the new value for v
    t1 = t + dt
    return np.array([y1, v1]), t1


def ODESolution(Derivatives, Step=StepEuler1, dt=0.1, maxt=30, initialConditions=(0,1)):
    """ Numerical solution of the differential equation with the specified integrator.

        Arguments:
        Derivatives -- a function returning the right-hand side of the ODE
        integrator -- function to make one step
        dt -- step size
        maxt -- integration performed from t=0 to t=maxt

        Returns:
        2D array with solution [[y0, v0], [y1, v1], ...]
        1D array with times [t0, t1, ...]
        name of the integrator
    """
    yv = np.array(initialConditions)        # Initial conditions
    yvs = [yv]                              # Array with results
    
    t = 0                                   # Actual time
    ts = [t]                                # Times
    
    while t < maxt:
        yv, t = Step(Derivatives, yv, t, dt) # Step
        
        yvs.append(yv)                      # Store position and velocity
        ts.append(t)                        # Store time
            
    return np.array(yvs), np.array(ts), Step.__name__


def ScipyODESolution(Derivatives, dt=0.1, maxt=30, initialConditions=(0,1)):
    """ Numerical solution of the differential equation with scipy solver """
    ts = np.arange(0, maxt, dt)
    yvs = odeint(Derivatives, initialConditions, ts)
        
    return yvs, ts, odeint.__name__


def ShowGraphSolutions(odeSolutions, ExactFunction=None, ylim=None):
    """ Shows multiple results in one graph.

        Arguments:
        odeSolutions -- a list of results returned from ODESolution
        ExactFunction -- if we want to show an extra panel with absolute deviations from the exact solution
        ylim -- limits for the vertical axis
    """
    if ExactFunction != None:
        plt.subplot(211)                    # First panel
        
    plt.xlabel('t')
    plt.ylabel('y')

    if ylim != None:
        plt.ylim(ylim)

    for yvs, ts, methodName in odeSolutions:       
        plt.plot(ts, yvs[:,0], label=methodName)  # We plot just the coordinate and disregard the velocity

    plt.legend()

    if ExactFunction != None:
        plt.subplot(212)                    # Second panel
        plt.xlabel('t')
        plt.ylabel('$\delta$y')
        if ylim != None:
            plt.ylim(ylim)
        
        for odeSolution in odeSolutions:
            plt.plot(odeSolution[1], LocalError(odeSolution, ExactFunction))
    
    plt.show()


def LocalError(odeSolution, ExactFunction):
    """ Calculates local error (deviation of the exact solution from the numerical solution).

        Arguments:
        odeSolution -- a result returned from ODESolution
    """
    yvs, ts, methodName = odeSolution
    return yvs[:,0] - ExactFunction(ts)


def CumulativeError(odeSolution, ExactFunction):
    """ Calculates average cumulative error (average deviation of the exact solution from the numerical solution).

        Arguments:
        odeSolution -- a result returned from ODESolution
    """
    residua = LocalError(odeSolution, ExactFunction)**2
    cumulativeError = np.sqrt(sum(residua) / len(residua))
    return cumulativeError

