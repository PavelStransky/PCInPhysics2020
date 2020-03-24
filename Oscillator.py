import numpy as np
from ODE import *
from scipy.stats import linregress

def Energy(yv):
    """ Return oscillator energy.
    
        Arguments:
        yv -- pairs in the form [position, velocity]
    """      
    y, v = yv
    return 0.5 * (y**2 + v**2)


def Derivatives(yv, t):
    """ Derivatives for the oscillator system """
    y, v = yv
    return np.array([v, -y])


def CompareMethods(dt = 0.1):
    """ Compare different methods """
    odeSolutions = []

    odeSolutions.append(ODESolution(Derivatives, Step=StepEuler1, dt=dt))
    odeSolutions.append(ODESolution(Derivatives, Step=StepEuler1AdvancedY, dt=dt))
    odeSolutions.append(ODESolution(Derivatives, Step=StepEuler2, dt=dt))
    odeSolutions.append(ODESolution(Derivatives, Step=StepVerlet, dt=dt))
    odeSolutions.append(ODESolution(Derivatives, Step=StepRungeKutta, dt=dt))
    odeSolutions.append(ScipyODESolution(Derivatives, dt=dt))

    #plt.figure(figsize=(8,8))
    ShowGraphSolutions(odeSolutions, np.sin)

    #plt.figure(figsize=(8,4))
    ShowGraphEnergy(odeSolutions)


def ShowGraphEnergy(odeSolutions):
    """ Shows energy of multiple results in one graph """
    plt.xlabel('t')
    plt.ylabel('E')

    for yvs, ts, methodName in odeSolutions:
        Es = [Energy(yv) for yv in yvs]
        plt.plot(ts, Es, label=methodName)

    plt.legend()
    plt.show()


def ShowGraphCumulativeErrors():
    dts = np.linspace(0.002, 0.1, 100)

    def OneCurve(Step):
        """ Auxiliary function to plot one curve into the graph """
        errors = [CumulativeError(ODESolution(Derivatives, Step=Step, dt=dt), np.sin) for dt in dts]
        slope, intercept, r_value, p_value, std_err = linregress(np.log(dts), np.log(errors)) # Linear regression
        plt.loglog(dts, errors, label="%s $\\alpha$=%.2f" % (Step.__name__, slope))
        
    #plt.figure(figsize=(8,5))

    OneCurve(StepEuler1)
    OneCurve(StepEuler2)
    OneCurve(StepVerlet)
    OneCurve(StepRungeKutta)
    OneCurve(StepEuler1AdvancedY)

    plt.xlabel("$\Delta$t")
    plt.ylabel("$\mathcal{E}$")
    plt.legend()
    plt.show()
