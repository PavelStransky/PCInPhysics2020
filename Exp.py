import numpy as np
from ODE import *
from scipy.stats import linregress

def Derivatives(yv, t):
    """ Derivatives for the exponential system """
    y, v = yv
    return np.array([v, y])


def ExpM(t):
    """ Auxiliary function -- exact solution for ODE given by Derivatives with initialConditions=(1,-1) """
    return np.exp(-t)


def CompareMethods(dt = 0.1):
    """ Compare different methods """
    odeSolutions = []
    ic = (1, -1)

    odeSolutions.append(ODESolution(Derivatives, Step=StepEuler1, dt=dt, initialConditions=ic))
    odeSolutions.append(ODESolution(Derivatives, Step=StepEuler1AdvancedY, dt=dt, initialConditions=ic))
    odeSolutions.append(ODESolution(Derivatives, Step=StepEuler1AdvancedV, dt=dt, initialConditions=ic))
    odeSolutions.append(ODESolution(Derivatives, Step=StepEuler2, dt=dt, initialConditions=ic))
    odeSolutions.append(ODESolution(Derivatives, Step=StepVerlet, dt=dt, initialConditions=ic))
    odeSolutions.append(ODESolution(Derivatives, Step=StepRungeKutta, dt=dt, initialConditions=ic))
    odeSolutions.append(ScipyODESolution(Derivatives, dt=dt, initialConditions=ic))

    #plt.figure(figsize=(8,8))
    ShowGraphSolutions(odeSolutions, ExpM, ylim=(-0.5,1))


def ShowGraphCumulativeErrors():
    dts = np.linspace(0.002, 0.1, 100)
    ic = (1, -1)

    def OneCurve(Step):
        """ Auxiliary function to plot one curve into the graph """
        errors = [CumulativeError(ODESolution(Derivatives, Step=Step, dt=dt, initialConditions=ic), ExpM) for dt in dts]
        slope, intercept, r_value, p_value, std_err = linregress(np.log(dts), np.log(errors)) # Linear regression
        plt.loglog(dts, errors, label="%s $\\alpha$=%.2f" % (Step.__name__, slope))
        
    #plt.figure(figsize=(8,4))

    OneCurve(StepEuler1)
    OneCurve(StepEuler2)
    OneCurve(StepVerlet)
    OneCurve(StepRungeKutta)
    OneCurve(StepEuler1AdvancedY)

    plt.xlabel("$\Delta$t")
    plt.ylabel("$\mathcal{E}$")
    plt.legend()
    plt.show()
