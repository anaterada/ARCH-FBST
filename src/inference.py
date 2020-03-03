import numpy as np

def evidence(simulations, unrestricted_loglikelihood_fn, restricted_maximum_loglikelihood_value):
    '''
        Calcula a medida de evidência com as simulações geradas, a função objetivo irrestrita e 
        o valor máximo da função objetivo restrita a hipótese do teste.
    '''
    unrestricted_loglikelihood_value = unrestricted_loglikelihood_fn(simulations)
    return (1 - np.average(unrestricted_loglikelihood_value >= restricted_maximum_loglikelihood_value))