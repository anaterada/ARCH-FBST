from tqdm import tqdm

class MetropolisSampler(object):
    import numpy as np

    '''
        Amostrador Metropolis-Hastings simples com kernel gaussiano multivariado
        * log_density_function: log densidade da distribuição alvo
        * warmup_iterations: Iterações de aquecimento da cadeia
        * kernel_sigma: lista de variâncias a serem utilizadas em um kernel gaussiano diagonal.
O número de elementos deve coincidir com a dimensão do espaço amostrado.
        * restrictions: Lista de tuplas indicando a região válida para cada uma das variáveis. 
Exemplo: [(-5, 5), (-2, 3)] delimita que as amostras satisfaçam -5 <= theta_1 <= 5 e -2 <= theta2 <= 3 simultaneamente.

    '''

    def __init__(self, log_density_function, starting_point, kernel_sigma=[], restrictions=None, warmup_iterations=500):
        self.log_density_function = log_density_function
        self.starting_point = starting_point
        self.D = len(self.starting_point)
        self.warmup_iterations = warmup_iterations
        self.kernel_sigma = kernel_sigma

        if restrictions is None:
            restrictions = []
            for _ in self.D:
                restrictions.append([-self.np.inf, self.np.inf])
        
        self.restrictions = restrictions

    def _sample(self, n, x0, show_progress=False):
        samples = []
        current_value = x0
        
        for _ in tqdm(range(n)) if show_progress else range(n):
            proposal = []
            for d in range(self.D):
                sampled_point = float(self.np.random.normal(current_value[d], self.kernel_sigma[d], 1))

                if self.restrictions[d][0] <= sampled_point <= self.restrictions[d][1]:
                    proposal.append(sampled_point)
            
            if len(proposal) < self.D:
                # Ponto proposto não satisfaz pelo menos uma das restrições
                # o ponto é rejeitado
                continue

            acc_alpha = self.np.exp(self.log_density_function(proposal) - self.log_density_function(current_value))

            if self.np.isnan(acc_alpha):
                continue

            if float(self.np.random.np.random.uniform(low=0.0, high=1.0, size=1)) <= acc_alpha:
                current_value = proposal

            samples.append(current_value)

        return self.np.array(samples)

    def warm(self, warmup_iterations):
        self.starting_point = self._sample(warmup_iterations, self.starting_point)[-1]

    def sample(self, n, show_progress=False):
        return self._sample(n, self.starting_point, show_progress)
