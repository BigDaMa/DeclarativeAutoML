import gpflow
import gpflowopt
import numpy as np

# Objective & constraint
def townsend(X):
    return -(np.cos((X[:,0]-0.1)*X[:,1])**2 + X[:,0] * np.sin(3*X[:,0]+X[:,1]))[:,None]

def constraint(X):
    return -(-np.cos(1.5*X[:,0]+np.pi)*np.cos(1.5*X[:,1])+np.sin(1.5*X[:,0]+np.pi)*np.sin(1.5*X[:,1]))[:,None]

# Setup input domain
domain = gpflowopt.domain.ContinuousParameter('x1', -2.25, 2.5) + \
         gpflowopt.domain.ContinuousParameter('x2', -2.5, 1.75)

# Initial evaluations
design = gpflowopt.design.LatinHyperCube(11, domain)
X = design.generate()
Yo = townsend(X)
Yc = constraint(X)

# Models
objective_model = gpflow.gpr.GPR(X, Yo, gpflow.kernels.Matern52(2, ARD=True))
objective_model.likelihood.variance = 0.01
constraint_model = gpflow.gpr.GPR(np.copy(X), Yc, gpflow.kernels.Matern52(2, ARD=True))
constraint_model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)
constraint_model.likelihood.variance = 0.01
constraint_model.likelihood.variance.prior =  gpflow.priors.Gamma(1./4.,1.0)

# Setup
ei = gpflowopt.acquisition.ExpectedImprovement(objective_model)
pof = gpflowopt.acquisition.ProbabilityOfFeasibility(constraint_model)
joint = ei * pof

# First setup the optimization strategy for the acquisition function
# Combining MC step followed by L-BFGS-B
acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, 200),
                                                   gpflowopt.optim.SciPyOptimizer(domain)])

# Then run the BayesianOptimizer for 50 iterations
optimizer = gpflowopt.BayesianOptimizer(domain, joint, optimizer=acquisition_opt, verbose=True)
result = optimizer.optimize([townsend, constraint], n_iter=50)

print(result)