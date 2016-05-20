Adam Optimizer
==============
A Method for Stochastic Optimization
------------------------------------

[PDF](http://arxiv.org/pdf/1412.6980.pdf)

#### Why strive to understand optimization methods in neural networks?

 1. In order to better select an optimizer for a given task 
 2. To fine tune its hyperparameters with more ease

#### Why do we need this algorithm?
* Gradient descent is already quite a good algorithm.
* Often objective functions are stochastic. E.g. they are composed of a sum
  of subfunctions evaluated at different subsamples of data. In this case 
	Stochastic Gradient Descent or ascent, is more appropriate.
* This algorithm aims 

#### How does it work at a logical level?

#### Python Style Psuedocode
```python
def Adam(self, stepsize, beta_1, beta_2, func, theta_0):
	m_0 = 0 # initialize our 1st moment vector
	v_0 = 0 # initialize our 2nd moment vector
	t   = 0 # initialize our timestep
	while theta_0 is not converged:
		 
	
```
