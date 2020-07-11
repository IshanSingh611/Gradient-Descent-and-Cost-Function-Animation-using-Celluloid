#MATHS BEHIND SIMPLE LINEAR REGRESSION FT. CELLULOID
#pip install celluloid

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

# gradient_descent function to reduce cost and find best coeff and intercept
def gradient_descent(x,y):
    #initialization of coeff and y_intercept
    coeff = y_intercept = 0 

    #initialization learning rate i.e. alpha
    alpha = 0.01

    #initialization of iterations or epochs
    epochs = 500 

    #initialize figure object
    fig  = plt.figure(figsize=(13,6))

    #allocate memory for Camera class i.e. Celluloid
    camera = Camera(fig)

    # iter_lst for storing number of iterations
    iter_lst = []

    # cost_lst for storing cost i.e. M.S.E.
    cost_lst = []
    n = len(x)
    for i in range(epochs):

        #random initialization of coeff and y-intercept
        y_predicted = coeff*x+y_intercept

        # calculating mean square error
        cost = (1/n)*np.sum(np.square(y-y_predicted))

        # append the cost & iterations in respective list
        cost_lst.append(cost)
        iter_lst.append(i)

        # creation of subplots to visualize cost & coeff,intercept  
        plt.subplot(121)
        plt.scatter(x,y,color='green')
        plt.text(x=1,y=0,s="y-intercept: {:1f} & coeff: {:1f}".format(y_intercept,coeff),fontdict={'fontsize':12})
        plt.xlabel('Independent Variable (x)')
        plt.ylabel('Response Variable (y)')
        plt.plot(x,y_predicted,color='red')
        plt.subplot(122)
        plt.xlabel('Number of epochs (x)')
        plt.ylabel('Cost function (y)')
        plt.plot(iter_lst, cost_lst,color='blue')
        plt.text(x=300,y=1,s="Cost: {:1f}".format(cost),fontdict={'fontsize':12})

        #capturing snapshot of each & every iteration
        camera.snap()

        #partial derivative of coeff
        coeff_derivative = -(2/n)*(sum(x*(y-y_predicted)))

        #parital derivative of intercept
        y_intercept_derivative = -(2/n)*(sum(y-y_predicted))

        #update coeff and intercept iteratively 
        #newweight = oldweight - learning_rate * partialderivatives
        coeff = coeff-alpha*coeff_derivative
        y_intercept = y_intercept-alpha*y_intercept_derivative

        #print("m {} b {} cost {} iters {}".format(m_curr,b_curr,cost,i))

    #plt.scatter(x,y,c='red')
    plt.suptitle("GRADIENT DESCENT & COST FUNCTION PLOT")

    #animate the snapshots
    animate=camera.animate()
    

    #plot visualization
    plt.show()

# numpy array with single feature vector (x) for easy understanding
x = np.array([1,2,3,5,6,8,10])

# target - y
y = np.array([2,1,2,3,3,4,4])

#function call
gradient_descent(x,y)
