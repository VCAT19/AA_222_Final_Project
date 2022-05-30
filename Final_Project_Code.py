import matplotlib.pyplot
import numpy
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.optimize import line_search

def main():

    def gradient_descent(fun, x0, alpha, epsilon):
        i = 0
        x = x0
        diff = 1000000
        der = numpy.polyder(fun)
        while diff > epsilon:
            x_prev = x
            g = der(x)
            x = x - alpha * g
            diff = x_prev - x
            i += 1
        return x

    # Import Data2
    data_arr_1 = genfromtxt('2021_4_FB_Data.csv', delimiter=',', names=True)
    fastball_mhp = data_arr_1['ff_avg_speed']; run_value_fastball = -data_arr_1['run_value']; whiff_fastball = data_arr_1['whiff_percent']

    # Construct MHP vs Run Value Surrogate Model
    polynomial_degree = 7
    x_grid_fastball = numpy.linspace(85, 100, num=182)
    trend_MHP_RV = numpy.polyfit(fastball_mhp, run_value_fastball, polynomial_degree)
    trendpoly_MHP_RV = numpy.poly1d(trend_MHP_RV)
    #g_MHP_RV = numpy.polyder(trendpoly_MHP_RV)
    opt_MHP_1 = gradient_descent(trendpoly_MHP_RV, 93, 2, .1)
    print('Optimal MHP for Run Value: '+str(opt_MHP_1))
    #print(line_search(trendpoly_MHP_RV, numpy.polyder(trendpoly_MHP_RV), 93,-1))


    # Construct MHP vs Whiff Rate Surrogate Model
    trend_MHP_whiff = numpy.polyfit(fastball_mhp, whiff_fastball, polynomial_degree)
    trendpoly_MHP_whiff = numpy.poly1d(trend_MHP_whiff)
    #g_MHP_RV = numpy.polyder(trendpoly_MHP_whiff)
    opt_MHP_2 = gradient_descent(trendpoly_MHP_whiff, 88.707, 2, .1)
    print('Optimal MHP for Whiff Rate: ' + str(opt_MHP_2))
    #print(line_search(trendpoly_MHP_whiff, numpy.polyder(trendpoly_MHP_whiff), 88,-1))

    # Plot ploy function over data points to verify it's even close
    plot_option = 'MHP_Whiff'
    if plot_option == 'MHP_RV':
        plt.plot(x_grid_fastball, trendpoly_MHP_RV(x_grid_fastball))
        plt.plot(fastball_mhp, run_value_fastball, 'o')
        plt.title('MHP vs Run Value of all Qualfying Pitchers from 2021 MLB Regular Season')
        plt.legend(['Polynomial Fit', 'Actual Data Points'])
        plt.xlabel('MHP')
        plt.ylabel('-Run Value')
        plt.grid()
        plt.show()
    elif plot_option == 'MHP_Whiff':
        plt.plot(x_grid_fastball, trendpoly_MHP_whiff(x_grid_fastball))
        plt.plot(fastball_mhp, whiff_fastball, 'o')
        plt.title('MHP vs Whiff Rate of all Qualfying Pitchers from 2021 MLB Regular Season')
        plt.legend(['Polynomial Fit', 'Actual Data Points'])
        plt.xlabel('MHP')
        plt.ylabel('Whiff Rate %')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
