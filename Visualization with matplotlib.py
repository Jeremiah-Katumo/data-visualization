from matplotlib import pyplot as plt
import numpy as np
import math
x=np.arange(0, math.pi*2, 0.5)
y=np.sin(x)
plt.plot(x,y)
plt.xlabel("angle")
plt.ylabel("sine")
plt.title('sine wave')

# The plot viewer window is invoked by the show() function
plt.show()

# The complete program is as follows
from matplotlib import pyplot as plt
import numpy as np
import math # needed for definition of pi
x=np.arange(0, math.pi*2, 0.05)
y=np.sin(x)
plt.plot(x,y)
plt.xlabel("angle")
plt.plot("sine")
plt.plot('sine wave')
plt.show()

# matplotlib- PyLab Basic Plotting
from numpy import *
from pylab import *
x= linspace(-3,3,30)
y=x**2
plot(x,y, 'r.')
plot(x,y, 'b.')
show()
# multiple plot command 
plot(x, sin(x))
plot(x, cos(x), 'b-')
plot(x, -sin(x), 'r--')
show()

# Object-oriented interface in matplotlib.axes.Axes
from matplotlib import pyplot as plt
import numpy as np
import math
x=np.arange(0, math.pi*2, 0.05)
y=np.sin(x)
fig=plt.figure() # instantiating the Figure object using figure() function
ax=fig.add_axes([0,0,1,1])
ax.plot()
ax.set_title("sine wave")
ax.set_xlabel('angle')
ax.set_ylabel('sine')
plt.show()

# Matplotlib-Axes Class
from matplotlib import pyplot as plt
y=[1,4,9,16,25,36,49,64]
x1=[1,16,30,42,55,68,77,88]
x2=[1,6,12,18,28,40,52,65]
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
l1=ax.plot(x1,y, 'ys-') # solid line with yellow colour and square marker
l2=ax.plot(x2,y, 'go--') # dash line with creen colour and circle marker
ax.legend(label=('tv', 'Smartohone'), loc='lower right') # legend placed at lower right
ax.set_title("Effect of advertisement on sales")
ax.set_xlabel('medium')
ax.set_ylabel('sales')
plt.show()

# Matplotlib - multiplots
from matplotlib import pyplot as plt
plt.subplot(subplot(nrows, ncols, index))
# plot a line by implicitly creating a subplot(111)
plt.plot([1,2,3])
# Then create a subplot that represents thetop of a grid with 2 rows and 1 column
# Since this subplot will overlap the first, the plot(andits axes) previously created, will be removed
plt.subplot(211)
plt.plot(range(12))
plt.subplot(212, facecolor='y') # creates second subplot with a yellow background
plt.plot(range(12))
# You can also use the add_subplot() but it will not overwrite the exixting plot
from matplotlib import pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.plot([1,2,3])
ax2=fig.add_subplot(221, facecolor='y') # creates ax2 subplot with a yellow background
ax2.plot([1,2,3])
# add an insert plot in the same figure by adding another axes object in the same figure canvas
import matplotlib.pyplot as plt
import numpy as np
import math
x=np.arange(0, math.pi*2, 0.5)
fig=plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8]) # main axes
axes2 = fig.add_axes([0.55,0.55,0.3,0.3]) # inset axes
y = np.sin(x)
axes1.plot(x, y, 'r')
axes2.plot(x, np.cos(x), 'g--')
axes1.set_title('sine')
axes2.set_title('cosine')
plt.show()

# create a subplot of 2 by 2 rows and columns respectively using subplots() and display 4 different plots in each subplot
import matplotlib.pyplot as plt
fig, a = plt.subplots(2,2)
import numpy as np
x = np.arange(1,5)
a[0][0].plot(x, x^2)
a[0][0].set_title('squares')
a[0][1].plot(x, np.sqrt(x))
a[0][1].set_title('square roots')
a[1][0].plot(x, np.log10(x))
a[1][0].set_title('logarithm')
a[1][1].plot(x, np.exp(x))
a[1][1].set_title('exponential')
plt.show()

# Using subplot2grid() function
import matplotlib.pyplot as plt
a1 = plt.subplot2grid((3,3), (0,0), colspan=2)
a2 = plt.subplot2grid((3,3),(0,1),rowspan=2)
a3 = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2)
import numpy as np
x = np.arange(1,10)
a2.plot(x, x^2)
a2.set_title('squares')
a1.plot(x, np.log10(x))
a1.set_title('logarithm')
a3.plot(x, np.exp(x))
a3.set_title('exponential')
plt.show()

# The grid() function
import matplotlib.pyplot as plt
import numpy as np
fig, axes =plt.subplots(1,3, figsize=(12,4))
x= np.arange(1,11)
axes[0].plot(x, x**3, 'b', lw=2)
axes[0].grid(True)
axes[0].set_title('The default grid')
axes[1].plot(x, np.log10(x), 'r')
axes[1].set_yscale('exp')
axes[1].grid(color='b', ls='-.', lw=0.5)
axes[1].set_title('Logarithmic scale (y)')
axes[1].set_xlabel("x axis")
axes[1].set_ylabel('y axis')
axes[2].plot(x,x)
axes[2].set_title('no grid')
fig.tight_layout()
plt.show()

# setting limits(minimum and maximum values of variables tobe displayed along x,y axes of a plot)
import matplotlib.pyplot as plt
fig = plt.figure()
a1 = fig.axes([0,0,1,1])
import numpy as np
x = np.arange(1,15)
a1.plot(x, np.exp(x))
a1.set_title('exponential')
a1.set_ylim(0,5000)
a1.set_xlim(1,20)
plt.show()

# Use of ticks and labels
import matplotlib.pyplot as plt
import numpy as np
import math
x = np.arange(0, math.pi*2, 0.06)
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8]) # these are main axes
y = np.sin(x)
ax.plot(x,y)
ax.set_xlabel('angles')
ax.set_ylabel('sine')
ax.set_xticks([0,2,4,6])
ax.set_yticklabels(['zero','two','four','six'])
ax.set_yticks([-2,-1,0,1,2])
plt.show()

# Bar plots
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.axes([0,0,1,1])
cars = ['Chevrolet', 'Audi', 'Maybach', 'Ferrarri', 'Audi', 'BMW']
workers = [20,13,26,9,35,31]
ax.bar(cars, workers)
plt.show()
data =[[30,25,50,20],[40,20,50,30],[25,32,35,19]]
x = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x+0.00, data[0], color='b', width=0.25)
ax.bar(x+0.25, data[1], color='r', width=0.25)
ax.set_xticks([0.25,1.25,2.25,3.25])
ax.set_xticklabels([2018,2019,2020,2021])
ax.legend(labels=['CS','IT','TC'])
plt.show()

# Histogram
from matplotlib import pyplot as plt
import numpy as np
fig,ax=plt.subplots(1,1)
a = np.array([20,86,71,63,96,23,45,64,31,20,51,41,79,31,37])
ax.hist(a, bins = [0,25,50,75,100])
ax.set_title("histogram of result")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('marks')
ax.set_ylabel('no. of drivers')
plt.show()

# Pie charts
from matplotlib import pyplot as plt
import numpy as np
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.axis('equal')
langs=['C', 'C++', 'Java', 'Python', 'PHP']
students=[23,17,35,29,12]
ax.pie(students, labels=langs,autopct='%1.2f%%')
plt.show()

# Scatter plot
import matplotlib.pyplot as plt
males_grades = [89, 90, 70, 80, 99, 80, 90, 100, 80, 54]
females_grades = [40, 30, 49, 58, 100, 48, 68, 45, 20, 30]
grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(grades_range, males_grades, color='b')
ax.scatter(grades_range, females_grades, color='g')
ax.set_xlabel('Grades Range')
ax.set_ylabel('Grades attained')
ax.set_title('Scatter Plot')
plt.show()

# Contour plot
import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-3.0, 3.0, 100)
ylist = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()

# Boxplot
np.random.seed(10)
collectn_1 = np.random.normal(700, 20, 200)
collectn_2 = np.random.normal(90, 30, 200)
collectn_3 = np.random.normal(90, 20, 200)
collectn_4 = np.random.normal(80, 15, 200)
fig = plt.figure()
# Create an axes instance
ax = fig.add_axes([0,0,1,1])
# Create the boxplot
bp = ax.boxplot(data_to_plot)
plt.show()

# Quiver plot
x,y = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .25))
z = x*np.exp(-x**2 - y**2)
v, u = np.gradient(z, .2, .2)
fig, ax = plt.subplots()
q = ax.quiver(x,y,u,v)
plt.show()

# 3D plotting(3D scatter plot)from mpl_toolkits import mplot3d
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
z = np.linspace(0, 1, 100)
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)
c = x + y
ax.scatter(x, y, z, c=c)
ax.set_title('3d Scatter plot')
plt.show()









