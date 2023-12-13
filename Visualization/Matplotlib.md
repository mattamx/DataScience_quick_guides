# Create Plot

```python
import matplotlib.pyplot as plt
```

## Figure
```python
fig = plt.figure()
fig2 = plt.figure(figsize=plt.figaspect(2.0))
```

## Axes
```python
fig.add_axes()
ax1 = fig.add_subplot(221) #row-col-num
ax3 = fig.add_subplot(212)
fig3, axes = plt.subplots(nrows=2,ncols=2)
fig4, axes2 = plt.subplots(ncols=3)
```

# Save Plot
```python
plt.savefig('foo.png') #Save figures
plt.savefig('foo.png',  transparent=True) #Save transparent figures
```

# Show Plot
```python
plt.show()
```

# Plotting Routines

## 1D Data
```python
fig, ax = plt.subplots()
lines = ax.plot(x,y) #Draw points with lines or markers connecting them
ax.scatter(x,y) #Draw unconnected points, scaled or colored
axes[0,0].bar([1,2,3],[3,4,5]) #Plot vertical rectangles (constant width)
axes[1,0].barh([0.5,1,2.5],[0,1,2]) #Plot horiontal rectangles (constant height)
axes[1,1].axhline(0.45) #Draw a horizontal line across axes
axes[0,1].axvline(0.65) #Draw a vertical line across axes
ax.fill(x,y,color='blue') #Draw filled polygons
ax.fill_between(x,y,color='yellow') #Fill between y values and 0
```

## 2D Data
```python
fig, ax = plt.subplots()
im = ax.imshow(img, #Colormapped or RGB arrays
      cmap= 'gist_earth', 
      interpolation= 'nearest',
      vmin=-2,
      vmax=2)
axes2[0].pcolor(data2) #Pseudocolor plot of 2D array
axes2[0].pcolormesh(data) #Pseudocolor plot of 2D array
CS = plt.contour(Y,X,U) #Plot contours
axes2[2].contourf(data1) #Plot filled contours
axes2[2]= ax.clabel(CS) #Label a contour plot
```

## Vector Fields
```python

```
