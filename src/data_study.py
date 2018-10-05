from util import *
import numpy as np
import matplotlib.pyplot as plt

train_data = listYearbook(train=True, valid=False)
d = {}
for x,y in train_data:
	if int(y) not in d:
		d[int(y)] = [0.0,0.0]
	if "M" in x:
		d[int(y)][0] += 1
	if "F" in x:
		d[int(y)][1] += 1


years = list(d.keys())
years.sort()


# Number of examples
array_m = []
array_f = []
for y in range(min(years),2017):
	if y in years:
		array_m.append(d[y][0])
		array_f.append(d[y][1])
	else:
		array_m.append(0)
		array_f.append(0)

N = len(array_m)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

plt.figure()
p1 = plt.bar(ind, array_m, width)
p2 = plt.bar(ind, array_f, width, bottom=array_m)

plt.ylabel('Number of Examples')
plt.title('Number of Examples by years and gender')
plt.xticks(ind[::10], np.arange(min(years),2017, step=10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.savefig('number_of_examples.png')

plt.figure()

# Ratio of examples
array_m = []
array_f = []
for y in range(min(years),2017):
	if y in years:
		total = d[y][0] + d[y][1]
		array_m.append(d[y][0]/total)
		array_f.append(d[y][1]/total)
	else:
		array_m.append(0)
		array_f.append(0)	

N = len(array_m)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, array_m, width)
p2 = plt.bar(ind, array_f, width, bottom=array_m)

plt.ylabel('Ratio of Examples')
plt.title('Ratio of Examples by gender per year')
plt.xticks(ind[::10], np.arange(min(years),2017, step=10))
plt.yticks(np.arange(0,1.1,step=0.1))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.axhline(y=0.5, color='r', linestyle='-')
plt.savefig('ratio_of_examples.png')
