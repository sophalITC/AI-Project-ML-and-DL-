import pylab
from sklearn import datasets
digit_dataset=datasets.load_digits()

print(digit_dataset.target[14])
pylab.imshow(digit_dataset.images[14],cmap=pylab.cm.gray_r)
pylab.show()
