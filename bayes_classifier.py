from sklearn.naive_bayes import GaussianNB
import numpy as np

gnb = GaussianNB()

# known data: h, w, shoes
X = [[160, 60, 38], [177, 70, 43],
     [171, 75, 42], [154, 54, 37],
     [166, 65, 40], [190, 90, 47],
     [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [181, 85, 43],
     [181, 80, 44], [189, 79, 45]]

Y = ['female', 'male',
     'male',   'female',
     'male',   'male',
     'female', 'female',
     'female', 'male',
     'male',   'male']

X = np.array(X)
Y = np.array(Y)

gnb = gnb.fit(X, Y)

# trying to predict
target = [150, 70, 35]
prediction = gnb.predict([target])

print(target, "is ", prediction)
