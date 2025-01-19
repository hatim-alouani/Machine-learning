import math

taille = [6,7,8,9,5]
poids = [150,170,200,210,140]
fruit = ["P","P","O","O", "P"]

newFruit =[7, 200]

k = 2

def getDistance(newFruit, taille, poids, i):
    return math.sqrt((newFruit[0] - taille[i])**2 + (newFruit[1] - poids[i])**2)

def prediction(taille, poids, newFruit):
    p = 0
    o = 0
    i = 0
    distances = []
    for i in range(len(taille)):
        dist = getDdistance(newFruit, taille, poids, i)
        distances.append((dist, fruit[i]))
    
    distances.sort()
    for i in range(k) :
        if (distances[i][1] == "P"):
            p += 1
        if (distances[i][1] == "O"):
            o += 1
    if p > o:
        return ("POMME")
    return ("ORANGE")
    
print(prediction(taille, poids, newFruit))