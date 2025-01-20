#идея похожа на прошлые задачи, но теперь используем энтропию, кстати она уже была у нас на ДМ
# энтропия показывает количество информации которое дает какой-то источник
# мне нравится аналогия с автоматом который генерирует некоторые символы с некоторой вероятностью
# и мы хотим посчитать сколько вопросов потратим, каждый раз спрашиваем это самый вероятный символ и получаем дерево
# число вопросов как раз взвешанное среднее  pi* (число вопросов)i, ну а число вопросов на заданной глубине сичтаем как log(числа исходов на этом уровне)
#  а это 1/pi так и получаем
# считаем через условную энтропию тк она измеряет оставшуюся неопределенность и если условная энтропия большая значит,
#X мало дает мало информации о Y и они слабо связаны, ну а дальше просто матожидание по энтропиям
# будем через словари считать частоты пар и значений 1 признака
import math
from collections import defaultdict

Kx, Ky = map(int, input().split())
n = int(input())


dictX = defaultdict(int)
pairsXY = defaultdict(dict)
for i in range(n):
    x, y = map(int, input().split())
    dictX[x] +=1
    if y not in pairsXY[x]:
        pairsXY[x][y] = 0
    pairsXY[x][y] +=1

ans = 0
for x in dictX:
    px = dictX[x]/n
    entr_x = 0
    for y in pairsXY[x]:
        py = pairsXY[x][y]/ dictX[x]
        entr_x += -1* py*math.log(py)
    ans += entr_x * px
print(f"{ans : .9f}")