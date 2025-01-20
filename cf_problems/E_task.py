# опять начну с описания интуитивной идеи, тут кажется идея очень похожа на то что было в задаче D,но теперь смотрим на доли вместо вероятностей и как бы рассматриваем
# теперь наши разницы (можно и дисперсии например, но у нас разность) рассматривать как компоненты общей разницы (или дисперсии) по X, кажется это лучше использовать если у нас неизвестно рапределение
# ну то есть через внутриклассовое расстояние смотрим на разницу значений внутри каждого класса Yi
# а через межклассовое расстояние смотрим на разницу между средними значениями внутри классов и общим средним
# и если получится что внутри групп разница небольшая а между ними большая то значит есть сильная зависимость от признака
# чтобы быстро считать заведем как бы массив префиксных сумм по группам для суммы внутри нее
# ( храним пару число элементов и сумму значений)  и сумма всех элементов, когда приходит новая пара то мы учтем его разность со всеми элементами внутри группы через разность значения*( на текущее число элементов в группе) и текущей суммы по группе ,
# а для внешней как разность значения*( на текущее число элементов всех - число в этой  группе) и (текущей суммы всех элементов - сумма по этой группе),
# и при рассмотрении следующих будем как раз найдем пары и с этим



num_classes = int(input())
num_objects = int(input())

all_x_values = []
class_x_values = [[] for _ in range(num_classes)]


for _ in range(num_objects):
    x, class_id = map(int, input().split())
    all_x_values.append(x)
    class_x_values[class_id - 1].append(x)

all_x_values.sort()

intra_class_distance = 0
for i in range(num_objects):
    total_objects = num_objects - 1
    intra_class_distance += 2 * i * all_x_values[i]
    intra_class_distance -= 2 * all_x_values[i] * total_objects
    intra_class_distance += i * 2 * all_x_values[i]

inter_class_distance = 0
for x_values in class_x_values:
    x_values.sort()
    for i in range(len(x_values)):
        total_class_objects = len(x_values) - 1
        inter_class_distance += 2 * i * x_values[i]
        inter_class_distance -= 2 * x_values[i] * total_class_objects
        inter_class_distance += i * 2 * x_values[i]

print(inter_class_distance)
print(intra_class_distance - inter_class_distance)
