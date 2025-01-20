# опять начну с описания интуитивной идеи, у нас есть корреляция пирсона которая показывает связь между столбцами (сравниваем на то как сильно отходят соответствующие значения от среднего + учитываем дисперсию)
# мы учитываем именно числовые данные (если смотреть на корреляцию именно для закодированного кат.признака  например номера цветов (1,2,3,3,2,3) то отход от середины не особо что-то показывает, ну то есть резкое увеличение номера не особо что-то показывает
# поэтому как бы и кодируем каждую составляющую категориального признаке бинарными значениями чтобы теперь смотреть на столбец только где есть этот компонент (чем-то похоже на идею из D c усл. дисперсиями)
# и так отхождения от среднего будет показывать связь (но вот если бы брали просто столбец частот то было бы не очень так как опять снижение увечение не особо что-то показывало так как не понятно что это за категория)
#в итоге считаем просто взвешанное среднее по все таким "столбцам наличия данного признака"  чтобы учесть не только связь для данной подкатегории но и частоту

import numpy as np

n, k = map(int, input().split())
kategory = np.array(list(map(int, input().split())))
vals = np.array(list(map(int, input().split())))
vals_mean = np.mean(vals)
vals_std = np.std(vals)
kateg_dif_vals = np.bincount(kategory - 1, minlength=k)
kateg_dif_probs = kateg_dif_vals / n

pirson_cor = np.zeros(k)

for kat in range(k):
    mask = (kategory == kat + 1)
    bin_kat_col_mean = np.mean(mask)
    bin_kat_col_std = np.sqrt(bin_kat_col_mean * (1 - bin_kat_col_mean))
    if bin_kat_col_std > 0 and vals_std > 0:
        cov = np.dot(mask - bin_kat_col_mean, vals - vals_mean) / n
        pirson_cor[kat] = cov / (bin_kat_col_std * vals_std)
probs_cors = pirson_cor * kateg_dif_probs
print(f"{np.sum(probs_cors):.12f}")
