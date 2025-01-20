from bisect import bisect_left

n = int(input())
items = []
que = []
for i in range(n):
    items.append(tuple(map(int, input().split())))
m = int(input())
for i in range(m):
    que.append(tuple(map(int, input().split())))


items.sort()
x_vals = []
for i in range(n):
    x_vals.append(items[i][0])

for x_cur, k_cur in que:

    pos = bisect_left(x_vals, x_cur)
    left, right = pos - 1, pos


    near = []
    while len(near) < k_cur and (left >= 0 or right < n):
        if left >= 0 and (right >= n or abs(x_cur - x_vals[left]) <= abs(x_vals[right] - x_cur)):
            near.append(items[left])
            left -= 1
        elif right < n:
            near.append(items[right])
            right += 1

    if len(near) == k_cur:
        last_distance = abs(x_cur - near[-1][0])
        if left >= 0 and abs(x_cur - x_vals[left]) == last_distance:
            print(-1)
            continue
        if right < n and abs(x_cur - x_vals[right]) == last_distance:
            print(-1)
            continue

    ansi = 0
    for i in range(k_cur):
        ansi += near[i][1]
    print(ansi / k_cur)
