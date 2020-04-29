import numpy as np

#insert the tables
M1 = np.array((
[100, 2200, 45],
[300, 400, 24],
[5,	6, 46],
[7,6,50],
[7,	8, 40],
[500, 500, 24]), float)

M2 = np.array((
[100, 2200, 45], #remain
[300, 400, 24], #remain
[8 ,8 ,86],     #collision
[500, 500, 24], #remain
[100000,1000000,5]), float)  #new baby

#1st strep : take only two first columns
M1_2col = np.empty((M1.shape[0],2), dtype=float)
for i  in range(M1.shape[0]):
    M1_2col[i, 0] = M1[i, 0]
    M1_2col[i, 1] = M1[i, 1]

M2_2col = np.empty((M2.shape[0],2), dtype=float)
for i  in range(M2.shape[0]):
    M2_2col[i, 0] = M2[i, 0]
    M2_2col[i, 1] = M2[i, 1]

#2nd step: take each row from M1 and check where it is in the M2
#instantly as soon as you find it take growth rate and store it

#define the distance function:
def distance(x1, y1, x2, y2):
    x_diff = 0
    y_diff = 0
    point_distance = 0
    x_diff = (x1 - x2) ** 2
    y_diff = (y1 - y2) ** 2
    point_distance = (x_diff + y_diff) ** 0.5
    return point_distance

distances_stored = np.empty((M1_2col.shape[0],M2_2col.shape[0]),dtype=object)
for i in range(M1_2col.shape[0]):
    d=0
    for j in range((M2_2col.shape[0])):
        d = distance(M1_2col[i][0], M1_2col[i][1], M2_2col[j][0], M2_2col[j][1])
        distances_stored[i][j] = round(d,2)
#print(distances_stored)
mini = np.empty((M1_2col.shape[0],M2_2col.shape[0]),dtype=object)
for i in range(M1_2col.shape[0]):
    for j in range((M2_2col.shape[0])):
        if distances_stored[i,j] == np.amin(distances_stored[i]):
            mini[i,j] = distances_stored[i,j]
        else:
            mini[i,j] = -1
#print(mini)
G_Rate_matrix = np.empty((M1.shape[0],1), dtype=object)
for i in range(M1_2col.shape[0]):
    for j in range((M2_2col.shape[0])):
        if mini[i,j] == -1:
            pass
        else:
            G_Rate_matrix[i] = abs(
                (M2[j][2] - M1[i][2]) / (M1[j][2])
            )
#print(G_Rate_matrix)
G_Rate_M_no0 = (G_Rate_matrix == 0).sum(1)
G_Rate_matrix_clean = G_Rate_matrix[G_Rate_M_no0 == 0, :]
G_Rate = round((np.mean(G_Rate_matrix_clean)), 3)
print(f'the total rate is: ', G_Rate)


