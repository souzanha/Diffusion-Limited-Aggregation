import numpy as np
import matplotlib.pyplot as plt

def set_grid(size,step):
    x = np.arange(0,size+step,step)
    y = np.arange(0,size+step,step)
    xx, yy = np.meshgrid(x, y)
    matrix_points = np.zeros((np.shape(yy)[0],np.shape(xx)[0]))
    return xx, yy, matrix_points

def set_outer_circle(matrix_points):
    circle_limit = np.shape(matrix_points)[0]/2

    for i in range(np.shape(matrix_points)[0]):
        for j in range(np.shape(matrix_points)[1]):

            if np.sqrt((i-circle_limit)**2+(j-circle_limit)**2) >= circle_limit:
                matrix_points[i][j] = 0.5

    return matrix_points

def set_inner_circle(matrix_points):

    circle_limit = np.shape(matrix_points)[0]/2.2

    center = np.shape(matrix_points)[0]/2 - circle_limit

    for i in range(np.shape(matrix_points)[0]):
        for j in range(np.shape(matrix_points)[1]):

            if np.sqrt((i-circle_limit-center)**2+(j-circle_limit-center)**2) >= circle_limit and matrix_points[i][j] != 0.5:
                matrix_points[i][j] = 0.2

    return matrix_points

def add_initial_seed(matrix_points,size,step): 
    #sets the initial seed, puts it at random somwhere close to the central
    random_point = [np.random.choice(np.shape(matrix_points)[0],1, replace=False)[0],
                    np.random.choice(np.shape(matrix_points)[0],1, replace=False)[0]]
    x_seed = xx[random_point[0]][random_point[1]]
    y_seed = yy[random_point[0]][random_point[1]]

    seed_limit_1 = (size/2)-(step*2)
    seed_limit_2 = (size/2)+(step*2)

    if seed_limit_2>x_seed>seed_limit_1 and seed_limit_2>y_seed>seed_limit_1:
        matrix_points[random_point[0]][random_point[1]] = 0.8
    else:
        while (x_seed>seed_limit_2 or x_seed<seed_limit_1) or (y_seed>seed_limit_2 or y_seed<seed_limit_1):
            random_point = [np.random.choice(np.shape(matrix_points)[0],1, replace=False)[0], 
                            np.random.choice(np.shape(matrix_points)[0],1, replace=False)[0]]
            x_seed = xx[random_point[0]][random_point[1]]
            y_seed = yy[random_point[0]][random_point[1]]
        matrix_points[random_point[0]][random_point[1]] = 0.8
    return matrix_points, random_point

def release_walker(matrix_points): #releases walker from the inner sphere vertix
    release = False 

    while release == False:
        random_point = [np.random.choice(np.shape(matrix_points)[0],1, replace=False)[0],
                        np.random.choice(np.shape(matrix_points)[0],1, replace=False)[0]]
        try:
            grannar = neighbours(matrix_points, random_point)
            occurency = int(np.count_nonzero(np.array(grannar) == 0.2))
            if 0.0 in grannar and (occurency == 2 or occurency == 1):
                release = True
        except:
            continue
    return random_point

def random_step(random_point):
    #there are four movements the walker can take
    scenarios = ["step_forward_x", "step_backward_x", 
    "step_up_y", "step_down_y"]
    which_scenario = np.random.choice(np.shape(scenarios)[0])
    if scenarios[which_scenario] == "step_forward_x":
        random_point[0] +=1
    if scenarios[which_scenario] == "step_backward_x":
        random_point[0] -=1
    if scenarios[which_scenario] == "step_up_y":
        random_point[1] +=1
    if scenarios[which_scenario] == "step_down_y":
        random_point[1] -=1
    return random_point

def neighbours(matrix_points, random_point):
    a = int(random_point[0]-1)
    b = int(random_point[0]+1)
    c = int(random_point[1]-1)
    d = int(random_point[1]+1)
    out = int(np.shape(matrix_points[0])[0])-1
    if a < 0 or a > out or b < 0 or b > out or c < 0 or c > out or d < 0 or d > out:
        raise Exception("outside")

    right = matrix_points[random_point[0]][random_point[1]+1]
    left = matrix_points[random_point[0]][random_point[1]-1]
    bottom = matrix_points[random_point[0]+1][random_point[1]]
    top = matrix_points[random_point[0]-1][random_point[1]]
    
    grannar = [right, left, top, bottom]

    return grannar

def fractal_dimension(k, matrix_points):
    div = []

    for i in range(2,k+1):
        div.append(2**i)
    #the pixels are in the array matrix_points

    cover = np.zeros(np.shape(div)[0])
    inverse_delta = np.zeros(np.shape(div)[0])

    def view_as_blocks(arr, BSZ): 
    # reference: https://stackoverflow.com/questions/44782476/split-a-numpy-array-both-horizontally-and-vertically
        # arr is input array, BSZ is block-size
        m,n = arr.shape
        M,N = BSZ
        return arr.reshape(m//M, M, n//N, N).swapaxes(1,2).reshape(-1,M,N)

    for i in range(np.shape(div)[0]):
        inverse_delta[i] += 1/div[i]
        boxes = view_as_blocks(matrix_points,(int(div[i]),int(div[i])))

        for j in range(np.shape(boxes)[0]): #loop over all boxes
            occupied = np.count_nonzero(boxes[j] == 1.0) #is the pixels in the grid occupied?
            if int(occupied) > 0: #if box is occupied count it as one 
                cover[i] += 1

    D = (np.log(cover)[1]-np.log(cover)[len(cover)-1])/(np.log(inverse_delta)[1]-np.log(inverse_delta)[len(inverse_delta)-1])

    print("The dimension of the fractal is:", D)

    

#-------DETERMINES THE SIZE OF OUR GRID SYSTEM-----------------------------
k = 7
#--------------------------------------------------------------------------

N = 2**k
step = 1/N
size = 1 - step

xx, yy, matrix_points = set_grid(size, step)

matrix_points, random_point = add_initial_seed(matrix_points,size,step)

matrix_points = set_outer_circle(matrix_points)

matrix_points = set_inner_circle(matrix_points)

N = 95000

particles = 0
        
for i in range(N):
    random_point = release_walker(matrix_points)
    try:
        while matrix_points[random_point[0]][random_point[1]] == 0.0:
            grannar = neighbours(matrix_points, random_point)
            
            if 1.0 in grannar or 0.8 in grannar: #checks if there is a neighbour == 1 or the seed == 0.8
                #now we reached our neighbour, lets settle down
                matrix_points[random_point[0]][random_point[1]] = 1.0
                particles += 1

            else:
                random_point = random_step(random_point)
                #outer circle limit
                if matrix_points[random_point[0]][random_point[1]] == 0.2: 
                    #we're now out of bounds
                    raise Exception("out!")

    except Exception:
        continue

plt.pcolormesh(xx, yy, matrix_points,cmap='Greys', edgecolors="white")
plt.axis("off")
plt.rcParams["figure.figsize"] = (10,10)

print(particles)

plt.show()

fractal_dimension(k, matrix_points)