import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
yun0 = 0
while yun0  < 21:
    yun = yun0 + 1
    print("程序运行第",yun,"次")
    #模型设置部分

    # 定义地势矩阵
    shan0 = []
    hang = []
    for i in range(50):
        hang.append(0)
    for j in range(50):
        shan0.append(hang)

    shan = np.array(shan0)

    #print(shan)
    # 绘制等高线图
    #plt.contour(shan, levels=20, colors='k')
    #plt.show()

    #设置信号需求的高度
    a = shan.shape
    gao =np.zeros(a)
    gao.fill(1) #先设置成所有的点都为2
    #print(np.sum(gao))
    ding = shan + gao   #ding表示问题范围的上界
    #生成村落（特殊点，必须覆盖)
    cunnum = 0
    num = 80
    #生成村落的函数
    def shengchengcunluo(quantity, side_length):
        coordinates = []  # 存储正方形的坐标
        k = 0
        # 随机生成第一个正方形的位置
        while k < quantity:    
            new_coordinate = (random.randint(5, a[0]-5), random.randint(5, a[1]-5))
            # 检查新生成的形状是否在已有形状的周围
            if new_coordinate not in coordinates:
                coordinates.append(new_coordinate)
                k += 1
    #def shengchengcunluo(quantity, side_length):
    #    coordinates = []  # 存储正方形的坐标

        # 随机生成第一个正方形的位置
    #    first_coordinate = (random.randint(5, a[0]-5), random.randint(5, a[1]-5))
    #    coordinates.append(first_coordinate)

    #    for i in range(1, quantity):
    #        last_coordinate = coordinates[i-1]
    #        while True:
    #            direction = random.choice(["up", "down", "left", "right"])
    #            if direction == "up":
    #                new_coordinate = (last_coordinate[0], last_coordinate[1] + side_length)
    #            elif direction == "down":
    #                new_coordinate = (last_coordinate[0], last_coordinate[1] - side_length)
    #            elif direction == "left":
    #                new_coordinate = (last_coordinate[0] - side_length, last_coordinate[1])
    #            elif direction == "right":
    #                new_coordinate = (last_coordinate[0] + side_length, last_coordinate[1])

                # 检查新生成的形状是否在已有形状的周围
    #            if new_coordinate not in coordinates:
    #                coordinates.append(new_coordinate)
    #                break
        # 绘制图形
        #fig, ax = plt.subplots()
        #for i, coordinate in enumerate(coordinates):
        #    x, y = coordinate
        #    square = Rectangle((x, y), side_length, side_length, fill=False)
        #    ax.add_patch(square)
        #    #ax.annotate(f"Square {i+1}", (x, y), textcoords="offset points", xytext=(0,10), ha='center') # 添加标签
        #plt.xlim(-15, 15)
        #plt.ylim(-15, 15)
        #plt.gca().set_aspect('equal', adjustable='box')
        #plt.show()
        # 输出生成的形状坐标
        return coordinates
    list0 = []
    #生成cunnum个随机分布的建筑群
    for i in range(cunnum):
        cunluo = shengchengcunluo(5,1)
        list0 +=cunluo
    #整理得到储存村落位置的列表fist
    flist =  [x for x in set(list0) if 0 <= x[0] < a[0] and 0 <= x[1] < a[1]]
    #print(flist)
    while len(flist) != 5*cunnum:
        list0 = []
        #生成cunnum个随机分布的建筑群
        for i in range(cunnum):
            cunluo = shengchengcunluo(5,1)
            list0 +=cunluo
        #整理得到储存村落位置的列表fist
        flist =  [x for x in set(list0) if 0 <= x[0] < a[0] and 0 <= x[1] < a[1]]
    #为生成的山区建筑拟定信号需求量，储存在列表plist中
    plist = []
    for i in range(len(flist)):
    #    p = int(i//10+1)
        p = i+2
        plist.append(p)
    #print("信号需求",plist)
    #设立禁区
    jin = np.zeros(a)#禁区
    #有建筑的地方不能设置基站，建筑周围九格内不能有基站
    for i in range(len(flist)):
        c = flist[i]
        #k = 0
        for i in range(-2,3):
            for j in range(-2,3):
                if c[1]+i>=0 and c[1]+i<a[1] and c[0]+j >= 0 and c[0]+j < a[0]:
                    jin[c[1]][c[0]] = 1
                #k+=1
        #通过直接取消ij的功能缩小禁区 可改
        #print(k)

    #基站部署部分
    import math
    import sympy as sp
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    #num 基站数量
    r = 5
    #覆盖村落的基站
    def fugaicunluo(list,shan,r):
        global plist
        tarlist = plist.copy()
        shape = shan.shape
        listsuo = [i for i in range(len(list))]
        #print(listsuo)
        bu = []
        while len(listsuo) > 0:
            f = 0
            #生成一个附近基站
            x0 = random.randint(-r+1,r-1)
            y = int((r**2-x0**2)**0.5)
            y0 = random.randint(-y+1,y-1)
            x1 = list[listsuo[0]][0]+x0
            x1 = max(0,x1)
            x1 = min(shape[1]-1,x1)
            y1 = list[listsuo[0]][1]+y0
            y1 = max(0,y1)
            y1 = min(shape[0]-1,y1)
            while jin[y1][x1] == 1:
                x0 = random.randint(-r+1,r-1)
                y = int((r**2-x0**2)**0.5)
                y0 = random.randint(-y+1,y-1)
                x1 = list[listsuo[0]][0]+x0
                x1 = max(0,x1)
                x1 = min(shape[0]-1,x1)
                y1 = list[listsuo[0]][1]+y0
                y1 = max(0,y1)
                y1 = min(shape[1]-1,y1)
            ji = (x1,y1)
            bu.append(ji)
            #print("1")
            #print(len(bu))
            #生成后遍历村落
            for i in range(len(listsuo)):
                xl = list[listsuo[i]][0]
                yl = list[listsuo[i]][1]
                a = (xl,yl)
                if flag(a,ji,r,shan) == 1:
                    k = listsuo[i]
                    tarlist[k] -= 1
                    if tarlist[k] == 0:
                        listsuo[i] = -1
                        #print(k+1,"号村落覆盖完成")
                        f += 1
            #遍历索引
            while f > 0:
                listsuo.remove(-1)
                #print(listsuo)
                f -= 1
            
        return bu

    #随机生成基站
    def shengchengjizhan(shan,list,num,r):
        jis = np.zeros(2*num)
        a = shan.shape
        bu = fugaicunluo(list,shan,r)
        while len(bu)>num:
            #print("基站不够,重新规划")
            bu = fugaicunluo(list,shan,r)
            #print("ccc")
        for i in range(num):
            if i < len(bu):
                jis[2*i] = bu[i][0]
                jis[2*i+1] = bu[i][1]
            else:
                x0 = random.randint(0,a[1]-2)
                y0 = random.randint(0,a[0]-2)
                while jin[y0][x0] ==1:
                    x0 = random.randint(0,a[1]-2)
                    y0 = random.randint(0,a[0]-2)
                jis[2*i] = x0
                jis[2*i+1] = y0
        return jis
    #检查更新后的粒子是否符合覆盖要求
    def jiancha(jis):
        #jis用的二维数组
        global r
        global plist
        global shan
        global flist
        global cunnum
        #print("开始检查")
        k = plist.copy()
        #print(k)
        for i in range(len(jis)):
            a = jis[i]
            for j in range(cunnum*5):
                ji = flist[j]
                if k[j]>0 and flag(a,ji,r,shan) == 1:
                    k[j] -= 1
                    #print(k)
                    #print("成功")
        f = 0
        #print(k)
        for kk in range(len(k)):
            f += k[kk]
        #print("违规：",f)
        return f
    #欧氏距离的平方
    def pingfanghe(a,b):
        k = 0
        for i in range(3):
            k += (a[i] - b[i])**2
        return k
    #判断是否覆盖(地表)
    def flag(a,ji,r,shan):
        f = 0
        xa = a[0]
        ya = a[1]
        xji = int(ji[0])
        yji = int(ji[1])
        ha = shan[int(ya)][int(xa)]
        hji = shan[yji][xji]
        a3 = (xa,ya,ha)
        ji3 = (ji[0],ji[1],hji)
        if pingfanghe(a3,ji3)<=r ** 2:
            f = 1
        return f
    #判断是否覆盖(空中)
    def flagtu(a,ji,r):
        f = 0
        if pingfanghe(a,ji)<=r ** 2:
            f = 1
        return f
    def func1 (jis):
        #jis用的一维数组
        global shan
        global gao
        global r
        
        shape = shan.shape
        num = int(len(jis)/2)
        tu = np.zeros((int(shan.shape[0]),int(shan.shape[1]),int(np.max(gao))))
        #print(num)
        k = 0
        for i in range(num):
            xj = int(jis[2*i])
            yj = int(jis[2*i+1])
            hj = shan[yj][xj]
            ji = [xj,yj,hj]
            for x in range(max(int(ji[0])-r,0),min(int(ji[0])+r,shape[0])):
                for y in range(max(int(ji[1])-r,0),min(int(ji[1])+r,shape[1])):
                    for z in range(int(gao[y][x])):
                        a = [x,y,shan[y][x]+z]
                        if flagtu(a,ji,r):
                            k+=1
                            tu[y][x][z] = 1

        fu = np.sum(tu)
        area = np.sum(gao)
        fit = fu/area
        #print(fit)
        fit = 1 - fit
        return fit
    
    #算法
    #库的导入
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    #from CEC2022 import cec2022_func

    fx_n = 5 #公式
    N=20   #种群数量
    iterators = 160   #迭代次数
    dim=num*2
    rangepop=[0,49]    #取值范围
    w=0.4   #惯性因子
    c1,c2=2,2
    fitness=np.zeros(N)
    fitness[:] = 10
    x = np.zeros((N,dim))
    for i in range(N):
        x[i] = shengchengjizhan(shan,flist,num,r)
    v = np.random.uniform(-1, 1, size=(N, dim))
    #CEC = cec2022_func(func_num = fx_n)

    #待求解问题
    def function(x):
        return func1(x)

    def find_nearest_distance_point(pos): #找与最优点欧几里德距离最小的点
        # 初始化最小距离和最近点的索引
        min_distance = float('inf')
        nearest_index = None
        
        # 遍历列表中的每个点
        for i in range(N):
            # 计算当前点与目标点的欧几里得距离
            distance = np.linalg.norm(x[i] - x[pos])
            # 如果当前距离小于最小距离，则更新最小距离和最近点的索引
            if distance < min_distance and pos != i:
                min_distance = distance
                nearest_index = i
        return nearest_index

    def find_nearest_fitness_point(fitnesses, target_fitness): #找到与当前点适应度最接近的点的索引
        diff = np.abs(fitnesses - target_fitness) # 计算差值
        return np.argmin(diff) # 找到最小值的索引

    u = np.zeros((N , dim)) #基于粒子自身的预测向量
    NC = np.zeros(N) #每个粒子的计数器
    Cur = np.zeros(N) #每个粒子使用SE策略的次数
    MaxCur = 0
    #fitness = function(np.transpose(x[0:N, :])) #矩阵 x 的前 N 行进行转置

    #Xgbest,Fgbest分别表示种群历史最优个体和适应度值
    Xgbest,Fgbest=x[fitness.argmin()].copy(),fitness.min()
    Xgworst,Fgworst=x[fitness.argmax()].copy(),fitness.max()
    #poppn,bestpn分别存储个体历史最优位置和适应度值
    Xpbest,Fpbest=x.copy(),fitness.copy()
    Xpworst,Fpworst=x.copy(),fitness.copy()
    #bestfitness用于存储每次迭代时的种群历史最优适应度值
    bestfitness=np.zeros(iterators+1)
    wmax = 1
    wmin = 0.4
    rk = np.random.random_integers(0, high=1, size=None)
    #开始迭代
    for t in range(1,iterators+1):
        print("generation:",t)
        rankings = np.argsort(fitness)
        for i in range(N):
            r1 = np.random.rand()
            r2 = np.random.rand()
            #计算移动速度
            # x1 = find_nearest_distance_point(i)
            x2 = find_nearest_fitness_point(fitness, fitness[i])
            x1 = fitness.argmin()
            u[i] = u[i] + (Xpbest[i] - x[i])
            ur = (u[x1]+u[x2]+u[i])/3  #这里求了均值
            if rankings[i]>=int(N*(1-15/100)):    #使用SE策略
                Fd = 1+dim/3
                Cur[i]+=1
                MaxCur+=1
                Fse = Cur[i]/MaxCur*Fd*ur
                v[i]=w*v[i]+c1*r1*(Xgbest-x[i])+c2*r2*Fse
            else:   #使用NEP策略
                Fnep = ur/t
                v[i]=w*v[i]+c1*r1*(Xgbest-x[i])+c2*r2*Fnep
            #计算新的位置
            x[i]=x[i]+v[i]
            #确保更新后的位置在取值范围内
            x[x<rangepop[0]]=rangepop[0] #对矩阵的每一行进项修正
            x[x>rangepop[1]]=rangepop[1]
        
        #计算适应度值
        fitness1 = fitness[:]
        for i in range(N):
            fitness[i] = function(x[i])
        for i in range(N):
            if abs(fitness[i]-fitness1[i])<1e8:
                NC[i]+=1
            #更新个体历史最优适应度值
            if fitness[i]<Fpbest[i]:
                Fpbest[i]=fitness[i]
                Xpbest[i]=x[i].copy()
            if fitness[i]>Fpworst[i]:
                Fpworst[i]=fitness[i]
                Xpworst[i]=x[i].copy()
        #更新种群历史最优适应度值
        if Fpbest.min()<Fgbest:
            Fgbest=Fpbest.min()
            Xgbest=Xpbest[Fpbest.argmin()].copy()
        if Fpworst.max()<Fgworst:
            Fgworst=Fpworst.min()
            Xgworst=Xpworst[Fpbest.argmax()].copy()
        bestfitness[t]=Fgbest
        print("the best fitness is:",1-bestfitness[t])

    #画图
    jis = Xgbest
    #画分布图
    plt.figure()
    #山地部分
    plt.contourf(shan, levels=20, cmap='terrain', zorder=-1)
    plt.colorbar()  # 添加颜色条
    plt.contour(shan, levels=20, colors='k', linewidths=0.5, zorder=0)



    #基站部分
    points = []  
    for i in range(num):
        points.append((jis[2*i],jis[2*i+1]))

    # 提取 x 和 y 坐标
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # 加载要插入的图片
    img = mpimg.imread('基站.png')

    # 创建新的图形

    for point in points:
        x_coord, y_coord = point
        plt.imshow(img, extent=[x_coord, x_coord+1, y_coord, y_coord+1], aspect='auto')

    # 隐藏坐标轴
    plt.axis('off')

    #村落部分
    def draw_square(x, y, size):
        x_coords = [x , x + size, x +size, x , x ]
        y_coords = [y , y , y + size, y + size, y ]
        plt.plot(x_coords, y_coords, 'black')  

    square_size = 1  # 设置正方形边长

    for coord in flist:
        draw_square(coord[0], coord[1], square_size)

    plt.axis('equal')  # 设置坐标轴比例相等，保证正方形形状正确
    plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴显示正方形
    plt.title("Distribution map of base stations")
    # 创建保存图片的文件夹
    file_path = f'分布图/{yun}.png'
    plt.savefig(file_path)
    #plt.show()
    plt.close()
    #绘制图像
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    plt.figure()
    #创建图形
    fig, ax = plt.subplots()

    #print(r0)
    #print(len(gbest))

    # 绘制基站位置
    for position in points:
        circle = Circle(position, radius=r,color = 'green',alpha=0.5)
        ax.add_patch(circle)
        #print(dir(circle))
        ax.plot(position[0], position[1], marker='+', color='blue', markersize=3)

    # 绘制目标点
        
    tar_x = [point[0] for point in flist]
    tar_y = [point[1] for point in flist]
    ax.scatter(tar_x, tar_y, color='red', marker='o', label='Target Points')

    # 绘制正方形
    pu = Rectangle((0, 0), 50, 50, fill=False, edgecolor='blue', linestyle='--')
    ax.add_patch(pu)

    # 设置坐标轴范围
    plt.xlim(-10, 60)
    plt.ylim(-10, 60)

    #通过标题报告本图适应度
    title = "best fitness: {:.4f}%".format((1-bestfitness[iterators-1])*100)
    plt.title(title)
    # 创建保存图片的文件夹
    file_path = f'覆盖图/{yun}.png'
    plt.savefig(file_path)
    #plt.show()
    plt.close()
    plt.close('all')
    import openpyxl
    wb = openpyxl.load_workbook('适应度.xlsx')
    ws = wb.active
    fitnesses = bestfitness
    for i in range(len(fitnesses)):
        cell = ws.cell(row=yun+1, column=i+1)
        cell.value = 1 - fitnesses[i]
    wb.save('适应度.xlsx')
    yun0 += 1