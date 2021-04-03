import numpy as np
import json
import torch

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    with open('performance.json','r') as f:
        content = json.loads(f.read())


    x=[]
    y=[]
    z = []
    for item in content:
        x.append(item[0])
        y.append(item[1])
        z.append(item[2])

    plt.plot(x, y, 'r-')
    plt.plot(x, z, 'b-')
    plt.show()
    # plt.savefig('result.png')
