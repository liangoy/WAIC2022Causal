import numpy as np
import pandas as pd


d={i:np.mean([np.load('O/'+str(i)+str(j)+'.npy')for j in range(6)],0) for i in range(3)}#如果模型数量不是6，需要把6改成相应的模型数量
p=pd.DataFrame({
    'ce_1':(d[1]-d[0]),
    'ce_2':(d[2]-d[0])
})
p.iloc[-5000:]=p.iloc[-5000:]-p.iloc[-5000:].mean()+p.iloc[:-5000].mean()#这里提升不大，去掉也行
p.to_csv('result.csv',index=False)
p.describe()