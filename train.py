import os
import numpy as np
import torch
import tools
import pandas as pd
from sklearn.utils import shuffle
import sys

# In[2]:


gpu_id=sys.argv[-1]
#os.environ['CUDA_VISIBLE_DEVICES']=gpu_id#如果有6个以上gpu，这里不用注释掉以提升速度，否者注释掉

# In[3]:
tools.setup_seed(int(gpu_id))

df_train = pd.read_csv('train.csv')
mean_list=list()
'''
按照treatment分组，对于每个组，使outcome均值为0
'''
for i in range(3):
    mean_list.append(df_train['outcome'][df_train.treatment==i].mean())
    df_train['outcome'][df_train.treatment==i]-=mean_list[-1]
df_test = pd.read_csv('test.csv')
df_test['treatment']=0
df_test['outcome']=0
df=pd.concat([df_train,df_test])

'''
C开头的代表连续变量，D开头的代表离散变量
'''

dic=dict()#用dic来保存变量列
for i in df.columns[:-2]:
    s=df[i]
    if len(s.unique())<10:
        s=np.array(pd.get_dummies(s)).argmax(-1)
        dic['DD_'+i.split('_')[-1]]=s
    else:
        #添加序特征
        v=s.argsort()
        v=(v-v.mean())/(v.std()+1e-7)
        dic['CC_'+i.split('_')[-1]]=v
'''
变量预处理
'''
dic['C_0']=df['V_0']
dic['C_1']=df['V_1']
dic['D_2']=(df['V_2']>0).astype(np.float32)
dic['C_3']=df['V_3']
dic['C_4']=df['V_4'].fillna(df['V_4'].mean())
dic['D_5']=df['V_5'].apply(lambda x:x if x==1 or x==2 else 0)
dic['C_6']=df['V_6']
def fn(x):
    if x>4.5:
        return 0
    elif x>2:
        return 1
    elif x>1.1:
        return 3
    else:
        return 4
dic['D_7']=df['V_7'].apply(fn)
dic['D_8']=(df['V_8']!='no').astype(np.float32)
dic['D_9']=(df['V_9']==999).astype(np.float32)
dic['D_10']=(df['V_10']!='no').astype(np.float32)
dic['C_11']=df['V_11']
dic['D_12']=df['V_12']
dic['C_13']=df['V_13']
dic['D_14']=(df['V_14']=='yes').astype(np.float32)
dic['D_15']=(df['V_15']!=0).astype(np.float32)
dic['D_16']=df['V_16']
dic['C_17']=df['V_17']
dic['C_18']=df['V_18']
dic['D_19']=df['V_19']
dic['C_21']=df['V_21']
dic['C_22']=df['V_22']
dic['C_23']=df['V_23']
dic['C_24']=df['V_24']
dic['C_25']=df['V_25']
dic['D_26']=(df['V_26']!='no').astype(np.float32)
dic['C_28']=df['V_28'].fillna(df['V_28'].mean())
dic['C_29']=df['V_29']
dic['C_30']=df['V_30']
dic['D_31']=df['V_31']
dic['C_32']=df['V_32']
dic['C_33']=df['V_33']
dic['C_34']=df['V_34']
dic['C_35']=df['V_35']
dic['C_36']=df['V_36']
def fn(x):
    if x>1.3:
        return 0
    elif x>1:
        return 1
    elif x>-1:
        return 3
    elif x>-2:
        return 4
    else:
        return 5
dic['D_37']=df['V_37'].apply(fn)
dic['D_38']=df['V_38'].apply(lambda x:x if x==0 or x==1 else 2)
dic['D_39']=df['V_39']
dic['X']=df.treatment
dic['Y']=df.outcome
'''
对于连续变量，将其标准化到0均值，1方差
'''
for i in dic:
    if 'C'in i:
        n=dic[i]
        n=(n-n.mean())/(n.std()+1e-7)
        dic[i]=n
    
df=pd.DataFrame(dic)

'''
根据V.md文件，提取WQ变量
'''
col_set={str(i)for  i in [2,5,6,7,8,9,10,12,14,15,16,19,21,24,25,26,28,31,36,37,39]}
col_list=sorted([i for i in df.columns if i.split('_')[-1] in col_set])+['X']#对变量排序，连续变量在前，离散变量在后

df=df[col_list]
    
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.c_len=sum(['C'in i for i in df.columns])#连续变量个数
        self.fc=torch.nn.Linear(self.c_len,128)#处理连续变量的函数
        self.dfcs=torch.nn.ModuleList([tools.Dfc(int(df[i].max())+1,128)for i in df.columns if 'C'not in i])#处理离散变量的函数
        
        self.fc1=torch.nn.Linear(128,128)
        self.fc2=torch.nn.Linear(128,1)
                
    def forward(self,x):
        #x:[batch_size,dims]
        out_x=self.fc(x[:,:self.c_len])#处理后的离散变量
        out_list=[fn(x[:,i]) for fn,i in zip(self.dfcs,range(self.c_len,x.shape[-1]))]#处理后的离散变量
        
        out=torch.stack([out_x]+out_list,0)
        out=torch.nn.functional.dropout(out,0.3,training=self.training)
        out=out.sum(0)/len(df.columns)

        out=out+tools.swish(self.fc1(out))
        
        out=self.fc2(out)[:,0]
        return out#[batch_size]
    
def get_loss(y_,y,T):
    #y_:真实值
    #y：预测值
    #T：treatment
    loss=torch.square(y_-y)
    loss_list=list()
    for i in range(3):
        cond=(T==i).type(torch.float32)
        _loss=(loss*cond).sum()/cond.sum()
        loss_list.append(_loss)
    return loss_list#第0个值为treatment为0时的loss，第1个值为treatment为1时的loss，第2个值为treatment为2时的loss
def train(weight_list=[0.1,1,0.1]):
    #weight_list:第0个值为treatment为0时的loss的权重，第1个值为treatment为1时的loss的权重，第2个值为treatment为2时的loss的权重
    if not model.training:
        model.train()

    x=X[:len(df_train)]
    y_=Y_[:len(df_train)]
    t=T[:len(df_train)]
    y=model(x)
    loss_list=get_loss(y_,y,t)
    loss=sum([i*j for i,j in zip(loss_list,weight_list)])
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
    optimizer.step()
    return loss_list[1].cpu().detach().numpy()

X=torch.tensor(np.asarray(df,dtype=np.float32)).cuda()
Y_=torch.tensor(np.asarray(df_train.outcome,dtype=np.float32)).cuda()
T=torch.tensor(np.asarray(df_train.treatment,dtype=np.float32)).cuda()
model=Model().cuda()

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
'''
训练
'''
for step in range(40000):
    optimizer.param_groups[0]['lr']=min(3e-4,3e-4/1000*(step+1))
    train_loss=train([1,0.1,1])
    if step%100==0:
        print(step)

'''
生成预测结果
'''
d_list=list()
for step in range(step,step+200):
    optimizer.param_groups[0]['lr']=min(3e-4,3e-4/1000*(step+1))
    train_loss=train([1,0.1,1])
    if step%2==0:
        if model.training:
            model.eval()
        
        d=dict()#d[0]为treatment为0时的预测值，d[1]为treatment为1时的预测值，d[2]为treatment为2时的预测值
        for i in range(3):
            x=torch.cat([X[:,:-1],torch.zeros([X.shape[0],1],device=X.device)+i],-1)
            d[i]=model(x).cpu().detach().numpy()+mean_list[i]
        d_list.append(d)

d=dict()#d[0]为treatment为0时的预测值，d[1]为treatment为1时的预测值，d[2]为treatment为2时的预测值
for i in range(3):
    d[i]=np.mean([dic[i]for dic in d_list],0)

'''
保存预测结果
'''
np.save('O/0'+gpu_id,d[0])
np.save('O/1'+gpu_id,d[1])
np.save('O/2'+gpu_id,d[2])
