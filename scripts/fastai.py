
'''

'''


from fastai.vision.all import *
from fastai.callback.hook import *

path = untar_data(URLs.PETS)/'images'

def is_cat(x):
    return x[0].isupper()

'''

'''

dls = ImageDataLoaders.from_name_func(path,get_image_files(path),
                                        valid_pct = .2,seed=42,label_func = is_cat,
                                        item_tfms = Resize(224))

'''

'''

learn = cnn_learner(dls,resnet34,metrics = error_rate)
learn.fine_tune(1)

'''

'''

path = untar_data(URLs.MNIST_SAMPLE)
time = torch.arange(0,20)
params = torch.randn(3).requires_grad_()

'''

'''

def apply_step(params,prn=True):
    speed = time*3 + (time-9.5)**2 + 1
    a,b,c = params
    pred = a*(time**2) + b*time + 1
    loss = ((pred - speed)**2).mean()
    loss.backward()
    lr = 1e-5
    params.grad
    params.data -= lr * params.grad.data
    params.grad = None
    if prn:
        print(loss.item())
        return pred

'''

'''

def L1_loss(average,real):
    result = (average - real).abs().mean()
    return result

'''

'''

def mean_sq_error_loss(average,real):
    result = ((average-real)**2).sqrt().mean()
    return result

'''

'''

# weigths
def init_params(size,std=1.0):
    params = (torch.randn(size)*std).requires_grad_()
    return params

'''

'''

# train
def linear1(xb):
    weights = xb@weights + bias
    return weights

'''

'''

# activation
def sigmoid(x):
    sig = 1/(1+torch.exp(-x))
    return sig

'''

'''

# loss
def mnist_loss(predictions,targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1,1-predictions,predictions).mean()

'''

'''

# train
def calc_grad(xb,yb,model):
    preds = model(xb)
    loss = mnist_loss(preds,yb)
    loss.backward()

'''

'''

def batch_accuracy(xb,yb):
    preds = xb.sigmoid()
    correct = (preds > .5) == yb
    result = correct.float().mean()
    return result

'''

'''

# metrics
def validate_epoch(model):
    accs = [batch_accuracy(model(xb),yb) for xb,yb in valid_dl]
    result = round(torch.stack(accs).mean().item(),4)
    return result

'''

'''

# train
def train_epoch(model,dl,opt):
    for xb,yb in dl:
        calc_grad(xb,yb,model)
        opt.step()
        opt.zero_grad()

'''

'''

# train
def train_model(model,epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model),end=' ')

'''

'''

# train
def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res

'''

'''

# loading data
def load_data(folder_name):
    training_tensor = [tensor(Image.open(i)) for i in folder_name]
    training_stack = ((torch.stack(training_tensor)).float())
    return training_stack

'''

'''

# transforming data
def training_data(*args):
    training = (torch.cat(args))
    return training

'''

'''

# data information
def size(training_stack):
    size = ((training_stack.shape)[1]) * (training_stack.shape[2])
    return size

'''

'''

# creating data
def init_weights(size):
    weights = (torch.randn(size)).requires_grad_()
    return weights

'''

'''

# creating data
def bias():
    bias = torch.randn(1)
    return bias

'''

'''

# transforming data
def transform_data_for_model(training_stack):
    result = training_stack[1] * training_stack[2]
    return result

'''

'''

# transforming data
def matrix_multiply(training_stack):
    new_training_stack = (training_stack).view(-1,784)
    pred = ((new_training_stack) @ weights) + bias
    return pred

'''

'''

# metric
def loss(pred,target):
    loss = (pred-target).abs().mean()
    return loss

'''

'''

# train
def update(lr):
    new_weights -= weights.grad * lr
    return new_weights

'''

'''

# data information
def size_of_image(image):
    image_size = image.shape
    return image_size

'''

'''

# data transformation
def apply_kernel(row,col,kernel):
    convolution = (img[row-1:row+2,col-1:col+2] * kernel).sum()
    return convolution

'''

'''

# transformation
def convolution_top():
    rng = (1,27)
    top_edge = tensor([[apply_kernel(i,j,top_edge) for j in rng] for i in rng])
    return top_edge

'''

'''

# information
def row(padding, stride, height):
    new_row = (height + padding) // stride
    return new_row

'''

'''

# information
def column(padding,stride,height):
    new_column = (height + padding) // stride
    return new_column

'''

'''

# information
def output_shape(w,n,p,f):
    output = int((W - K + (2*P))/(S + 1))
    new_output = (w - n + (2*p) - f) + 1
    return new_output

'''

'''

# creating kernels
def top_edge():
    top_edge = (tensor([1,1,1],[0,0,0],[-1,-1,-1])).float()
    return top_edge

'''

'''

# creating kernels
def bottom_edge():
    bottom_edge = (tensor([-1,-1,-1],[0,0,0],[1,1,1])).float()
    return bottom_edge

'''

'''

# creating kernels
def right_edge():
    right_edge = (tensor([-1,0,1],[-1,0,1],[-1,0,1])).float()
    return right_edge

'''

'''

# creating kernels
def left_edge():
    left_edge = (tensor([1,0,-1],[1,0,-1],[1,0,-1])).float()
    return left_edge

'''

'''

# creating kernels
def diag1_edge():
    diag1_edge = (tensor([1,0,-1],[0,1,0],[-1,0,1])).float()
    return diag1_edge

'''

'''

class BasicOptim:

    def __init__(self,params,lr):
        self.params,self.lr = list(params),lr

    def step(self,*args,**kwargs):
        for p in self.params:
            p.data-=p.grad.data *self.lr

    def zero_grad(self,*args,**kwargs):
        for p in self.params:
