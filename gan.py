# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:57:15 2018

@author: mtm916
"""


#import numpy as np
import random
from PIL import Image
from math import floor
import numpy as np
import time

def zero():
    return np.random.uniform(0.99, 1.0, size = [1])

def one():
    return np.random.uniform(-1.0, -0.99, size = [1])

def noise(n, s): #s gradually decreases
    return np.random.normal(0, s, size = [n, 64])

def adjust_hue(image, amount):
    t0 = Image.fromarray(np.uint8(image*255))
    t1 = t0.convert('HSV')
    t2 = np.array(t1, dtype='float32')
    t2 = t2 / 255
    t2[...,0] = (t2[...,0] + amount) % 1
    t3 = Image.fromarray(np.uint8(t2*255), mode = "HSV")
    t4 = np.array(t3.convert('RGB'), dtype='float32') / 255
    
    return t4

#Import Images Function
def import_images(loc, n):
    
    out = []
    
    for n in range(1, n + 1):
        temp = Image.open("data/"+loc+"/im ("+str(n)+").png").convert('RGB')
        
        temp = np.array(temp.convert('RGB'), dtype='float32') / 255
        
        out.append(temp)
        
        for i in range(4):
            temp = adjust_hue(temp, 0.2)
            out.append(temp)
            
    return out

    
from keras.layers import Conv2D, BatchNormalization, Dense, AveragePooling2D, LeakyReLU
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam



def g_block(f, b = True):
    temp = Sequential()
    temp.add(UpSampling2D())
    temp.add(Conv2D(filters = f, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform'))
    if b:
        temp.add(BatchNormalization(momentum = 0.9))
    temp.add(Activation('relu'))
    
    return temp

def d_block(f, b = True, p = True):
    temp = Sequential()
    temp.add(Conv2D(filters = f, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform'))
    if b:
        temp.add(BatchNormalization(momentum = 0.9))
    temp.add(LeakyReLU(0.2))
    if p:
        temp.add(AveragePooling2D())
    
class GAN(object):
    
    def __init__(self):
        
        #Models
        self.D = None
        self.G = None
        self.E = None
        
        self.OD = None
        self.OG = None
        
        self.DM = None
        self.AM = None
        self.VM = None
        self.KM = None
        self.ZM = None
        
        #Config
        self.LR = 0.0002
        self.steps = 1
        self.clip_value = 1
        
        #Experience Replay Models
        self.ERD = []
        self.ERG = []
        
    def discriminator(self):
        
        if self.D:
            return self.D
        
        self.D = Sequential()
        
        self.D.add(Activation('linear', input_shape = [32, 32, 3]))
        
        #512
        self.D.add(d_block(8, b = False)) #32
        self.D.add(d_block(16, b = False)) #16
        self.D.add(d_block(32)) #8
        self.D.add(d_block(64, p = False)) #4
        self.D.add(Flatten())
        
        #8192
        
        self.D.add(Dropout(0.6))
        self.D.add(Dense(1, activation = 'linear'))
        
        return self.D
    
    def generator(self):
        
        if self.G:
            return self.G
        
        self.G = Sequential()
        
        self.G.add(Dense(1024, input_shape = [64]))
        
        self.G.add(Reshape(target_shape = [4, 4, 64]))
        
        #4x4
        self.G.add(Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = "relu", kernel_initializer = 'he_uniform'))
        self.G.add(g_block(32)) #8x8
        self.G.add(g_block(16)) #16x16
        self.G.add(g_block(8)) #32x32
        
        #32x32
        self.G.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same'))
        self.G.add(Conv2D(filters = 3, kernel_size = 1, padding = 'same'))
        self.G.add(Activation('sigmoid'))
        
        return self.G
    
    def encoder(self):
        
        if self.E:
            return self.E
        
        #32
        self.E = Sequential()
        self.E.add(Activation('linear', input_shape = [32, 32, 3]))
        self.E.add(d_block(8, b = False)) #32
        self.E.add(d_block(16, b = False)) #16
        self.E.add(d_block(32)) #8
        self.E.add(d_block(64, p = False)) #4
        self.E.add(Flatten())
        
        self.E.add(Dense(128, activation = "relu", kernel_initializer = 'he_uniform'))
        self.E.add(Dense(64))
        
        return self.E
    
    def DisModel(self):
        
        if self.DM == None:
            self.DM = Sequential()
            self.DM.add(self.discriminator())
        
        self.DM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))), loss = 'mse')
        
        return self.DM
    
    def AdModel(self):
        
        if self.AM == None:
            self.AM = Sequential()
            self.AM.add(self.generator())
            self.AM.add(self.discriminator())
            
        self.AM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))), loss = 'mse')
        
        return self.AM
    
    def VAEModel(self):
        
        if self.VM == None:
            self.VM = Sequential()
            self.VM.add(self.encoder())
            self.VM.add(self.generator())
            
            self.KM = Sequential()
            self.KM.add(self.encoder())
            
        self.VM.compile(optimizer = Adam(lr = self.LR * 2 * (0.9 ** floor(self.steps / 10000))), loss = 'mae')
        self.KM.compile(optimizer = Adam(lr = self.LR * 0.01 * (0.9 ** floor(self.steps / 10000))), loss = 'kld')
        
        return self.VM
    
    def ZDModel(self):
        
        if self.ZM == None:
            
            self.ZM = Sequential()
            self.ZM.add(self.generator())
            self.ZM.add(self.encoder())
            
        self.ZM.compile(optimizer = Adam(lr = self.LR * (0.9 ** floor(self.steps / 10000))), loss = 'mae')
        
        return self.ZM
    
    def sod(self):
        
        self.OD = self.D.get_weights()
        
    def lod(self):
        
        self.D.set_weights(self.OD)
        
        



class WGAN(object):
    
    def __init__(self, steps = -1, silent = True):
        
        self.GAN = GAN()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.generator = self.GAN.generator()
        self.VAEModel = self.GAN.VAEModel()
        self.KModel = self.GAN.KM
        self.ZDModel = self.GAN.ZDModel()
        
        if steps >= 0:
            self.GAN.steps = steps
        
        #Standard Deviation
        self.std_dev = 1
        
        self.lastblip = time.clock()
        
        self.noise_level = 0
        
        self.ImagesA = import_images("Sprites", 80)
        #self.instance_noise()
        
        self.silent = silent
        
    def train(self, batch = 8):
        
        (a, b) = self.train_dis(batch)
        c = self.train_gen(batch)
        d = self.train_vae(batch)
        e = self.train_zd(batch)
        
        if self.GAN.steps % 20 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(a))
            print("D: " + str(b))
            print("G: " + str(c))
            print("V: " + str(d))
            print("Z: " + str(e))
            s = round((time.clock() - self.lastblip) * 1000) / 1000
            print("Time::: " + str(s) + "sec")
            self.lastblip = time.clock()
        
        if self.GAN.steps % 500 == 0:
            #self.GAN.save_checkpoint()
            #self.instance_noise()
            self.save(floor(self.GAN.steps / 10000))
            
        if self.GAN.steps % 5000 == 0:
            self.GAN.AM = None
            self.GAN.DM = None
            self.GAN.VM = None
            self.GAN.KM = None
            self.AdModel = self.GAN.AdModel()
            self.DisModel = self.GAN.DisModel()
            self.VAEModel = self.GAN.VAEModel()
            self.KModel = self.GAN.KM
            self.ZDModel = self.GAN.ZDModel()
        
        self.GAN.steps = self.GAN.steps + 1
        
        #Set self.std_dev
        self.std_dev = 1
        
    def train_dis(self, batch):
        
        #Get Real Images
        train_data = []
        label_data = []
        for i in range(batch):
            im_no = random.randint(0, len(self.ImagesA) - 1)
            train_data.append(self.ImagesA[im_no])
            label_data.append(one())
            
        d_loss_real = self.DisModel.train_on_batch(np.array(train_data), np.array(label_data))
        
        #Get Fake Images
        train_data = self.generator.predict(noise(batch, self.std_dev))
        label_data = []
        for i in range(batch):
            label_data.append(zero())
            
        d_loss_fake = self.DisModel.train_on_batch(train_data, np.array(label_data))
        
        return (d_loss_real, d_loss_fake)
        
    def train_gen(self, batch):
        
        self.GAN.sod()
        
        label_data = []
        for i in range(int(batch)):
            label_data.append(one())
        
        g_loss = self.AdModel.train_on_batch(noise(batch, self.std_dev), np.array(label_data))
        
        self.GAN.lod()
        
        return g_loss
    
    def train_vae(self, batch):
        
        train_data = []
        label_data = []
        for i in range(batch):
            im_no = random.randint(0, len(self.ImagesA) - 1)
            train_data.append(self.ImagesA[im_no])
            label_data.append(self.ImagesA[im_no])
        
        g_loss = self.VAEModel.train_on_batch(np.array(train_data), np.array(label_data))
        
        self.KModel.train_on_batch(np.array(train_data), noise(batch, self.std_dev))
        
        return g_loss
    
    def train_zd(self, batch):
        
        s = noise(batch, self.std_dev)
        
        g_loss = self.ZDModel.train_on_batch(s, s)
        
        return g_loss
    
    def evaluate(self, num = 0, trunc = 1.0):
        
        n2 = noise(32, self.std_dev)
        n3 = noise(32, 1)
        
        im2 = self.generator.predict(n2)
        im3 = self.generator.predict(n3)
        
        r12 = np.concatenate(im2[:8], axis = 1)
        r22 = np.concatenate(im2[8:16], axis = 1)
        r32 = np.concatenate(im2[16:24], axis = 1)
        r42 = np.concatenate(im2[24:32], axis = 1)
        r13 = np.concatenate(im3[:8], axis = 1)
        r23 = np.concatenate(im3[8:16], axis = 1)
        r33 = np.concatenate(im3[16:24], axis = 1)
        r43 = np.concatenate(im3[24:32], axis = 1)
        
        c1 = np.concatenate([r12, r22, r32, r42, r13, r23, r33, r43], axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255))
        
        x.save("Results/i"+str(num)+".png")
    
    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)
            
        model.save_weights("Models/"+name+"_"+str(num)+".h5")
        
    def loadModel(self, name, num):
        
        file = open("Models/"+name+".json", 'r')
        json = file.read()
        file.close()
        
        mod = model_from_json(json)
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")
        
        return mod
    
    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)
        self.saveModel(self.GAN.E, "enc", num)
        

    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Load Models
        self.GAN.G = self.loadModel("gen", num)
        self.GAN.D = self.loadModel("dis", num)
        
        self.GAN.steps = steps1

        #Reinitialize
        self.GAN.steps = steps1
        
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.VAEModel = self.GAN.VAEModel()
        self.KModel = self.GAN.KM
        self.ZDModel = self.GAN.ZDModel()
        
        
        self.std_dev = 1
    
        
    def sample(self, n):
        
        return self.generator.predict(noise(n, self.std_dev))
    
    def instance_noise(self):
        
        self.AmagesA = np.array(self.AmagesA)
        
        self.ImagesA = self.AmagesA + np.random.uniform(-self.noise_level, self.noise_level, size = self.AmagesA.shape)
        
        
        
        
if __name__ == "__main__":
    model = WGAN(silent = False)
    
    while(model.GAN.steps < 200000):
        
        #model.eval()
        model.train(4)
        
        if model.GAN.steps % 1000 == 0:
            model.evaluate(int(model.GAN.steps / 1000))


    
    

