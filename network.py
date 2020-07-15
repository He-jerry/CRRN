import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class featureExtractionB(nn.Module):

	def __init__(self, in_planes):
		super(featureExtractionB, self).__init__()
		self.path1 = nn.Sequential(
			BasicConv2d(in_planes, 128, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(128, 192, kernel_size = 7, stride = 1, padding = 3)
		)
		
		self.path2 = nn.Sequential(
			BasicConv2d(in_planes, 128, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(128, 192, kernel_size = 3, stride = 1, padding = 1)
		)
		
		self.path3 = nn.Sequential(
			BasicConv2d(in_planes, 128, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
		)
		
		self.path4 = nn.Sequential(
			BasicConv2d(in_planes, 128, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
			BasicConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
		)

	def forward(self, x):
		Path1 = self.path1(x)
		Path2 = self.path2(x)
		Path3 = self.path3(x)
		Path4 = self.path4(x)
		
		out = torch.cat((Path1, Path2, Path3, Path4), 1)
		#print(out.shape)
		return out

class featureExtrationA(nn.Module): #192, k256/2, l256/2, m192/3, n192/3, p96/3, q192/3
		
	def __init__(self, in_planes):
		super(featureExtrationA, self).__init__() 
		
		self.path1 = nn.Sequential(
			BasicConv2d(in_planes, 96, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(96, 192, kernel_size = 7, stride = 1, padding = 3)
		)
		
		self.path2 = BasicConv2d(in_planes, 192, kernel_size = 3, stride = 1, padding = 1)
		
		self.path3 = nn.Sequential(
			BasicConv2d(in_planes, 256, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
			BasicConv2d(256, 192, kernel_size = 3, stride = 1, padding = 1)
		)
		
	def forward(self, x):
		x1 = self.path1(x)
		x2 = self.path2(x)
		x3 = self.path3(x)
		out = torch.cat((x1, x2, x3), 1)
		#print(out.shape)
		return out
   
class BasicConv2d(nn.Module):
  def __init__(self, in_planes, out_planes, kernel_size, stride, padding = 0):
   super(BasicConv2d, self).__init__()
   self.conv = nn.Conv2d(in_planes, out_planes,
								kernel_size = kernel_size, stride = stride,
								padding = padding, bias = False
							 ) # verify bias false
   self.bn = nn.BatchNorm2d(out_planes,
								eps=0.001, # value found in tensorflow
								momentum=0.1, # default pytorch value
								affine=True)
   self.relu = nn.ReLU(inplace = True)

  def forward(self, x):
   x = self.relu(self.bn(self.conv(x)))
   return x

class BasicTransConv2d(nn.Module):
  def __init__(self, in_planes, out_planes, kernel_size, stride, padding = 0):
   super(BasicTransConv2d, self).__init__()
   self.transconv =  nn.ConvTranspose2d(in_planes, out_planes,
											kernel_size = kernel_size, stride = stride,
											padding = padding, bias = False)
   self.bn = nn.BatchNorm2d(out_planes,
								eps=0.001, # value found in tensorflow
								momentum=0.1, # default pytorch value
								affine=True)
   self.relu = nn.ReLU(inplace = True)
		
  def forward(self, x):
   x = self.relu(self.bn(self.transconv(x)))
   return x
   
class Gin(nn.Module):
  def __init__(self):
    super(Gin,self).__init__()
    
    #encoder
    self.en1conv1=nn.Conv2d(4,32,kernel_size=3,stride=1)
    self.en1conv2=nn.Conv2d(32,32,kernel_size=4,stride=2,padding=2)

    self.en2conv1=nn.Conv2d(32,64,kernel_size=3,stride=1)
    self.en2conv2=nn.Conv2d(64,64,kernel_size=4,stride=2,padding=2)

    self.en3conv1=nn.Conv2d(64,128,kernel_size=3,stride=1)
    self.en3conv2=nn.Conv2d(128,128,kernel_size=4,stride=2,padding=2)

    self.en4conv1=nn.Conv2d(128,256,kernel_size=3,stride=1)
    self.en4conv2=nn.Conv2d(256,256,kernel_size=4,stride=2,padding=2)

    self.en5conv1=nn.Conv2d(256,512,kernel_size=3,stride=1)
    self.en5conv2=nn.Conv2d(512,512,kernel_size=4,stride=2,padding=2)

    #bridge
    self.bconv1=nn.Conv2d(512,1024,kernel_size=7,padding=2)
    self.bconv2=nn.Conv2d(1024,512,kernel_size=1,padding=1)

    #decoder
    self.de1conv1=nn.Conv2d(512,256,kernel_size=3,padding=1)
    self.de1deconv1=nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1)

    self.de2conv1=nn.Conv2d(256+256,128,kernel_size=3,padding=1)
    self.de2deconv1=nn.ConvTranspose2d(128,128,kernel_size=4,stride=2,padding=1)

    self.de3conv1=nn.Conv2d(128+128,64,kernel_size=3,padding=1)
    self.de3deconv1=nn.ConvTranspose2d(64,64,kernel_size=4,stride=2,padding=1)

    self.de4conv1=nn.Conv2d(64+64,32,kernel_size=3,padding=1)
    self.de4deconv1=nn.ConvTranspose2d(32,32,kernel_size=4,stride=2,padding=1)

    #end
    self.econv1=nn.Conv2d(32+32,64,kernel_size=4,stride=2,padding=1)
    self.econv2=nn.Conv2d(64,1,kernel_size=5,stride=2,padding=2)
    self.sig=nn.Sigmoid()

  def forward(self,x):
    en1=self.en1conv2(self.en1conv1(x))
    #print('en1',en1.shape)
    en2=self.en2conv2(self.en2conv1(en1))
    #print('en2',en2.shape)
    en3=self.en3conv2(self.en3conv1(en2))
    #print('en3',en3.shape)
    en4=self.en4conv2(self.en4conv1(en3))
    print('en4',en4.shape)
    en5=self.en5conv2(self.en5conv1(en4))
    #print('en5',en5.shape)

    bd=self.bconv1(en5)
    #print('bd',bd.shape)
    bd=self.bconv2(bd)
    #print('bd',bd.shape)

    de1=self.de1conv1(bd)
    #print('de1',de1.shape)
    de1=self.de1deconv1(de1)
    #print('de1',de1.shape)
    de1out=torch.cat((de1,en4),1)
    #print('de1',de1out.shape)

    de2=self.de2deconv1(self.de2conv1(de1out))
    de2out=torch.cat((de2,en3),1)
    #print('de2',de2out.shape)

    de3=self.de3deconv1(self.de3conv1(de2out))
    de3out=torch.cat((de3,en2),1)
    #print('de3',de3out.shape)

    de4=self.de4deconv1(self.de4conv1(de3out))
    de4out=torch.cat((de4,en1),1)
    #print('de4',de4out.shape)

    ends=self.econv1(de4out)
    #print('ends',ends.shape)
    ends=self.econv2(ends)
    #print('ends',ends.shape)
    output=self.sig(ends)
    output=F.interpolate(output,scale_factor=8)
    #print('output',output.shape)
    return de1,de2,de3,de4,output
    
class liN(nn.Module):
  def __init__(self):
    super(liN,self).__init__()
    backmodel=torchvision.models.vgg16(pretrained=True).features
    self.mlist=list(backmodel.children())
    self.convs1R = nn.Sequential(*self.mlist[0:5])#64*64
    self.convs2R = nn.Sequential(*self.mlist[5:10])
    self.convs3R = nn.Sequential(*self.mlist[10:17])#16*16
    self.convs4R = nn.Sequential(*self.mlist[17:24])#8*8
    self.convs5R = nn.Sequential(*self.mlist[24:31]) #4*4
    self.featA=featureExtrationA(512)#576

    self.b1tc1=BasicTransConv2d(576,256,4,2,1)#*3 times
    self.b2tc1=BasicTransConv2d(1280,128,4,2,1)#256*3+512+256=1536
    self.featB=featureExtractionB(768)#128*3+128+256=768
    self.b3tc1=BasicTransConv2d(640,64,4,2,1)#192+192+128+128=514
    self.b4tc1=BasicTransConv2d(384,32,4,2,1)#64*3+64+128=384
    self.b5tc1=BasicTransConv2d(192,16,4,2,1)#32*3+32+64=192
    self.e1=BasicConv2d(48,16,3,1,1)#16*3=48
    self.e2=BasicConv2d(16,3,3,1,1)
  def forward(self,img,g1,g2,g3,g4):
    cv1=self.convs1R(img)
    cv2=self.convs2R(cv1)
    cv3=self.convs3R(cv2)
    cv4=self.convs4R(cv3)
    cv5=self.convs5R(cv4)
    #print('cv1',cv1.shape)
    #print('cv2',cv2.shape)
    #print('cv3',cv3.shape)
    #print('cv4',cv4.shape)
    #print('cv5',cv5.shape)
    feata=self.featA(cv5)
    b1c11=self.b1tc1(feata)
    b1c12=self.b1tc1(feata)
    b1c13=self.b1tc1(feata)
    #print('b1c11',b1c11.shape)
    cv3=F.interpolate(cv3,scale_factor=0.5)
    #print('cv3',cv3.shape)
    #print('g1',g1.shape)
    b1o1=torch.cat((b1c11,b1c12,b1c13,cv3,g1),1)
    #print(b1o1.shape)
    b2c11=self.b2tc1(b1o1)
    b2c12=self.b2tc1(b1o1)
    b2c13=self.b2tc1(b1o1)
    #print('b2c11',b2c11.shape)
    cv3=F.interpolate(cv3,scale_factor=2)
    #print('cv3',cv3.shape)

    #print('g2',g2.shape)
    b2o1=torch.cat((b2c11,b2c12,b2c13,cv3,g2),1)
    
    b2o1=self.featB(b2o1)
    #print(b2o1.shape)
    b3c11=self.b3tc1(b2o1)
    b3c12=self.b3tc1(b2o1)
    b3c13=self.b3tc1(b2o1)
    #print('b3c11',b3c11.shape)
    #print('cv2',cv2.shape)
    #print('g3',g3.shape)
    b3o1=torch.cat((b3c11,b3c12,b3c13,cv2,g3),1)
    #print(b3o1.shape)
    b4c11=self.b4tc1(b3o1)
    b4c12=self.b4tc1(b3o1)
    b4c13=self.b4tc1(b3o1)
    #print('b4c11',b4c11.shape)
    #print('cv1',cv1.shape)
    #print('g4',g4.shape)
    b4o1=torch.cat((b4c11,b4c12,b4c13,cv1,g4),1)
    #print(b4o1.shape)
    b5c11=self.b5tc1(b4o1)
    b5c12=self.b5tc1(b4o1)
    b5c13=self.b5tc1(b4o1)
    b5o1=torch.cat((b5c11,b5c12,b5c13),1)
    #print(b5o1.shape)
    e1=self.e1(b5o1)
    e2=self.e2(e1)
    #print('e2',e2.shape)
    r2=torch.sub(img,e2)
    return e2,r2

class totalnet(nn.Module):
  def __init__(self):
    super(totalnet,self).__init__()
    self.LiN=liN()
    self.GiN=Gin()
  def forward(self,img,grad):
    fedgin=torch.cat((img,grad),1)
    o1,o2,o3,o4,e3=self.GiN(fedgin)
    #print(o1.shape,',',o2.shape,o3.shape,o4.shape)
    t2,r2=self.LiN(img,o1,o2,o3,o4)
    return t2,r2,e3