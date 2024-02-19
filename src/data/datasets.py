from torch.utils.data import Dataset
import torch
import os
from pil import Image
import numpy as np
from .transforms import ColorizeTransform

class originalReflax(Dataset):
      # need to be able to pull the following.
    def __init__(self, data, processor,transforms=None,pathToImages=None,dataframe=None, headerForSpecificFile=None,mode="original"):
        self.data = data
        self.processor = processor
        self.pathToImages=pathToImages
        self.dataframe = dataframe
        self.headerForSpecificFile=headerForSpecificFile
        self.transforms=transforms
        self.transcriptionHeader="transcriptions"
        self.mode = mode
    def __len__(self):
      return len(self.data)
    def __getitem__(self, idx):
      if torch.is_tensor(idx):
            idx = idx.tolist()

      #if images are paths need find image.
      if self.mode == "contrastive":
        return self.get_contrastive_pair(idx)

      image=os.path.join(self.pathToImages,self.data.loc[idx, self.headerForSpecificFile].values[0]+".jpg")
      imagePixels = Image.open(image)
      if self.transforms:
          imagePixels = self.transforms(imagePixels)
          if imagePixels.size(0) == 1:
    # Squeeze the tensor along the batch dimension
            imagePixels = torch.squeeze(imagePixels, dim=0)
          else:
    # If the batch size is not 1, keep the tensor as is
            imagePixels = imagePixels

      transcriptions=self.data.loc[idx,self.transcriptionHeader].values[0]

      encodingdict=self.processor(text=transcriptions,images=imagePixels,padding="max-length",return_tensors="pt")

      encodingdict["id"]=self.data.loc[idx,"id"].values[0]






      return encodingdict
    def get_contrastive_pair(self,idx,negative_category=None):
      """
      Current working mechanism is image path, pixels retieved, idx

      """
      # print("here:   \t",self.data.loc[idx, self.headerForSpecificFile])

      image=os.path.join(self.pathToImages,self.data.loc[idx, self.headerForSpecificFile]+".jpg")
      anchorImagePixels=Image.open(image)
      id=self.data.loc[idx,"id"]
      # if negative category given
      if negative_category:

        try:
          #
          negativeSample=np.random.choice(self.dataframe[self.dataframe[negative_category]!=self.dataframe[self.dataframe['id']==id][negative_category].values[0]][negative_category])
          imageNegative=os.path.join(self.pathToImages,negativeSample.values[0]+".jpg")
        except:
          negativeSample= os.path.join(self.pathToImages,np.random.choice(self.data[self.headerForSpecificFile]).values[0]+".jpg")
      else:
        negativeSample= os.path.join(self.pathToImages,np.random.choice(self.data[self.headerForSpecificFile])+".jpg")
      negativeImagePixels = Image.open(negativeSample)

      if self.transforms:
        # shape needs to be

        negative = self.transforms(negativeImagePixels.convert("RGB"))
        positive = self.transforms(anchorImagePixels.convert("RGB"))
      transcriptions=self.data.loc[idx,self.transcriptionHeader]

      encodingdict=self.processor(text=transcriptions,images=anchorImagePixels,padding="max_length",return_tensors="pt")

      # encodingdict["id"]=self.data.loc[idx,"id"].values[0]

      encodingdict["negative"]=negative
      encodingdict["positive"]=positive
      encodingdict['labels']=self.processor.tokenizer(text=transcriptions,padding="max_length",return_tensors="pt").input_ids
      encodingdict["labels"][encodingdict["labels"] == self.processor.tokenizer.pad_token_id] = -100
      # create positives
      # the following will be available

      encodingdict = {k:v.squeeze() for k,v in encodingdict.items()}
      return encodingdict