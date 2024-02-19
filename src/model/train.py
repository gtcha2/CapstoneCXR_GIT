from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torch
import os

class Trainer:
    def __init__(self, model, trainDataset, testDataset, tripletLoss, num_epochs=10, batch_size=4):
        self.model = model
        self.trainDataset = trainDataset
        self.testDataset = testDataset
        self.tripletLoss = tripletLoss
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.initialize_optimizers()
        
    
    def initialize_optimizers(self, filterFuncCLM=None, filterFuncTripletLoss=None):
        num_workers = os.cpu_count()  # Get the number of cores available
    
        self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        # Use custom filter functions if provided, otherwise default to lambda functions
        if filterFuncCLM is None:
            filterFuncCLM = lambda p: "ia3" in p[0] or "modules_to_save" in p[0]
        self.optimizerCLM = self.createOptimizer(filterFunc=filterFuncCLM)
        
        if filterFuncTripletLoss is None:
            filterFuncTripletLoss = lambda p: (("ia3" in p[0] and "image_encoder" in p[0]) or "fc2" in p[0])
        self.optimizerTripletLoss = self.createOptimizer(filterFunc=filterFuncTripletLoss)
        
        self.scheduler1 = lr_scheduler.CosineAnnealingLR(self.optimizerCLM, T_max=self.num_epochs//5, eta_min=0)
        self.scheduler2 = lr_scheduler.CosineAnnealingLR(self.optimizerTripletLoss, T_max=self.num_epochs//5, eta_min=0)
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}")
        
        #
    
    
     
    
    
    def build_image_embeddings_git_model(self, pixelValues):
        # this wont work in certain states___ need to fix
        tokenState = self.model.git.image_encoder(pixelValues)
        return self.model.git.visual_projection(tokenState.last_hidden_state)

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        loaderTqdm = tqdm(self.trainDataLoader)
        for i, batch in enumerate(loaderTqdm):
            # handle what type of training? contrastive vs jsut clm vs anything 
            
            self.train_batch(epoch, i, batch, loaderTqdm)
    def setGradPerOptimizerParams(self, optimizer):
        for param in optimizer.param_groups[0]['params']:
            param.reuires_grad=True
    def createOptimizer(self, filterFunc,**Optimizerkwargs):
        # this function should create an optimizer based off the filters you hand i
        # lambda p: "ia3" in p[0] or "modules_to_save" in p[0]
        
        optimizer=torch.optim.AdamW(
            (name[1] for name in filter(filterFunc, self.model.named_parameters())), **Optimizerkwargs)
        return optimizer
    
    def train_batch(self, epoch, batch_idx, batch, loaderTqdm):
        # Set all parameters to not require gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable gradients for parameters in optimizerCLM
        self.setGradPerOptimizerParams(self.optimizerCLM)
        # for param in self.optimizerCLM.param_groups[0]['params']:
        #     param.requires_grad = True

        input_ids = batch.pop("input_ids").to(self.device)
        pixel_values = batch.pop("pixel_values").to(self.device)
        attention_masks = batch.pop("attention_mask").to(self.device)
        labels = batch.pop("labels").to(self.device)

        # Forward pass for the main model
        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=labels, attention_mask=attention_masks)
        loss = outputs.loss
        loss.backward()
        self.optimizerCLM.step()
        self.optimizerCLM.zero_grad()

        # Set all parameters to not require gradients again
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable gradients for parameters in optimizerTripletLoss
        self.setGradPerOptimizerParams(self.optimizerTripletLoss)
        # for param in self.optimizerTripletLoss.param_groups[0]['params']:
        #     param.requires_grad = True

        # Prepare data for triplet loss calculation
        pixel_values_positive = batch.pop('positive').to(self.device)
        pixel_values_negative = batch.pop('negative').to(self.device)

        # Compute embeddings for triplet loss
        A_embedding = self.build_image_embeddings_git_model(pixel_values)
        P_embedding = self.build_image_embeddings_git_model(pixel_values_positive)
        N_embedding = self.build_image_embeddings_git_model(pixel_values_negative)

        # Calculate triplet loss
        tripletLossValue = self.tripletLoss(A_embedding, P_embedding, N_embedding)
        tripletLossValue.backward()

        loaderTqdm.set_description(f"Epoch {epoch} - Batch {batch_idx} - Loss: {loss.item():.4f} - Triplet Loss: {tripletLossValue.item():.4f}")

        self.optimizerTripletLoss.step()
        self.optimizerTripletLoss.zero_grad()

        # Perform any additional operations such as logging, model evaluation, etc.
        # ...

        # Update learning rate schedulers
        self.scheduler1.step()
        self.scheduler2.step()

# Usage
# model = [Your model]
# trainDataset = [Your training dataset]
# testDataset = [Your testing dataset]
# tripletLoss = [Your triplet loss function]
# trainer = Trainer(model, trainDataset, testDataset, tripletLoss)
# trainer.train()
