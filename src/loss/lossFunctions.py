import torch

class tripletLoss(torch.nn.Module):
      # extract first token and reshape inputs to 1,768 or what ever dimension you have embeddings

  def __init__(self,margin=1.0):
    super(tripletLoss, self).__init__()
    self.margin = margin
  def calculateEuclidean(self,x1,x2):
    #basically you have two embeddings, you calcualte the squared difference between the two.
    dist=(x1-x2).pow(2).sum(1)

    return dist
  def forward(self,anchor, positive, negative):
    #
    marginalPositiveEu=self.calculateEuclidean(anchor,positive)
    marginalNegativeEu=self.calculateEuclidean(anchor,negative)
    relu=torch.nn.ReLU()
    loss = relu(marginalPositiveEu-marginalNegativeEu+self.margin)
    # for the following
    return loss.mean()
  def calculateCosine(self,x1,x2):
    # The following needs to be completed
    # one needs

    return