import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import re
from operator import attrgetter
from torch.nn import functional as F

#TODO: import the model, load in the other sections 

model=AutoModelForCausalLM.from_pretrained("microsoft/git-large")

# need to fix ia3 vectors if feedforward...

class MoV(nn.module):
    def __init__(self, linear_layer, num_experts):
        super(MoV, self).__init__()
        # Original linear layer
        self.original_layer = linear_layer
        self.router = Router(self.original_layer.in_features, num_experts)
        self.experts = nn.Parameter(torch.ones(num_experts, linear_layer.out_features))
    def prepare_model_gradients(self):
        self.experts.requires_grad_(True)
        self.router.feedForward.weight.requires_grad_(True)
    def forward(self, x):
        frozen_output = self.original_layer(x)
        _, gating_probs = self.router(x)
        # Compute the weighted sum of expert outputs
        # mov_combined = torch.einsum("bse,ed->bsd", gating_probs, self.experts)
        # Assuming gating_probs is of shape [b, s, e] and self.experts is of shape [e, d]
        mov_combined = torch.matmul(gating_probs, self.experts)  # Performs batched matrix multiplication
        mov_combined = mov_combined.transpose(1, 2)  # Transposes the last two dimensions
        return frozen_output * mov_combined

class Router(torch.nn.Module):
    
  def __init__(self,input_dim,num_experts):
    super().__init__()
    self.num_experts = num_experts
    self.feedForward=torch.nn.Linear(input_dim,num_experts)
  def __repr__(self):
      # Call the superclass's __repr__ method and customize the result
      superclass_repr = super().__repr__()

      modified_repr = str(self.num_experts)+"x "+superclass_repr
      return modified_repr
  def forward(self,input):
    logits = self.feedForward(input)
    probs = F.softmax(logits,dim=-1)
    return logits, probs


# need to make custom class for loading..
# create the appropriate regex. 
regexForImage = "(.*q_proj.weight)|(.)"




class MOE_adapter:
    def __init__(self, model="microsoft/git-large",experts=1,regex="(.*query.weight)|(.*intermediate.dense.weight)"):
        self.model = AutoModelForCausalLM.from_pretrained(model)
    
    def adapt_model_with_moe(self,experts, regex):
        
        self.regex=regex
        regex_match = regex#"(.*(self_attn|LlamaAttention).(k_proj|v_proj).weight)|(.*LlamaMLP.down_proj.weight)"

        for n, _ in self.model.named_parameters():
            if re.search(regex_match, n) is None:
                continue
            # Get module that the parameter belongs to
            module_name = ".".join(n.split(".")[:-1])
            module = attrgetter(module_name)(self.model)
            module_parent_name = ".".join(n.split(".")[:-2])
            module_key_name = n.split(".")[-2]
            module_parent = attrgetter(module_parent_name)(self.model)
            setattr(module_parent, module_key_name, MoV(module, experts))

        # Freeze base model and set MoV weights as tunable
        for m in self.model.modules():
            m.requires_grad_(False)
            if isinstance(m, MoV):
                m.prepare_model_gradients()
    def change_num_experts(self,experts):
        assert not self.regex is None
        self.remove_model_moe_peft()
        self.adapt_model_with_moe(experts,self.regex)
        return
    def remove_model_moe_peft(self):
        regex_match = "(.*original_layer.weight)"
        namedParams=set()
        for n, _ in self.model.named_parameters():
        
            # check for filter
            if re.search(regex_match, n) is None:
                continue
            namedParams.add(n)
        for n in namedParams:   
            # module name is everythng but last ie exclude .weight
            module_name = ".".join(n.split(".")[:-2])
            
          
            
            # print(module)
            # now get the parent name everything but query and weight
            module_parent_name = ".".join(n.split(".")[:-3])
            # module key name... ie query in this case
            module_key_name = n.split(".")[-3]
        
            # print(module_key_name)
            # actually retrieve the module
            module_parent = attrgetter(module_parent_name)(self.model)
            original_layer= attrgetter(module_name+".original_layer")(self.model)
            # from module ie self set query to MoV and key name ie the module to 
            setattr(module_parent, module_key_name, original_layer)

        # Freeze base model and set MoV weights as tunable
        for m in self.model.modules():
            m.requires_grad_(True)
        
        
      
        
        
    def build_image_embeddings_git_model(self, pixelValues):
        # this will only work if the model is git model 
        # this 
        tokenState = self.model.git.image_encoder(pixelValues)
        return self.model.git.visual_projection(tokenState.last_hidden_state)
    
        
        
        
        