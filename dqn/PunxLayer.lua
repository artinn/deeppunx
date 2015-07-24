require 'nn'

do

    local convLayer, parent = torch.class('nn.PunxLayer', 'nn.convLayer')
   
    -- override
the constructor to have the additional range of initialization
    function convLayer:__init(inputSize, outputSize, mean, std)
        parent.__init(self,inputSize,outputSize)
               
        self:reset(mean,std)
    end
   
    -- override the :reset method to use custom weight initialization.        
    function convLayer:reset(mean,stdv)
       
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:normal(mean,stdv)
        else
            self.weight:normal(0,1)
            self.bias:normal(0,1)
        end
    end

end

