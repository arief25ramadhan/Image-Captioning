import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        # Word embedding layer
        self.word_embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM( input_size = embed_size,
                             hidden_size = hidden_size,
                             num_layers = num_layers,
                             dropout = 0,
                             batch_first = True
                           )
        
        # Linear layer
        self.linear_fc = nn.Linear(hidden_size, vocab_size)
        
        
    def forward(self, features, captions):
        
        # Get the captions, without the <end> word
        captions = captions[:, :-1] 
        
        # Pass image captions to the word embedding layer
        captions = self.word_embedding_layer(captions)
        
        # Input for LSTM layer
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)  
        
        # LSTM layer 
        lstm_output, _ = self.lstm(inputs)
        
        # Output of LSTM layer as linear layer input
        outputs = self.linear_fc(lstm_output)
        
        return outputs

    
    def sample(self, inputs, states=None, max_len=20):
        
        outputs = []   
        
        for i in range(max_len):
            
            # LSTM layer
            output, states = self.lstm(inputs,states)
            
            # Linear layer
            output = self.linear_fc(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            
            # convert cuda tensor to CPU first, because numpy doesnt support CUDA (GPU)
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            # Note that <end> has index_value = 1. Go out of the loop when <end> is detected
            if (predicted_index == 1):
                break
            
            inputs = self.word_embedding_layer(predicted_index)   
            inputs = inputs.unsqueeze(1)

        return outputs