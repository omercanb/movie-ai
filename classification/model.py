import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, 
                blocks,
                filters,
                kernel_size,
                embedding_dim,
                dropout_rate,
                pool_size,
                input_shape,
                num_features,):
        super(Classifier, self).__init__()


        self.embedding = nn.Embedding(num_embeddings = num_features, embedding_dim = embedding_dim, padding_idx=0)

        self.conv_stack = nn.Sequential()

        self.conv_stack.append(nn.Dropout(dropout_rate))
        self.conv_stack.append(SeperableConv1D(embedding_dim, filters, kernel_size))
        self.conv_stack.append(nn.ReLU())
        self.conv_stack.append(SeperableConv1D(embedding_dim, filters, kernel_size))
        self.conv_stack.append(nn.ReLU())
        self.conv_stack.append(nn.MaxPool1d(pool_size))

        for _ in range(blocks-2):
            self.conv_stack.append(nn.Dropout(dropout_rate))
            self.conv_stack.append(SeperableConv1D(filters, filters, kernel_size))
            self.conv_stack.append(nn.ReLU)
            self.conv_stack.append(SeperableConv1D(filters, filters, kernel_size))
            self.conv_stack.append(nn.ReLU)
            self.conv_stack.append(nn.MaxPool1d(pool_size))

        self.conv_stack.append(SeperableConv1D(filters * 2, filters * 2, kernel_size))
        self.conv_stack.append(nn.ReLU)
        self.conv_stack.append(SeperableConv1D(filters * 2, filters * 2, kernel_size))
        self.conv_stack.append(nn.ReLU)

        self.global_avg_pool = nn.AvgPool1d(kernel_size=input_shape[0])
        self.dropout_final = nn.Dropout(dropout_rate)
        self.linear_output = nn.Linear(in_features=filters * 2, out_features=1)


    def forward(self, x):
        x = self.embedding(x)
        x = self.conv_stack(x)
        x = self.global_avg_pool(x)
        x = self.dropout_final(x)
        x = self.linear_output(x)
        return x



class DepthwiseConvolution1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseConvolution1D, self).__init__()
        self.layer = nn.Conv1d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                groups=in_channels,)
        

    def forward(self, x):
        return self.layer(x)
    

class PiecewiseConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PiecewiseConvolution, self).__init__()
        self.layer = nn.Conv1d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=1,)
        

    def forward(self, x):
        return self.layer(x)
    

class SeperableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeperableConv1D, self).__init__()
        self.depthwise = DepthwiseConvolution1D(in_channels=in_channels, 
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,)
        self.piecewise = PiecewiseConvolution(in_channels=out_channels,
                                                out_channels=out_channels,)
        

    def forward(self, x):
        x = self.depthwise(x)
        x = self.piecewise(x)
        return x
            