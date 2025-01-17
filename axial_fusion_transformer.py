import torch.nn as nn
import torch 
from einops.layers.torch import Rearrange
from utils import  append_neighboring_slices, convert3d_image2_slices

class conv_norm_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True) # Doesn't create a new tensor for the output.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True) # For saving memory

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out

class PixelEmbedding(nn.Module):
    def __init__(self,in_channels,pixel_size,embedding_size):
        self.pix_size = pixel_size
#         self.in_channels = in_channels
#         self.embedding_size = embedding_size
        super().__init__()
        self.inter_slice_projection = nn.Sequential(
            Rearrange('batch channel neighbor (h pix1) (w pix2) -> batch (h w) neighbor (pix1 pix2 channel)', pix1=pixel_size, pix2=pixel_size),
            nn.Linear(pixel_size * pixel_size * in_channels, embedding_size)
        )
    def forward(self,x):
#         two embedding must be created because different embeddings are generated for intra slice attention and inter slice attention after passing to linear layer
#         print(x.shape)
        inter_slice_embedding = self.inter_slice_projection(x)
        return inter_slice_embedding


# in inter attention we reshaped the tensor and generate embedding for each pixel across its neighbors, so there are 16 pixel in 4x4 image, so 
# and shape of the tensor is 8,16,9,512 where 9 is the neighbor and 512 is channel,8 is batchsize, 
class InterAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.dim = dim
#         In the figure we see a normalization step before attention so, we have done this
        self.norm = nn.LayerNorm(normalized_shape = self.dim)
        self.num_heads = num_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=self.dim,
                                               num_heads=self.num_heads,
                                               dropout=dropout, 
                                               batch_first=True,)
        
    def forward(self, x):
        # note the extra dimension is from the extra neighbour 
        #  Since MHA only takes three dimension, so we want to merge batch size and extra dimension into one.
        x = self.norm(x)
        batch_size, extra_dimension = x.shape[0], x.shape[1]
#         print(batch_size)
#         print(extra_dimension)
        x_reshaped = x.reshape(-1, x.shape[2], x.shape[3])
#         print(x_reshaped.shape)
        attn_output, attn_output_weights = self.att(x_reshaped,x_reshaped, x_reshaped,need_weights=True)
#         print(attn_output.shape) # torch.Size([1, 16, 128]) # N,L,E where N is the batch size, L is the target sequence length,and E is the embedding dimension 
#         print(attn_output_weights.shape) # torch.Size([1, 16, 16]) # (N,L,S), where N is the batch size, L is the target sequence length, and S is the source sequence length.
        
#         after calculating attention we reshape into original shape

        attn_output = attn_output.reshape(batch_size,extra_dimension, attn_output.shape[1],attn_output.shape[2])
        attn_output_weights = attn_output_weights.reshape(batch_size,extra_dimension, attn_output_weights.shape[1],attn_output_weights.shape[2])
        return attn_output


class IntraAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=self.dim,
                                               num_heads=self.num_heads,
                                               dropout=dropout, 
                                               batch_first=True)
    def forward(self, x):
#         print(x_reshaped.shape)
        attn_output, attn_output_weights = self.att(x,x,x,need_weights=True)
#         print(attn_output.shape) # torch.Size([1, 16, 128]) # N,L,E where N is the batch size, L is the target sequence length,and E is the embedding dimension 
#         print(attn_output_weights.shape) # torch.Size([1, 16, 16]) # (N,L,S), where N is the batch size, L is the target sequence length, and S is the source sequence length.
        
#         after calculating attention we reshape into original shape
        return attn_output
#         return attn_output, attn_output_weights

class Embedding2Pixel(nn.Module):
    def __init__(self,out_channels, embedding_dim, image_size):
        self.embedding_dim = embedding_dim
#         self.in_channels = in_channels
        super().__init__()
        self.embedding_2_img = nn.Sequential(
            nn.Linear(embedding_dim, out_channels),
            Rearrange('batch (h w) out_channels -> batch out_channels h w', h = image_size, w = image_size),
            
        )
    def forward(self,x):
#         two embedding must be created because different embeddings are generated for intra slice attention and inter slice attention after passing to linear layer
        embedding_2_images = self.embedding_2_img(x)
        return embedding_2_images

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    


class Transformer(nn.Module):
    def __init__(self, dim, num_layers, multihead_attention_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                InterAttention(dim, multihead_attention_heads, dropout = dropout),
                IntraAttention(dim, multihead_attention_heads, dropout = dropout),
                FeedForward(dim, dim, dropout = dropout)
            ]))

    def forward(self, x):
        for inter_attn,intra_attn, ff in self.layers:
            x_inter = inter_attn(x)
            x_inter_res = x_inter + x
#             print(x_inter_res.shape) # torch.Size([8, 16, 9, 128])
            # Only taking the inter 4th slice because hamle 4th wala slice ko MHA 4th embedding ma pauxam ani aru slices haru ko MHA hamle arko step ma calculate garihalxam nee ta
            x_intra = intra_attn(x_inter_res[:,:,4,:])
#             print(x_intra.shape) # torch.Size([8, 16, 128])
#             print(x_inter_res.shape) # torch.Size([8, 16, 9, 128])
            x_intra_res = x_intra + x_inter_res[:,:,4,:]
#             print(x_intra_res.shape)
#             print(ff(x_intra_res).shape) #torch.Size([8, 16, 128])
#             print(x_intra_res.shape) # torch.Size([8, 16, 128])
            x_final = ff(x_intra_res) + x_intra_res
        return self.norm(x_final)
    


class VitWithBottleneck(nn.Module):
    def __init__(self,img_size, patch_size, embedding_dim, multihead_attention_heads, num_layers, channels, dropout):
        super().__init__()
        self.pixel_embedding = PixelEmbedding(channels, patch_size, embedding_dim)
        num_patches = (img_size // patch_size) ** 2 # multiply by 2 becoz for both height and width
#         Making positional embedding a learnable parameter
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim)) # num_patches + 1 is done in original paper but we don't need cls token here
        self.transformer = Transformer(embedding_dim, num_layers, multihead_attention_heads, dropout)
        self.embedding2pixel = Embedding2Pixel(out_channels = channels,embedding_dim = embedding_dim, image_size=img_size)
        
    def forward(self, img):
        # print('PRINT VIT WITH BOTTLENECT ----------------------------')
        # print(print(img.shape))
        x = self.pixel_embedding(img)
        # print("value of x ")
        # print(x.shape) #torch.Size([8, 16, 9, 128])
        b,_,n,_ = x.shape
        # print(x.shape)
        # print(self.pos_embedding[:, :(n)].shape)
        x += self.pos_embedding[:, :(n)]
#         print(x.shape) #torch.Size([8, 16, 9, 128])     
        x = self.transformer(x)
        x = self.embedding2pixel(x)
        return x


class axial_fusion_transformer(nn.Module):
    def __init__(self, Na, Nf, num_classes,num_channels_before_training,init_features=64):
        super().__init__()
        """ Encoder"""
#         Number of blocks = 5
        self.Na = Na
        self.Nf = Nf
        self.num_classes = num_classes
        self.num_channels_before_training = num_channels_before_training
        self.features = init_features

        self.convert3d_image2_slices = convert3d_image2_slices
        self.append_neighboring_slices = append_neighboring_slices


        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = conv_norm_relu(self.num_channels_before_training,self.features)
        self.down_conv2 = conv_norm_relu(self.features,self.features*2)
        self.down_conv3 = conv_norm_relu(self.features*2,self.features*4)
        self.down_conv4 = conv_norm_relu(self.features*4,self.features*8)

        """Bottleneck"""
        self.down_conv5 = conv_norm_relu(self.features*8,self.features*16)
        """Transformer Block"""
        # change img size for shape which is after encoder and embedding dimension as shape of batch i.e zth dimension term. which is 128 for 128,128,128 object and 32 for 32,32,32 object
        self.vit_with_bottleneck = VitWithBottleneck(img_size=4, patch_size=1,embedding_dim = 128,multihead_attention_heads = 8, num_layers = 5, channels = init_features*16, dropout = 0.)
        
        """ Decoder"""

        self.up_transpose1 = nn.ConvTranspose2d(self.features*16, self.features*8, kernel_size=2, stride=2)
        self.up_conv1 = conv_norm_relu(self.features*16, self.features*8)
        self.up_transpose2 = nn.ConvTranspose2d(self.features*8, self.features*4, kernel_size=2, stride=2)
        self.up_conv2 = conv_norm_relu(self.features*8, self.features*4)
        self.up_transpose3 = nn.ConvTranspose2d(self.features*4, self.features*2, kernel_size=2, stride=2)
        self.up_conv3 = conv_norm_relu(self.features*4, self.features*2)
        self.up_transpose4 = nn.ConvTranspose2d(self.features*2, self.features, kernel_size=2, stride=2)
        self.up_conv4 = conv_norm_relu(self.features*2, self.features)

        self.out = nn.Conv2d(in_channels=self.features, out_channels=self.num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, a_3d_image):
        ct_scan_slices = self.convert3d_image2_slices(a_3d_image)
#         print(ct_scan_slices.shape) # torch.Size([128, 1, 128, 128])
        
        ct_scan_neighboring = self.append_neighboring_slices(self.Na, self.Nf, ct_scan_slices.shape[-1], ct_scan_slices)
        # print(ct_scan_neighboring.shape) # torch.Size([128, 9, 1, 128, 128])

        # If you get error like this : ValueError: too many values to unpack (expected 5) please change batch_size
        
        d, Na, c, h, w = ct_scan_neighboring.shape
        ct_scan_neighbouring = ct_scan_neighboring.reshape(-1,c,h,w)
        down_1 = self.down_conv1(ct_scan_neighbouring)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_conv2(down_2) 
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_conv3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_conv4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_conv5(down_8) 

        # print(down_2.shape) # torch.Size([1152, 64, 64, 64])                                                  
        # print(down_9.shape) # torch.Size([1152, 1024, 8, 8])                  # 1152 = 128 x 9

        down_9_reshaped = down_9.reshape(d, Na, down_9.shape[1], down_9.shape[2], down_9.shape[3])
#         print(p5_reshaped.shape) # torch.Size([128, 9, 128, 4, 4])
        down_9_permute = down_9_reshaped.permute(0,2,1,3,4) # depth->batch_size, channel, Na, h,w     # depth has turned into batchsize and original batchsize is 1 
        
        
        # print(down_9_permute.shape)
        
        """VIT Step"""
#         print(p5_permute.shape) # torch.Size([128, 128, 9, 4, 4]) # depth or batchsize , channel , neighbour , height, width
        after_axial_vit = self.vit_with_bottleneck(down_9_permute) # depth or batchsize , channel , height , width
        # print(after_axial_vit.shape) # torch.Size([128, 1024, 8, 8])

        """ Decoder"""
#         print(p5_permute.shape) # torch.Size([128, 128, 9, 4, 4])

#         print(p4.reshape(d,Na,p4.shape[1],p4.shape[2],p4.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:].shape) # torch.Size([128, 64, 8, 8])
        up_1 = self.up_transpose1(after_axial_vit)
        skip_1 = down_7.reshape(d,Na,down_7.shape[1],down_7.shape[2],down_7.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:]
        x = self.up_conv1(torch.cat([skip_1, up_1], 1))

        up_2 = self.up_transpose2(x)
        skip_2 = down_5.reshape(d,Na,down_5.shape[1],down_5.shape[2],down_5.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:]
        x = self.up_conv2(torch.cat([skip_2, up_2], 1))

        up_3 = self.up_transpose3(x)
        skip_3 = down_3.reshape(d,Na,down_3.shape[1],down_3.shape[2],down_3.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:]
        x = self.up_conv3(torch.cat([skip_3, up_3], 1))

        up_4 = self.up_transpose4(x)
        skip_4 = down_1.reshape(d,Na,down_1.shape[1],down_1.shape[2],down_1.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:]
        x = self.up_conv4(torch.cat([skip_4, up_4], 1))

        out = self.out(x)

        # print(out.shape)

#         print(d4.shape) # torch.Size([128, 4, 128, 128])
#         print(d4.permute(1,2,3,0).shape)  # torch.Size([4, 128, 128, 128])
        out_permute = out.permute(1,2,3,0)
        # print(out_permute.shape) # torch.Size([5, 128, 128, 128])
        # final_softmax = self.softmax(d5_permute)
        return out_permute #final_softmax


if __name__ == "__main__": 
    a = torch.rand((1, 1, 64,64,64))
    axial = axial_fusion_transformer(Na=9,Nf=1, num_classes=5,num_channels_before_training=1)
    y = axial(a) # torch.Size([5, 128, 128, 128])
    print(y.shape)
