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
    
class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_norm_relu(in_channels,out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return p
    

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_norm_relu(in_channels,out_channels)
        self.up_sample = nn.Upsample(scale_factor=2)
        
    def forward(self, x, skip):
        x = x + skip #torch.cat([x,skip],axis=1)
        x = self.up_sample(x)
        x = self.conv(x)
        return x

class PixelEmbedding(nn.Module):
    def __init__(self,in_channels,pixel_size,embedding_size):
        self.pix_size = pixel_size
#         self.in_channels = in_channels
#         self.embedding_size = embedding_size
        super().__init__()
        self.inter_slice_projection = nn.Sequential(
            Rearrange('batch channel neighbor (h pix1) (w pix2) -> batch (h w) neighbor (pix1 pix2 channel)', pix1=pixel_size, pix2=pixel_size),
            nn.Linear(pixel_size*pixel_size * in_channels, embedding_size)
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
        x = self.pixel_embedding(img)
#         print(x.shape) #torch.Size([8, 16, 9, 128])
        b,_,n,_ = x.shape
        x += self.pos_embedding[:, :(n)]
#         print(x.shape) #torch.Size([8, 16, 9, 128])     
        x = self.transformer(x)
        x = self.embedding2pixel(x)
        return x


class axial_fusion_transformer(nn.Module):
    def __init__(self, Na, Nf, num_classes,num_channels_before_training,init_features=8):
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
        
        self.e1 = encoder_block(self.num_channels_before_training,self.features)
        self.e2 = encoder_block(self.features,self.features*2)
        self.e3 = encoder_block(self.features*2,self.features*4)
        self.e4 = encoder_block(self.features*4,self.features*8)
        self.e5 = encoder_block(self.features*8,self.features*16)
        
        """Transformer Block"""
        self.vit_with_bottleneck = VitWithBottleneck(img_size=4, patch_size=1,embedding_dim = 128,multihead_attention_heads = 8, num_layers = 5, channels = init_features*16, dropout = 0.)
        
        """ Decoder"""
        self.d1 = decoder_block(self.features*16, self.features*8)
        self.d2 = decoder_block(self.features*8, self.features*4)
        self.d3 = decoder_block(self.features*4, self.features*2)
        self.d4 = decoder_block(self.features*2, self.features)
        self.d5 = decoder_block(self.features, self.num_classes)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, a_3d_image):
        ct_scan_slices = self.convert3d_image2_slices(a_3d_image)
#         print(ct_scan_slices.shape) # torch.Size([128, 1, 128, 128])
        
        ct_scan_neighboring = self.append_neighboring_slices(self.Na, self.Nf, ct_scan_slices.shape[-1], ct_scan_slices)
#         print(ct_scan_neighboring.shape) # torch.Size([128, 8, 1, 128, 128])
        d, Na, c, h, w = ct_scan_neighboring.shape
        ct_scan_neighbouring = ct_scan_neighboring.reshape(-1,c,h,w)
        p1 = self.e1(ct_scan_neighbouring)
        p2 = self.e2(p1)
        p3 = self.e3(p2)
        p4 = self.e4(p3)
        p5 = self.e5(p4)
#         print(p5.shape) # torch.Size([1152, 128, 4, 4]) # 1152 = 128 x 9
        p5_reshaped = p5.reshape(d, Na, p5.shape[1], p5.shape[2], p5.shape[3])
#         print(p5_reshaped.shape) # torch.Size([128, 8, 128, 4, 4])
        p5_permute = p5_reshaped.permute(0,2,1,3,4) # depth->batch_size, channel, Na, h,w     # depth has turned into batchsize and original batchsize is 1 
        """VIT Step"""
#         print(p5_permute.shape) # torch.Size([128, 128, 9, 4, 4]) # depth or batchsize , channel , neighbour , height, width
        after_axial_vit = self.vit_with_bottleneck(p5_permute) # depth or batchsize , channel , height , width
#         print(after_axial_vit.shape) # torch.Size([128, 128, 4, 4])

        """ Decoder"""
#         print(p5_permute.shape) # torch.Size([128, 128, 9, 4, 4])
        d1 = self.d1(after_axial_vit, p5_permute[:,:,4,:,:]) # Skip Connections FROM  S5 
#         print(d1.shape) # torch.Size([128, 64, 8, 8])
#         print(p4.reshape(d,Na,p4.shape[1],p4.shape[2],p4.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:].shape) # torch.Size([128, 64, 8, 8])
        d2 = self.d2(d1, p4.reshape(d,Na,p4.shape[1],p4.shape[2],p4.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:]) # Skip Connections FROM S4
        d3 = self.d3(d2, p3.reshape(d,Na,p3.shape[1],p3.shape[2],p3.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:]) # Skip Connections FROM S3
        d4 = self.d4(d3, p2.reshape(d,Na,p2.shape[1],p2.shape[2],p2.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:]) # Skip Connections FROM S2
        d5 = self.d5(d4, p1.reshape(d,Na,p1.shape[1],p1.shape[2],p1.shape[3]).permute(0,2,1,3,4)[:,:,4,:,:]) # Skip Connections FROM S1

#         print(d5.shape) # torch.Size([128, 4, 128, 128])
#         print(d5.permute(1,2,3,0).shape)  # torch.Size([4, 128, 128, 128])
        d5_permute = d5.permute(1,2,3,0)
        # final_softmax = self.softmax(d5_permute)
        return d5_permute #final_softmax