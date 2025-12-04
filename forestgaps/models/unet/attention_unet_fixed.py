"""Quick fix for attention_unet forward pass - corrected decoder loop"""

# FIX: In forward(), replace lines 201-215 with:

# Passage à travers le décodeur
for i in range(len(self.decoder_blocks)):
    # Récupérer les caractéristiques de l'encodeur
    skip = encoder_features[-(i+1)]

    # IMPORTANT: Appliquer l'attention AVANT upsampling
    # pour que le signal de guidage ait la bonne taille
    skip_attention = self.attention_gates[i](g=x, x=skip)

    # Sur-échantillonnage
    x = self.upsample_blocks[i](x)

    # Vérifier et ajuster les dimensions si nécessaire avant concaténation
    if x.shape[2:] != skip_attention.shape[2:]:
        skip_attention = F.interpolate(skip_attention, size=x.shape[2:], mode='bilinear', align_corners=False)

    # Concaténation
    x = torch.cat([x, skip_attention], dim=1)

    # Convolution du décodeur
    x = self.decoder_blocks[i](x)
