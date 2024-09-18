import mast3r.utils.path_to_dust3r
from dust3r.inference import features_extract
from dust3r.utils.image import load_images

from mast3r.model import AsymmetricMASt3R


import torch

class MasterFeaturesExtractor:
    def __init__(
        self, size = 256, device=None, batch_size=50, num_workers=1
    ):  
        self.model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        self.size = size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AsymmetricMASt3R.from_pretrained(self.model_name).to(device)

    @torch.no_grad()
    def extract_features_from_pairs(self, pairs, mode):
        return features_extract(pairs, self.model, self.device, batch_size=self.batch_size, verbose=True, mode=mode)

    @torch.no_grad()
    def extract_features_single_source_multiple_targets(self, img_source, novel_paths, mode):

        img_novel =  load_images(novel_paths, size=self.size)

        source_novel_pairs = [(source, novel) for source, novel in zip(img_source, img_novel)]

        source_novel_feature = features_extract(source_novel_pairs, self.model, self.device, batch_size=self.batch_size, verbose=True, mode=mode)

        return source_novel_feature
    
