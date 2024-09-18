from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.demo import get_3D_model_from_scene


if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    # images = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png'], size=512)
    images = load_images(['../zero123/zero123/data/gso-30-360/alarm/000.png', '../zero123/zero123/data/gso-30-360/alarm/008.png'], size=256)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)

    from scipy.spatial.transform import Rotation
    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    r = Rotation.from_matrix(poses[1,:3,:3].cpu().numpy())
    angles = r.as_euler("zyx",degrees=True)
    print(angles)
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # visualize reconstruction
    # scene.show()
    outdir = './figures'
    outfile = get_3D_model_from_scene(outdir, silent=False, scene=scene, min_conf_thr=3,
                                      as_pointcloud=False, mask_sky=False, clean_depth=True, transparent_cams=False, cam_size=0.05)

    print("3D model saved to:", outfile)