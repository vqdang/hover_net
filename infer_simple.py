import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
import numpy as np
import cv2
import openslide
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.affinity import scale
from shapely.ops import unary_union
from PIL import ImageDraw
import h5py

from run_utils.utils import convert_pytorch_checkpoint
from models.hovernet.net_desc import HoVerNet
from models.hovernet.post_proc import process


def segment_tissue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    _, img_prepped = cv2.threshold(
        img_med, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy


def detect_foreground(contours, hierarchy):
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    # find foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

    all_holes = []
    for cont_idx in hierarchy_1:
        all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

    hole_contours = []
    for hole_ids in all_holes:
        holes = [contours[idx] for idx in hole_ids]
        hole_contours.append(holes)

    return foreground_contours, hole_contours


def construct_tissue_polygon(foreground_contours, hole_contours, min_area):
    polys = []
    for foreground, holes in zip(foreground_contours, hole_contours):
        # We remove all contours that consist of fewer than 3 points, as these won't work with the Polygon constructor.
        if len(foreground) < 3:
            continue

        # remove redundant dimensions from the contour and convert to Shapely Polygon
        poly = Polygon(np.squeeze(foreground))

        # discard all polygons that are considered too small
        if poly.area < min_area:
            continue

        if not poly.is_valid:
            # This is likely becausee the polygon is self-touching or self-crossing.
            # Try and 'correct' the polygon using the zero-length buffer() trick.
            # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
            poly = poly.buffer(0)

        # Punch the holes in the polygon
        for hole_contour in holes:
            if len(hole_contour) < 3:
                continue

            hole = Polygon(np.squeeze(hole_contour))

            if not hole.is_valid:
                continue

            # ignore all very small holes
            if hole.area < min_area:
                continue

            poly = poly.difference(hole)

        polys.append(poly)

    if len(polys) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")

    # If we have multiple polygons, we merge any overlap between them using unary_union().
    # This will result in a Polygon or MultiPolygon with most tissue masks.
    return unary_union(polys)


def make_tile_QC_fig(tile_sets, slide, level, line_width_pix):
    # Render the tiles on an image derived from the specified zoom level
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    downsample = 1 / slide.level_downsamples[level]

    draw = ImageDraw.Draw(img, 'RGBA')
    colors = ['lightgreen', 'red']
    assert len(tile_sets) <= len(colors), 'define more colors'
    for tiles, color in zip(tile_sets, colors):
        for tile in tiles:
            bbox = tuple(np.array(tile.bounds) * downsample)
            draw.rectangle(bbox, outline=color, width=line_width_pix)

    img = img.convert('RGB')
    return img


def create_tissue_mask(wsi,
                       seg_level,

                       # A tissue 'island' should
                       # have a minimum surface area of 1/x the total slide area at this level.
                       # If it is smaller, it is discarded.
                       # Note that this value should be sensible in the context of the chosen tile size.
                       min_rel_surface_area=500
                       ):
    # Determine the best level to determine the segmentation on
    level_dims = wsi.level_dimensions[seg_level]

    img = np.array(wsi.read_region((0, 0), seg_level, level_dims))
    contours, hierarchy = segment_tissue(img)
    foreground_contours, hole_contours = detect_foreground(contours, hierarchy)

    # Get the total surface area of the slide level that was used
    level_area = level_dims[0] * level_dims[1]

    # Minimum surface area of tissue polygons (in pixels)
    min_area = level_area / min_rel_surface_area

    tissue_mask = construct_tissue_polygon(
        foreground_contours, hole_contours, min_area)

    # Scale the tissue mask polygon to be in the coordinate space of the slide's level 0
    scale_factor = wsi.level_downsamples[seg_level]
    tissue_mask_scaled = scale(
        tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0))

    return tissue_mask_scaled


def create_tiles_in_mask(tissue_mask_scaled, tile_size_pix, stride, padding=0):
    # Generate tiles covering the entire mask
    minx, miny, maxx, maxy = tissue_mask_scaled.bounds

    # Add an additional tile size to the range stop to prevent tiles being cut off at the edges.
    cols = range(int(minx) - tile_size_pix, int(maxx + tile_size_pix), stride)
    rows = range(int(miny) - tile_size_pix, int(maxy + tile_size_pix), stride)
    rects = []
    for x in cols:
        for y in rows:
            # (minx, miny, maxx, maxy)
            rect = box(
                x - padding,
                y - padding,
                x + tile_size_pix + padding,
                y + tile_size_pix + padding,
            )

            # Retain only the tiles that partially overlap with the tissue mask.
            if tissue_mask_scaled.intersects(rect):
                rects.append(rect)

    return rects


def load_model(model_path, model_args, device):
    net = HoVerNet(**model_args)
    saved_state_dict = convert_pytorch_checkpoint(
        torch.load(model_path)["desc"]
    )
    net.load_state_dict(saved_state_dict, strict=True)
    net = torch.nn.DataParallel(net).to(device)
    return net


def crop_rect_from_slide(slide, rect):
    minx, miny, maxx, maxy = rect.bounds
    # Note that the y-axis is flipped in the slide: the top of the shapely polygon is y = ymax,
    # but in the slide it is y = 0. Hence: miny instead of maxy.
    top_left_coords = (int(minx), int(miny))
    return slide.read_region(top_left_coords, 0, (int(maxx - minx), int(maxy - miny)))


class BagOfTiles(Dataset):
    def __init__(self, wsi, tiles):
        self.wsi = wsi
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        img = crop_rect_from_slide(self.wsi, tile)

        # Convert from RGBA to RGB
        img = img.convert('RGB')

        # Ensure we have a square tile in our hands.
        # We can't handle non-squares currently, as this would requiring changes to
        # the aspect ratio when resizing.
        width, height = img.size
        assert width == height, 'input image is not a square'

        # Turn the PIL image into a (C x H x W) torch.FloatTensor (32 bit by default)
        # in the range [0.0, 1.0].
        img = transforms.functional.to_tensor(img)

        # TODO: the model's forward() weirldy expects images to be in domain [0.0, 255.0]
        # This is hard to change because it affects training dataloader as well.
        img = img * 255

        coords = np.array(tile.bounds).astype(np.int32)
        return img, coords


def infer_batch(batch_imgs, model, device):
    batch_imgs = batch_imgs.to(device, non_blocking=True)
    with torch.no_grad():
        pred_dict = model(batch_imgs)
        # Restructure the tensor: move the 'values' to the last dimension.
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()]
             for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)
    return pred_output.cpu().numpy()


def write_to_h5(file, asset_dict):
    for key, val in asset_dict.items():
        if key not in file:
            maxshape = (None, ) + val.shape[1:]
            dset = file.create_dataset(
                key,
                shape=val.shape,
                maxshape=maxshape,
                dtype=val.dtype
            )
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + val.shape[0], axis=0)
            dset[-val.shape[0]:] = val


def postprocess(features_file, nr_types, chunk_padding, patch_margin):
    with h5py.File(features_file, 'r') as hdf5_file:
        chunk_shape = box(*hdf5_file['chunk'])
        patch_coords = hdf5_file['coords'][:]
        patch_features = hdf5_file['features'][:]

    # Normalize patch coordinates to origin == 0
    minx = np.min(patch_coords[:, 0])
    miny = np.min(patch_coords[:, 1])
    maxx = np.max(patch_coords[:, 2])
    maxy = np.max(patch_coords[:, 3])
    normed_coords = (patch_coords - (minx, miny, minx, miny))

    # Assemble all patch-level feature maps into a single feature map
    # The feature map shape =/= the chunk shape! A patch does not have to be fully inside the chunk to be considered 'in the chunk'.
    # Hence, many patches will be in multiple chunks.
    feature_map = np.zeros(dtype=np.float32, shape=(
        int(maxy - miny), int(maxx - minx), 4))

    for c, f in zip(normed_coords, patch_features):
        cropped = c + (patch_margin, patch_margin, -
                       patch_margin, -patch_margin)
        _minx, _miny, _maxx, _maxy = cropped
        feature_map[_miny:_maxy, _minx:_maxx] = f

    # Apply Sobel/watershed postprocessing to the feature map
    pred_inst, inst_info_dict = process(
        feature_map,
        nr_types=nr_types,
        return_centroids=True
    )

    # De-normalize the coordinates that are currently expressed relative to the chunk shape back to WSI space
    shift = np.array([minx, miny], dtype=np.int32)
    normalized_instances = []
    for inst in inst_info_dict.values():
        centroid = inst['centroid'] + shift
        inst_in_wsi_coords = {
            "centroid": centroid,
            "contour": inst['contour'] + shift,
            "bbox": inst['bbox'] + shift,
            "type_prob": inst['type_prob'],
            "type": inst['type'],
        }
        normalized_instances.append(inst_in_wsi_coords)

    # Remove the cells that are inside the chunk padding
    chunk_minx, chunk_miny, chunk_maxx, chunk_maxy = np.array(chunk_shape.bounds) + np.array([
        chunk_padding,
        chunk_padding,
        - chunk_padding,
        - chunk_padding,
    ], dtype=np.int32)

    filtered_instances = []
    for inst in normalized_instances:
        x, y = inst['centroid']
        if x > chunk_minx and x < chunk_maxx and y > chunk_miny and y < chunk_maxy:
            filtered_instances.append(inst)

    return filtered_instances


if __name__ == '__main__':
    import os
    import argparse
    import time
    from multiprocessing import Pool, cpu_count
    import csv

    parser = argparse.ArgumentParser(description='HoVer-Net inference script')
    parser.add_argument('input_slide', type=str, help='Path to input WSI file')
    parser.add_argument('output_dir', type=str,
                        help='Directory to save output data (and temporary files)')
    parser.add_argument('model_path', type=str,
                        help='Path to the model checkpoint')
    parser.add_argument('nr_types', type=int,
                        help='Number of nuclei types to predict. Dependent on model checkpoint.', default=0)
    parser.add_argument('model_mode', type=str, choices=['fast', 'original'],
                        help='Model architecture. Dependent on model checkpoint.'
                        )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nr_inference_workers',
                        type=int,
                        default=cpu_count()-1,
                        help='Number of workers to use for the pytorch DataLoader')
    parser.add_argument('--nr_post_proc_workers',
                        type=int,
                        default=cpu_count(),
                        help='Number of workers to use for postprocessing (recommended: use all cores)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Derive the slide ID from its name
    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))

    # Open the slide for reading
    wsi = openslide.open_slide(args.input_slide)

    chunk_size = 3000
    chunk_padding = 100
    if args.model_mode == 'fast':
        # These params are hardcoded in the Hovernet model code.
        patch_input_shape = 256  # In pixels. Patches are squares.
        # Margin along each edge of the square. Hardcoded in the model architecture.
        patch_margin = 46
        patch_output_shape = 164
    else:
        patch_input_shape = 270
        patch_output_shape = 80
        # TODO: hardcode correct patch margin value
        patch_margin = (patch_input_shape - patch_output_shape) / 2

    # Decide on which slide level we want to base the segmentation
    seg_level = wsi.get_best_level_for_downsample(64)

    # Run the segmentation and tiling procedure
    start_time = time.time()
    tissue_mask = create_tissue_mask(wsi, seg_level)
    patches = create_tiles_in_mask(
        tissue_mask,
        tile_size_pix=patch_input_shape,
        stride=patch_output_shape
    )
    chunks = create_tiles_in_mask(
        tissue_mask,
        tile_size_pix=chunk_size,
        stride=chunk_size,
        padding=chunk_padding,
    )

    # Build a figure for quality control purposes; to check if the tiles are where we expect them.
    qc_img = make_tile_QC_fig([patches, chunks], wsi, seg_level, 1)
    qc_img_target_width = 1920
    qc_img = qc_img.resize((qc_img_target_width, int(
        qc_img.height / (qc_img.width / qc_img_target_width))))
    qc_img_file_path = os.path.join(args.output_dir, slide_id + '_tile_QC.png')
    qc_img.save(qc_img_file_path)
    print(f"Finished creating tissue tiles in {time.time() - start_time}s")

    model_args = dict(
        input_ch=3,
        freeze=True,  # disable gradients

        # Model checkpoint-specific
        nr_types=args.nr_types,
        mode=args.model_mode,
    )

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = load_model(
        args.model_path,
        model_args,
        device
    )
    model.eval()

    # Use multiple workers if running on the GPU, otherwise we'll need all workers for
    # evaluating the model.
    loader_kwargs = {'num_workers': args.nr_inference_workers,
                     'pin_memory': True} if device.type == "cuda" else {}

    start_time = time.time()
    chunk_feature_files = []
    for chunk_id, chunk in enumerate(chunks):
        chunk_time = time.time()
        chunk_features_file_path = os.path.join(
            args.output_dir, slide_id + f'_chunk_{chunk_id}_features.h5')
        patches_intersecting_chunk = [
            p for p in patches if chunk.intersects(p)]
        loader = DataLoader(
            dataset=BagOfTiles(wsi, patches_intersecting_chunk),
            batch_size=args.batch_size,
            **loader_kwargs,
        )
        chunk_features = []
        coords = []
        for batch_id, (batch, c) in enumerate(loader):
            print(
                f'Chunk {chunk_id}/{len(chunks)} -- inferring batch {batch_id}/{len(loader)}...')
            chunk_features.append(infer_batch(batch, model, device))
            coords.append(c)

        with h5py.File(chunk_features_file_path, 'w') as file:
            write_to_h5(file, {
                'features': np.concatenate(chunk_features),
                'coords': np.vstack(coords),
                'chunk': np.array(chunk.bounds).astype(np.int32)
            })

        chunk_feature_files.append(chunk_features_file_path)
        print(
            f"Finished chunk {chunk_id} in {((time.time() - chunk_time) / 60):.2f} mins")

    print(
        f"Finished inference on all chunks in {((time.time() - start_time) / 60):.2f} mins")
    print(
        f"Postprocessing {len(chunk_feature_files)} chunk feature maps using {args.nr_post_proc_workers} workers...")

    def postproc_chunk(filename):
        return postprocess(filename, args.nr_types, chunk_padding, patch_margin)

    start_time = time.time()
    results = []
    output_file = os.path.join(args.output_dir, slide_id + '_nuclei.csv')
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=['type', 'type_prob', 'centroid_x', 'centroid_y']
        )
        writer.writeheader()
        with Pool(processes=args.nr_post_proc_workers) as pool:
            for instances in pool.imap_unordered(postproc_chunk, chunk_feature_files, 1):
                # TODO: save complete results object in a streamlined format. Storing only nuclei coordinates and class for now.
                writer.writerows([
                    {
                        'type': i['type'],
                        'type_prob': i['type_prob'],
                        'centroid_x': int(i['centroid'][0]),
                        'centroid_y':int(i['centroid'][1])
                    } for i in instances
                ])

    # cleanup temporary files
    for f in chunk_feature_files:
        os.remove(f)

    print(
        f"Finished postprocessing of all chunks in {((time.time() - start_time) / 60):.2f} mins")
