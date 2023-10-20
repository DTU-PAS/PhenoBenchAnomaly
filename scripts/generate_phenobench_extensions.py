from phenobench.phenobench_loader import PhenoBench
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import hydra

def get_tiles(path_to_phenobench, tiles, split="train"):
    train_data = PhenoBench(path_to_phenobench, split=split,
                    target_types=["semantics"])
    
    weed_free_tiles = []
    weed_tiles = []
    crop_pixels = 0
    for i in range(len(train_data)):
        semantic_mask = train_data[i]["semantics"]
        semantic_mask[semantic_mask == 3] = 1
        semantic_mask[semantic_mask == 4] = 2
        for j, tile in enumerate(tiles):
            sub_mask = semantic_mask[tile[0][0]:tile[1][0], tile[0][1]:tile[1][1]]
            if np.sum(sub_mask == 2) == 0:
                weed_free_tiles.append({"image_name": train_data[i]["image_name"],
                                        "tile": tile,})
                crop_pixels += np.sum(sub_mask == 1)
            else:
                weed_tiles.append({"image_name": train_data[i]["image_name"],
                                        "tile": tile, "weed_pixels": np.sum(sub_mask == 2)})
    return weed_free_tiles, crop_pixels, weed_tiles

def generate_penobench_anomaly(path_to_phenobench, tiles, weed_percentages):
    weed_free_tiles, crop_pixels, weed_tiles = get_tiles(path_to_phenobench, tiles)

    for weed_percentage in weed_percentages:
        random.shuffle(weed_tiles)
        all_tiles = weed_free_tiles
        weed_pixels = 0
        weed_tile_id = 0
        while weed_pixels/crop_pixels < weed_percentage:
            weed_pixels += weed_tiles[weed_tile_id]["weed_pixels"]
            all_tiles.append(weed_tiles[weed_tile_id])
            weed_tile_id += 1
            if weed_tile_id == len(weed_tiles):
                break
        if weed_tile_id == len(weed_tiles):
            continue
        # write tiles to txt file
        with open(f"PhenoBench_extensions/train/phenobench_anomaly_{weed_percentage}.txt", "w") as f:
            for tile in all_tiles:
                f.write(f"{tile['image_name']} {tile['tile'][0][0]} {tile['tile'][0][1]} {tile['tile'][1][0]} {tile['tile'][1][1]}\n")

    # Let's save validation data if not yet done.
    if not os.path.exists("PhenoBench_extensions/val/phenobench_anomaly.txt"):
        weed_free_tiles, crop_pixels, weed_tiles= get_tiles(path_to_phenobench, tiles, split="val")
        all_tiles = weed_free_tiles + weed_tiles
        # write tiles to txt file
        with open(f"PhenoBench_extensions/val/phenobench_anomaly.txt", "w") as f:
            for tile in all_tiles:
                f.write(f"{tile['image_name']} {tile['tile'][0][0]} {tile['tile'][0][1]} {tile['tile'][1][0]} {tile['tile'][1][1]}\n")


@hydra.main(config_path="../config", config_name="config.yaml")
def main(cfg):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tiles_2_2 = [
                [[0, 0], [512, 512]], 
                [[512, 0], [1024, 512]], 
                [[0, 512], [512, 1024]], 
                [[512, 512], [1024, 1024]]
                ]
    tiles_3_3 = [
                [[0, 0], [341, 341]],
                [[341, 0], [682, 341]],
                [[682, 0], [1024, 341]],
                [[0, 341], [341, 682]],
                [[341, 341], [682, 682]],
                [[682, 341], [1024, 682]],
                [[0, 682], [341, 1024]],
                [[341, 682], [682, 1024]],
                [[682, 682], [1024, 1024]]
                ]              
    tiles = tiles_3_3
    # path_to_phenobench = "/home/ronja/data/PhenoBench-v100/PhenoBench"
    weed_percentages = [0.0, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    generate_penobench_anomaly(cfg.phenobench_root, tiles, weed_percentages)


if __name__ == "__main__":
    main()

