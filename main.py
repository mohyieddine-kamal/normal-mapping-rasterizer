import argparse
import time

import numpy as np
from PIL import Image

from camera import Camera
from graphicPipeline import GraphicPipeline
from projection import Projection
from readply import readply


def parse_args():
    parser = argparse.ArgumentParser(description="Software rasterizer with normal mapping")
    parser.add_argument("--scene", default="Scenes/flat_big.ply", help="PLY mesh (x y z nx ny nz s t)")
    parser.add_argument("--texture", default="ColorMap/stone.png", help="color texture")
    parser.add_argument("--normal-map", default="NormalMap/stone_NormalMap.png", help="tangent-space normal map")
    parser.add_argument("--no-normal-map", action="store_true", help="render without normal mapping, for comparison")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", help="save the render to this PNG file instead of opening a window")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = GraphicPipeline(args.width, args.height)

    position = np.array([1.1, 1.1, 1.1])
    lookAt = np.array([-0.577, -0.577, -0.577])
    up = np.array([0.33333333, 0.33333333, -0.66666667])
    right = np.array([-0.57735027, 0.57735027, 0.])
    cam = Camera(position, lookAt, up, right)

    proj = Projection(near=0.1, far=10.0, fov=1.91986, aspectRatio=args.width / args.height)

    vertices, triangles = readply(args.scene)

    data = {
        'viewMatrix': cam.getMatrix(),
        'projMatrix': proj.getMatrix(),
        'cameraPosition': position,
        'lightPosition': np.array([10, 0, 10]),
        'texture': np.asarray(Image.open(args.texture)),
        'normalMap': np.asarray(Image.open(args.normal_map)),
        'useNormalMap': not args.no_normal_map,
    }

    start = time.time()
    pipeline.draw(vertices, triangles, data)
    print(f"Rendered in {time.time() - start:.1f} s")

    if args.output:
        # Same orientation as the matplotlib window (row 0 at the top)
        img = (np.clip(pipeline.image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(args.output)
        print(f"Saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.imshow(pipeline.image)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
