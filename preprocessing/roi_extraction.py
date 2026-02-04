import rasterio
from rasterio.windows import Window

def extract_roi(img_path, out_path, x, y, w, h):
    with rasterio.open(img_path) as src:
        window = Window(x, y, w, h)
        data = src.read(window=window)
        transform = src.window_transform(window)

        profile = src.profile
        profile.update({
            "height": h,
            "width": w,
            "transform": transform
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
