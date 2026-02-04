import rasterio
from rasterio.enums import Resampling

def resample_image(src_path, out_path, scale_factor):
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * scale_factor),
                int(src.width * scale_factor)
            ),
            resampling=Resampling.bilinear
        )

        transform = src.transform * src.transform.scale(
            src.width / data.shape[-1],
            src.height / data.shape[-2]
        )

        profile = src.profile
        profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
