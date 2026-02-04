import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def georegister(src_path, ref_path, out_path):
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, ref_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": ref_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )
