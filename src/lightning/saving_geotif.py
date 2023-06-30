import numpy as np
import rasterio
import pathlib
import os

def save_output(a_file,probs,experiment_settings_dict,show=False):
    """
    Save the prediction as a geotiff
    """
    output_folder=pathlib.Path(experiment_settings_dict["output_folder"])
    # make sure outputfolder exists
    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(a_file) as src:
        # make a copy of the geotiff metadata so we can save the prediction/probabilities as the same kind of geotif as the input image
        new_meta = src.meta.copy()
        new_xform = src.transform

    if show:
        # show input image
        tmp_numpy = np.array(Image.open(a_file))
        # nir will be visualized as alpha channel. We remove it to make image visualizable
        tmp_numpy = tmp_numpy[:, :, 0:3]
        Image.fromarray(tmp_numpy).show()


    if experiment_settings_dict["crop_size"]:
        y_index_start = int((probs.shape[1] - experiment_settings_dict["crop_size"]) / 2)
        x_index_start = int((probs.shape[2] - experiment_settings_dict["crop_size"]) / 2)
        # create a translation transform to shift the pixel coordinates
        crop_translation = rasterio.Affine.translation(x_index_start, y_index_start)
        # prepend the pixel translation to the original geotiff transform
        new_xform = new_xform * crop_translation
        new_meta['width'] = int(experiment_settings_dict["crop_size"])
        new_meta['height'] = int(experiment_settings_dict["crop_size"])
        new_meta['transform'] = new_xform
        # set the number of channels in the output
        new_meta["count"] = probs.shape[0]

        y_index_end = y_index_start + int(experiment_settings_dict["crop_size"])
        x_index_end = x_index_start + int(experiment_settings_dict["crop_size"])

        #crop the infered prediction to remove the untrusted boarder areas
        probs = probs[:, y_index_start:y_index_end, x_index_start:x_index_end]
    # write the geotiff to disk
    path_to_probabilities = output_folder / pathlib.Path("PROBS_" + a_file.name)
    if experiment_settings_dict["save_probs"]:
        # probabilities are floats
        new_meta["count"] = probs.shape[0]
        new_meta["dtype"] = np.float32

        with rasterio.open(path_to_probabilities, "w", **new_meta) as dest:
            dest.write(probs)


    if experiment_settings_dict["save_preds"]:
        preds = np.array(probs.argmax(dim=0), dtype=np.uint8)

        new_meta["count"] = 1
        new_meta["dtype"] = np.uint8
        path_to_predictions = pathlib.Path(output_folder) / pathlib.Path(a_file.name)

        with rasterio.open(path_to_predictions, "w", **new_meta) as dest:
            dest.write(np.expand_dims(preds, axis=0))
