from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pathlib

def visualize_input_output_and_label(raw_input,un_normalized_transformed_input,infered_prediction=None,label=None,file_name="a_filename",n_classes=10):
    print("visualizing :" + str(file_name))

    data = {pathlib.Path(file_name).name:raw_input,"un_normalized_transformed_input":un_normalized_transformed_input,"infered_prediction":infered_prediction,"label":label}
    #filter away all data == None
    things_to_show= [data_name for data_name in data if not data[data_name] is None]
    #prepare matplot lib for visualizing the data
    nr_of_things_to_show = len(things_to_show)

    fig, axes = plt.subplots(1, nr_of_things_to_show)
    

    for idx,data_name in enumerate(things_to_show):
        axes[idx].set_title(data_name)
        if data_name in ["infered_prediction","label"]:
            axes[idx].imshow(data[data_name], cmap="tab20", vmin=0, interpolation='nearest', vmax=n_classes)
        else:
            axes[idx].imshow(data[data_name])
    plt.show()


def un_normalize(normalized_img,means,stds):

    """
    Reversing the transforms.Normalize(mean, std) operation that was aplied as a transform in the dataset
    @output: data as uint8
    The output data should be posible to load as a normal [0-255] uint8 image
    The output should be posible to compare with the untransformed input
    If the output has artifacts that looks unnatural and isnt representative of the training set we have a
    problem with transforms.

    Normalize does the following
    img = (img - mean * max_pixel_value) / (std * max_pixel_value)
    #https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.transforms.Normalize

    So we need to do:un_normalized_img=  normalized_img* (std * max_pixel_value) +mean * max_pixel_value
    The code below coresponds to the pseudocode above
    """
    max_pixel_value = 255
    scaled_stds=  (stds * max_pixel_value)
    scaled_means = (means * max_pixel_value)
    img_shape = normalized_img.shape
    # in order for numpy to be able to broadcast properly we need to reshape the data,  and then reshape back after multiplcatin/addition
    un_normalized_img = (normalized_img.reshape(img_shape[0],-1) * scaled_stds.reshape([-1,1]) + scaled_means.reshape([-1,1])).reshape(img_shape)

    #after un-normalization the data should now be unit8 in range [0,255]
    un_normalized_img = np.clip(un_normalized_img, 0, 255)
    un_normalized_img=np.array(un_normalized_img,dtype=np.uint8)
    return un_normalized_img

def visualize_dataset(dataset,experiment_settings,n_classes=10):
    """
    A function for visualizing the data in the dataset
    showing the data as it looks after the transforms have been aplied
    """

    def simple_normalize(data,info):
        data = np.array(255 * ((data - data.min()) / (data.max() - data.min())), dtype=np.uint8)
        return data
    dataset.setup()
    #end matplotlib
    for sample_idx in range(experiment_settings["nr_of_images_to_visualize"]):
        for augmetnation_examples in range(experiment_settings["nr_of_transformed_versions_to_visualize"]):

            imglabel = dataset.dataset_train[sample_idx]
            try:
                (img, label)=imglabel
            except:
                #some datasets have no label
                img= imglabel
                label = None

            channels_to_show = [0, 1, 2]
            channelsdata = simple_normalize(img[channels_to_show],info="rgbdata")
            # showing first 3 channels
            rgb_img = Image.fromarray(channelsdata.transpose([1, 2, 0]), 'RGB')
            #label_img = Image.fromarray(simple_normalize(label,info="label"))
            #visualizing the input by unnormalizing it
            means= np.array(experiment_settings["means"])[channels_to_show]
            stds= np.array(experiment_settings["stds"])[channels_to_show]

            un_normalized = un_normalize(img[channels_to_show], means, stds).transpose([1,2,0])

            filename = dataset.image_paths_train[sample_idx]

            raw_input = np.array(Image.open(filename))


            visualize_input_output_and_label(raw_input=raw_input,un_normalized_transformed_input=un_normalized,infered_prediction=None,label=label,file_name=filename)

            """

            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(label,cmap="tab20",vmin=0,interpolation='nearest', vmax=n_classes)
            axes[0].set_title('label_img')
            axes[1].imshow(un_normalized)
            axes[1].set_title('rgb after transforms and un_normalization')
            plt.show()
            
            """
            # showing all channels one by one?
            show_all_channels= False
            if show_all_channels:
                for channel in range(len(experiment_settings["means"])):
                    normalized = un_normalized = un_normalize(img[channel], np.array(experiment_settings["means"][channel]), np.array(experiment_settings["stds"][channel]))
                    Image.fromarray(normalized).show()
           






