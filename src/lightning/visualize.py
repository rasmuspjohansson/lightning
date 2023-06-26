from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
def visualize(dataset,experiment_settings):
    """
    A function for visualizing the data in the dataset
    showing the data as it looks after the transforms have been aplied
    """
    nr_of_visualizations = experiment_settings["nr_of_visualizations"]

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
        print("normalized_img:" + str(normalized_img))

        scaled_stds=  (stds * max_pixel_value)
        scaled_means = (means * max_pixel_value)
        img_shape = normalized_img.shape
        # in order
        un_normalized_img = (normalized_img.reshape(img_shape[0],-1) * scaled_stds.reshape([-1,1]) + scaled_means.reshape([-1,1])).reshape(img_shape)


        print("un_normalized_img:"+str(un_normalized_img))
        un_normalized_img = np.clip(un_normalized_img, 0, 255)

        un_normalized_img=np.array(un_normalized_img,dtype=np.uint8)

        return un_normalized_img





    def simple_normalize(data,info):

        data = np.array(255 * ((data - data.min()) / (data.max() - data.min())), dtype=np.uint8)
        return data

    dataset.setup()






    #end matplotlib

    for sample_idx in range(nr_of_visualizations):
        for augmetnation_examples in range(3):
            img, label = dataset.dataset_train[sample_idx]

            channels_to_show = [0, 1, 2]
            channelsdata = simple_normalize(img[channels_to_show],info="rgbdata")
            # showing first 3 channels



            rgb_img = Image.fromarray(channelsdata.transpose([1, 2, 0]), 'RGB')
            label_img = Image.fromarray(simple_normalize(label,info="label"))


            #visualizing the input by unnormalizing it
            means= np.array([experiment_settings["means"][channel] for channel in channels_to_show])
            stds= np.array([experiment_settings["stds"][channel] for channel in channels_to_show])
            un_normalized = un_normalize(img[channels_to_show], means, stds).transpose([1,2,0])
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(label_img)
            axes[0].set_title('label_img')
            axes[1].imshow(un_normalized)
            axes[1].set_title('rgb after transforms and un_normalization')
            plt.show()

            # showing all channels one by one?
            show_all_channels= False
            if show_all_channels:
                for channel in range(len(experiment_settings["means"])):

                    normalized = un_normalized = un_normalize(img[channel], np.array(experiment_settings["means"][channel]), np.array(experiment_settings["stds"][channel]))
                    Image.fromarray(normalized).show()

            nr_of_visualizations = nr_of_visualizations - 1
            if nr_of_visualizations < 1:
                return 0






