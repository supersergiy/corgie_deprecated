Downsample
^^^^^^^^^^
Next, we want to downsample the MIP6 data that we copied for easier visualization. By downsampling we meed producing coarser data at higher MIP levels. In this tutorial, we will downsample the layer from MIP6 to MIP6. In order to do that, rught the following commands:

Normalize image:

.. code-block:: bash 

   corgie downsample \
   --src_layer_spec '{
      "path": "gs://corgie/demo/my_first_stack/img/unaligned"
      }' \
   --mip_start 6 --mip_end 8 \
   --start_coord "150000, 150000, 17000" \
   --end_coord "200000, 200000, 17010" \
   --chunk_xy 1024


Normalize fold mask:

.. code-block:: bash 

   corgie downsample \
   --src_layer_spec '{
      "path": "gs://corgie/demo/my_first_stack/mask/fold_mask",
      "type": "mask"
      }' \
   --mip_start 6 --mip_end 8 \
   --start_coord "150000, 150000, 17000" \
   --end_coord "200000, 200000, 17010" \
   --chunk_xy 1024

Most of the parameters of ``downample`` command are same as with the ``copy`` command we used earlier -- we specify start and end coordinates, chunk size, and the source layer. Unlike ``copy`` command, we do not have to provice the destination parameter. When ``--dst_layer_spec`` is not specified, the downsampled data will be written to the source layer. 

Downsampling for the mask and the image layer have to be done separately, becuase image and mask data require different downsampling strategies. To learn more about downsampling strategies, please refer to TOD:downsampling_strategies.

To learn more about ``downsample`` command, please refer to TODO:downsample_command.


