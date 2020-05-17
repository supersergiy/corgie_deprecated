Normalize
^^^^^^^^^

The last step before we start the actual alignment is normalization of images. In many cases alignment procedures perform better when each input section has `0.0` mean and `1.0` variance. However, simply normalizing all of the pixels in each section for the specified bounding cube can produce biases result when part of the section is missing or defected. For this reason, we use several masks in the normalization commad:

.. code-block:: bash 

   corgie normalize \
   --src_layer_spec '{
      "path": "gs://corgie/demo/my_first_stack/img/unaligned",
         "name": "img"
         }' \
   --src_layer_spec '{
      "path":"gs://corgie/demo/my_first_stack/mask/fold_mask", 
         "type": "mask",
         "name": "fold_mask"
      }' \
      --src_layer_spec '{
         "path":"gs://corgie/demo/my_first_stack/img/unaligned", 
         "args": {"binarization": ["eq", 0.0]},
         "type": "mask",
         "name": "black_mask"
      }' \
   --dst_folder gs://corgie/demo/my_first_stack \
   --stats_mip 7 --mip_start 6 --mip_end 8 \
   --start_coord "150000, 150000, 17000" \
   --end_coord "200000, 200000, 17010" \
   --chunk_xy 2048 \
   --suffix normalized \
   --recompute_stats

The first mask that we use is the fold mask we copied from the reference stack. The second mask is obtained by applying ``binarization: ["eq", 0,0]`` to the source image, which will mask out all the ``0`` valued pixels in the image. 

``normalize`` command works in two steps -- first it computes mean and variance for each section, and then it normalizes each section individually. The MIP at which mean and variance are calculated is specified by ``--stats_mip``. 

We also speciffy the ``--suffix`` to be used for the resulting layer -- in this case, the normalized image will be written out to ``gs://corgie/demo/my_first_stack/img/img_normalized``.

To learn more about ``normalize`` command, please refer to TODO:normalize_command.

