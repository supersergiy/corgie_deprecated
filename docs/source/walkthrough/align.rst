Align
^^^^^

After we copied, downsampled, and normalized the image stack, we are ready to align. This time we will be aligning using a pre-build ``corgie`` Block-Matching model: 


.. code-block:: bash 

   corgie align-block \
   --src_layer_spec '{"path": "gs://corgie/demo/my_first_stack/img/img_normalized"}' \
   --dst_folder gs://corgie/demo/my_first_stack/aligned \
   --start_coord "100000, 100000, 17000" \
   --end_coord "150000, 150000, 170010" \
   --chunk_xy 2048 \
   --suffix run_x0 \
   --processor_spec '{"ApplyModel": {
      "params": {
         "path": "gs://corgie/models/blockmatch",
         "tile_size": 128,
         "tile_step": 64,
         "max_disp": 48,
         "r_delta": 1.3
     }}}' \
   --processor_mip 7 

The ``--processor_spec`` specifies with "processor" to use for alignment, and ``--processor_mip`` specifies what resolution to apply it to. You can can change the ``tile_size``, ``tile_step``, ``max_disp`` and ``r_delta`` parameters and see how it affects your result. To avoid rewriting data from previous runs, use different ``--dst_folder`` and/or ``--suffix``. 

To learn more about the meaning of ``tile_size``, ``tile_step``, ``max_disp`` and ``r_delta`` parameters, please refer to `this link <https://imagej.net/Elastic_Alignment_and_Montage>`_.

To learn more about processor specification, please refer to TODO:processor_spec.

To learn more about ``align-block`` command, please refer to TODO:align_block_command.
