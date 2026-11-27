EXIT CODES
0 success (including a closed stdout pipe)
2 usage
3 invalid scene/program
4 device, GPU memory, encoder, setup or uncategorized failure
5 filesystem I/O (including other stdout write failures)
A refused frame (e.g. non-finite radiance) stops the render, including videos and sequences, and leaves written frames without meta.json.
A canvas PNG/EXR sequence skips non-finite view/compositor frames, finishes the
range, then exits nonzero naming refused frames, without meta.json.

If setup cannot open a GPU on Linux, it checks Vulkan ICD driver files and names
libraries that cannot load or expose Vulkan entry points. A GPU visible in
nvidia-smi can still have a broken ICD. Use VK_LOADER_DEBUG=error,warn vork setup
for loader details; see docs/REMOTE-GPU.md for headless NVIDIA driver discovery.
