
Unzip/git clone, into the ./modules/ folder of godot 4 repository.

Currently only works for linux, but it should be possible for it to work for windows

When running the created executable, you may need to run it from the terminal like so:

`LD_LIBRARY_PATH=$(pwd)/modules/rwkv/libtorch/lib/ ./bin/godot.linuxbsd.tools.x86_64`

Will be working on fixing this or something soon
