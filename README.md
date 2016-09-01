# tfs_dnn
C++ Deep Neural Net library - Version 1.01 is now available.

The currently implemented features work well, but there is still some more work to do.


This implementation was inspired by a Java Script library called ConvNetJS by Andrej Karpathy.

http://cs.stanford.edu/people/karpathy/convnetjs/

The library is configured to use Cmake.  Cmake configures the makefiles for your particular machine.
We build "out of source", which means that we do not mix our intermediate and output files with our source.

Here is one way to use Cmake with this project:

cd tfs_dnn

mkdir .build

cd .build

cmake ..

make

This command sequence will generate Makefiles, build a tfs_dnn library and a test_tfs_dnn application.  

The library is found in tfs_dnn/.build/src/libfts_dnn.a

The test application is found in tfs_dnn/.build/test/test_tfs_dnn

It takes about 15 seconds for the test application to execute on my 2009 MacBook Pro.

There are also some hand written Makefiles in the src and test directories.  I use these for my Kaggle projects and have kept them in the source tree in case you prefer not to use Cmake.   These Makefiles may then serve as a starting point for your customization.


I hope that you have fun with this library.  I have developed it as I learned about DNNs.

Please let me know if you have found it useful.

Regards,

Barrett Davis

http://thefrog.com/barrett/


https://github.com/barrettd/tfs_dnn.git



