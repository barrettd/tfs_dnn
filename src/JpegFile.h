//
//  JpegFile.hpp
//
//  Created by Barrett Davis on 6/19/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "Matrix.h"

#ifndef JpegFile_h
#define JpegFile_h

namespace tfs {
    
    Matrix* readJpeg( const char *fileName );
    bool    writeJpeg( const Matrix &matrix, const char *fileName, int quality = 100 ); // quality 0 (bad) to 100 (good), inclusive
    
}  // tfs namespace

#endif /* JpegFile_h */
