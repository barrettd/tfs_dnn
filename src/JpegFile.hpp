//
//  JpegFile.hpp
//
//  Created by Barrett Davis on 6/19/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "Matrix.hpp"

#ifndef JpegFile_hpp
#define JpegFile_hpp

namespace tfs {
    
    Matrix* readJpeg( const char *fileName );
    bool    writeJpeg( const Matrix &matrix, const char *fileName, int quality = 100 ); // quality 0 (bad) to 100 (good), inclusive
    
}  // tfs namespace

#endif /* JpegFile_hpp */
