//
//  testIO.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/31/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "DnnBuilder.h"
#include "DnnTrainerSGD.h"
#include "Error.h"
#include "testIO.hpp"

namespace tfs {
    
    static bool
    localTestIO( void ) {
        log_info( "Test I/O - Start" );
        log_info( "Test I/O - End" );
        return true;
    }
    
    
}  // tfs namespace

bool
testIO( void ) {
    return tfs::localTestIO();
}
