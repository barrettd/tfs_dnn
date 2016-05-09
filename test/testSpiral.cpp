//
//  testSpiral.cpp
//  TestNeuralNet
//
//  Created by Barrett Davis on 5/9/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "Dnn.h"
#include "Error.h"
#include "testSpiral.hpp"

namespace tfs {
    
    static bool
    localTestSpiral( void ) {
        log_info( "Test Spiral - Start" );
        log_info( "Test Spiral - End" );
        return true;
    }
    
}  // tfs namespace


bool
testSpiral( void ) {
    return tfs::localTestSpiral();
}