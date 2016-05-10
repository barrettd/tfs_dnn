// --------------------------------------------------------------------
//  DnnLayerSupportVectorMachine.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerSupportVectorMachine.h"

namespace tfs {
    
    static const char *NAME = "svm";

    const char*
    DnnLayerSupportVectorMachine::className( void ) {
        return NAME;
    }

    DnnLayerSupportVectorMachine::DnnLayerSupportVectorMachine( void ):
    DnnLayer( NAME ) {
        // Constructor
    }
    
    DnnLayerSupportVectorMachine::~DnnLayerSupportVectorMachine( void ) {
        // Destructor
    }
    
}   // namespace tfs

