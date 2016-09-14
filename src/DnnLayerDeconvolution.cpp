//
//  DnnLayerDeconvolution.cpp
//
//  Created by Barrett Davis on 6/24/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerDeconvolution.hpp"
#include "Error.hpp"

namespace tfs {
    
    DnnLayerDeconvolution::DnnLayerDeconvolution( DnnLayer *previousLayer, const bool trainable ):
    DnnLayer( LAYER_MAXOUT, previousLayer ) {
        // Constructor
    }
    
    DnnLayerDeconvolution::~DnnLayerDeconvolution( void ) {
        // Destructor
    }
    
    bool
    DnnLayerDeconvolution::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }
    
    bool
    DnnLayerDeconvolution::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
        return log_warn( "Not implemented yet" );
    }
    
}   // namespace tfs


