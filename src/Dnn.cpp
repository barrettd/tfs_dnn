// --------------------------------------------------------------------
//  Dnn.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#include "Dnn.h"
#include "DnnLayers.h"

namespace tfs {
   
    Dnn::Dnn( void ) {
        // Constructor
    }
    
    Dnn::~Dnn( void ) {
        // Destructor
        clear();
    }
    
    void
    Dnn::clear( void ) {
        // Clean up the layers.
        std::vector< DnnLayer* >::const_iterator layer_end = m_layers.end();
        for( std::vector< DnnLayer* >::const_iterator it = m_layers.begin(); it != layer_end; it++ ) {
            DnnLayer *layer = *it;
            delete layer;
        }
        m_layers.clear();
        return;
    }
    
    bool
    Dnn::addLayerInput( unsigned long xx, unsigned long yy, unsigned long zz ) {
        // Add an input layer
        if( xx < 1 || yy < 1 || zz < 1 ) {
            return false;
        }
        DnnLayerInput *layer = new DnnLayerInput( xx, yy, zz );
        m_layers.push_back( layer );
        
        return true;
    }
    
    bool
    Dnn::addLayerConvolution( void ) {
        // Convolution Layer
        return true;
    }
    
    bool
    Dnn::addLayerDropout( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerFullyConnected( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerLocalResponseNormalization( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerMaxout( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerPool( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerRectifiedLinearUnit( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerRegression( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerSigmoid( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerSoftmax( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerSupportVectorMachine( void ) {
        return true;
    }
    
    bool
    Dnn::addLayerTanh( void ) {
        return true;
    }
    
    void
    Dnn::randomize( void ) {
        // Randomize weights and bias.
        return;
    }
    
    bool
    Dnn::forward(  void ) {
        // Forward propagate while training
        return true;
    }
    
    bool
    Dnn::backward( void ) {
        // Back propagate while training
        return true;
    }
    
    bool
    Dnn::predict( void ) {
        // Forward progagate when predicting
        return true;
    }

    bool
    Dnn::save( const char *file_path ) const  {
        if( file_path == 0 || *file_path == 0 ) {
            return false;
        }
        return true;
    }
    
    bool
    Dnn::load( const char *file_path ) {
        if( file_path == 0 || *file_path == 0 ) {
            return false;
        }
        return true;
    }
    

    
}   // namespace tfs
