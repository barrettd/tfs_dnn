// --------------------------------------------------------------------
//  Dnn.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#include "Dnn.h"
#include "DnnLayers.h"
#include "Error.h"

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
            return log_error( "bad args" );
        }
        DnnLayerInput *layer = new DnnLayerInput( xx, yy, zz );
        m_layers.push_back( layer );
        
        return true;
    }
    
    bool
    Dnn::addLayerConvolution( unsigned long side, unsigned long filters, unsigned long stride, unsigned long pad ) {
        // ADD A Convolution Layer
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
    Dnn::addLayerPool( unsigned long side, unsigned long stride ) {
        return true;
    }
    
    bool
    Dnn::addLayerRectifiedLinearUnit( void ) {
        // Add a ReLu layer
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
        std::vector< DnnLayer* >::const_iterator layer_end = m_layers.end();
        for( std::vector< DnnLayer* >::const_iterator it = m_layers.begin(); it != layer_end; it++ ) {
            DnnLayer *layer = *it;
            if( layer != 0 ) {
                layer->randomize();
            }
        }
        return;
    }
    
    bool
    Dnn::forward(  void ) {
        // Forward propagate while training
        std::vector< DnnLayer* >::const_iterator layer_end = m_layers.end();
        for( std::vector< DnnLayer* >::const_iterator it = m_layers.begin(); it != layer_end; it++ ) {
            DnnLayer *layer = *it;
            if( layer != 0 ) {
                if( !layer->forward()) {
                    return log_error( "forward propagate failed" );
                }
            }
        }
        return true;
    }
    
    bool
    Dnn::backprop( void ) {
        // Back propagate while training
        std::vector< DnnLayer* >::const_iterator layer_end = m_layers.end();
        for( std::vector< DnnLayer* >::const_iterator it = m_layers.begin(); it != layer_end; it++ ) {
            DnnLayer *layer = *it;
            if( layer != 0 ) {
                if( !layer->backprop()) {
                    return log_error( "back propagate failed" );
                }
            }
        }
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
            return log_error( "bad file path" );
        }
        return true;
    }
    
    bool
    Dnn::load( const char *file_path ) {
        if( file_path == 0 || *file_path == 0 ) {
            return log_error( "bad file path" );
        }
        return true;
    }
    

    
}   // namespace tfs
