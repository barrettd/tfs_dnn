// --------------------------------------------------------------------
//  DnnBuilder.cpp
//
//  Created by Barrett Davis on 5/21/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnBuilder.h"


namespace tfs {
    
    DnnBuilder::DnnBuilder( Dnn &dnn, Activation activation ):
    m_dnn( dnn ),
    m_activation( activation ) {
        // Constructor
        if( m_activation == ACTIVATION_CURRENT ) {
            m_activation = ACTIVATION_DEFAULT;
        }
    }
    
    DnnBuilder::~DnnBuilder( void ) {
        // Destructor
    }
        
    Activation
    DnnBuilder::activation( void ) const {
        return m_activation;
    }
    
    Activation
    DnnBuilder::activation( Activation value ) {
        if( value == ACTIVATION_CURRENT ) {
            value = ACTIVATION_DEFAULT;
        }
        return m_activation = value;
    }
        
    bool
    DnnBuilder::addLayerInput( unsigned long xx, unsigned long yy, unsigned long zz ) {
        return m_dnn.addLayerInput( xx, yy, zz );
    }
    
    bool
    DnnBuilder::addActivation( Activation activation ) {
        if( activation == ACTIVATION_CURRENT ) {
            activation = m_activation;
        }
        switch( activation ) {
            case ACTIVATION_SIGMOID:    return m_dnn.addLayerSigmoid();
            case ACTIVATION_TANH:       return m_dnn.addLayerTanh();
            case ACTIVATION_RELU:       return m_dnn.addLayerRectifiedLinearUnit();
            case ACTIVATION_MAXOUT:     return m_dnn.addLayerMaxout();
            default: log_error( "Unexpected actication type: %d", activation );
        }
        return log_error( "Unrecognized actication type" );
    }
    
    bool
    DnnBuilder::addLayerConvolution( unsigned long side, unsigned long filters, unsigned long stride, unsigned long pad, Activation activation ) {
        if( !m_dnn.addLayerConvolution( side, filters, stride, pad )) {
            return false;
        }
        return addActivation( activation );
    }
   
    bool
    DnnBuilder::addLayerDropout( void ) {
        return true;
    }
    
    bool
    DnnBuilder::addLayerFullyConnected( unsigned long neuronCount, Activation activation ) {
        if( neuronCount < 1 ) {
            return log_error( "Neuron Count of classes < 1" );
        }
        if( !m_dnn.addLayerFullyConnected( neuronCount )) {
            return false;
        }
        return addActivation( activation );
    }
    
    bool
    DnnBuilder::addLayerLocalResponseNormalization( void ) {
        return true;
    }
    
    bool
    DnnBuilder::addLayerMaxout( void ) {
        return true;
    }
    
    bool
    DnnBuilder::addLayerPool( unsigned long side, unsigned long stride ) {
        return m_dnn.addLayerPool( side, stride );
    }
    
    bool
    DnnBuilder::addLayerRegression( unsigned long neuronCount ) {
        if( neuronCount < 1 ) {
            return log_error( "Neuron Count of classes < 1" );
        }
        if( !m_dnn.addLayerFullyConnected( neuronCount )) {
            return false;
        }
        return m_dnn.addLayerRegression();
    }
    
    bool
    DnnBuilder::addLayerSupportVectorMachine( void ) {
        return true;
    }
    
    bool
    DnnBuilder::addLayerSoftmax( unsigned long numberOfClasses ) {
        if( numberOfClasses < 1 ) {
            return log_error( "Number of classes < 1" );
        }
        if( !m_dnn.addLayerFullyConnected( numberOfClasses )) {
            return false;
        }
        return m_dnn.addLayerSoftmax();
    }


}   // namespace tfs
