// --------------------------------------------------------------------
//  DnnBuilder.hpp
//  This is a helper class that add activation layers and fully connected
//  layers in some cases to simplify the DNN building process.
//
//  Created by Barrett Davis on 5/21/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "Dnn.hpp"

#ifndef DnnBuilder_hpp
#define DnnBuilder_hpp

namespace tfs {
    
    enum Activation {
        ACTIVATION_CURRENT = 0, // Use the DnnBuilder current activation type.
        ACTIVATION_SIGMOID,
        ACTIVATION_TANH,
        ACTIVATION_RELU,
        ACTIVATION_MAXOUT,
        ACTIVATION_DEFAULT = ACTIVATION_TANH
    };
    
    class DnnBuilder {
    protected:
        Dnn       &m_dnn;
        Activation m_activation;    // Current activation type
        
        bool addActivation( Activation activation );
        
    public:
        DnnBuilder( Dnn &dnn, Activation activation = ACTIVATION_DEFAULT );
       ~DnnBuilder( void );
        
        Activation activation( void ) const;
        Activation activation( Activation value );

        bool addLayerInput( unsigned long xx, unsigned long yy = 1, unsigned long zz = 1 );
        
        bool addLayerConvolution( unsigned long side, unsigned long filters, unsigned long stride = 1, unsigned long pad = 0, Activation activation = ACTIVATION_CURRENT );
        
        bool addLayerFullyConnected( unsigned long neuronCount, Activation activation = ACTIVATION_CURRENT );

        bool addLayerDropout( DNN_NUMERIC probability = 0.5 );
        bool addLayerLocalResponseNormalization( void );
        bool addLayerPool( unsigned long side, unsigned long stride = 1 );
        bool addLayerRegression( unsigned long neuronCount );
        
        bool addLayerSupportVectorMachine( unsigned long numberOfClasses );

        bool addLayerSoftmax( unsigned long numberOfClasses );
        
    };

}   // namespace tfs

#endif /* DnnBuilder_hpp */
