// --------------------------------------------------------------------
//  DnnLayerFullyConnected.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerFullyConnected_h
#define DnnLayerFullyConnected_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerFullyConnected : public DnnLayer {
    protected:
        unsigned long m_neuron_count;   // Always contains a bias for each neuron.
        
        void setup( const bool trainable = true );
        void zeroBiases( void );

    public:
        static const char *className( void );

        DnnLayerFullyConnected( DnnLayer *previousLayer, unsigned long neuronCount, const bool trainable = true );
        virtual ~DnnLayerFullyConnected( void );

        virtual void initialize( void );                // Zero activations, gradiant and randomize weights. Forward calling.

        unsigned long neuronCount( void ) const;
        
        virtual bool forward(  void );                  // Forward propagate while training
        virtual bool backprop( void );                  // Back propagate while training
        
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting

    };
    
    
}   // namespace tfs


#endif /* DnnLayerFullyConnected_h */
