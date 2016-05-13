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
    
    class DnnLayerFullyConnected : public DnnLayer {    // Always contains a bias for each neuron.
    protected:
        unsigned long m_neuron_count;
        DNN_NUMERIC   m_l1_decay_mul;
        DNN_NUMERIC   m_l2_decay_mul;
        
    public:
        static const char *className( void );

        DnnLayerFullyConnected( DnnLayer *previousLayer, unsigned long neuronCount, const bool trainable = true );
        virtual ~DnnLayerFullyConnected( void );

        unsigned long neuronCount( void ) const;
        
        DNN_NUMERIC l1DecayMultiplier( void ) const;
        DNN_NUMERIC l1DecayMultiplier( DNN_NUMERIC value );

        DNN_NUMERIC l2DecayMultiplier( void ) const;
        DNN_NUMERIC l2DecayMultiplier( DNN_NUMERIC value );

        virtual bool forward( const Matrix &data );     // Forward propagate while training
        virtual bool backprop( void );                  // Back propagate while training
        
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting

    };
    
    
}   // namespace tfs


#endif /* DnnLayerFullyConnected_h */
